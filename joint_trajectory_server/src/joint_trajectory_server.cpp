#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include <control_msgs/action/follow_joint_trajectory.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <std_msgs/msg/string.hpp>
#include <urdf/model.hpp>

using FollowJT = control_msgs::action::FollowJointTrajectory;
using GoalHandle = rclcpp_action::ServerGoalHandle<FollowJT>;
using Vec = std::vector<double>;

namespace
{

// Piecewise cubic Hermite spline over all controllable joints.
struct Spline
{
    Vec times;
    std::vector<Vec> pos, vel;

    struct Sample { Vec p, v, a; };

    double duration() const { return times.back(); }

    Sample eval(double t) const
    {
        t = std::clamp(t, 0.0, duration());
        size_t k = std::upper_bound(times.begin(), times.end(), t) - times.begin();
        k = std::min<size_t>(k == 0 ? 0 : k - 1, times.size() - 2);
        const double h = times[k + 1] - times[k];
        const double u = (t - times[k]) / h;
        const double u2 = u * u, u3 = u2 * u;
        const size_t n = pos[k].size();
        Sample s{Vec(n), Vec(n), Vec(n)};
        for (size_t j = 0; j < n; ++j) {
            const double p0 = pos[k][j], p1 = pos[k + 1][j];
            const double m0 = vel[k][j], m1 = vel[k + 1][j];
            s.p[j] = (2 * u3 - 3 * u2 + 1) * p0 + (u3 - 2 * u2 + u) * h * m0 +
                     (-2 * u3 + 3 * u2) * p1 + (u3 - u2) * h * m1;
            s.v[j] = ((6 * u2 - 6 * u) * p0 + (-6 * u2 + 6 * u) * p1) / h +
                     (3 * u2 - 4 * u + 1) * m0 + (3 * u2 - 2 * u) * m1;
            s.a[j] = ((12 * u - 6) * p0 + (-12 * u + 6) * p1) / (h * h) +
                     ((6 * u - 4) * m0 + (6 * u - 2) * m1) / h;
        }
        return s;
    }
};

}  // namespace

class JointTrajectoryServer : public rclcpp::Node
{
public:
    JointTrajectoryServer() : Node("joint_trajectory_server")
    {
        joints_ = declare_parameter<std::vector<std::string>>("joints", std::vector<std::string>{});
        max_vel_ = declare_parameter<Vec>("max_velocity", Vec{});
        max_acc_ = declare_parameter<Vec>("max_acceleration", Vec{});
        rate_ = declare_parameter<double>("command_rate", 50.0);
        max_start_dev_ = declare_parameter<double>("max_start_deviation", 0.1);
        goal_tol_ = declare_parameter<double>("goal_tolerance", 0.02);
        progress_timeout_ = declare_parameter<double>("progress_timeout", 2.0);
        progress_thresh_ = declare_parameter<double>("progress_threshold", 0.002);
        state_timeout_ = declare_parameter<double>("state_timeout", 1.0);
        const auto action_name = declare_parameter<std::string>("action_name", "follow_joint_trajectory");
        const auto command_topic = declare_parameter<std::string>("command_topic", "commands");
        const auto states_topic = declare_parameter<std::string>("joint_states_topic", "joint_states");
        const auto desc_topic = declare_parameter<std::string>("robot_description_topic", "/robot_description");

        if (joints_.empty() || max_vel_.size() != joints_.size() || max_acc_.size() != joints_.size()) {
            throw std::runtime_error(
                "'joints', 'max_velocity' and 'max_acceleration' must be non-empty and of equal size");
        }
        if (rate_ <= 0.0) {
            throw std::runtime_error("command_rate must be positive");
        }
        for (size_t j = 0; j < joints_.size(); ++j) {
            if (max_vel_[j] <= 0.0 || max_acc_[j] <= 0.0) {
                throw std::runtime_error("max_velocity and max_acceleration must be positive");
            }
            index_[joints_[j]] = j;
        }
        limits_.resize(joints_.size());

        cmd_pub_ = create_publisher<std_msgs::msg::Float64MultiArray>(command_topic, 10);
        state_sub_ = create_subscription<sensor_msgs::msg::JointState>(
            states_topic, rclcpp::SensorDataQoS(),
            [this](sensor_msgs::msg::JointState::SharedPtr msg) { onJointState(*msg); });
        desc_sub_ = create_subscription<std_msgs::msg::String>(
            desc_topic, rclcpp::QoS(1).transient_local().reliable(),
            [this](std_msgs::msg::String::SharedPtr msg) { onDescription(msg->data); });

        server_ = rclcpp_action::create_server<FollowJT>(
            this, action_name,
            [this](const rclcpp_action::GoalUUID &, std::shared_ptr<const FollowJT::Goal> goal) {
                return handleGoal(*goal);
            },
            [](std::shared_ptr<GoalHandle>) { return rclcpp_action::CancelResponse::ACCEPT; },
            [this](std::shared_ptr<GoalHandle> gh) {
                executing_ = true;
                std::thread([this, gh] { execute(gh); }).detach();
            });
        RCLCPP_INFO(get_logger(), "Waiting for joint states and robot description");
    }

private:
    // ---------------------------------------------------------------- inputs
    void onJointState(const sensor_msgs::msg::JointState & msg)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        for (size_t i = 0; i < msg.name.size() && i < msg.position.size(); ++i) {
            if (index_.count(msg.name[i])) {
                current_[msg.name[i]] = msg.position[i];
            }
        }
        if (current_.size() == joints_.size()) {
            last_state_time_ = std::chrono::steady_clock::now();
        }
    }

    void onDescription(const std::string & xml)
    {
        urdf::Model model;
        if (!model.initString(xml)) {
            RCLCPP_ERROR(get_logger(), "Failed to parse robot description");
            return;
        }
        std::vector<std::optional<std::pair<double, double>>> limits(joints_.size());
        for (size_t j = 0; j < joints_.size(); ++j) {
            auto joint = model.getJoint(joints_[j]);
            if (!joint) {
                RCLCPP_ERROR(get_logger(), "Joint '%s' not found in robot description", joints_[j].c_str());
                return;
            }
            if ((joint->type == urdf::Joint::REVOLUTE || joint->type == urdf::Joint::PRISMATIC) && joint->limits) {
                limits[j] = std::make_pair(joint->limits->lower, joint->limits->upper);
            }
        }
        std::lock_guard<std::mutex> lock(mutex_);
        limits_ = limits;
        have_limits_ = true;
        RCLCPP_INFO(get_logger(), "Joint limits loaded from robot description");
    }

    std::optional<Vec> currentPositions()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (current_.size() != joints_.size()) {
            return std::nullopt;
        }
        if (std::chrono::steady_clock::now() - last_state_time_ > std::chrono::duration<double>(state_timeout_)) {
            return std::nullopt;
        }
        Vec p(joints_.size());
        for (size_t j = 0; j < joints_.size(); ++j) {
            p[j] = current_[joints_[j]];
        }
        return p;
    }

    // ------------------------------------------------------------ validation
    std::optional<Spline> buildSpline(const FollowJT::Goal & goal, const Vec & current, std::string & error)
    {
        const auto & traj = goal.trajectory;
        const size_t n = joints_.size();
        if (traj.points.empty()) {
            error = "trajectory is empty";
            return std::nullopt;
        }
        // map trajectory column -> controllable joint index
        std::vector<size_t> col(traj.joint_names.size());
        std::vector<bool> covered(n, false);
        for (size_t c = 0; c < traj.joint_names.size(); ++c) {
            auto it = index_.find(traj.joint_names[c]);
            if (it == index_.end() || covered[it->second]) {
                error = "unknown or duplicate joint '" + traj.joint_names[c] + "'";
                return std::nullopt;
            }
            col[c] = it->second;
            covered[it->second] = true;
        }

        Spline sp;
        if (traj.points.size() == 1) {
            const auto & pt = traj.points[0];
            if (pt.positions.size() != traj.joint_names.size()) {
                error = "point has wrong number of positions";
                return std::nullopt;
            }
            if (!pt.velocities.empty() && pt.velocities.size() != traj.joint_names.size()) {
                error = "point has wrong number of velocities";
                return std::nullopt;
            }
            Vec p = current;  // joints not in the trajectory hold their current position
            for (size_t c = 0; c < col.size(); ++c) {
                p[col[c]] = pt.positions[c];
            }
            double t = rclcpp::Duration(pt.time_from_start).seconds();
            if (t <= 0.0) {
                double min_t = 1.0 / rate_;
                for (size_t j = 0; j < n; ++j) {
                    const double d = std::abs(p[j] - current[j]);
                    min_t = std::max(min_t, 1.5 * d / max_vel_[j]);
                    min_t = std::max(min_t, std::sqrt(6.0 * d / max_acc_[j]));
                }
                t = min_t;
            } else if (t < 1.0 / rate_) {
                t = 1.0 / rate_;
            }
            sp.times = {0.0, t};
            sp.pos = {current, p};
            sp.vel = {Vec(n, 0.0), Vec(n, 0.0)};
            if (!pt.velocities.empty()) {
                for (size_t c = 0; c < col.size(); ++c) {
                    sp.vel[1][col[c]] = pt.velocities[c];
                }
            }
        } else {
            bool all_vel = true;
            const double t0 = rclcpp::Duration(traj.points.front().time_from_start).seconds();
            for (const auto & pt : traj.points) {
                if (pt.positions.size() != traj.joint_names.size()) {
                    error = "point has wrong number of positions";
                    return std::nullopt;
                }
                const double t = rclcpp::Duration(pt.time_from_start).seconds() - t0;
                if (!sp.times.empty() && t <= sp.times.back()) {
                    error = "time_from_start must be strictly increasing";
                    return std::nullopt;
                }
                Vec p = current;  // joints not in the trajectory hold their current position
                for (size_t c = 0; c < col.size(); ++c) {
                    p[col[c]] = pt.positions[c];
                }
                sp.times.push_back(t);
                sp.pos.push_back(p);
                all_vel = all_vel && pt.velocities.size() == traj.joint_names.size();
            }

            const size_t m = sp.times.size();
            sp.vel.assign(m, Vec(n, 0.0));
            for (size_t k = 0; k < m; ++k) {
                if (all_vel) {
                    for (size_t c = 0; c < col.size(); ++c) {
                        sp.vel[k][col[c]] = traj.points[k].velocities[c];
                    }
                } else if (k > 0 && k + 1 < m) {
                    for (size_t j = 0; j < n; ++j) {
                        sp.vel[k][j] = (sp.pos[k + 1][j] - sp.pos[k - 1][j]) / (sp.times[k + 1] - sp.times[k - 1]);
                    }
                }
            }

            for (size_t j = 0; j < n; ++j) {
                if (std::abs(sp.pos[0][j] - current[j]) > max_start_dev_) {
                    error = "start of trajectory too far from current state for joint '" + joints_[j] + "'";
                    return std::nullopt;
                }
            }
        }

        // Stretch time uniformly until velocity/acceleration limits hold.
        const double dt = 1.0 / rate_;
        for (int iter = 0; iter < 20; ++iter) {
            double scale = 1.0;
            for (double t = 0.0;; t += dt) {
                auto s = sp.eval(t);
                for (size_t j = 0; j < n; ++j) {
                    scale = std::max(scale, std::abs(s.v[j]) / max_vel_[j]);
                    scale = std::max(scale, std::sqrt(std::abs(s.a[j]) / max_acc_[j]));
                }
                if (t >= sp.duration()) {
                    break;
                }
            }
            if (scale <= 1.0 + 1e-9) {
                break;
            }
            scale *= 1.01;
            for (auto & t : sp.times) {
                t *= scale;
            }
            for (auto & v : sp.vel) {
                for (auto & x : v) {
                    x /= scale;
                }
            }
        }
        return sp;
    }

    bool checkLimits(const Spline & sp, std::string & error)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        const double dt = 1.0 / rate_;
        for (double t = 0.0;; t += dt) {
            auto s = sp.eval(t);
            for (size_t j = 0; j < joints_.size(); ++j) {
                if (limits_[j] && (s.p[j] < limits_[j]->first - 1e-9 || s.p[j] > limits_[j]->second + 1e-9)) {
                    error = "joint '" + joints_[j] + "' violates position limits at t=" + std::to_string(t);
                    return false;
                }
            }
            if (t >= sp.duration()) {
                break;
            }
        }
        return true;
    }

    rclcpp_action::GoalResponse handleGoal(const FollowJT::Goal & goal)
    {
        auto reject = [this](const std::string & why) {
            RCLCPP_WARN(get_logger(), "Rejecting trajectory: %s", why.c_str());
            return rclcpp_action::GoalResponse::REJECT;
        };
        if (executing_.load()) {
            return reject("another trajectory is being executed");
        }
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (!have_limits_) {
                return reject("robot description / joint limits not available");
            }
        }
        auto current = currentPositions();
        if (!current) {
            return reject("no recent joint states for all controllable joints");
        }
        std::string error;
        auto sp = buildSpline(goal, *current, error);
        if (!sp) {
            return reject(error);
        }
        if (!checkLimits(*sp, error)) {
            return reject(error);
        }
        return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
    }

    // ------------------------------------------------------------- execution
    void publish(const Vec & p)
    {
        std_msgs::msg::Float64MultiArray msg;
        msg.data = p;
        cmd_pub_->publish(msg);
    }

    // Publishes current positions as target so the arm stops trying to move.
    void holdCurrent()
    {
        if (auto cur = currentPositions()) {
            publish(*cur);
        }
    }

    void execute(std::shared_ptr<GoalHandle> gh)
    {
        using clock = std::chrono::steady_clock;
        clock::time_point execution_start = clock::now();
        auto result = std::make_shared<FollowJT::Result>();
        auto done = [&](int code, const std::string & msg) {
            const double execution_time = std::chrono::duration<double>(clock::now() - execution_start).count();
            if (msg.empty()) {
                RCLCPP_INFO(
                    get_logger(), "Trajectory execution finished successfully in %.3f s", execution_time);
            } else {
                RCLCPP_INFO(
                    get_logger(), "Trajectory execution finished after %.3f s: %s", execution_time, msg.c_str());
            }
            result->error_code = code;
            result->error_string = msg;
            executing_ = false;
        };

        auto current = currentPositions();
        std::string error = "no joint states";
        std::optional<Spline> sp;
        if (current) {
            sp = buildSpline(*gh->get_goal(), *current, error);
        }
        if (!sp) {
            done(FollowJT::Result::INVALID_GOAL, error);
            gh->abort(result);
            return;
        }

        const size_t n = joints_.size();
        const auto period = std::chrono::duration_cast<clock::duration>(std::chrono::duration<double>(1.0 / rate_));
        const Vec final_target = sp->pos.back();

        Vec ref_pos = *current;
        std::vector<clock::time_point> ref_time(n, clock::now());
        Vec prev_cmd = sp->pos.front();

        auto next = clock::now();
        const auto start = next;
        execution_start = start;
        RCLCPP_INFO(
            get_logger(), "Starting trajectory execution with %zu points; estimated duration %.3f s",
            gh->get_goal()->trajectory.points.size(), sp->duration());
        while (rclcpp::ok()) {
            if (gh->is_canceling()) {
                holdCurrent();
                done(FollowJT::Result::SUCCESSFUL, "canceled");
                gh->canceled(result);
                return;
            }
            const auto now = clock::now();
            const double t = std::chrono::duration<double>(now - start).count();
            const bool in_motion = t < sp->duration();
            auto s = sp->eval(t);
            publish(s.p);

            auto cur = currentPositions();
            if (!cur) {
                holdCurrent();
                done(FollowJT::Result::PATH_TOLERANCE_VIOLATED, "joint states lost");
                gh->abort(result);
                return;
            }

            bool reached = true;
            for (size_t j = 0; j < n; ++j) {
                const bool cmd_moving = std::abs(s.p[j] - prev_cmd[j]) > 1e-6;
                const bool off_target = std::abs((*cur)[j] - final_target[j]) > goal_tol_;
                reached = reached && !off_target;
                const bool should_move = in_motion ? cmd_moving : off_target;

                if (std::abs((*cur)[j] - ref_pos[j]) > progress_thresh_ || !should_move) {
                    ref_pos[j] = (*cur)[j];
                    ref_time[j] = now;
                } else if (std::chrono::duration<double>(now - ref_time[j]).count() > progress_timeout_) {
                    RCLCPP_ERROR(get_logger(), "Aborting: no progress on joint '%s'", joints_[j].c_str());
                    holdCurrent();
                    done(FollowJT::Result::GOAL_TOLERANCE_VIOLATED, "no progress on joint '" + joints_[j] + "'");
                    gh->abort(result);
                    return;
                }
            }
            prev_cmd = s.p;

            if (!in_motion && reached) {
                done(FollowJT::Result::SUCCESSFUL, "");
                gh->succeed(result);
                return;
            }

            next += period;
            std::this_thread::sleep_until(next);
        }
        done(FollowJT::Result::INVALID_GOAL, "shutdown");
    }

    // ---------------------------------------------------------------- members
    std::vector<std::string> joints_;
    Vec max_vel_, max_acc_;
    std::map<std::string, size_t> index_;
    double rate_, max_start_dev_, goal_tol_, progress_timeout_, progress_thresh_, state_timeout_;

    std::mutex mutex_;
    std::map<std::string, double> current_;
    std::chrono::steady_clock::time_point last_state_time_{};
    std::vector<std::optional<std::pair<double, double>>> limits_;
    bool have_limits_{false};
    std::atomic<bool> executing_{false};

    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr cmd_pub_;
    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr state_sub_;
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr desc_sub_;
    rclcpp_action::Server<FollowJT>::SharedPtr server_;
};

int main(int argc, char ** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<JointTrajectoryServer>();
    try {
        rclcpp::spin(node);
    } catch (const rclcpp::exceptions::RCLError &) {
    }
    return 0;
}
