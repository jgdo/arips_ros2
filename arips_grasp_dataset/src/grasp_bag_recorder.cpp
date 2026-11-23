// Records a fixed set of topics into one bag per episode, controlled by services.
//
// The subscriptions are generic (serialized messages, nothing is deserialized) and stay alive
// for the lifetime of the node, so every episode starts without a discovery gap. Messages are
// only written between ~/start and ~/stop. Each topic gets its own mutually exclusive callback
// group: topics are processed in parallel (a large image does not block the joint states),
// while the messages of one topic stay in order.
//
// A topic is subscribed as soon as it has a publisher. The subscription is reliable unless a
// publisher of the topic is best effort (a reliable subscription would not match it).
//
// Parameters:
//   topics           topics to record
//   storage_id       rosbag2 storage plugin (default mcap)
//   discovery_period_s  how often unsubscribed topics are looked up
//   queue_depth      subscription history depth
//
// Services:
//   ~/start  (arips_action_msgs/srv/StartBagRecording)  open a new bag; fails while a topic is
//            not subscribed yet or a bag is already open
//   ~/stop   (arips_action_msgs/srv/StopBagRecording)   close the bag, return per-topic counts
//            and first/last receive stamps (node clock, i.e. sim time with use_sim_time)

#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "arips_action_msgs/srv/start_bag_recording.hpp"
#include "arips_action_msgs/srv/stop_bag_recording.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rosbag2_cpp/writer.hpp"
#include "rosbag2_storage/storage_options.hpp"
#include "rosbag2_storage/topic_metadata.hpp"

using StartBagRecording = arips_action_msgs::srv::StartBagRecording;
using StopBagRecording = arips_action_msgs::srv::StopBagRecording;

namespace
{
const std::vector<std::string> kDefaultTopics = {
  "/feetech_joint_state_broadcaster/joint_states",
  "/arm_forward_position_controller/commands",
  "/kinect/rgb/image_raw/compressed",
  "/kinect/depth/image_raw/compressedDepth",
  "/kinect/rgb/camera_info",
  "/kinect/depth/camera_info",
};
}  // namespace

class GraspBagRecorder : public rclcpp::Node
{
public:
  GraspBagRecorder()
  : Node("grasp_bag_recorder")
  {
    topic_names_ = declare_parameter<std::vector<std::string>>("topics", kDefaultTopics);
    storage_id_ = declare_parameter<std::string>("storage_id", "mcap");
    queue_depth_ = declare_parameter<int>("queue_depth", 100);
    const double discovery_period = declare_parameter<double>("discovery_period_s", 0.5);

    for (const auto & name : topic_names_) {
      topics_[name] = std::make_shared<Topic>();
    }

    service_group_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    start_srv_ = create_service<StartBagRecording>(
      "~/start",
      [this](
        const std::shared_ptr<StartBagRecording::Request> request,
        std::shared_ptr<StartBagRecording::Response> response) {on_start(request, response);},
      rclcpp::ServicesQoS(), service_group_);
    stop_srv_ = create_service<StopBagRecording>(
      "~/stop",
      [this](
        const std::shared_ptr<StopBagRecording::Request>,
        std::shared_ptr<StopBagRecording::Response> response) {on_stop(response);},
      rclcpp::ServicesQoS(), service_group_);

    discovery_timer_ = create_wall_timer(
      std::chrono::duration<double>(discovery_period), [this]() {discover();},
      service_group_);
    discover();
    RCLCPP_INFO(get_logger(), "Grasp bag recorder ready, %zu topics", topic_names_.size());
  }

private:
  struct Topic
  {
    std::string type;
    rclcpp::GenericSubscription::SharedPtr subscription;
    rclcpp::CallbackGroup::SharedPtr group;
    rclcpp::QoS qos{10};
    // Statistics of the current bag, protected by bag_mutex_
    uint32_t count = 0;
    rclcpp::Time first_stamp;
    rclcpp::Time last_stamp;
  };

  void discover()
  {
    const auto names_and_types = get_topic_names_and_types();
    for (auto & [name, topic] : topics_) {
      if (topic->subscription) {
        continue;
      }
      const auto it = names_and_types.find(name);
      if (it == names_and_types.end() || it->second.empty()) {
        continue;
      }
      const auto publishers = get_publishers_info_by_topic(name);
      if (publishers.empty()) {
        continue;
      }
      bool reliable = true;
      for (const auto & info : publishers) {
        if (info.qos_profile().reliability() != rclcpp::ReliabilityPolicy::Reliable) {
          reliable = false;
        }
      }
      rclcpp::QoS qos(static_cast<size_t>(queue_depth_));
      if (reliable) {
        qos.reliable();
      } else {
        qos.best_effort();
      }
      topic->type = it->second.front();
      topic->qos = qos;
      topic->group = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
      rclcpp::SubscriptionOptions options;
      options.callback_group = topic->group;
      Topic * raw = topic.get();
      const std::string topic_name = name;
      topic->subscription = create_generic_subscription(
        name, topic->type, qos,
        [this, raw, topic_name](std::shared_ptr<const rclcpp::SerializedMessage> message) {
          on_message(topic_name, *raw, message);
        },
        options);
      RCLCPP_INFO(
        get_logger(), "Subscribed to %s [%s] (%s)", name.c_str(), topic->type.c_str(),
        reliable ? "reliable" : "best effort");
    }
  }

  void on_message(
    const std::string & name, Topic & topic,
    std::shared_ptr<const rclcpp::SerializedMessage> message)
  {
    const rclcpp::Time stamp = now();
    std::lock_guard<std::mutex> lock(bag_mutex_);
    if (!writer_) {
      return;
    }
    try {
      writer_->write(message, name, topic.type, stamp);
    } catch (const std::exception & error) {
      RCLCPP_ERROR(get_logger(), "Writing %s failed: %s", name.c_str(), error.what());
      write_errors_++;
      return;
    }
    if (topic.count == 0) {
      topic.first_stamp = stamp;
    }
    topic.last_stamp = stamp;
    topic.count++;
  }

  void on_start(
    const std::shared_ptr<StartBagRecording::Request> request,
    std::shared_ptr<StartBagRecording::Response> response)
  {
    std::vector<std::string> missing;
    for (const auto & [name, topic] : topics_) {
      if (!topic->subscription) {
        missing.push_back(name);
      }
    }
    if (!missing.empty()) {
      response->success = false;
      response->message = "Not subscribed yet (no publisher):";
      for (const auto & name : missing) {
        response->message += " " + name;
      }
      return;
    }

    std::lock_guard<std::mutex> lock(bag_mutex_);
    if (writer_) {
      response->success = false;
      response->message = "A bag is already open: " + uri_;
      return;
    }
    rosbag2_storage::StorageOptions storage;
    storage.uri = request->uri;
    storage.storage_id = storage_id_;
    auto writer = std::make_unique<rosbag2_cpp::Writer>();
    try {
      writer->open(storage);
      for (const auto & [name, topic] : topics_) {
        rosbag2_storage::TopicMetadata metadata;
        metadata.name = name;
        metadata.type = topic->type;
        metadata.serialization_format = "cdr";
        metadata.offered_qos_profiles = {topic->qos};
        writer->create_topic(metadata);
      }
    } catch (const std::exception & error) {
      response->success = false;
      response->message = std::string("Cannot open bag: ") + error.what();
      return;
    }
    for (auto & [name, topic] : topics_) {
      topic->count = 0;
    }
    write_errors_ = 0;
    uri_ = request->uri;
    writer_ = std::move(writer);
    response->success = true;
    response->message = "Recording to " + uri_;
    RCLCPP_INFO(get_logger(), "%s", response->message.c_str());
  }

  void on_stop(std::shared_ptr<StopBagRecording::Response> response)
  {
    std::unique_ptr<rosbag2_cpp::Writer> writer;
    {
      std::lock_guard<std::mutex> lock(bag_mutex_);
      if (!writer_) {
        response->success = false;
        response->message = "No bag is open";
        return;
      }
      writer = std::move(writer_);
      for (const auto & name : topic_names_) {
        const auto & topic = *topics_.at(name);
        response->topics.push_back(name);
        response->counts.push_back(topic.count);
        response->first_stamps.push_back(topic.count ? topic.first_stamp : rclcpp::Time(0));
        response->last_stamps.push_back(topic.count ? topic.last_stamp : rclcpp::Time(0));
      }
    }
    // Closing writes the remaining data and metadata.yaml; done outside the lock.
    try {
      writer->close();
    } catch (const std::exception & error) {
      response->success = false;
      response->message = std::string("Closing the bag failed: ") + error.what();
      return;
    }
    response->success = write_errors_ == 0;
    response->message = write_errors_ == 0 ?
      "Closed " + uri_ : "Closed " + uri_ + " with " + std::to_string(write_errors_) +
      " write errors";
    RCLCPP_INFO(get_logger(), "%s", response->message.c_str());
  }

  std::vector<std::string> topic_names_;
  std::string storage_id_;
  int64_t queue_depth_;
  std::map<std::string, std::shared_ptr<Topic>> topics_;

  std::mutex bag_mutex_;
  std::unique_ptr<rosbag2_cpp::Writer> writer_;
  std::string uri_;
  size_t write_errors_ = 0;

  rclcpp::CallbackGroup::SharedPtr service_group_;
  rclcpp::Service<StartBagRecording>::SharedPtr start_srv_;
  rclcpp::Service<StopBagRecording>::SharedPtr stop_srv_;
  rclcpp::TimerBase::SharedPtr discovery_timer_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<GraspBagRecorder>();
  rclcpp::executors::MultiThreadedExecutor executor(rclcpp::ExecutorOptions(), 4);
  executor.add_node(node);
  executor.spin();
  rclcpp::shutdown();
  return 0;
}
