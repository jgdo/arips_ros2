#include <cmath>
#include <string>

#include "behaviortree_cpp/action_node.h"
#include "behaviortree_cpp/bt_factory.h"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "nav2_behavior_tree/bt_utils.hpp"

namespace arips_nav2_bt_plugins
{

// Shifts a pose by (dx, dy) in its own frame, i.e. along its x and y axes.
// The orientation is kept. Assumes a planar pose (rotation about z only).
class OffsetPose : public BT::SyncActionNode
{
public:
  OffsetPose(const std::string & name, const BT::NodeConfiguration & conf)
  : BT::SyncActionNode(name, conf)
  {
  }

  static BT::PortsList providedPorts()
  {
    return {
      BT::InputPort<geometry_msgs::msg::PoseStamped>("input_pose", "Pose to shift"),
      BT::InputPort<double>("dx", 0.0, "Shift along the pose's x axis [m]"),
      BT::InputPort<double>("dy", 0.0, "Shift along the pose's y axis [m]"),
      BT::OutputPort<geometry_msgs::msg::PoseStamped>("output_pose", "Shifted pose")
    };
  }

  BT::NodeStatus tick() override
  {
    geometry_msgs::msg::PoseStamped pose;
    if (!getInput("input_pose", pose)) {
      return BT::NodeStatus::FAILURE;
    }
    double dx = 0.0;
    double dy = 0.0;
    getInput("dx", dx);
    getInput("dy", dy);

    const auto & q = pose.pose.orientation;
    const double yaw = std::atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z));
    pose.pose.position.x += dx * std::cos(yaw) - dy * std::sin(yaw);
    pose.pose.position.y += dx * std::sin(yaw) + dy * std::cos(yaw);

    setOutput("output_pose", pose);
    return BT::NodeStatus::SUCCESS;
  }
};

}  // namespace arips_nav2_bt_plugins

BT_REGISTER_NODES(factory)
{
  factory.registerNodeType<arips_nav2_bt_plugins::OffsetPose>("OffsetPose");
}
