#include <cstdint>
#include <string>

#include "arips_action_msgs/action/move_arm.hpp"
#include "behaviortree_cpp/bt_factory.h"
#include "nav2_behavior_tree/bt_action_node.hpp"

namespace arips_nav2_bt_plugins
{

// Moves the arm to an SRDF group state (named_target) or the door gripper in a straight
// line along its pointing direction (linear_distance, used when named_target is empty).
class MoveArm
  : public nav2_behavior_tree::BtActionNode<arips_action_msgs::action::MoveArm>
{
public:
  using Action = arips_action_msgs::action::MoveArm;

  MoveArm(const std::string & xml_tag_name, const BT::NodeConfiguration & conf)
  : BtActionNode(xml_tag_name, "/move_arm", conf)
  {
  }

  static BT::PortsList providedPorts()
  {
    return providedBasicPorts({
      BT::InputPort<std::string>("named_target", "", "SRDF group state to move to"),
      BT::InputPort<double>(
        "linear_distance", 0.0,
        "Straight gripper move along its pointing direction [m] (if named_target is empty)")
    });
  }

  void on_tick() override
  {
    getInput("named_target", goal_.named_target);
    getInput("linear_distance", goal_.linear_distance);
  }

  BT::NodeStatus on_success() override
  {
    if (!result_.result) {
      return BT::NodeStatus::FAILURE;
    }
    setOutput("error_code_id", static_cast<uint16_t>(result_.result->error_code));
    setOutput("error_msg", result_.result->error_str);
    return result_.result->error_code == Action::Result::SUCCESS ?
           BT::NodeStatus::SUCCESS : BT::NodeStatus::FAILURE;
  }
};

}  // namespace arips_nav2_bt_plugins

BT_REGISTER_NODES(factory)
{
  factory.registerNodeType<arips_nav2_bt_plugins::MoveArm>("MoveArm");
}
