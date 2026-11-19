#include <cstdint>
#include <string>

#include "arips_action_msgs/action/drive_forward.hpp"
#include "behaviortree_cpp/bt_factory.h"
#include "nav2_behavior_tree/bt_action_node.hpp"

namespace arips_nav2_bt_plugins
{

class DriveForward
  : public nav2_behavior_tree::BtActionNode<arips_action_msgs::action::DriveForward>
{
public:
  using Action = arips_action_msgs::action::DriveForward;

  DriveForward(const std::string & xml_tag_name, const BT::NodeConfiguration & conf);

  static BT::PortsList providedPorts()
  {
    return providedBasicPorts({
      BT::InputPort<double>("distance", 0.0, "Distance to drive forward [m] (0 = server default)")
    });
  }

  void on_tick() override;
  BT::NodeStatus on_success() override;
};

DriveForward::DriveForward(
  const std::string & xml_tag_name,
  const BT::NodeConfiguration & conf)
: BtActionNode(xml_tag_name, "/drive_forward", conf)
{
}

void DriveForward::on_tick()
{
  getInput("distance", goal_.distance);
}

BT::NodeStatus DriveForward::on_success()
{
  if (!result_.result) {
    return BT::NodeStatus::FAILURE;
  }
  setOutput("error_code_id", static_cast<uint16_t>(result_.result->error_code));
  setOutput("error_msg", result_.result->error_str);
  return result_.result->error_code == Action::Result::SUCCESS ?
    BT::NodeStatus::SUCCESS : BT::NodeStatus::FAILURE;
}

}  // namespace arips_nav2_bt_plugins

BT_REGISTER_NODES(factory)
{
  factory.registerNodeType<arips_nav2_bt_plugins::DriveForward>("DriveForward");
}
