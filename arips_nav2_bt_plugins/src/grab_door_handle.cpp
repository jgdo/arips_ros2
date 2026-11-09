#include <cstdint>
#include <string>

#include "arips_action_msgs/action/grab_door_handle.hpp"
#include "behaviortree_cpp/bt_factory.h"
#include "nav2_behavior_tree/bt_action_node.hpp"

namespace arips_nav2_bt_plugins
{

class GrabDoorHandle
  : public nav2_behavior_tree::BtActionNode<arips_action_msgs::action::GrabDoorHandle>
{
public:
  using Action = arips_action_msgs::action::GrabDoorHandle;

  GrabDoorHandle(
    const std::string & xml_tag_name,
    const BT::NodeConfiguration & conf);

  static BT::PortsList providedPorts()
  {
    return providedBasicPorts({
      BT::InputPort<std::string>(
        "prompt", "", "Policy prompt (empty = server default_prompt)"),
      BT::InputPort<double>(
        "max_duration_s", 0.0, "Max policy run time in seconds (0 = policy node default)")
    });
  }

  void on_tick() override;
  BT::NodeStatus on_success() override;
};

GrabDoorHandle::GrabDoorHandle(
  const std::string & xml_tag_name,
  const BT::NodeConfiguration & conf)
: BtActionNode(xml_tag_name, "/grab_door_handle", conf)
{
}

void GrabDoorHandle::on_tick()
{
  getInput("prompt", goal_.prompt);
  getInput("max_duration_s", goal_.max_duration_s);
}

BT::NodeStatus GrabDoorHandle::on_success()
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
  factory.registerNodeType<arips_nav2_bt_plugins::GrabDoorHandle>("GrabDoorHandle");
}
