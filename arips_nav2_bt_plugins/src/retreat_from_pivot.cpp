#include <cstdint>
#include <exception>
#include <string>

#include "arips_action_msgs/action/retreat_from_pivot.hpp"
#include "behaviortree_cpp/bt_factory.h"
#include "door_metadata.hpp"
#include "nav2_behavior_tree/bt_action_node.hpp"

namespace arips_nav2_bt_plugins
{

// Releases the door handle, folds the arm and drives away from the door pivot.
class RetreatFromPivot
  : public nav2_behavior_tree::BtActionNode<arips_action_msgs::action::RetreatFromPivot>
{
public:
  using Action = arips_action_msgs::action::RetreatFromPivot;

  RetreatFromPivot(const std::string & xml_tag_name, const BT::NodeConfiguration & conf);

  static BT::PortsList providedPorts()
  {
    return providedBasicPorts({
      BT::InputPort<std::string>("metadata_json", "Door metadata as JSON"),
      BT::InputPort<double>(
        "forward_distance", 0.0, "Distance to drive away from the pivot [m] (0 = server default)")
    });
  }

  void on_tick() override;
  BT::NodeStatus on_success() override;
};

RetreatFromPivot::RetreatFromPivot(
  const std::string & xml_tag_name,
  const BT::NodeConfiguration & conf)
: BtActionNode(xml_tag_name, "/retreat_from_pivot", conf)
{
}

void RetreatFromPivot::on_tick()
{
  std::string metadata_json;
  if (!getInput("metadata_json", metadata_json)) {
    setOutput("error_msg", "Missing metadata_json input");
    should_send_goal_ = false;
    return;
  }
  try {
    goal_.pivot = door_pivot_pose(parse_door_metadata(metadata_json));
  } catch (const std::exception & error) {
    setOutput("error_msg", std::string("Invalid door metadata_json: ") + error.what());
    setOutput("error_code_id", static_cast<uint16_t>(Action::Result::FAILURE));
    should_send_goal_ = false;
    return;
  }
  getInput("forward_distance", goal_.forward_distance);
}

BT::NodeStatus RetreatFromPivot::on_success()
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
  factory.registerNodeType<arips_nav2_bt_plugins::RetreatFromPivot>("RetreatFromPivot");
}
