#include <cstdint>
#include <exception>
#include <string>

#include "arips_action_msgs/action/cross_door_step.hpp"
#include "behaviortree_cpp/bt_factory.h"
#include "nav2_behavior_tree/bt_action_node.hpp"
#include "nlohmann/json.hpp"

namespace arips_nav2_bt_plugins
{

class CrossDoorStep : public nav2_behavior_tree::BtActionNode<arips_action_msgs::action::CrossDoorStep>
{
public:
  using Action = arips_action_msgs::action::CrossDoorStep;

  CrossDoorStep(
    const std::string & xml_tag_name,
    const BT::NodeConfiguration & conf);

  static BT::PortsList providedPorts()
  {
    return providedBasicPorts({
      BT::InputPort<std::string>("segment_type", "Semantic route segment type"),
      BT::InputPort<std::string>("metadata_json", "Door metadata as JSON")
    });
  }

  void on_tick() override;
  BT::NodeStatus on_success() override;
};

CrossDoorStep::CrossDoorStep(
  const std::string & xml_tag_name,
  const BT::NodeConfiguration & conf)
: BtActionNode(xml_tag_name, "/cross_door_step", conf)
{
}

void CrossDoorStep::on_tick()
{
  std::string segment_type;
  std::string metadata_json;
  if (!getInput("segment_type", segment_type) || segment_type != "door") {
    setOutput("error_msg", "CrossDoorStep requires segment_type 'door'");
    should_send_goal_ = false;
    return;
  }

  if (!getInput("metadata_json", metadata_json)) {
    setOutput("error_msg", "Missing metadata_json input");
    should_send_goal_ = false;
    return;
  }

  try {
    const auto metadata = nlohmann::json::parse(metadata_json);
    goal_.door.pivot.x = metadata.at("pivot").at("x").get<double>();
    goal_.door.pivot.y = metadata.at("pivot").at("y").get<double>();
    goal_.door.extent.x = metadata.at("extent").at("x").get<double>();
    goal_.door.extent.y = metadata.at("extent").at("y").get<double>();
    goal_.door.open_angle_deg = metadata.at("open_angle_deg").get<double>();
  } catch (const std::exception & error) {
    setOutput("error_msg", std::string("Invalid door metadata_json: ") + error.what());
    setOutput("error_code_id", static_cast<uint16_t>(Action::Result::FAILURE));
    should_send_goal_ = false;
  }
}

BT::NodeStatus CrossDoorStep::on_success()
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
  factory.registerNodeType<arips_nav2_bt_plugins::CrossDoorStep>("CrossDoorStep");
}