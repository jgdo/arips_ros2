#include <exception>
#include <memory>
#include <string>

#include "arips_action_msgs/srv/get_door_state.hpp"
#include "behaviortree_cpp/bt_factory.h"
#include "door_metadata.hpp"
#include "nav2_behavior_tree/bt_service_node.hpp"

namespace arips_nav2_bt_plugins
{

// Asks the door state detector (laser based) whether the door of the current semantic route
// segment is open. SUCCESS when the state could be determined (see door_open), FAILURE if
// the door could not be observed well enough or the request failed. door_open is set to
// false at every tick and only becomes true if the door was detected open, so on FAILURE
// it is false (i.e. the caller falls back to opening the door).
class GetDoorState : public nav2_behavior_tree::BtServiceNode<
    arips_action_msgs::srv::GetDoorState>
{
public:
  using Service = arips_action_msgs::srv::GetDoorState;

  GetDoorState(const std::string & xml_tag_name, const BT::NodeConfiguration & conf)
  : BtServiceNode(xml_tag_name, conf, "/get_door_state")
  {
  }

  static BT::PortsList providedPorts()
  {
    return providedBasicPorts({
      BT::InputPort<std::string>("metadata_json", "Door metadata as JSON"),
      BT::OutputPort<bool>("door_open", "True if the door is open"),
      BT::OutputPort<double>("door_angle_deg", "Estimated door leaf angle [deg]"),
      BT::OutputPort<std::string>("error_msg", "Error message")
    });
  }

  void on_tick() override
  {
    setOutput("door_open", false);
    std::string metadata_json;
    if (!getInput("metadata_json", metadata_json)) {
      setOutput("error_msg", "Missing metadata_json input");
      should_send_request_ = false;
      return;
    }
    try {
      request_->door = parse_door_metadata(metadata_json);
    } catch (const std::exception & error) {
      setOutput("error_msg", std::string("Invalid door metadata_json: ") + error.what());
      should_send_request_ = false;
    }
  }

  BT::NodeStatus on_completion(std::shared_ptr<Service::Response> response) override
  {
    setOutput("error_msg", response->error_message);
    if (!response->success) {
      return BT::NodeStatus::FAILURE;
    }
    setOutput("door_open", response->is_open);
    setOutput("door_angle_deg", response->angle_deg);
    return BT::NodeStatus::SUCCESS;
  }
};

}  // namespace arips_nav2_bt_plugins

BT_REGISTER_NODES(factory)
{
  factory.registerNodeType<arips_nav2_bt_plugins::GetDoorState>("GetDoorState");
}
