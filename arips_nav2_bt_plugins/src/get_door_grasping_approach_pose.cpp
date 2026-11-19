#include <exception>
#include <memory>
#include <string>

#include "arips_action_msgs/srv/get_door_grasping_approach_pose.hpp"
#include "behaviortree_cpp/bt_factory.h"
#include "door_metadata.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "nav2_behavior_tree/bt_service_node.hpp"

namespace arips_nav2_bt_plugins
{

class GetDoorGraspingApproachPose : public nav2_behavior_tree::BtServiceNode<
    arips_action_msgs::srv::GetDoorGraspingApproachPose>
{
public:
  using Service = arips_action_msgs::srv::GetDoorGraspingApproachPose;

  GetDoorGraspingApproachPose(
    const std::string & xml_tag_name, const BT::NodeConfiguration & conf);

  static BT::PortsList providedPorts()
  {
    return providedBasicPorts({
      BT::InputPort<std::string>("metadata_json", "Door metadata as JSON"),
      BT::OutputPort<geometry_msgs::msg::PoseStamped>(
        "approach_pose", "Door handle grasping approach pose (map frame)"),
      BT::OutputPort<std::string>("error_msg", "Error message")
    });
  }

  void on_tick() override;
  BT::NodeStatus on_completion(std::shared_ptr<Service::Response> response) override;
};

GetDoorGraspingApproachPose::GetDoorGraspingApproachPose(
  const std::string & xml_tag_name,
  const BT::NodeConfiguration & conf)
: BtServiceNode(xml_tag_name, conf, "/get_door_grasping_approach_pose")
{
}

void GetDoorGraspingApproachPose::on_tick()
{
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

BT::NodeStatus GetDoorGraspingApproachPose::on_completion(
  std::shared_ptr<Service::Response> response)
{
  setOutput("error_msg", response->error_message);
  if (!response->success) {
    return BT::NodeStatus::FAILURE;
  }
  setOutput("approach_pose", response->pose);
  return BT::NodeStatus::SUCCESS;
}

}  // namespace arips_nav2_bt_plugins

BT_REGISTER_NODES(factory)
{
  factory.registerNodeType<arips_nav2_bt_plugins::GetDoorGraspingApproachPose>(
    "GetDoorGraspingApproachPose");
}
