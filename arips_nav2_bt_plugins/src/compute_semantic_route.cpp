#include <memory>
#include <string>

#include "arips_semantic_map_msgs/srv/compute_semantic_route.hpp"
#include "behaviortree_cpp/bt_factory.h"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "nav2_behavior_tree/bt_service_node.hpp"

namespace arips_nav2_bt_plugins
{

class ComputeSemanticRoute : public nav2_behavior_tree::BtServiceNode<
    arips_semantic_map_msgs::srv::ComputeSemanticRoute>
{
public:
  using Service = arips_semantic_map_msgs::srv::ComputeSemanticRoute;

  ComputeSemanticRoute(const std::string & xml_tag_name, const BT::NodeConfiguration & conf);

  static BT::PortsList providedPorts()
  {
    return providedBasicPorts({
      BT::InputPort<geometry_msgs::msg::PoseStamped>("start", "Start pose for route planning"),
      BT::InputPort<geometry_msgs::msg::PoseStamped>("goal", "Goal pose for route planning"),
      BT::OutputPort<arips_semantic_map_msgs::msg::SemanticRoute>("semantic_route", "Computed semantic route"),
      BT::OutputPort<int>("error_code_id", "Route planning result code"),
      BT::OutputPort<std::string>("error_msg", "Route planning error message")
    });
  }

  void on_tick() override;
  BT::NodeStatus on_completion(std::shared_ptr<Service::Response> response) override;
};

ComputeSemanticRoute::ComputeSemanticRoute(
  const std::string & xml_tag_name,
  const BT::NodeConfiguration & conf)
: BtServiceNode(xml_tag_name, conf, "/compute_semantic_route")
{
}

void ComputeSemanticRoute::on_tick()
{
  if (!getInput("start", request_->start_pose) || !getInput("goal", request_->goal_pose)) {
    setOutput("error_code_id", Service::Response::UNKNOWN_ERROR);
    setOutput("error_msg", "Missing start or goal pose");
    should_send_request_ = false;
  }
}

BT::NodeStatus ComputeSemanticRoute::on_completion(std::shared_ptr<Service::Response> response)
{
  setOutput("semantic_route", response->semantic_route);
  setOutput("error_code_id", response->error_code);
  setOutput("error_msg", response->error_msg);
  return response->error_code == Service::Response::SUCCESS ?
    BT::NodeStatus::SUCCESS : BT::NodeStatus::FAILURE;
}

}  // namespace arips_nav2_bt_plugins

BT_REGISTER_NODES(factory)
{
  factory.registerNodeType<arips_nav2_bt_plugins::ComputeSemanticRoute>("ComputeSemanticRoute");
}