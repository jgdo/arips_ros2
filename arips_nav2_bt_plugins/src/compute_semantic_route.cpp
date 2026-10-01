#include "arips_nav2_bt_plugins/compute_semantic_route.hpp"

#include "behaviortree_cpp/bt_factory.h"

#include <memory>

namespace arips_nav2_bt_plugins
{

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