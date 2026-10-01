#include "arips_nav2_bt_plugins/pop_next_semantic_route_segment.hpp"

#include "behaviortree_cpp/bt_factory.h"
#include <iostream>

namespace arips_nav2_bt_plugins
{

PopNextSemanticRouteSegment::PopNextSemanticRouteSegment(
  const std::string & xml_tag_name,
  const BT::NodeConfiguration & conf)
: BT::ActionNodeBase(xml_tag_name, conf)
{
}

BT::NodeStatus PopNextSemanticRouteSegment::tick()
{
  arips_semantic_map_msgs::msg::SemanticRoute semantic_route;
  if (!getInput("semantic_route", semantic_route) || semantic_route.segments.empty()) {
    return BT::NodeStatus::FAILURE;
  }

  const auto segment = semantic_route.segments.front();
  semantic_route.segments.erase(semantic_route.segments.begin());

  setOutput("semantic_route", semantic_route);
  setOutput("next_goal", segment.end_pose);
  setOutput("segment_type", segment.segment_type);
  setOutput("metadata_json", segment.metadata_json);

  std::cout << "PopNextSemanticRouteSegment: next_goal position=("
            << segment.end_pose.pose.position.x << ", "
            << segment.end_pose.pose.position.y << ", "
            << segment.end_pose.pose.position.z << "), segment_type="
            << segment.segment_type << ", metadata_json=" << segment.metadata_json
            << std::endl;

  return BT::NodeStatus::SUCCESS;
}

}  // namespace arips_nav2_bt_plugins

BT_REGISTER_NODES(factory)
{
  factory.registerNodeType<arips_nav2_bt_plugins::PopNextSemanticRouteSegment>(
    "PopNextSemanticRouteSegment");
}