#include <iostream>
#include <string>

#include "arips_semantic_map_msgs/msg/semantic_route.hpp"
#include "behaviortree_cpp/action_node.h"
#include "behaviortree_cpp/bt_factory.h"
#include "geometry_msgs/msg/pose_stamped.hpp"

namespace arips_nav2_bt_plugins
{

class PopNextSemanticRouteSegment : public BT::ActionNodeBase
{
public:
  PopNextSemanticRouteSegment(
    const std::string & xml_tag_name, const BT::NodeConfiguration & conf);

  static BT::PortsList providedPorts()
  {
    return {
      BT::BidirectionalPort<arips_semantic_map_msgs::msg::SemanticRoute>(
        "semantic_route", "Semantic route being traversed"),
      BT::OutputPort<geometry_msgs::msg::PoseStamped>("next_goal", "End pose of the next segment"),
      BT::OutputPort<std::string>("segment_type", "Type of the next segment"),
      BT::OutputPort<std::string>("metadata_json", "Metadata for the next segment")
    };
  }

  void halt() override {}
  BT::NodeStatus tick() override;
};

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