#ifndef ARIPS_NAV2_BT_PLUGINS__POP_NEXT_SEMANTIC_ROUTE_SEGMENT_HPP_
#define ARIPS_NAV2_BT_PLUGINS__POP_NEXT_SEMANTIC_ROUTE_SEGMENT_HPP_

#include <string>

#include "arips_semantic_map_msgs/msg/semantic_route.hpp"
#include "behaviortree_cpp/action_node.h"
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

}  // namespace arips_nav2_bt_plugins

#endif  // ARIPS_NAV2_BT_PLUGINS__POP_NEXT_SEMANTIC_ROUTE_SEGMENT_HPP_