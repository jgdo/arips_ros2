#ifndef ARIPS_NAV2_BT_PLUGINS__POP_NEXT_EDGE_HPP_
#define ARIPS_NAV2_BT_PLUGINS__POP_NEXT_EDGE_HPP_

#include <string>

#include "behaviortree_cpp/action_node.h"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "nav2_msgs/msg/route.hpp"

namespace arips_nav2_bt_plugins
{

class PopNextEdge : public BT::ActionNodeBase
{
public:
  PopNextEdge(const std::string & xml_tag_name, const BT::NodeConfiguration & conf);

  static BT::PortsList providedPorts()
  {
    return {
      BT::BidirectionalPort<nav2_msgs::msg::Route>("route", "Route being traversed"),
      BT::OutputPort<geometry_msgs::msg::PoseStamped>("next_goal", "Pose of the next route node"),
      BT::OutputPort<std::string>("transition_type", "Type of the next transition"),
      BT::OutputPort<int>("door_id", "ID of the next route node")
    };
  }

  void halt() override {}
  BT::NodeStatus tick() override;
};

}  // namespace arips_nav2_bt_plugins

#endif  // ARIPS_NAV2_BT_PLUGINS__POP_NEXT_EDGE_HPP_