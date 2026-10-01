#ifndef ARIPS_NAV2_BT_PLUGINS__ROUTE_COUNTER_HPP_
#define ARIPS_NAV2_BT_PLUGINS__ROUTE_COUNTER_HPP_

#include <cstdint>
#include <string>

#include "behaviortree_cpp/action_node.h"
#include "nav2_msgs/msg/route.hpp"

namespace arips_nav2_bt_plugins
{

class RouteCounter : public BT::ActionNodeBase
{
public:
  RouteCounter(const std::string & xml_tag_name, const BT::NodeConfiguration & conf);

  static BT::PortsList providedPorts()
  {
    return {
      BT::InputPort<nav2_msgs::msg::Route>("route", "Route to count"),
      BT::InputPort<std::uint16_t>("next_node_id", "ID of the next node"),
      BT::InputPort<std::uint16_t>("current_edge_id", "ID of the current edge"),
      BT::OutputPort<int>("edge_count", "Number of edges in the route"),
      BT::OutputPort<int>("node_count", "Number of nodes in the route")
    };
  }

  void halt() override {}
  BT::NodeStatus tick() override;
};

}  // namespace arips_nav2_bt_plugins

#endif  // ARIPS_NAV2_BT_PLUGINS__ROUTE_COUNTER_HPP_