#include "arips_nav2_bt_plugins/route_counter.hpp"

#include "behaviortree_cpp/bt_factory.h"
#include "rclcpp/rclcpp.hpp"

namespace arips_nav2_bt_plugins
{

RouteCounter::RouteCounter(
  const std::string & xml_tag_name,
  const BT::NodeConfiguration & conf)
: BT::ActionNodeBase(xml_tag_name, conf)
{
}

BT::NodeStatus RouteCounter::tick()
{
  nav2_msgs::msg::Route route;
  if (!getInput("route", route)) {
    return BT::NodeStatus::SUCCESS;
  }

  uint16_t node_id, edge_id;
  if(!getInput("next_node_id", node_id)) {
    return BT::NodeStatus::SUCCESS;
  }

  if(!getInput("current_edge_id", edge_id)) {
    return BT::NodeStatus::SUCCESS;
  }

  setOutput("edge_count", static_cast<int>(route.edges.size()));
  setOutput("node_count", static_cast<int>(route.nodes.size()));


  const auto route_clock = rclcpp::Time(route.header.stamp);

    std::cout << "RouteCounter node " << node_id << " on edge " << edge_id << std::endl;

  return route.nodes.size()? BT::NodeStatus::RUNNING : BT::NodeStatus::SUCCESS;
}

}  // namespace arips_nav2_bt_plugins

BT_REGISTER_NODES(factory)
{
  factory.registerNodeType<arips_nav2_bt_plugins::RouteCounter>("RouteCounter");
}