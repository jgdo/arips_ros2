#include <cstdint>
#include <iostream>
#include <string>

#include "behaviortree_cpp/action_node.h"
#include "behaviortree_cpp/bt_factory.h"
#include "nav2_msgs/msg/route.hpp"
#include "rclcpp/rclcpp.hpp"

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