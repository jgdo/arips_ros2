#include "arips_nav2_bt_plugins/pop_next_edge.hpp"

#include "behaviortree_cpp/bt_factory.h"
#include <iostream>

namespace arips_nav2_bt_plugins
{

PopNextEdge::PopNextEdge(
  const std::string & xml_tag_name,
  const BT::NodeConfiguration & conf)
: BT::ActionNodeBase(xml_tag_name, conf)
{
}

BT::NodeStatus PopNextEdge::tick()
{
  nav2_msgs::msg::Route route;
  if (!getInput("route", route) || route.nodes.empty()) {
    return BT::NodeStatus::FAILURE;
  }

  const auto node = route.nodes.front();
  route.nodes.erase(route.nodes.begin());

  geometry_msgs::msg::PoseStamped next_goal;
  next_goal.header = route.header;
  next_goal.pose.position = node.position;
  next_goal.pose.orientation.w = 1.0;

  const int door_id = static_cast<int>(node.nodeid);
  const std::string transition_type = "foobar";

  setOutput("route", route);
  setOutput("next_goal", next_goal);
  setOutput("transition_type", transition_type);
  setOutput("door_id", door_id);

  std::cout << "PopNextEdge: next_goal position=(" << next_goal.pose.position.x << ", "
            << next_goal.pose.position.y << ", " << next_goal.pose.position.z
            << "), transition_type=" << transition_type << ", door_id=" << door_id
            << std::endl;

  return BT::NodeStatus::SUCCESS;
}

}  // namespace arips_nav2_bt_plugins

BT_REGISTER_NODES(factory)
{
  factory.registerNodeType<arips_nav2_bt_plugins::PopNextEdge>("PopNextEdge");
}