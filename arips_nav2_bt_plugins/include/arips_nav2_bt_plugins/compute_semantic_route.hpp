#ifndef ARIPS_NAV2_BT_PLUGINS__COMPUTE_SEMANTIC_ROUTE_HPP_
#define ARIPS_NAV2_BT_PLUGINS__COMPUTE_SEMANTIC_ROUTE_HPP_

#include <string>

#include "arips_semantic_map_msgs/srv/compute_semantic_route.hpp"
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

}  // namespace arips_nav2_bt_plugins

#endif  // ARIPS_NAV2_BT_PLUGINS__COMPUTE_SEMANTIC_ROUTE_HPP_