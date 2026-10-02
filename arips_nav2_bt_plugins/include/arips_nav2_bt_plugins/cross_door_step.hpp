#ifndef ARIPS_NAV2_BT_PLUGINS__CROSS_DOOR_STEP_HPP_
#define ARIPS_NAV2_BT_PLUGINS__CROSS_DOOR_STEP_HPP_

#include <string>

#include "arips_action_msgs/action/cross_door_step.hpp"
#include "nav2_behavior_tree/bt_action_node.hpp"

namespace arips_nav2_bt_plugins
{

class CrossDoorStep : public nav2_behavior_tree::BtActionNode<arips_action_msgs::action::CrossDoorStep>
{
public:
  using Action = arips_action_msgs::action::CrossDoorStep;

  CrossDoorStep(
    const std::string & xml_tag_name,
    const BT::NodeConfiguration & conf);

  static BT::PortsList providedPorts()
  {
    return providedBasicPorts({
      BT::InputPort<std::string>("segment_type", "Semantic route segment type"),
      BT::InputPort<std::string>("metadata_json", "Door metadata as JSON")
    });
  }

  void on_tick() override;
  BT::NodeStatus on_success() override;
};

}  // namespace arips_nav2_bt_plugins

#endif  // ARIPS_NAV2_BT_PLUGINS__CROSS_DOOR_STEP_HPP_