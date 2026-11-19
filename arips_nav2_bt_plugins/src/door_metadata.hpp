#pragma once

#include <string>

#include "arips_semantic_map_msgs/msg/door.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "nlohmann/json.hpp"

namespace arips_nav2_bt_plugins
{

// Parse the metadata_json of a 'door' semantic route segment, e.g.
// {"door_id": 0, "pivot": {"x":..,"y":..}, "extent": {"x":..,"y":..}, "open_angle_deg": ..}
// Throws std::exception on invalid input.
inline arips_semantic_map_msgs::msg::Door parse_door_metadata(const std::string & metadata_json)
{
  const auto metadata = nlohmann::json::parse(metadata_json);
  arips_semantic_map_msgs::msg::Door door;
  door.pivot.x = metadata.at("pivot").at("x").get<double>();
  door.pivot.y = metadata.at("pivot").at("y").get<double>();
  door.extent.x = metadata.at("extent").at("x").get<double>();
  door.extent.y = metadata.at("extent").at("y").get<double>();
  door.open_angle_deg = metadata.at("open_angle_deg").get<double>();
  return door;
}

// Door pivot as a pose in the map frame (zero stamp = latest transform).
inline geometry_msgs::msg::PoseStamped door_pivot_pose(
  const arips_semantic_map_msgs::msg::Door & door)
{
  geometry_msgs::msg::PoseStamped pose;
  pose.header.frame_id = "map";
  pose.pose.position.x = door.pivot.x;
  pose.pose.position.y = door.pivot.y;
  pose.pose.orientation.w = 1.0;
  return pose;
}

}  // namespace arips_nav2_bt_plugins
