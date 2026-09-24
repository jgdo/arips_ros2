#ifndef ARIPS_SEMANTIC_MAP_RVIZ__ADD_DOOR_TOOL_HPP_
#define ARIPS_SEMANTIC_MAP_RVIZ__ADD_DOOR_TOOL_HPP_

#include <memory>

#include <OgreVector.h>

#include <QObject>

#include <arips_semantic_map_msgs/srv/add_door.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rviz_common/tool.hpp>

namespace rviz_rendering
{
class BillboardLine;
}

namespace arips_semantic_map_rviz
{

class AddDoorTool : public rviz_common::Tool
{
  Q_OBJECT

public:
  AddDoorTool();
  ~AddDoorTool() override;

  void onInitialize() override;
  void activate() override;
  void deactivate() override;
  int processMouseEvent(rviz_common::ViewportMouseEvent & event) override;

  void setReturnTool(rviz_common::Tool * tool);

private:
  enum class InteractionState
  {
    Idle,
    DrawingDoor,
    ChoosingAngle,
  };

  bool getMapGroundPoint(
    const rviz_common::ViewportMouseEvent & event,
    Ogre::Vector3 & point) const;
  void updateDoorPreview(const Ogre::Vector3 & extent);
  void updateAnglePreview(const Ogre::Vector3 & clicked_point);
  void clearPreview();
  void submitDoor(double open_angle_deg);
  void restoreReturnTool();

  rclcpp::Node::SharedPtr node_;
  rclcpp::Client<arips_semantic_map_msgs::srv::AddDoor>::SharedPtr add_door_client_;
  std::unique_ptr<rviz_rendering::BillboardLine> closed_line_;
  std::unique_ptr<rviz_rendering::BillboardLine> opened_line_;
  std::unique_ptr<rviz_rendering::BillboardLine> angle_arc_;
  rviz_common::Tool * return_tool_{nullptr};
  Ogre::Vector3 pivot_{Ogre::Vector3::ZERO};
  Ogre::Vector3 extent_{Ogre::Vector3::ZERO};
  InteractionState state_{InteractionState::Idle};
};

}  // namespace arips_semantic_map_rviz

#endif  // ARIPS_SEMANTIC_MAP_RVIZ__ADD_DOOR_TOOL_HPP_
