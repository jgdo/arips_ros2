#include "arips_semantic_map_rviz/add_door_tool.hpp"

#include <algorithm>
#include <cmath>
#include <utility>

#include <OgreCamera.h>
#include <OgrePlane.h>
#include <OgreSceneManager.h>
#include <OgreSceneNode.h>
#include <OgreRay.h>

#include <QEvent>

#include <pluginlib/class_list_macros.hpp>
#include <rviz_common/display_context.hpp>
#include <rviz_common/render_panel.hpp>
#include <rviz_common/tool_manager.hpp>
#include <rviz_common/view_controller.hpp>
#include <rviz_common/view_manager.hpp>
#include <rviz_common/viewport_mouse_event.hpp>
#include <rviz_common/ros_integration/ros_node_abstraction_iface.hpp>
#include <rviz_rendering/objects/billboard_line.hpp>

namespace arips_semantic_map_rviz
{

namespace
{
constexpr double kClickDistance = 0.3;
constexpr double kClickExtent = 0.8;
constexpr double kMinimumAnglePointDistance = 1e-6;
constexpr double kArcStepDegrees = 5.0;
constexpr double kPi = 3.14159265358979323846;
}

AddDoorTool::AddDoorTool()
{
  shortcut_key_ = 'd';
  setName("Add door");
  setDescription("Draw a door on the map ground plane");
}

AddDoorTool::~AddDoorTool() = default;

void AddDoorTool::onInitialize()
{
  auto node_abstraction = context_->getRosNodeAbstraction().lock();
  if (node_abstraction) {
    node_ = node_abstraction->get_raw_node();
    add_door_client_ = node_->create_client<arips_semantic_map_msgs::srv::AddDoor>(
      "/semantic_map_server/add_door");
  }

  closed_line_ = std::make_unique<rviz_rendering::BillboardLine>(scene_manager_);
  closed_line_->setLineWidth(0.04f);
  closed_line_->setColor(1.0f, 0.75f, 0.1f, 1.0f);
  closed_line_->getSceneNode()->setVisible(false);

  opened_line_ = std::make_unique<rviz_rendering::BillboardLine>(scene_manager_);
  opened_line_->setLineWidth(0.03f);
  opened_line_->setColor(0.2f, 1.0f, 0.3f, 1.0f);
  opened_line_->getSceneNode()->setVisible(false);

  angle_arc_ = std::make_unique<rviz_rendering::BillboardLine>(scene_manager_);
  angle_arc_->setLineWidth(0.02f);
  angle_arc_->setColor(1.0f, 0.4f, 0.1f, 1.0f);
  angle_arc_->getSceneNode()->setVisible(false);
}

void AddDoorTool::activate()
{
  state_ = InteractionState::Idle;
  clearPreview();
  setStatus("Click and drag on the map to add a door");
}

void AddDoorTool::deactivate()
{
  state_ = InteractionState::Idle;
  clearPreview();
}

void AddDoorTool::setReturnTool(rviz_common::Tool * tool)
{
  return_tool_ = tool;
}

bool AddDoorTool::getMapGroundPoint(
  const rviz_common::ViewportMouseEvent & event,
  Ogre::Vector3 & point) const
{
  if (context_ == nullptr || event.panel == nullptr ||
    context_->getFixedFrame() != QStringLiteral("map"))
  {
    return false;
  }

  auto * view_controller = event.panel->getViewController();
  if (view_controller == nullptr || view_controller->getCamera() == nullptr) {
    return false;
  }

  const float width = static_cast<float>(event.panel->width());
  const float height = static_cast<float>(event.panel->height());
  if (width <= 0.0f || height <= 0.0f) {
    return false;
  }

  const Ogre::Real viewport_x = static_cast<Ogre::Real>(event.x) / width;
  const Ogre::Real viewport_y = static_cast<Ogre::Real>(event.y) / height;
  const Ogre::Ray ray = view_controller->getCamera()->getCameraToViewportRay(
    viewport_x, viewport_y);
  const auto intersection = ray.intersects(Ogre::Plane(Ogre::Vector3::UNIT_Z, 0.0f));
  if (!intersection.first || intersection.second < 0.0f) {
    return false;
  }

  point = ray.getPoint(intersection.second);
  point.z = 0.0f;
  return true;
}

void AddDoorTool::updateDoorPreview(const Ogre::Vector3 & extent)
{
  if (!closed_line_) {
    return;
  }

  closed_line_->clear();
  closed_line_->addPoint(pivot_);
  closed_line_->addPoint(extent);
  closed_line_->finishLine();
  closed_line_->getSceneNode()->setVisible(true);
  opened_line_->getSceneNode()->setVisible(false);
  angle_arc_->getSceneNode()->setVisible(false);
}

void AddDoorTool::updateAnglePreview(const Ogre::Vector3 & clicked_point)
{
  if (!closed_line_ || !opened_line_ || !angle_arc_) {
    return;
  }

  const Ogre::Vector3 closed_vector = extent_ - pivot_;
  const Ogre::Vector3 clicked_vector = clicked_point - pivot_;
  const double clicked_distance = std::hypot(clicked_vector.x, clicked_vector.y);
  if (clicked_distance < kMinimumAnglePointDistance) {
    opened_line_->getSceneNode()->setVisible(false);
    angle_arc_->getSceneNode()->setVisible(false);
    return;
  }

  const double angle_rad = std::atan2(
    closed_vector.x * clicked_vector.y - closed_vector.y * clicked_vector.x,
    closed_vector.x * clicked_vector.x + closed_vector.y * clicked_vector.y);
  const double closed_angle_rad = std::atan2(closed_vector.y, closed_vector.x);
  const double closed_length = std::hypot(closed_vector.x, closed_vector.y);
  const Ogre::Vector3 opened_vector(
    static_cast<float>(closed_vector.x * std::cos(angle_rad) - closed_vector.y * std::sin(angle_rad)),
    static_cast<float>(closed_vector.x * std::sin(angle_rad) + closed_vector.y * std::cos(angle_rad)),
    0.0f);

  opened_line_->clear();
  opened_line_->addPoint(pivot_);
  opened_line_->addPoint(pivot_ + opened_vector);
  opened_line_->finishLine();
  opened_line_->getSceneNode()->setVisible(true);

  angle_arc_->clear();
  const int arc_steps = std::max(
    1, static_cast<int>(std::ceil(std::abs(angle_rad) * 180.0 / kPi / kArcStepDegrees)));
  for (int step = 0; step <= arc_steps; ++step) {
    const double step_angle = closed_angle_rad + angle_rad * static_cast<double>(step) /
      static_cast<double>(arc_steps);
    const Ogre::Vector3 arc_point(
      static_cast<float>(pivot_.x + closed_length * std::cos(step_angle)),
      static_cast<float>(pivot_.y + closed_length * std::sin(step_angle)),
      0.0f);
    angle_arc_->addPoint(arc_point);
  }
  angle_arc_->finishLine();
  angle_arc_->getSceneNode()->setVisible(true);
}

void AddDoorTool::clearPreview()
{
  if (!closed_line_ || !opened_line_ || !angle_arc_) {
    return;
  }

  closed_line_->clear();
  opened_line_->clear();
  angle_arc_->clear();
  closed_line_->getSceneNode()->setVisible(false);
  opened_line_->getSceneNode()->setVisible(false);
  angle_arc_->getSceneNode()->setVisible(false);
}

void AddDoorTool::submitDoor(double open_angle_deg)
{
  if (!add_door_client_ || !add_door_client_->service_is_ready()) {
    if (node_) {
      RCLCPP_WARN(node_->get_logger(), "Semantic map add-door service is unavailable");
    }
    return;
  }

  auto request = std::make_shared<arips_semantic_map_msgs::srv::AddDoor::Request>();
  request->door.pivot.x = pivot_.x;
  request->door.pivot.y = pivot_.y;
  request->door.extent.x = extent_.x;
  request->door.extent.y = extent_.y;
  request->door.open_angle_deg = open_angle_deg;

  add_door_client_->async_send_request(
    request,
    [node = node_](rclcpp::Client<arips_semantic_map_msgs::srv::AddDoor>::SharedFuture future) {
      const auto response = future.get();
      if (!response->success) {
        if (node) {
          RCLCPP_WARN(node->get_logger(), "Could not add semantic map door: %s",
            response->error_message.c_str());
        }
        return;
      }

      if (node) {
        RCLCPP_INFO(node->get_logger(), "Added semantic map door %s",
          response->new_door_name.c_str());
      }
    });
}

void AddDoorTool::restoreReturnTool()
{
  if (context_ == nullptr) {
    return;
  }

  auto * tool_manager = context_->getToolManager();
  auto * target = return_tool_;
  if (target == nullptr || target == this) {
    target = tool_manager->getDefaultTool();
  }
  if (target != nullptr && target != this) {
    tool_manager->setCurrentTool(target);
  }
}

int AddDoorTool::processMouseEvent(rviz_common::ViewportMouseEvent & event)
{
  if (event.leftDown() && state_ == InteractionState::Idle) {
    if (!getMapGroundPoint(event, pivot_)) {
      setStatus("Could not find the map ground plane");
      return Render;
    }

    state_ = InteractionState::DrawingDoor;
    updateDoorPreview(pivot_);
    return Render;
  }

  if (state_ == InteractionState::DrawingDoor && event.type == QEvent::MouseMove &&
    (event.buttons_down & Qt::LeftButton) != Qt::NoButton)
  {
    if (getMapGroundPoint(event, extent_)) {
      updateDoorPreview(extent_);
    }
    return Render;
  }

  if (state_ == InteractionState::DrawingDoor && event.leftUp()) {
    if (!getMapGroundPoint(event, extent_)) {
      state_ = InteractionState::Idle;
      clearPreview();
      restoreReturnTool();
      return Render;
    }

    const Ogre::Vector3 delta = extent_ - pivot_;
    if (std::hypot(delta.x, delta.y) < kClickDistance) {
      extent_ = pivot_ + Ogre::Vector3(kClickExtent, 0.0f, 0.0f);
    }

    state_ = InteractionState::ChoosingAngle;
    updateDoorPreview(extent_);
    setStatus("Move the mouse to preview the opening angle, then click");
    return Render;
  }

  if (state_ != InteractionState::ChoosingAngle) {
    return Render;
  }

  if (event.type == QEvent::MouseMove) {
    Ogre::Vector3 clicked_point;
    if (getMapGroundPoint(event, clicked_point)) {
      updateAnglePreview(clicked_point);
    }
    return Render;
  }

  if (event.leftUp()) {
    Ogre::Vector3 clicked_point;
    if (!getMapGroundPoint(event, clicked_point)) {
      setStatus("Could not find the map ground plane");
      return Render;
    }

    const Ogre::Vector3 clicked_vector = clicked_point - pivot_;
    if (std::hypot(clicked_vector.x, clicked_vector.y) < kMinimumAnglePointDistance) {
      setStatus("Click away from the pivot to set the opening angle");
      return Render;
    }

    const Ogre::Vector3 closed_vector = extent_ - pivot_;
    const double angle_rad = std::atan2(
      closed_vector.x * clicked_vector.y - closed_vector.y * clicked_vector.x,
      closed_vector.x * clicked_vector.x + closed_vector.y * clicked_vector.y);
    submitDoor(angle_rad * 180.0 / kPi);
    state_ = InteractionState::Idle;
    clearPreview();
    restoreReturnTool();
  }

  return Render;
}

}  // namespace arips_semantic_map_rviz

PLUGINLIB_EXPORT_CLASS(arips_semantic_map_rviz::AddDoorTool, rviz_common::Tool)
