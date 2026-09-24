#include "arips_semantic_map_rviz/semantic_map_panel.hpp"

#include <string>

#include <QHBoxLayout>
#include <QLineEdit>
#include <QPushButton>
#include <QVBoxLayout>

#include <pluginlib/class_list_macros.hpp>
#include <rviz_common/display_context.hpp>
#include <rviz_common/ros_integration/ros_node_abstraction_iface.hpp>
#include <rviz_common/tool_manager.hpp>

#include "arips_semantic_map_rviz/add_door_tool.hpp"

namespace arips_semantic_map_rviz
{

SemanticMapPanel::SemanticMapPanel(QWidget * parent)
: rviz_common::Panel(parent), load_path_(new QLineEdit(this)), save_path_(new QLineEdit(this))
{
  auto * layout = new QVBoxLayout(this);
  layout->setContentsMargins(4, 4, 4, 4);

  auto * load_row = new QHBoxLayout();
  load_path_->setPlaceholderText("Semantic map path");
  auto * load_button = new QPushButton("Load", this);
  load_row->addWidget(load_path_);
  load_row->addWidget(load_button);
  layout->addLayout(load_row);

  auto * save_row = new QHBoxLayout();
  save_path_->setPlaceholderText("Semantic map path");
  auto * save_button = new QPushButton("Save", this);
  save_row->addWidget(save_path_);
  save_row->addWidget(save_button);
  layout->addLayout(save_row);

  auto * add_door_button = new QPushButton("Add door", this);
  layout->addWidget(add_door_button);

  connect(load_button, &QPushButton::clicked, this, &SemanticMapPanel::loadMap);
  connect(save_button, &QPushButton::clicked, this, &SemanticMapPanel::saveMap);
  connect(add_door_button, &QPushButton::clicked, this, &SemanticMapPanel::addDoor);
}

void SemanticMapPanel::onInitialize()
{
  context_ = getDisplayContext();
  auto node_abstraction = context_->getRosNodeAbstraction().lock();
  if (!node_abstraction) {
    return;
  }

  node_ = node_abstraction->get_raw_node();
  load_client_ = node_->create_client<arips_semantic_map_msgs::srv::HandleMapFile>(
    "/semantic_map_server/load_map");
  save_client_ = node_->create_client<arips_semantic_map_msgs::srv::HandleMapFile>(
    "/semantic_map_server/save_map");

  auto * tool = context_->getToolManager()->addTool(
    "arips_semantic_map_rviz/AddDoorTool");
  add_door_tool_ = dynamic_cast<AddDoorTool *>(tool);
  if (add_door_tool_ == nullptr) {
    RCLCPP_ERROR(node_->get_logger(), "Could not create the semantic map Add door tool");
  }
}

void SemanticMapPanel::loadMap()
{
  handleMapFile(load_path_->text(), false);
}

void SemanticMapPanel::saveMap()
{
  handleMapFile(save_path_->text(), true);
}

void SemanticMapPanel::addDoor()
{
  if (context_ == nullptr || add_door_tool_ == nullptr) {
    if (node_) {
      RCLCPP_WARN(node_->get_logger(), "The semantic map Add door tool is unavailable");
    }
    return;
  }

  auto * tool_manager = context_->getToolManager();
  add_door_tool_->setReturnTool(tool_manager->getCurrentTool());
  tool_manager->setCurrentTool(add_door_tool_);
}

void SemanticMapPanel::handleMapFile(const QString & path, bool save)
{
  if (!node_ || path.trimmed().isEmpty()) {
    if (node_) {
      RCLCPP_WARN(node_->get_logger(), "A semantic map path is required");
    }
    return;
  }

  auto client = save ? save_client_ : load_client_;
  if (!client || !client->service_is_ready()) {
    reportError(save ? "save" : "load", "service is unavailable");
    return;
  }

  auto request = std::make_shared<arips_semantic_map_msgs::srv::HandleMapFile::Request>();
  request->file_path = path.trimmed().toStdString();
  const std::string operation = save ? "save" : "load";
  client->async_send_request(
    request,
    [node = node_, operation](
      rclcpp::Client<arips_semantic_map_msgs::srv::HandleMapFile>::SharedFuture future) {
      const auto response = future.get();
      if (!response->success) {
        if (node) {
          RCLCPP_WARN(node->get_logger(), "Could not %s semantic map: %s",
            operation.c_str(), response->error_message.c_str());
        }
        return;
      }

      if (node) {
        RCLCPP_INFO(node->get_logger(), "Semantic map %sed successfully",
          operation.c_str());
      }
    });
}

void SemanticMapPanel::reportError(const std::string & operation, const std::string & error)
{
  if (node_) {
    RCLCPP_WARN(node_->get_logger(), "Could not %s semantic map: %s",
      operation.c_str(), error.c_str());
  }
}

}  // namespace arips_semantic_map_rviz

PLUGINLIB_EXPORT_CLASS(arips_semantic_map_rviz::SemanticMapPanel, rviz_common::Panel)
