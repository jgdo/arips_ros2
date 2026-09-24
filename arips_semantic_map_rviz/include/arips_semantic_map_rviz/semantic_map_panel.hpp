#ifndef ARIPS_SEMANTIC_MAP_RVIZ__SEMANTIC_MAP_PANEL_HPP_
#define ARIPS_SEMANTIC_MAP_RVIZ__SEMANTIC_MAP_PANEL_HPP_

#include <memory>

#include <QWidget>

#include <arips_semantic_map_msgs/srv/handle_map_file.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rviz_common/panel.hpp>

class QLineEdit;

namespace rviz_common
{
class DisplayContext;
}

namespace arips_semantic_map_rviz
{

class AddDoorTool;

class SemanticMapPanel : public rviz_common::Panel
{
  Q_OBJECT

public:
  explicit SemanticMapPanel(QWidget * parent = nullptr);

  void onInitialize() override;

private Q_SLOTS:
  void loadMap();
  void saveMap();
  void addDoor();

private:
  void handleMapFile(const QString & path, bool save);
  void reportError(const std::string & operation, const std::string & error);

  QLineEdit * load_path_;
  QLineEdit * save_path_;
  rviz_common::DisplayContext * context_{nullptr};
  rclcpp::Node::SharedPtr node_;
  rclcpp::Client<arips_semantic_map_msgs::srv::HandleMapFile>::SharedPtr load_client_;
  rclcpp::Client<arips_semantic_map_msgs::srv::HandleMapFile>::SharedPtr save_client_;
  AddDoorTool * add_door_tool_{nullptr};
};

}  // namespace arips_semantic_map_rviz

#endif  // ARIPS_SEMANTIC_MAP_RVIZ__SEMANTIC_MAP_PANEL_HPP_
