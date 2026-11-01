#!/bin/bash

rsync --numeric-ids -az --delete /home/rosuser/colcon_ws/src/arips_ros2/ jgdo@arips:/home/jgdo/colcon_ws/src/arips_ros2
rsync --numeric-ids -az --delete /home/rosuser/colcon_ws/src/lerobot-rosetta/ jgdo@arips:/home/jgdo/colcon_ws/src/lerobot-rosetta
rsync --numeric-ids -az --delete /home/rosuser/colcon_ws/src/rosetta/ jgdo@arips:/home/jgdo/colcon_ws/src/rosetta
rsync --numeric-ids -az --delete /home/rosuser/colcon_ws/src/rosetta_recorder_cpp/ jgdo@arips:/home/jgdo/colcon_ws/src/rosetta_recorder_cpp
rsync --numeric-ids -az --delete /home/rosuser/colcon_ws/src/lerobot-robot-rosetta/ jgdo@arips:/home/jgdo/colcon_ws/src/lerobot-robot-rosetta
rsync --numeric-ids -az --delete /home/rosuser/colcon_ws/src/rosetta_interfaces/ jgdo@arips:/home/jgdo/colcon_ws/src/rosetta_interfaces