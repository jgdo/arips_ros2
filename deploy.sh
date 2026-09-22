#!/bin/bash

rsync --numeric-ids -az --delete /home/rosuser/colcon_ws/src/arips_ros2/ jgdo@arips:/home/jgdo/colcon_ws/src/arips_ros2

