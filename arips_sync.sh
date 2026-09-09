#!/bin/bash

rsync --numeric-ids -az --delete ./ jgdo@arips:/home/jgdo/colcon_ws/src/arips_ros2

