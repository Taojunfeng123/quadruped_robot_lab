# Copyright (c) 2026 taojunfeng123
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 2.8.3)

project(newdog7)

find_package(catkin REQUIRED)

catkin_package()

find_package(roslaunch)

foreach(dir config launch meshes urdf)
	install(DIRECTORY ${dir}/
		DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/${dir})
endforeach(dir)
