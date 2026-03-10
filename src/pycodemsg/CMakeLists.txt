cmake_minimum_required(VERSION 3.8)
project(pycodemsg)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)
 

find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/FileEntry.msg"
  "msg/CommEntry.msg"
  "msg/NodeEntry.msg"
  "srv/FileOperation.srv"
  "srv/NodeList.srv"
  "srv/FileList.srv"
  "srv/CommString.srv"
  "srv/NodeCommand.srv"
  "action/CmdOperation.action"
  "action/CommString.action"
)

ament_package()
