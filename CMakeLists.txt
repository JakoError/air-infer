cmake_minimum_required(VERSION 3.8)
project(air_infer)

# Install non-Python resources (launch files, etc.)
install(DIRECTORY resource
    DESTINATION share/${PROJECT_NAME}
)

ament_python_install_package(${PROJECT_NAME})

ament_package()
