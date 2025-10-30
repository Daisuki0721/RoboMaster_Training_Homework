cmake_minimum_required(VERSION 3.10.0)
project(ArmorSim VERSION 0.1.0 LANGUAGES C CXX)

find_package(OpenCV REQUIRED)

include_directories(include)

add_executable(ArmorSim
    src/armorsim.cpp
    src/simulate.cpp
)

add_executable(KaimanFilter
    src/KaimanFilter.cpp
)

target_include_directories(KaimanFilter PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(KaimanFilter ${OpenCV_LIBS})

include(CTest)
enable_testing()

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
