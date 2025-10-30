cmake_minimum_required(VERSION 3.10.0)
project(opencv_test VERSION 0.1.0 LANGUAGES C CXX)

find_package(OpenCV REQUIRED)

include_directories(include)

add_executable(opencv_basic 
    src/ImageProcessor.cpp
    src/segmentation.cpp
)

add_executable(detector
    src/ArmorDetector.cpp
    src/detector.cpp
)

target_include_directories(opencv_basic PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(opencv_basic ${OpenCV_LIBS})

target_include_directories(detector PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(detector ${OpenCV_LIBS})

include(CTest)
enable_testing()

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
