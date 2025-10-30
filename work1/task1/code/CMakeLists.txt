cmake_minimum_required(VERSION 3.10)
project(MatrixTransform)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(include)

# 矩阵基本运算
add_executable(matrix_basic_ops 
    src/matrix.cpp
    src/basic_matrix_ops.cpp
)

# 坐标变换
add_executable(matrix_transform 
    src/matrix.cpp
    src/shift_rotate.cpp
)

# 设置编译器标志
if(MSVC)
    target_compile_options(matrix_basic_ops PRIVATE /W4)
    target_compile_options(matrix_transform PRIVATE /W4)
else()
    target_compile_options(matrix_basic_ops PRIVATE -Wall -Wextra -pedantic)
    target_compile_options(matrix_transform PRIVATE -Wall -Wextra -pedantic)
endif()