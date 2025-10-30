#include "matrix.hpp"
#include <iostream>
#include <vector>
#include <iomanip>
#include <cmath>

// 直接先旋转后平移（不使用齐次坐标）
std::vector<float> transform2D(const std::vector<float>& point, float angle, float tx, float ty) {
    if (point.size() != 2) {
        throw std::invalid_argument("输入点必须是2维向量");
    }
    
    Matrix rot = Matrix::rotation2D(angle);
    
    // 点转换为矩阵形式
    Matrix point_mat(2, 1);
    point_mat.setElement(0, 0, point[0]);
    point_mat.setElement(1, 0, point[1]);
    
    // 旋转
    Matrix rotated = rot * point_mat;
    
    // 平移
    std::vector<float> result(2);
    result[0] = rotated.getElement(0, 0) + tx;
    result[1] = rotated.getElement(1, 0) + ty;
    
    return result;
}

// 使用齐次坐标实现组合变换
std::vector<float> transform2DHomogeneous(const std::vector<float>& point, float angle, float tx, float ty) {
    if (point.size() != 2) {
        throw std::invalid_argument("输入点必须是2维向量");
    }
    
    // 将原二维向量转换为齐次坐标
    Matrix point_homo = Matrix::vectorToHomogeneous(point);
    
    // 创建旋转矩阵（2x2）
    Matrix rot_2d = Matrix::rotation2D(angle);
    
    // 构造包含旋转和平移的齐次变换矩阵（3x3）
    Matrix transform(3, 3);
    
    // 设置旋转 & 平移 & 齐次坐标
    transform.setElement(0, 0, rot_2d.getElement(0, 0));
    transform.setElement(0, 1, rot_2d.getElement(0, 1));
    transform.setElement(1, 0, rot_2d.getElement(1, 0));
    transform.setElement(1, 1, rot_2d.getElement(1, 1));

    transform.setElement(0, 2, tx);
    transform.setElement(1, 2, ty);

    transform.setElement(2, 2, 1.0f);
    
    // 矩阵乘法齐次坐标变换
    Matrix result_homo = transform * point_homo;
    
    // 结果转换回二维坐标
    return Matrix::homogeneousToVector(result_homo);
}

void demonstrateCoordinateTransforms() {
    std::cout << "---------------坐标变换---------------" << std::endl;
    
    try {
        std::vector<float> original_point = {1.0f, 0.0f}; // 原始点 (1, 0)
        float rotation_angle = 90.0f; // 旋转90度
        float translation_x = 2.0f;   // x方向平移2
        float translation_y = 1.0f;   // y方向平移1
        
        std::cout << "原始点: (" << original_point[0] << ", " << original_point[1] << ")" << std::endl;
        std::cout << "旋转角度: " << rotation_angle << "度" << std::endl;
        std::cout << "平移向量: (" << translation_x << ", " << translation_y << ")" << std::endl;
        std::cout << std::endl;
        
        // 方法1：直接先旋转后平移
        std::cout << "---直接先旋转后平移---" << std::endl;
        std::vector<float> result1 = transform2D(original_point, rotation_angle, translation_x, translation_y);
        std::cout << "变换后坐标: (" << result1[0] << ", " << result1[1] << ")" << std::endl;
        std::cout << std::endl;
        
        // 方法2：使用齐次坐标
        std::cout << "---使用齐次坐标---" << std::endl;
        std::vector<float> result2 = transform2DHomogeneous(original_point, rotation_angle, translation_x, translation_y);
        std::cout << "变换后坐标: (" << result2[0] << ", " << result2[1] << ")" << std::endl;
        std::cout << std::endl;
        
        // 演示齐次变换矩阵的构造
        std::cout << "---齐次变换矩阵---" << std::endl;
        
        Matrix rotation_matrix = Matrix::rotation2D(rotation_angle);
        std::cout << "旋转矩阵 (2x2):" << std::endl;
        rotation_matrix.print();
        
        Matrix homogeneous_transform(3, 3);
        
        // 设置旋转部分
        homogeneous_transform.setElement(0, 0, rotation_matrix.getElement(0, 0));
        homogeneous_transform.setElement(0, 1, rotation_matrix.getElement(0, 1));
        homogeneous_transform.setElement(1, 0, rotation_matrix.getElement(1, 0));
        homogeneous_transform.setElement(1, 1, rotation_matrix.getElement(1, 1));
        
        // 设置平移部分
        homogeneous_transform.setElement(0, 2, translation_x);
        homogeneous_transform.setElement(1, 2, translation_y);
        
        // 设置齐次坐标部分
        homogeneous_transform.setElement(2, 2, 1.0f);
        
        std::cout << "齐次变换矩阵 (3x3):" << std::endl;
        homogeneous_transform.print();
    } catch (const std::exception& e) {
        std::cout << "出现未预期错误: " << e.what() << std::endl;
    }
}

int main() {

    demonstrateCoordinateTransforms();
    
    return 0;
}