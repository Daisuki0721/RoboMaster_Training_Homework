#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <vector>
#include <stdexcept>
#include <cmath>
#include <iostream>

class Matrix {
private:
    std::vector<std::vector<float>> data;
    int rows;
    int cols;

public:
    Matrix(int r, int c);    // 构造函数1：指定行数和列数，初始化为0
    
    Matrix(float** arr, int r, int c);  // 构造函数2：使用二维数组初始化
    
    Matrix(const std::vector<std::vector<float>>& vec);    // 构造函数3：使用vector<vector<float>>初始化

    Matrix() : rows(0), cols(0) {}    // 默认构造函数
    
    int getRows() const { return rows; }    // 获取行数
    
    int getCols() const { return cols; }    // 获取列数
    
    void setElement(int i, int j, float value);    // 设置指定位置的元素值
    
    float getElement(int i, int j) const;    // 获取指定位置的元素值
    
    Matrix operator+(const Matrix& other) const;    // 矩阵加法
    
    Matrix operator-(const Matrix& other) const;    // 矩阵减法
    
    Matrix operator*(const Matrix& other) const;    // 矩阵乘法
    
    void print() const;    // 输出矩阵
    
    static Matrix identity(int size);    // 创建单位矩阵
    
    static Matrix rotation2D(float angle);    // 创建二维旋转矩阵
    
    static Matrix translation2D(float tx, float ty);    // 创建二维平移矩阵（齐次坐标）
    
    static Matrix vectorToHomogeneous(const std::vector<float>& vec);    // 向量转换为齐次坐标
    
    static std::vector<float> homogeneousToVector(const Matrix& homogenous);    // 齐次坐标转换为向量
};

// 坐标变换主函数
std::vector<float> transform2D(const std::vector<float>& point, float angle, float tx, float ty);
std::vector<float> transform2DHomogeneous(const std::vector<float>& point, float angle, float tx, float ty);

#endif