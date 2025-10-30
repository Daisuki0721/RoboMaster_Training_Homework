#include "matrix.hpp"

// 构造函数1：指定行数和列数，初始化为0
Matrix::Matrix(int r, int c) : rows(r), cols(c) {
    if (r <= 0 || c <= 0) {
        throw std::invalid_argument("行数和列数必须大于0");
    }
    data.resize(rows, std::vector<float>(cols, 0.0f));
}

// 构造函数2：使用二维数组初始化
Matrix::Matrix(float** arr, int r, int c) : rows(r), cols(c) {
    if (r <= 0 || c <= 0) {
        throw std::invalid_argument("行数和列数必须大于0");
    }
    data.resize(rows, std::vector<float>(cols, 0.0f));
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            data[i][j] = arr[i][j];
        }
    }
}

// 构造函数3：使用vector<vector<float>>初始化
Matrix::Matrix(const std::vector<std::vector<float>>& vec) {
    if (vec.empty() || vec[0].empty()) {
        throw std::invalid_argument("输入矩阵不能为空");
    }
    rows = vec.size();
    cols = vec[0].size();
    data = vec;
}

// 设置指定位置的元素值
void Matrix::setElement(int i, int j, float value) {
    if (i < 0 || i >= rows || j < 0 || j >= cols) {
        throw std::out_of_range("索引超出矩阵范围");
    }
    data[i][j] = value;
}

// 获取指定位置的元素值
float Matrix::getElement(int i, int j) const {
    if (i < 0 || i >= rows || j < 0 || j >= cols) {
        throw std::out_of_range("索引超出矩阵范围");
    }
    return data[i][j];
}

// 矩阵加法
Matrix Matrix::operator+(const Matrix& other) const {
    if (rows != other.rows || cols != other.cols) {
        throw std::invalid_argument("矩阵维度不匹配，无法相加");
    }
    
    Matrix result(rows, cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result.data[i][j] = data[i][j] + other.data[i][j];
        }
    }
    return result;
}

// 矩阵减法
Matrix Matrix::operator-(const Matrix& other) const {
    if (rows != other.rows || cols != other.cols) {
        throw std::invalid_argument("矩阵维度不匹配，无法相减");
    }
    
    Matrix result(rows, cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result.data[i][j] = data[i][j] - other.data[i][j];
        }
    }
    return result;
}

// 矩阵乘法
Matrix Matrix::operator*(const Matrix& other) const {
    if (cols != other.rows) {
        throw std::invalid_argument("矩阵维度不匹配，无法相乘");
    }
    
    Matrix result(rows, other.cols);
    for (int i = 0; i < result.rows; i++) {
        for (int j = 0; j < result.cols; j++) {
            for (int k = 0; k < cols; k++) {
                result.data[i][j] += data[i][k] * other.data[k][j];
            }
        }
    }
    return result;
}

// 输出矩阵
void Matrix::print() const {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << data[i][j] << " ";
        }
        std::cout << std::endl;
    }
}

// 创建单位矩阵
Matrix Matrix::identity(int size) {
    Matrix result(size, size);
    for (int i = 0; i < size; i++) {
        result.setElement(i, i, 1.0f);
    }
    return result;
}

// 创建二维旋转矩阵
Matrix Matrix::rotation2D(float angle) {
    Matrix rot(2, 2);
    float rad = angle * M_PI / 180.0f; // 角度转弧度
    rot.setElement(0, 0, std::cos(rad));
    rot.setElement(0, 1, -std::sin(rad));
    rot.setElement(1, 0, std::sin(rad));
    rot.setElement(1, 1, std::cos(rad));
    return rot;
}

// 创建二维平移矩阵（齐次坐标）
Matrix Matrix::translation2D(float tx, float ty) {
    Matrix trans(3, 3);
    trans.setElement(0, 0, 1.0f);
    trans.setElement(0, 2, tx);
    trans.setElement(1, 1, 1.0f);
    trans.setElement(1, 2, ty);
    trans.setElement(2, 2, 1.0f);
    return trans;
}

// 向量转换为齐次坐标
Matrix Matrix::vectorToHomogeneous(const std::vector<float>& vec) {
    if (vec.size() != 2) {
        throw std::invalid_argument("输入向量必须是2维");
    }
    Matrix homogenous(3, 1);
    homogenous.setElement(0, 0, vec[0]);
    homogenous.setElement(1, 0, vec[1]);
    homogenous.setElement(2, 0, 1.0f);
    return homogenous;
}

// 齐次坐标转换为向量
std::vector<float> Matrix::homogeneousToVector(const Matrix& homogenous) {
    if (homogenous.getRows() != 3 || homogenous.getCols() != 1) {
        throw std::invalid_argument("输入必须是3x1的齐次坐标");
    }
    std::vector<float> result(2);
    result[0] = homogenous.getElement(0, 0);
    result[1] = homogenous.getElement(1, 0);
    return result;
}