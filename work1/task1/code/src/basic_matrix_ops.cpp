#include "matrix.hpp"
#include <iostream>
#include <vector>
#include <iomanip>

void testBasicMatrixOperations() {
    try {
        // 行数列数创建矩阵
        Matrix A(3, 3);
        A.setElement(0, 0, 1.0f);
        A.setElement(1, 1, 2.0f);
        A.setElement(2, 2, 3.0f);
        
        std::cout << "Matrix A:" << std::endl;
        A.print();
        
        // 二维数组创建矩阵
        float** arr = new float*[2];
        for (int i = 0; i < 2; ++i) {
            arr[i] = new float[2];
        }
        arr[0][0] = 1.0f;
        arr[0][1] = 2.0f;
        arr[1][0] = 3.0f;
        arr[1][1] = 4.0f;
        
        Matrix B(arr, 2, 2);
        std::cout << "\nMatrix B:" << std::endl;
        B.print();
        
        for (int i = 0; i < 2; ++i) {
            delete[] arr[i];
        }
        delete[] arr;
        
        // vector创建矩阵
        std::vector<std::vector<float>> vec = {
            {1.1f, 2.2f, 3.3f}, 
            {4.4f, 5.5f, 6.6f}
        };
        Matrix C(vec);
        std::cout << "\nMatrix C:" << std::endl;
        C.print();
        
        // 矩阵加法 & 减法
        vec = {{1.0f, 2.0f}, {3.0f, 4.0f}};
        Matrix D(vec);
        
        vec = {{5.0f, 6.0f}, {7.0f, 8.0f}};
        Matrix E(vec);
        
        std::cout << "Matrix D:" << std::endl;
        D.print();
        std::cout << "Matrix E:" << std::endl;
        E.print();
        
        Matrix F = D + E;
        std::cout << "D + E:" << std::endl;
        F.print();
        
        Matrix G = D - E;
        std::cout << "D - E:" << std::endl;
        G.print();
        
        // 矩阵乘法
        vec = {{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}};
        Matrix H(vec);
        
        vec = {{7.0f, 8.0f}, {9.0f, 10.0f}, {11.0f, 12.0f}};
        Matrix I(vec);

        std::cout << "Matrix H (2x3):" << std::endl;
        H.print();
        std::cout << "Matrix I (3x2):" << std::endl;
        I.print();
        
        Matrix J = H * I;
        std::cout << "H * I (2x2):" << std::endl;
        J.print();

        
        //错误处理
        std::cout << "Error Handling:" << std::endl;
        
        try {
            Matrix N1(2, 2);
            Matrix N2(3, 3);
            Matrix N3 = N1 + N2;
            std::cout << "Error: Expected an error but didn't catch one!" << std::endl;
        } catch (const std::exception& e) {
            std::cout << "Caught the adding dimention error correctly: " << e.what() << std::endl;
        }
        
        try {
            Matrix N1(2, 3);
            Matrix N2(2, 3);
            Matrix N3 = N1 * N2;
            std::cout << "Error: Expected an error but didn't catch one!" << std::endl;
        } catch (const std::exception& e) {
            std::cout << "Caught the multipling dimention error correctly:: " << e.what() << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cout << "Unnexpected Error: " << e.what() << std::endl;
    }
}

int main() {
    std::cout << "Matrix basic calculation" << std::endl;
    
    testBasicMatrixOperations();
    
    return 0;
}