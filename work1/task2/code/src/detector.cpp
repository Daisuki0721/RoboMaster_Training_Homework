#include "ArmorDetector.hpp"

#include <iostream>
#include <random>

void processImage(const std::string& imagePath, const std::string& binaryImagePath) {
    std::cout << "处理图像: " << imagePath << std::endl;

    // 读取图像
    cv::Mat image = cv::imread(imagePath);
    if (image.empty()) {
        std::cout << "无法读取图像: " << imagePath << std::endl;
        return;
    }

	// 读取二值化图像
	cv::Mat binaryImage = cv::imread(binaryImagePath, cv::IMREAD_GRAYSCALE);
    if (binaryImage.empty()) {
        std::cout << "无法读取图像: " << binaryImagePath << std::endl;
        return;
    }
    // 创建检测器
    armor_lightbar_detector detector;

    // 根据实际情况调整参数
    detector.set_min_area(400);
    detector.set_aspect_ratio_range(0.1f, 10.0f);

    // 检测灯条
    cv::Mat resultImage;
    std::vector<LightBar> lightBars;
    detector.detect_lightbars(binaryImage, image, resultImage, lightBars);

    // 打印检测信息
    detector.print_detection(lightBars);

    // 显示结果
    cv::imshow("原始图像", image);
    cv::imshow("二值化图像", binaryImage);
    cv::imshow("检测结果", resultImage);
    
    // 保存结果
    cv::imwrite("detection_result.jpg", resultImage);
    cv::waitKey(0);
}

int main() {
    processImage("segmentation.jpg", "LightBarBinary.jpg");
    cv::destroyAllWindows();
    return 0;
}