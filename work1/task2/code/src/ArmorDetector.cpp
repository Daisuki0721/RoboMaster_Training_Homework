#include "ArmorDetector.hpp"
#include <iostream>

armor_lightbar_detector::armor_lightbar_detector() {
    min_area_ = 50.0;
    max_area_ = 10000.0;
    min_aspect_ratio_ = 2.0f;
    max_aspect_ratio_ = 8.0f;
    min_width_ = 5.0f;
    min_height_ = 15.0f;
}

void armor_lightbar_detector::detect_lightbars(const cv::Mat& binaryImage,
    const cv::Mat& originalImage,
    cv::Mat& resultImage,
    std::vector<LightBar>& lightBars) {
    // 清空之前的检测结果
    lightBars.clear();

    // 复制原始图像用于绘制结果
    resultImage = originalImage.clone();

    // 查找轮廓
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(binaryImage, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    for (size_t i = 0; i < contours.size(); i++) {
        // 计算轮廓面积
        double area = cv::contourArea(contours[i]);

        // 面积筛选
        if (area < min_area_ || area > max_area_) {
            continue;
        }

        // 获取最小外接矩形
        cv::RotatedRect rect = cv::minAreaRect(contours[i]);

        // 获取矩形的宽度和高度
        float width = rect.size.width;
        float height = rect.size.height;

        // 确保高度大于宽度
        if (width > height) {
            std::swap(width, height);
        }

        // 跳过太小的情况
        if (width < min_width_ || height < min_height_) {
            continue;
        }

        // 计算长宽比
        float aspect_ratio = (width > 0) ? height / width : 0;

        // 长宽比筛选
        if (aspect_ratio < min_aspect_ratio_ || aspect_ratio > max_aspect_ratio_) {
            continue;
        }

        // 创建灯条信息
        LightBar lightBar;
        lightBar.rect = rect;
        lightBar.contour = contours[i];
        lightBar.center = rect.center;
        lightBar.width = width;
        lightBar.height = height;
        lightBar.aspect_ratio = aspect_ratio;
        lightBar.area = area;

        lightBars.push_back(lightBar);

        // 在图像上绘制灯条
        draw_lightbar(resultImage, lightBar, static_cast<int>(lightBars.size() - 1));
    }
}

void armor_lightbar_detector::draw_lightbar(cv::Mat& image, const LightBar& lightBar, int index) {
    // 绘制外接矩形
    cv::Point2f vertices[4];
    lightBar.rect.points(vertices);

    for (int i = 0; i < 4; i++) {
        cv::line(image, vertices[i], vertices[(i + 1) % 4], cv::Scalar(0, 255, 0), 2);
    }

    // 绘制中心点
    cv::circle(image, lightBar.center, 3, cv::Scalar(0, 0, 255), -1);

    // 添加标签显示信息
    std::string label = "Bar" + std::to_string(index) + ": AR" +
        std::to_string(static_cast<int>(lightBar.aspect_ratio * 10) / 10.0f).substr(0, 3);

    cv::putText(image, label,
        cv::Point(static_cast<int>(lightBar.center.x) + 10,
            static_cast<int>(lightBar.center.y)),
        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
}

void armor_lightbar_detector::print_detection(const std::vector<LightBar>& lightBars) {
    std::cout << "检测到 " << lightBars.size() << " 个灯条:" << std::endl;
    for (size_t i = 0; i < lightBars.size(); i++) {
        const LightBar& bar = lightBars[i];
        std::cout << "灯条 " << i << ": 中心(" << bar.center.x << ", " << bar.center.y
            << "), 长宽比:" << bar.aspect_ratio << ", 面积:" << bar.area << std::endl;
    }
}

