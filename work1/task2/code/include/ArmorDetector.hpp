
#ifndef _ARMOR_DETECTOR_H_
#define _ARMOR_DETECTOR_H_

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

struct LightBar {
    cv::RotatedRect rect;
    std::vector<cv::Point> contour;
    cv::Point2f center;
    float width;
    float height;
    float aspect_ratio;
    double area;
};

class armor_lightbar_detector {
    private:
        void draw_lightbar(cv::Mat& image, const LightBar& lightBar, int index);    // 绘制单个灯条
    
        double min_area_;
        double max_area_;
        float min_aspect_ratio_;
        float max_aspect_ratio_;
        float min_width_;
        float min_height_;
    public:
        armor_lightbar_detector();

        void detect_lightbars(const cv::Mat& binaryImage, const cv::Mat& originalImage,
            cv::Mat& resultImage, std::vector<LightBar>& lightBars);    // 检测函数
        
        void print_detection(const std::vector<LightBar>& lightBars);   // 打印检测信息
        
        //设置参数
        void set_min_area(double min_area) { min_area_ = min_area; }
        void set_max_area(double max_area) { max_area_ = max_area; }
        void set_aspect_ratio_range(float min_ar, float max_ar) {
            min_aspect_ratio_ = min_ar;
            max_aspect_ratio_ = max_ar;
        }
        void set_min_width(float min_width) { min_width_ = min_width; }
        void set_min_height(float min_height) { min_height_ = min_height; }
};

#endif