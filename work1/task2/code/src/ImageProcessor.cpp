#include<ImageProcessor.hpp>

ImageProcessorException::ImageProcessorException(const std::string& message)
    : std::runtime_error(message) {}

//构造函数：加载图像
ImageProcessor::ImageProcessor(const std::string& path) : image_path(path) {
    image = cv::imread(image_path, cv::IMREAD_COLOR);
    if (image.empty()) {
        throw ImageProcessorException("无法加载图像: " + image_path +
            " - 文件不存在或格式不支持");
    }
}

//获取图像尺寸
cv::Size ImageProcessor::GetSize() const {
    return cv::Size(image.cols, image.rows);
}


//获取通道数
int ImageProcessor::GetChannels() const {
    return image.channels();
}

//获取像素数据
cv::Vec3b ImageProcessor::GetPixels(int x, int y) const {
    if (x < 0 || x >= image.cols || y < 0 || y >= image.rows) {
        throw ImageProcessorException("像素坐标越界");
    }
    return image.at<cv::Vec3b>(y, x);
}

//保存图像
void ImageProcessor::SaveImage(const std::string& outpath, const cv::Mat& img) {
    if (!cv::imwrite(outpath, img)) {
        throw ImageProcessorException("图像保存失败：" + outpath);
    }
}

//显示图像
void ImageProcessor::ShowImage(const std::string& WinName, const cv::Mat& img) {
    cv::imshow(WinName, img);
}

// 灯条阈值分割 - 基于亮度（灰度阈值）
cv::Mat ImageProcessor::LightBarSegmentation(int threshold_value) const {
    cv::Mat gray, binary;

    // 转换为灰度图
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

    // 二值化处理 - 灯条区域为前景（白色），其余为背景（黑色）
    cv::threshold(gray, binary, threshold_value, 255, cv::THRESH_BINARY);

    // 去除噪声和连接断点
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));

    // 开运算去噪
    cv::morphologyEx(binary, binary, cv::MORPH_OPEN, kernel);

    // 闭运算填充
    cv::morphologyEx(binary, binary, cv::MORPH_CLOSE, kernel);
    return binary;
}

// 灯条阈值分割 - 基于颜色（HSV 空间）
cv::Mat ImageProcessor::ColorSegmentationHSV(int h_low, int h_high, int s_low, int s_high, int v_low, int v_high) const {
    cv::Mat hsv, mask, binary;

    // 转换到 HSV 颜色空间
    cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);

    // 定义灯条颜色的 HSV 范围
    cv::Scalar lower_bound(h_low, s_low, v_low);
    cv::Scalar upper_bound(h_high, s_high, v_high);

    // 应用阈值分割，生成mask
    cv::inRange(hsv, lower_bound, upper_bound, mask);

    // 去除噪声和连接断点
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));

    // 开运算去噪
    cv::morphologyEx(mask, binary, cv::MORPH_OPEN, kernel);

    // 闭运算填充
    cv::morphologyEx(binary, binary, cv::MORPH_CLOSE, kernel);
    return binary;
}

//RGB转灰度图
cv::Mat ImageProcessor::ToGray() const {
    cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    return gray;
}

//均值模糊去噪
cv::Mat ImageProcessor::MeanBlur(int ksize) const {
    cv::Mat blurred;
    cv::blur(image, blurred, cv::Size(ksize, ksize));
    return blurred;
}
