#ifndef _IMAGEPROCESSOR_HPP_
#define _IMAGEPROCESSOR_HPP_

#include <opencv2/opencv.hpp>
#include <iostream>

//图像处理异常类
class ImageProcessorException : public std::runtime_error {
	public:
		explicit ImageProcessorException(const std::string& message);
};

//图像处理工具类
class ImageProcessor {
	private:
		cv::Mat image;
		std::string image_path;
	public:
		ImageProcessor(const std::string& path);						//构造函数：加载图像

		cv::Size GetSize() const;										//获取图像尺寸

		int GetChannels() const;										//获取通道数

		cv::Vec3b GetPixels(int x, int y) const;						//获取像素数据

		void SaveImage(const std::string& outpath, const cv::Mat& img);	//保存图像

		void ShowImage(const std::string& WinName, const cv::Mat& img);	//显示图像

		cv::Mat LightBarSegmentation(int threshold_value = 200) const;	// 灯条阈值分割 - 基于亮度（灰度阈值）

		// 灯条阈值分割 - 基于颜色（HSV 空间）
		cv::Mat ColorSegmentationHSV(int h_low = 0, int h_high = 20, int s_low = 100, int s_high = 255, int v_low = 100, int v_high = 255) const;

		cv::Mat ToGray() const;											//RGB转灰度图

		cv::Mat MeanBlur(int ksize = 5) const;							//均值模糊去噪
};

#endif
