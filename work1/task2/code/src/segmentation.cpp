#include"ImageProcessor.hpp"

int main() {
	try {
		ImageProcessor processor("test_image.jpg");
		std::cout << "图像尺寸：" << processor.GetSize() << std::endl;
		std::cout << "图像通道数：" << processor.GetChannels() << std::endl;
		cv::Vec3b pixel = processor.GetPixels(20, 20);
		std::cout << "像素[20,20]:[" << (int)pixel[0] << "," << (int)pixel[1] << "," << (int)pixel[2] << "]" << std::endl;
		cv::Mat gray = processor.ToGray();
		cv::Mat blurred = processor.MeanBlur(7);
		processor.ShowImage("GrayImage.jpg", gray);
		processor.ShowImage("BlurredImage.jpg", blurred);
		processor.SaveImage("GrayImage.jpg", gray);
		processor.SaveImage("BlurredImage.jpg", blurred);

		cv::waitKey(0);
		cv::destroyAllWindows();
	} catch (const std::exception& e) {
		std::cerr << "错误: " << e.what() << std::endl;
	}
	
	try {
		ImageProcessor processor2("segmentation.jpg");
		cv::Mat lightbar_binary = processor2.ColorSegmentationHSV();
		processor2.ShowImage("LightBarBinary.jpg", lightbar_binary);
		processor2.SaveImage("LightBarBinary.jpg", lightbar_binary);
		cv::waitKey(0);
		cv::destroyAllWindows();
	} catch (const std::exception& e) {
		std::cerr << "错误: " << e.what() << std::endl;
	}
	return 0;
}