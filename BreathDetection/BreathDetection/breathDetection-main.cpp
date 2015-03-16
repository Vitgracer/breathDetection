/* breath-detection launch file */

#define WIDTH 640
#define HEIGHT 480

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "breathDetection.h"


int main() {
	cv::Mat imgL = cv::imread("D:/MyDOC/Диплом/BreathDetection/BreathDetection/images/Tsukuba/left_picture.png");
	cv::Mat imgR = cv::imread("D:/MyDOC/Диплом/BreathDetection/BreathDetection/images/Tsukuba/right_picture.png");
	cv::Mat disparity;

	// prepare images 
	cv::resize(imgL, imgL, cv::Size(WIDTH, HEIGHT));
	cv::resize(imgR, imgR, cv::Size(WIDTH, HEIGHT));
	
	// launch engine to calculate dusparity
	breathDetection engine;
	engine._calculateDisparity(imgL, imgR, &disparity);

	return 0;
}