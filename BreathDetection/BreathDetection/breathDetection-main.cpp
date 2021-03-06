/* breath-detection launch file */

#define WIDTH 640
#define HEIGHT 480

#include <ctime>
#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "breathDetection.h"


int main() {
	cv::Mat imgL = cv::imread("D:/MyDOC/������/BreathDetection/BreathDetection/images/Tsukuba/left_picture.png");
	cv::Mat imgR = cv::imread("D:/MyDOC/������/BreathDetection/BreathDetection/images/Tsukuba/right_picture.png");
	//cv::Mat imgL = cv::imread("D:/MyDOC/������/BreathDetection/BreathDetection/images/Cegles/left_picture.png");
	//cv::Mat imgR = cv::imread("D:/MyDOC/������/BreathDetection/BreathDetection/images/Cegles/right_picture.png");
	//cv::Mat imgL = cv::imread("D:/MyDOC/������/BreathDetection/BreathDetection/images/Cones/left_picture.png");
	//cv::Mat imgR = cv::imread("D:/MyDOC/������/BreathDetection/BreathDetection/images/Cones/right_picture.png");
	//cv::Mat imgL = cv::imread("D:/MyDOC/������/BreathDetection/BreathDetection/images/Teddy/left_picture.png");
	//cv::Mat imgR = cv::imread("D:/MyDOC/������/BreathDetection/BreathDetection/images/Teddy/right_picture.png");
	//cv::Mat imgL = cv::imread("D:/MyDOC/������/BreathDetection/BreathDetection/images/Venus/left_picture.png");
	//cv::Mat imgR = cv::imread("D:/MyDOC/������/BreathDetection/BreathDetection/images/Venus/right_picture.png");

	cv::Mat disparity;

	// prepare images 
	cv::resize(imgL, imgL, cv::Size(WIDTH, HEIGHT));
	cv::resize(imgR, imgR, cv::Size(WIDTH, HEIGHT));

	// launch engine to calculate dusparity
	breathDetection engine;

	// prepare all opencl options 
	engine._prepareOpenCL();

	std::clock_t timer = std::clock();
	engine._calculateDisparity(imgL, imgR, &disparity);
	std::cout << "\nTotal: " << std::clock() - timer << " ms\n";

	return 0;
}