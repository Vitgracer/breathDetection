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
	cv::Mat imgL = cv::imread("D:/MyDOC/Диплом/BreathDetection/BreathDetection/images/Tsukuba/left_picture.png");
	cv::Mat imgR = cv::imread("D:/MyDOC/Диплом/BreathDetection/BreathDetection/images/Tsukuba/right_picture.png");
	//cv::Mat imgL = cv::imread("D:/MyDOC/Диплом/BreathDetection/BreathDetection/images/Cegles/left_picture.png");
	//cv::Mat imgR = cv::imread("D:/MyDOC/Диплом/BreathDetection/BreathDetection/images/Cegles/right_picture.png");

	cv::Mat disparity;

	// prepare images 
	cv::resize(imgL, imgL, cv::Size(WIDTH, HEIGHT));
	cv::resize(imgR, imgR, cv::Size(WIDTH, HEIGHT));

	// launch engine to calculate dusparity
	breathDetection engine;

	// prepare all opencl options 
	engine._prepareOpenCL();

	cv::VideoCapture capL("D:/MyDOC/Диплом/BreathDetection/BreathDetection/Video/left.mp4");
	cv::VideoCapture capR("D:/MyDOC/Диплом/BreathDetection/BreathDetection/Video/right.mp4");

	while (true) {
		cv::Mat l;
		cv::Mat r;
		capL >> l;
		capR >> r;
		
		cv::resize(l, l, cv::Size(WIDTH, HEIGHT));
		cv::resize(r, r, cv::Size(WIDTH, HEIGHT));
		engine._calculateDisparity(l, r, &disparity);
		int a = 2;
	}

	std::clock_t timer = std::clock();
	engine._calculateDisparity(imgL, imgR, &disparity);
	std::cout << "\nTotal: " << std::clock() - timer << " ms\n";

	return 0;
}