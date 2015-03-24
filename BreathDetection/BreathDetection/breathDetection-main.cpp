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
	//cv::Mat imgL = cv::imread("D:/MyDOC/Диплом/BreathDetection/BreathDetection/images/Tsukuba/left_picture.png");
	//cv::Mat imgR = cv::imread("D:/MyDOC/Диплом/BreathDetection/BreathDetection/images/Tsukuba/right_picture.png");
	//cv::Mat imgL = cv::imread("D:/MyDOC/Диплом/BreathDetection/BreathDetection/images/Cegles/left_picture.png");
	//cv::Mat imgR = cv::imread("D:/MyDOC/Диплом/BreathDetection/BreathDetection/images/Cegles/right_picture.png");

	cv::Mat disparity;

	// prepare images 
	//cv::resize(imgL, imgL, cv::Size(WIDTH, HEIGHT));
	//cv::resize(imgR, imgR, cv::Size(WIDTH, HEIGHT));

	// launch engine to calculate dusparity
	breathDetection engine;

	// prepare all opencl options 
	engine._prepareOpenCL();

	cv::VideoCapture capL(0);
	cv::VideoCapture capR(1);

	while (true) {
		cv::Mat l;
		cv::Mat r;
		capL >> l;
		capR >> r;

		cv::resize(l, l, cv::Size(WIDTH, HEIGHT));
		cv::resize(r, r, cv::Size(WIDTH, HEIGHT));
		

		int shiftValue = 86;
		cv::Mat shiftedR = cv::Mat::zeros(r.size(), r.type());
		r(cv::Rect(0, shiftValue, r.cols, r.rows - shiftValue)).copyTo(shiftedR(cv::Rect(0, 0, r.cols, r.rows - shiftValue)));

		cv::line(l, cv::Point(0, 160), cv::Point(640, 160), cv::Scalar(255));
		cv::line(l, cv::Point(0, 320), cv::Point(640, 320), cv::Scalar(255));
		cv::line(r, cv::Point(0, 160), cv::Point(640, 160), cv::Scalar(255));
		cv::line(r, cv::Point(0, 320), cv::Point(640, 320), cv::Scalar(255));
		cv::line(shiftedR, cv::Point(0, 160), cv::Point(640, 160), cv::Scalar(255));
		cv::line(shiftedR, cv::Point(0, 320), cv::Point(640, 320), cv::Scalar(255));

		

		cv::medianBlur(l, l, 9);
		cv::medianBlur(shiftedR, shiftedR, 9);

		cv::imshow("l", l);
		//cv::imshow("r", r);

		cv::imshow("rs", shiftedR);

		engine._calculateDisparity(l, shiftedR, &disparity);
		cv::normalize(disparity, disparity, 0, 1, cv::NORM_MINMAX);
		cv::imshow("disp", disparity);

		cv::waitKey(1);
	}

	return 0;
}