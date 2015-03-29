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

	cv::Mat CM1 = cv::Mat(3, 3, CV_64FC1);
	cv::Mat CM2 = cv::Mat(3, 3, CV_64FC1);
	cv::Mat D1, D2;
	cv::Mat R, T, E, F;
	cv::Mat R1, R2, P1, P2, Q;

	cv::VideoWriter outputVideoL("D:/MyDOC/Диплом/Результаты/7/l.avi", -1, 6, cv::Size(640, 480));
	cv::VideoWriter outputVideoR("D:/MyDOC/Диплом/Результаты/7/r.avi", -1, 6, cv::Size(640, 480));

	cv::FileStorage fs1("D:/MyDOC/Диплом/BreathDetection/BreathDetection/BreathDetection/mystereocalibGOOD.yml", cv::FileStorage::READ);
	fs1["CM1"] >> CM1;
	fs1["CM2"] >> CM2;
	fs1["D1"] >> D1;
	fs1["D2"] >> D2;
	fs1["R1"] >> R1;
	fs1["R2"] >> R2;
	fs1["P1"] >> P1;
	fs1["P2"] >> P2;

	cv::Mat maplx, maply, maprx, mapry;
	cv::Mat imgUl, imgUr;

	initUndistortRectifyMap(CM1, D1, R1, P1, cv::Size(WIDTH, HEIGHT), CV_32FC1, maplx, maply);
	initUndistortRectifyMap(CM2, D2, R2, P2, cv::Size(WIDTH, HEIGHT), CV_32FC1, maprx, mapry);

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

		cv::remap(l, imgUl, maplx, maply, cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar());
		cv::remap(r, imgUr, maprx, mapry, cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar());

		engine._calculateDisparity(imgUr, imgUl, &disparity);
		
		outputVideoL << imgUr;
		outputVideoR << imgUl;

		cv::line(imgUl, cv::Point(0, 160), cv::Point(640, 160), cv::Scalar(255));
		cv::line(imgUl, cv::Point(0, 320), cv::Point(640, 320), cv::Scalar(255));
		cv::line(imgUr, cv::Point(0, 160), cv::Point(640, 160), cv::Scalar(255));
		cv::line(imgUr, cv::Point(0, 320), cv::Point(640, 320), cv::Scalar(255));

		cv::line(l, cv::Point(0, 160), cv::Point(640, 160), cv::Scalar(255));
		cv::line(l, cv::Point(0, 320), cv::Point(640, 320), cv::Scalar(255));
		cv::line(r, cv::Point(0, 160), cv::Point(640, 160), cv::Scalar(255));
		cv::line(r, cv::Point(0, 320), cv::Point(640, 320), cv::Scalar(255));

		cv::imshow("ld", imgUl);
		cv::imshow("rd", imgUr);
		//cv::imshow("l", l);
		//cv::imshow("r", r);
		cv::normalize(disparity, disparity, 0, 1, cv::NORM_MINMAX);
		cv::imshow("disp", disparity);

		if (cv::waitKey(1) == 27) break;
	}

	return 0;
}