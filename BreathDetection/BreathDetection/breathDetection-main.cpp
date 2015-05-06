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
	
	cv::Mat disparity;

	

	// prepare images 
	//cv::resize(imgL, imgL, cv::Size(WIDTH, HEIGHT));
	//cv::resize(imgR, imgR, cv::Size(WIDTH, HEIGHT));

	// launch engine to calculate dusparity
	breathDetection engine;

	// prepare all opencl options 
	engine._prepareOpenCL();
	cv::Mat graph = cv::Mat(cv::Size(1500, 500), CV_8UC3);
	cv::rectangle(graph, cv::Rect(0, 0, 1500, 500), cv::Scalar(0), -1);
	cv::VideoCapture capL("D:/MyDOC/Диплом/Результаты/6/l.avi");
	cv::VideoCapture capR("D:/MyDOC/Диплом/Результаты/6/r.avi");
	int i = 0;
	int counter = 0;
	cv::Point start = cv::Point(0, 0);
	while (true) {
		counter++;
		cv::Mat l;
		cv::Mat r;
		capL >> l;
		capR >> r;
		if (l.empty()) break;

		if (counter < 170) continue;
		i+= 2;

		cv::resize(l, l, cv::Size(WIDTH, HEIGHT));
		cv::resize(r, r, cv::Size(WIDTH, HEIGHT));
		if (!l.data) break;
	
		engine._calculateDisparity(l, r, &disparity);
		cv::Rect breathArea = cv::Rect(200, 200, 200, 80);
		cv::rectangle(disparity, breathArea, cv::Scalar(128));

		float sum = cv::sum(disparity(breathArea))[0] / breathArea.width / breathArea.height * 10;
		std::cout << sum << "\n";

		int minDiap = 785;
		int maxDiap = 810;
		
		int yVal = (sum - minDiap) * 500 / (maxDiap - minDiap);

		cv::Point end = cv::Point(i, 500 - yVal);
		cv::line(graph, start, end, cv::Scalar(255, 255, 255));
		start = end;

		cv::imshow("graph", graph);
		cv::imshow("ld", l);
		cv::imshow("rd", r);
		//cv::imshow("l", l);
		//cv::imshow("r", r);
		cv::normalize(disparity, disparity, 0, 1, cv::NORM_MINMAX);
		cv::imshow("disp", disparity);

		if (cv::waitKey(1) == 27) break;
	}

	return 0;
}