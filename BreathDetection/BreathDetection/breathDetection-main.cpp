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

	std::vector<cv::Point> sumValues;
	int breathNumber = 0;

	int shrinkHeight = 500.0 + 480.0 / (640.0 / 500.0);
	cv::Mat toShow = cv::Mat(shrinkHeight, 1500, CV_8UC3);

	while (true) {
		counter++;
		cv::Mat l;
		cv::Mat r;
		capL >> l;
		capR >> r;
		if (l.empty()) break;

		if (counter < 180) continue;
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

		sumValues.push_back(end);

		// red circles 
		if (sumValues.size() > 10) {
			bool drawPointHigh = true;
			bool drawPointLow = true;
			for (int sumC = -5; sumC < 5; sumC++) {
				if (sumValues[sumValues.size() - 5].y > sumValues[sumValues.size() - 5 + sumC].y) {
					drawPointHigh = false;
				}
				if (sumValues[sumValues.size() - 5].y < sumValues[sumValues.size() - 5 + sumC].y) {
					drawPointLow = false;
				}
			}
			if (drawPointHigh) {
				breathNumber++;
				cv::circle(graph, sumValues[sumValues.size() - 5], 4, cv::Scalar(0, 0, 255), -1);
				cv::String breathText= "Breaths: " + std::to_string(breathNumber);
				cv::String breathTextErase = "Breaths: " + std::to_string(breathNumber - 1);
				cv::putText(graph, breathTextErase, cv::Point(1100, 50), 2, 2, cv::Scalar(0, 0, 0));
				cv::putText(graph, breathText, cv::Point(1100, 50), 2, 2, cv::Scalar(255, 0, 255));
			}
			if (drawPointLow) {
				cv::circle(graph, sumValues[sumValues.size() - 5], 4, cv::Scalar(0, 255, 255), -1);
			}
		}

		cv::line(graph, start, end, cv::Scalar(255, 255, 255));
		start = end;

		// create toShow 
		cv::resize(l, l, cv::Size(500, 480.0 / (640.0 / 500.0)));
		cv::resize(r, r, cv::Size(500, 480.0 / (640.0 / 500.0)));
		cv::resize(disparity, disparity, cv::Size(500, 480.0 / (640.0 / 500.0)));
		graph.copyTo(toShow(cv::Rect(0, shrinkHeight - 500, graph.cols, graph.rows))); 
		l.copyTo(toShow(cv::Rect(0,0, l.cols, l.rows)));
		r.copyTo(toShow(cv::Rect(500, 0, l.cols, l.rows)));
		cvtColor(disparity, disparity, cv::COLOR_GRAY2RGB);
		disparity.copyTo(toShow(cv::Rect(1000, 0, l.cols, l.rows)));
		cv::rectangle(toShow, cv::Rect(0, 0, toShow.cols, toShow.rows), cv::Scalar(255, 255, 255), 3);
		cv::rectangle(toShow, cv::Rect(500, 0, l.cols, l.rows), cv::Scalar(255, 255, 255), 3);
		cv::rectangle(toShow, cv::Rect(1000, 0, l.cols, l.rows), cv::Scalar(255, 255, 255), 3);
		cv::rectangle(toShow, cv::Rect(0, 0, l.cols, l.rows), cv::Scalar(255, 255, 255), 3);
		cv::rectangle(toShow, cv::Rect(0, shrinkHeight, graph.cols, graph.rows), cv::Scalar(255, 255, 255), 3);
		/*cv::imshow("graph", graph);
		cv::imshow("ld", l);
		cv::imshow("rd", r);*/
		//cv::imshow("l", l);
		//cv::imshow("r", r);
		cv::normalize(disparity, disparity, 0, 1, cv::NORM_MINMAX);
		cv::imshow("Breath detection", toShow);

		if (cv::waitKey(1) == 27) break;
	}

	return 0;
}