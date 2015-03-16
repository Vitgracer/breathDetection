/* breath-detection launch file */

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "breathDetection.h"

int main() {
	breathDetection engine;
	engine._prepareOpenCL();

	return 0;
}