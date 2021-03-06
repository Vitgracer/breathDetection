/* header file for breathDetection class */

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <CL/cl.hpp>

class breathDetection {
public:
	void _calculateDisparity(const cv::Mat imgL, const cv::Mat imgR, cv::Mat* disparity);
	void _prepareOpenCL();

private:
	// openCL launch variables 
	cl::Program _program;
	cl::Context _context;
	cl::CommandQueue _queue;

	// kernel functions to call 
	void _launchKernel(const char* kernelName, const int width, const int height, const int nArgs, ...);
	void _launchKernel(const char* kernelName, const int width, const int height, const int depth, const int nArgs, ...);
};