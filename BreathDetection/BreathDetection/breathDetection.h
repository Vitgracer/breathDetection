/* header file for breathDetection class */

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <CL/cl.hpp>

class breathDetection {
public:
	void _calculateDisparity(cv::Mat imgL, cv::Mat imgR, cv::Mat* disparity);

private:
	// openCL launch variables 
	std::vector<cl::Platform> _platforms;
	std::vector<cl::Device> _devices;
	cl::Program _program;
	cl::Context _context;
	cl::CommandQueue _queue;
	
	void _prepareOpenCL();

	// kernel functions to call 
	void _launchKernel(const char* kernelName, const int width, const int height, const int depth, const int nArgs, ...);
};