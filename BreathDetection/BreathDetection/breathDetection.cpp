/* organize class for breath detection */
#include <iostream>
#include <fstream>
#define WIDTH 640	
#define HEIGHT 480
#define SQUARE (WIDTH * HEIGHT)
#define DISP_MAX 100
#define DISP_MIN 0
#define DIFF (DISP_MAX - DISP_MIN)

#include <streambuf>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "breathDetection.h"

void breathDetection::_launchKernel(const char* kernelName, const int width, const int height, const int depth, const int nArgs, ...) {
	/* funtion to call kernel with current name and width * height * depth threads */

	// standard procedure to work with variable number of variables 
	va_list args;
	va_start(args, nArgs);

	cl::Kernel kernel = cl::Kernel( _program , kernelName);
	
	// put buffers in the kernel  
	for (char bufNum = 0; bufNum < nArgs; bufNum++) {
		kernel.setArg(bufNum, va_arg(args, cl::Buffer));
	}

	va_end(args);
	
	//organize kernel computations 
	_queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(width, height, depth));
	_queue.finish();
}

void breathDetection::_launchKernel(const char* kernelName, const int width, const int height, const int nArgs, ...) {
	/* funtion to call kernel with current name and width * height  threads */

	// standard procedure to work with variable number of variables 
	va_list args;
	va_start(args, nArgs);

	cl::Kernel kernel = cl::Kernel(_program, kernelName);

	// put buffers in the kernel  
	for (char bufNum = 0; bufNum < nArgs; bufNum++) {
		kernel.setArg(bufNum, va_arg(args, cl::Buffer));
	}

	va_end(args);

	//organize kernel computations 
	_queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(width, height));
	_queue.finish();
}

void breathDetection::_prepareOpenCL() {
	/* prepare opencl device to work */

	// get info about platform 
	cl::Platform::get(&_platforms);
	cl::Platform defaultPlatform = _platforms[0];
	std::cout << "Using platform: " << defaultPlatform.getInfo<CL_PLATFORM_NAME>();

	// get info about devices 
	defaultPlatform.getDevices(CL_DEVICE_TYPE_ALL, &_devices);
	cl::Device default_device = _devices[0];
	std::cout << "\nDevice: " << default_device.getInfo<CL_DEVICE_NAME>();

	// organize context 
	_context = cl::Context(default_device);
	cl::Program::Sources sources;

	// read opencl file with kernels 
	std::ifstream kernelCL("ADCensus-kernel.cl.c");
	std::string kernel_code((std::istreambuf_iterator<char>(kernelCL)), std::istreambuf_iterator<char>());
	kernelCL.close();

	sources.push_back({ kernel_code.c_str(), kernel_code.length() });
	_program = cl::Program(_context, sources);

	// use compiler logs 
	if (_program.build(_devices, "") == CL_SUCCESS) std::cout << "\nSuccessfully built! " << _program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(default_device) << "\n---------------------------------------------";
											   else std::cout << "\nErrors!\n----------------------------------------" << _program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(_devices[0]) << "\n";
}

void breathDetection::_calculateDisparity(const cv::Mat imgL, const cv::Mat imgR, cv::Mat* disparity) {
	/* func to calculate disparity using left and right images from stereo-pair */
	
	// prepare all opencl options 
	_prepareOpenCL();

	// allocate buffer memory for images 
	cl::Buffer bL = cl::Buffer(_context, CL_MEM_READ_ONLY, sizeof(uchar) * SQUARE * 3);
	cl::Buffer bR = cl::Buffer(_context, CL_MEM_READ_ONLY, sizeof(uchar) * SQUARE * 3);
	cl::Buffer bDisp(_context, CL_MEM_WRITE_ONLY, sizeof(float)* SQUARE);

	// allocate buffer memory to store 3D costs 
	cl::Buffer bCosts(_context, CL_MEM_READ_WRITE, sizeof(float) * SQUARE * DIFF);

	// write images in the buffer 
	_queue.enqueueWriteBuffer(bL, CL_TRUE, 0, sizeof(uchar)* SQUARE * 3, imgL.data);
	_queue.enqueueWriteBuffer(bR, CL_TRUE, 0, sizeof(uchar)* SQUARE * 3, imgR.data);

	_launchKernel("kComputeCosts", WIDTH, HEIGHT, DIFF, 3, bL, bR, bCosts);
	_launchKernel("kGetDisparityMap", WIDTH, HEIGHT, 2, bCosts, bDisp);

	// read disparity result 
	std::vector<float> dispData(SQUARE);

	_queue.enqueueReadBuffer(bDisp, CL_TRUE, 0, sizeof(float)* SQUARE, &dispData[0]);
	cv::Mat(HEIGHT, WIDTH, CV_32FC1, &dispData[0]).copyTo((*disparity));
}