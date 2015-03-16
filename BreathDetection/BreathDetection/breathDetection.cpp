/* organize class for breath detection */
#include <iostream>
#include <fstream>
#include <streambuf>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "breathDetection.h"

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