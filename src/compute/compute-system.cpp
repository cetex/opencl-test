#include "compute-system.h"

#include <iostream>

ComputeSystem::ComputeSystem(DeviceType type)
{
	// ==============
	// Load Platform
	// ==============
	
	std::vector<cl::Platform> allPlatforms;

	cl::Platform::get(&allPlatforms);

	if (allPlatforms.empty())
	{
		throw std::runtime_error(std::string("[compute] No platforms found. Check your OpenCL Installation."));
	}

	_platform = allPlatforms.front();

	// ===============
	// Load Device
	// ===============
	
	std::vector<cl::Device> allDevices;

	switch(type)
	{
		case _cpu:
			_platform.getDevices(CL_DEVICE_TYPE_CPU, &allDevices);
			break;
		case _gpu:
			_platform.getDevices(CL_DEVICE_TYPE_GPU, &allDevices);
			break;
		case _all:
			_platform.getDevices(CL_DEVICE_TYPE_ALL, &allDevices);
			break;
	}

	if (allDevices.empty())
	{
		throw std::runtime_error(std::string("[compute] No devices found. Check your OpenCL Installation."));
	}

	_device = allDevices.front();
	
	// ===============
	// Load Context
	// ===============
	// This is cheating, we should create a proper context here.
	_context = _device;

	// ===============
	// Load Queue
	// ===============
	
	_queue = cl::CommandQueue(_context, _device);	

}

void ComputeSystem::printCLInfo()
{
	std::cout << "[compute] OpenCL Version: " << _platform.getInfo<CL_PLATFORM_VERSION>() << std::endl;
	std::cout << "[compute] OpenCL Platform: " << _platform.getInfo<CL_PLATFORM_NAME>() << std::endl;
	std::cout << "[compute] OpenCL device: " << _device.getInfo<CL_DEVICE_NAME>() << std::endl;
}
