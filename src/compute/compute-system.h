// =================
// Compute-system.h
// =================

#ifndef COMPUTE_SYSTEM_H
#define COMPUTE_SYSTEM_H

#define CL_HPP_MINIMUM_OPENCL_VERSION 200
#define CL_HPP_TARGET_OPENCL_VERSION 200

#include <CL/cl2.hpp>

class ComputeSystem
{
	public:
		enum DeviceType{_cpu, _gpu, _all};

		ComputeSystem(DeviceType type);
		void printCLInfo();

		cl::Platform getPlatform()
		{
			return _platform;
		}
		
		cl::Device getDevice()
		{
			return _device;
		}

		cl::Context getContext()
		{
			return _context;
		}

		cl::CommandQueue getQueue()
		{
			return _queue;
		}
	
	private:
		cl::Platform _platform;
		cl::Device _device;
		cl::Context _context;
		cl::CommandQueue _queue;
};

#endif
