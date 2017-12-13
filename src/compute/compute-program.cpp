// ====================
// compute-program.cpp
// ====================

#include "compute-program.h"

#include <fstream>
#include <iostream>

ComputeProgram::ComputeProgram(ComputeSystem &cs, const std::string &fileName)
{
	std::ifstream sourceFile(fileName);

	if (!sourceFile.is_open())
	{
		throw std::runtime_error(std::string("[compute] Could not open file: " + fileName + "!"));
	}

	std::string kernel = "";

	while (!sourceFile.eof() && sourceFile.good())
	{
		std::string line;

		std::getline(sourceFile, line);

		kernel += line + "\n";

	}
	
	_program = cl::Program(cs.getContext(), kernel);

	if (_program.build(std::vector<cl::Device>(1, cs.getDevice())) != CL_SUCCESS)
	{
		throw std::runtime_error(std::string("[compute] Error building: " + _program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(cs.getDevice())));
	}
	std::vector<cl::Kernel> kernels;
	_program.createKernels(&kernels);
	std::cout << "[compute/compute-program] Got kernels: " << std::to_string(kernels.size()) << std::endl;
	for (unsigned int i = 0; i < kernels.size(); i++) {
		std::cout << "[compute/compute-program] ";
		std::cout << std::to_string(i);
		std::cout << ": ";
		std::cout << kernels[i].getInfo<CL_KERNEL_FUNCTION_NAME>(NULL);
		std::cout << ", numargs: ";
		std::cout << kernels[i].getInfo<CL_KERNEL_NUM_ARGS>(NULL) << std::endl;
	}
	std::cout << std::string("[compute] built: " + _program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(cs.getDevice())) << std::endl;
	sourceFile.close();
}
