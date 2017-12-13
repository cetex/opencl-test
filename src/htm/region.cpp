// ==============
// region.cpp
// ==============

#include "region.h"

namespace HTM {

Region::Region(ComputeSystem &cs, InputLayer &input, Vec2i columnsDim, Vec2i columnInputDim)
{
	/*
	 * A region needs to keep track of input, the number of columns, and the overlap between columns (synapse-wise)
	 * It will need to know the input dim (X / Y) as well as be able to set the input data-buffer on every time step if required.
	 * This means someone externally (user) needs an option to repoint the input-SDR buffer as long as the dimensions of it is correct.
	 * We need a class which wraps buffers ... ...
	*/
	
	// Store compute-system for later use
	_cs = &cs;

	// Create instance of compute-program with the opencl code
	_cp = new ComputeProgram(cs, std::string("kernel.cl"));
	
	// Store pointer to InputLayer
	_input = &input;

	// Set dimensions of columns
	_columnsDim = Vec2i(columnsDim.x, columnsDim.y);
	cl_uint columnsBuffDim[2] = { _columnsDim.x, _columnsDim.y };

	// Set the Radius that each column sees (each column only sees a subset of the input SDR)
	_columnInputDim = Vec2i(columnInputDim.x, columnInputDim.y);
	cl_uint columnInputBuffRadius[2] = { _columnInputDim.x, _columnInputDim.y };

	// Proximal dendrites (feed forward to column from SDR)
	// Multiply size of SDR by the radius each column sees. (input.size * columnInput)
	_proximalDendritesDim = Vec2i(_input->getSDRDim().x * _columnInputDim.x, _input->getSDRDim().y * _columnInputDim.y);


	cl_uint proximalDendritesBuffDim[2] = { _proximalDendritesDim.x, _proximalDendritesDim.y };

	// Create proximalDendritesBuffRand with initial, random, values.
	std::mt19937::result_type seed = time(0);
	auto _rand = std::bind(std::uniform_int_distribution<int>(80, 160), std::mt19937(seed));

	uint8_t *proximalDendritesBuffRand = new uint8_t[_proximalDendritesDim.x * _proximalDendritesDim.y];
	for (unsigned int i = 0; i < (_proximalDendritesDim.x * _proximalDendritesDim.y); i++) {
		proximalDendritesBuffRand[i] = _rand();
	}

	// Create memory-buffer for columns
	_columnsBuff = new cl::Buffer(_cs->getContext(), CL_MEM_READ_WRITE,
			_columnsDim.x * _columnsDim.y * sizeof(uint8_t), NULL, NULL);
	
	// Create memory-buffer for columns dimensions (x, y)
	_columnsBuffDim = new cl::Buffer(_cs->getContext(), CL_MEM_READ_ONLY,
			2 * sizeof(cl_uint), NULL, NULL);

        // Write dimensions of columns to _columnBuffDim
        _cs->getQueue().enqueueWriteBuffer(*_columnsBuffDim, CL_TRUE, 0,
                        2 * sizeof(cl_uint),
                        columnsBuffDim, NULL, NULL);

	// Create memory-buffer for input-dimensions for each column (how many bits from the SDR it sees)
	_columnInputBuffDim = new cl::Buffer(_cs->getContext(), CL_MEM_READ_WRITE,
			2 * sizeof(cl_uint), NULL, NULL);

	// Create buffer to store proximal dendrites in
	_proximalDendritesBuff = new cl::Buffer(_cs->getContext(), CL_MEM_READ_WRITE,
			_proximalDendritesDim.x * _proximalDendritesDim.y * sizeof(uint8_t), NULL, NULL);

	// Create memory-buffer for proximal dendrites dimensions (x, y)
	_proximalDendritesBuffDim = new cl::Buffer(_cs->getContext(), CL_MEM_READ_WRITE,
			2 * sizeof(cl_uint), NULL, NULL);

	// Write column input-dimensions to buffer
	int clret = _cs->getQueue().enqueueWriteBuffer(*_columnInputBuffDim, CL_TRUE, 0,
			2 * sizeof(cl_uint), columnInputBuffRadius, NULL, NULL);
	if (clret != CL_SUCCESS) throw std::runtime_error(std::string("[htm/region] write buffer failed: " + std::to_string(clret)));
	
	// Write proximalDendrites to cl device
	clret = _cs->getQueue().enqueueWriteBuffer(*_proximalDendritesBuff, CL_TRUE, 0,
			_proximalDendritesDim.x * _proximalDendritesDim.y * sizeof(uint8_t),
		       proximalDendritesBuffRand, NULL, NULL);
	if (clret != CL_SUCCESS) throw std::runtime_error(std::string("[htm/region] write random values to _proximalDendritesBuff failed: " + std::to_string(clret)));

	// It's sent of, delete data (free memory)
	delete proximalDendritesBuffRand;

        // Write dimensions of columns to _columnBuffDim
        _cs->getQueue().enqueueWriteBuffer(*_proximalDendritesBuffDim, CL_TRUE, 0,
                        2 * sizeof(cl_uint),
                        proximalDendritesBuffDim, NULL, NULL);

	// Create reference to the opencl kernel for overlap calculations
	_kernelOverlap = new cl::Kernel(_cp->getProgram(), "CalculateOverlap", &clret);
	if (clret != CL_SUCCESS) {
		throw std::runtime_error(std::string("[htm/region] Setup kernel CalculateOverlap failed, return code: " + std::to_string(clret)));
	}

	std::cout << "[htm/region] columns dimensions: " << _columnsDim.x << ", " << _columnsDim.y << std::endl;
	std::cout << "[htm/region] Columns size: " << _columnsDim.x * _columnsDim.y << std::endl;
	std::cout << "[htm/region] columnInputDim: " << columnInputBuffRadius[0] << ", " << columnInputBuffRadius[1] << std::endl;
	std::cout << "[htm/region] proximalDendritesDim dimensions: " << _proximalDendritesDim.x << ", " << _proximalDendritesDim.y << std::endl;
	std::cout << "[htm/region] proximalDendrites size: " << _proximalDendritesDim.x * _proximalDendritesDim.y << std::endl;
}

void Region::overlap()
{
	int clret = 0;
	/*
	clret = _kernelOverlap->setArg(0, *(_input->getSDRBuff()));
	if (clret != CL_SUCCESS) throw std::runtime_error(std::string("[htm/region] kernel setarg 0 failed, return code: " + std::to_string(clret)));

	clret = _kernelOverlap->setArg(1, *(_input->getSDRBuffDim()));
	if (clret != CL_SUCCESS) throw std::runtime_error(std::string("[htm/region] kernel setarg 1 failed, return code: " + std::to_string(clret)));

	clret = _kernelOverlap->setArg(2, *_columnsBuff);
	if (clret != CL_SUCCESS) throw std::runtime_error(std::string("[htm/region] kernel setarg 2 failed, return code: " + std::to_string(clret)));

	clret = _kernelOverlap->setArg(3, *_columnsBuffDim);
	if (clret != CL_SUCCESS) throw std::runtime_error(std::string("[htm/region] kernel setarg 3 failed, return code: " + std::to_string(clret)));

	clret = _kernelOverlap->setArg(4, *_proximalDendritesBuff);
	if (clret != CL_SUCCESS) throw std::runtime_error(std::string("[htm/region] kernel setarg 4 failed, return code: " + std::to_string(clret)));

	clret = _kernelOverlap->setArg(5, *_proximalDendritesBuffDim);
	if (clret != CL_SUCCESS) throw std::runtime_error(std::string("[htm/region] kernel setarg 5 failed, return code: " + std::to_string(clret)));
	*/
	clret = _kernelOverlap->setArg(0, *_columnsBuff);
	if (clret != CL_SUCCESS) throw std::runtime_error(std::string("[htm/region] kernel setarg 0 failed, return code: " + std::to_string(clret)));

	clret = _kernelOverlap->setArg(1, *_columnsBuffDim);
	if (clret != CL_SUCCESS) throw std::runtime_error(std::string("[htm/region] kernel setarg 1 failed, return code: " + std::to_string(clret)));

	clret = _kernelOverlap->setArg(2, *_columnInputBuffDim);
	if (clret != CL_SUCCESS) throw std::runtime_error(std::string("[htm/region] kernel setarg 2 failed, return code: " + std::to_string(clret)));

	clret = _kernelOverlap->setArg(3, *(_input->getSDRBuff()));
	if (clret != CL_SUCCESS) throw std::runtime_error(std::string("[htm/region] kernel setarg 3 failed, return code: " + std::to_string(clret)));

	clret = _kernelOverlap->setArg(4, *(_input->getSDRBuffDim()));
	if (clret != CL_SUCCESS) throw std::runtime_error(std::string("[htm/region] kernel setarg 4 failed, return code: " + std::to_string(clret)));

	clret = _kernelOverlap->setArg(5, *_proximalDendritesBuff);
	if (clret != CL_SUCCESS) throw std::runtime_error(std::string("[htm/region] kernel setarg 5 failed, return code: " + std::to_string(clret)));

	clret = _kernelOverlap->setArg(6, *_proximalDendritesBuffDim);
	if (clret != CL_SUCCESS) throw std::runtime_error(std::string("[htm/region] kernel setarg 6 failed, return code: " + std::to_string(clret)));

	std::cout << "[htm/region] About to enqueue Overlap kernel, dimensions: " << _columnsDim.x << ", " << _columnsDim.y << std::endl;
	int ret = _cs->getQueue().enqueueNDRangeKernel(*_kernelOverlap, cl::NullRange, cl::NDRange(_columnsDim.x, _columnsDim.y), cl::NullRange);
	if (ret != CL_SUCCESS) std::cout << "[htm/region] Run kernel overlap failed, return code: " << std::to_string(ret) << std::endl;
}

void Region::stepOne()
{
	overlap();
}

cl::Buffer* Region::getSDRBuff()
{
	return _columnsBuff;
}

cl::Buffer* Region::getSDRBuffDim()
{
	return _columnsBuffDim;
}

// 
// getSDRMat returns a cv::Mat of the sdr buffer
//
cv::Mat Region::getSDRMat()
{
	// Create placeholder for returned grayscale image from opencl (8-bit Unsigned Char, 1 colour-channel)
	// Returned data will in reality be binary, 1 or 0.
	cv::Mat sdr = cv::Mat(_columnsDim.x, _columnsDim.y, CV_8UC1);

	// Read back SDR data from SDR buffer
	_cs->getQueue().enqueueReadBuffer(*_columnsBuff, CL_TRUE, 0,
			_columnsDim.x * _columnsDim.y,
			sdr.data, NULL, NULL);
	std::cout << "pixel 0 is: " << +sdr.data[0] << std::endl;
	return sdr;
}
};
