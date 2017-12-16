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
	// TEST!, make columns a multiplier of (SDR divided by InputDim)
	//_columnsDim = Vec2i(columnsDim.x, columnsDim.y);
	_columnsDim = Vec2i(
			columnsDim.x * (_input->getSDRDim().x / columnInputDim.x),
			columnsDim.y * (_input->getSDRDim().y / columnInputDim.y));
	cl_uint columnsBuffDim[2] = { _columnsDim.x, _columnsDim.y };

	// Set the Radius that each column sees (each column only sees a subset of the input SDR)
	_columnInputDim = Vec2i(columnInputDim.x, columnInputDim.y);
	cl_uint columnInputBuffRadius[2] = { _columnInputDim.x, _columnInputDim.y};

	// We want 2% sparsity, get how much 2% of columns are
	// These 2% of columns should be active
	_sparsity = (_columnsDim.x * _columnsDim.y) * 0.02; 
       	// Inhibition will be applied every _stepSize range of columns, round it up.
	_stepSize = (_columnsDim.x * _columnsDim.y) / _sparsity + ((_columnsDim.x * _columnsDim.y) % _sparsity != 0);

	// Proximal dendrites (feed forward to column from SDR)
	// Multiply size of SDR by the radius each column sees. (input.size * columnInput)
	_proximalDendritesDim = Vec2i(_columnInputDim.x * _columnsDim.x, _columnInputDim.y * _columnsDim.y);
	std::cout << "Proximal dendrites dimensions: " << _proximalDendritesDim.x << ", " << _proximalDendritesDim.y << ", sum: " << _proximalDendritesDim.x * _proximalDendritesDim.y << std::endl;

	cl_uint proximalDendritesBuffDim[2] = { _proximalDendritesDim.x, _proximalDendritesDim.y };

	// Create proximalDendritesBuffRand with initial, random, values.
	std::mt19937::result_type seed = time(0);
	auto _rand = std::bind(std::uniform_int_distribution<int>(40, 200), std::mt19937(seed));

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

	// Create memory-buffer for stepsize (inhibit kernel)
	_stepSizeBuff = new cl::Buffer(_cs->getContext(), CL_MEM_READ_WRITE,
			sizeof(cl_uint), NULL, NULL);

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
	
	// Write stepsize (steps in which to run inhibit kernel) to buffer
	clret = _cs->getQueue().enqueueWriteBuffer(*_stepSizeBuff, CL_TRUE, 0,
			sizeof(cl_uint), &_stepSize, NULL, NULL);
	if (clret != CL_SUCCESS) throw std::runtime_error(std::string("[htm/region] write buffer failed: " + std::to_string(clret)));
	
	// Write proximalDendrites to cl device
	clret = _cs->getQueue().enqueueWriteBuffer(*_proximalDendritesBuff, CL_TRUE, 0,
			_proximalDendritesDim.x * _proximalDendritesDim.y * sizeof(uint8_t),
		       proximalDendritesBuffRand, NULL, NULL);
	if (clret != CL_SUCCESS) throw std::runtime_error(std::string("[htm/region] write random values to _proximalDendritesBuff failed: " + std::to_string(clret)));

	// It's sent of, delete data (free memory)
	delete [] proximalDendritesBuffRand;

        // Write dimensions of columns to _columnBuffDim
        _cs->getQueue().enqueueWriteBuffer(*_proximalDendritesBuffDim, CL_TRUE, 0,
                        2 * sizeof(cl_uint),
                        proximalDendritesBuffDim, NULL, NULL);

	// Create reference to the opencl kernel for overlap calculations
	_kernelOverlap = new cl::Kernel(_cp->getProgram(), "Overlap", &clret);
	if (clret != CL_SUCCESS) {
		throw std::runtime_error(std::string("[htm/region] Setup kernel Overlap failed, return code: " + std::to_string(clret)));
	}

	// Create reference to the opencl kernel for inhibition calculations
	_kernelInhibit = new cl::Kernel(_cp->getProgram(), "Inhibit", &clret);
	if (clret != CL_SUCCESS) {
		throw std::runtime_error(std::string("[htm/region] Setup kernel Inhibit failed, return code: " + std::to_string(clret)));
	}

	// Create reference to the opencl kernel for learning calculations
	_kernelLearn = new cl::Kernel(_cp->getProgram(), "Learn", &clret);
	if (clret != CL_SUCCESS) {
		throw std::runtime_error(std::string("[htm/region] Setup kernel learn failed, return code: " + std::to_string(clret)));
	}

	// Create reference to the opencl kernel for forgetting calculations
	_kernelForget = new cl::Kernel(_cp->getProgram(), "Forget", &clret);
	if (clret != CL_SUCCESS) {
		throw std::runtime_error(std::string("[htm/region] Setup kernel Forget failed, return code: " + std::to_string(clret)));
	}

	std::cout << "[htm/region] columns dimensions: " << _columnsDim.x << ", " << _columnsDim.y << std::endl;
	std::cout << "[htm/region] Columns size: " << _columnsDim.x * _columnsDim.y << std::endl;
	std::cout << "[htm/region] columnInputDim: " << columnInputBuffRadius[0] << ", " << columnInputBuffRadius[1] << std::endl;
	std::cout << "[htm/region] sparsity is: " << _sparsity << std::endl;
	std::cout << "[htm/region] Stepsize is: " << _stepSize << std::endl;
	std::cout << "[htm/region] proximalDendritesDim dimensions: " << _proximalDendritesDim.x << ", " << _proximalDendritesDim.y << std::endl;
	std::cout << "[htm/region] proximalDendrites size: " << _proximalDendritesDim.x * _proximalDendritesDim.y << std::endl;
}

void Region::overlap()
{
	int clret = 0;
	int arg   = 0;
	
	clret = _kernelOverlap->setArg(arg++, *_columnsBuff);
	if (clret != CL_SUCCESS) throw std::runtime_error(std::string("[htm/region] kernel setarg " + std::to_string(arg) + " failed, return code: " + std::to_string(clret)));

	clret = _kernelOverlap->setArg(arg++, *_columnsBuffDim);
	if (clret != CL_SUCCESS) throw std::runtime_error(std::string("[htm/region] kernel setarg " + std::to_string(arg) + " failed, return code: " + std::to_string(clret)));

	clret = _kernelOverlap->setArg(arg++, *_columnInputBuffDim);
	if (clret != CL_SUCCESS) throw std::runtime_error(std::string("[htm/region] kernel setarg " + std::to_string(arg) + " failed, return code: " + std::to_string(clret)));

	clret = _kernelOverlap->setArg(arg++, *(_input->getSDRBuff()));
	if (clret != CL_SUCCESS) throw std::runtime_error(std::string("[htm/region] kernel setarg " + std::to_string(arg) + " failed, return code: " + std::to_string(clret)));

	clret = _kernelOverlap->setArg(arg++, *(_input->getSDRBuffDim()));
	if (clret != CL_SUCCESS) throw std::runtime_error(std::string("[htm/region] kernel setarg " + std::to_string(arg) + " failed, return code: " + std::to_string(clret)));

	clret = _kernelOverlap->setArg(arg++, *_proximalDendritesBuff);
	if (clret != CL_SUCCESS) throw std::runtime_error(std::string("[htm/region] kernel setarg " + std::to_string(arg) + " failed, return code: " + std::to_string(clret)));

	clret = _kernelOverlap->setArg(arg++, *_proximalDendritesBuffDim);
	if (clret != CL_SUCCESS) throw std::runtime_error(std::string("[htm/region] kernel setarg " + std::to_string(arg) + " failed, return code: " + std::to_string(clret)));

	std::cout << "[htm/region] About to enqueue Overlap kernel, dimensions: " << _columnsDim.x << ", " << _columnsDim.y << std::endl;
	int ret = _cs->getQueue().enqueueNDRangeKernel(*_kernelOverlap, cl::NullRange, cl::NDRange(_columnsDim.x, _columnsDim.y), cl::NullRange);
	if (ret != CL_SUCCESS) std::cout << "[htm/region] Run kernel overlap failed, return code: " << std::to_string(ret) << std::endl;
}

void Region::inhibit()
{
	int clret = 0;
	int arg   = 0;
	
	clret = _kernelInhibit->setArg(arg++, *_columnsBuff);
	if (clret != CL_SUCCESS) throw std::runtime_error(std::string("[htm/region] kernel setarg " + std::to_string(arg) + " failed, return code: " + std::to_string(clret)));

	clret = _kernelInhibit->setArg(arg++, *_columnsBuffDim);
	if (clret != CL_SUCCESS) throw std::runtime_error(std::string("[htm/region] kernel setarg " + std::to_string(arg) + " failed, return code: " + std::to_string(clret)));

	clret = _kernelInhibit->setArg(arg++, *_stepSizeBuff);
	if (clret != CL_SUCCESS) throw std::runtime_error(std::string("[htm/region] kernel setarg " + std::to_string(arg) + " failed, return code: " + std::to_string(clret)));

	std::cout << "[htm/region] About to enqueue Inhibit kernel, dimensions: " << _columnsDim.x << ", " << _columnsDim.y << std::endl;
	int ret = _cs->getQueue().enqueueNDRangeKernel(*_kernelInhibit, cl::NullRange, cl::NDRange(_sparsity), cl::NullRange);
	if (ret != CL_SUCCESS) std::cout << "[htm/region] Run kernel Inhibit failed, return code: " << std::to_string(ret) << std::endl;
}

void Region::learn()
{
	int clret = 0;
	int arg   = 0;
	
	clret = _kernelLearn->setArg(arg++, *_columnsBuff);
	if (clret != CL_SUCCESS) throw std::runtime_error(std::string("[htm/region] kernel setarg " + std::to_string(arg) + " failed, return code: " + std::to_string(clret)));

	clret = _kernelLearn->setArg(arg++, *_columnsBuffDim);
	if (clret != CL_SUCCESS) throw std::runtime_error(std::string("[htm/region] kernel setarg " + std::to_string(arg) + " failed, return code: " + std::to_string(clret)));

	clret = _kernelLearn->setArg(arg++, *_columnInputBuffDim);
	if (clret != CL_SUCCESS) throw std::runtime_error(std::string("[htm/region] kernel setarg " + std::to_string(arg) + " failed, return code: " + std::to_string(clret)));

	clret = _kernelLearn->setArg(arg++, *(_input->getSDRBuff()));
	if (clret != CL_SUCCESS) throw std::runtime_error(std::string("[htm/region] kernel setarg " + std::to_string(arg) + " failed, return code: " + std::to_string(clret)));

	clret = _kernelLearn->setArg(arg++, *(_input->getSDRBuffDim()));
	if (clret != CL_SUCCESS) throw std::runtime_error(std::string("[htm/region] kernel setarg " + std::to_string(arg) + " failed, return code: " + std::to_string(clret)));

	clret = _kernelLearn->setArg(arg++, *_proximalDendritesBuff);
	if (clret != CL_SUCCESS) throw std::runtime_error(std::string("[htm/region] kernel setarg " + std::to_string(arg) + " failed, return code: " + std::to_string(clret)));

	clret = _kernelLearn->setArg(arg++, *_proximalDendritesBuffDim);
	if (clret != CL_SUCCESS) throw std::runtime_error(std::string("[htm/region] kernel setarg " + std::to_string(arg) + " failed, return code: " + std::to_string(clret)));

	std::cout << "[htm/region] About to enqueue Learn kernel, dimensions: " << _columnsDim.x << ", " << _columnsDim.y << std::endl;
	int ret = _cs->getQueue().enqueueNDRangeKernel(*_kernelLearn, cl::NullRange, cl::NDRange(_columnsDim.x, _columnsDim.y), cl::NullRange);
	if (ret != CL_SUCCESS) std::cout << "[htm/region] Run kernel Learn failed, return code: " << std::to_string(ret) << std::endl;
}

void Region::forget()
{
	int clret = 0;
	int arg   = 0;
	
	clret = _kernelInhibit->setArg(arg++, *_proximalDendritesBuff);
	if (clret != CL_SUCCESS) throw std::runtime_error(std::string("[htm/region] kernel setarg " + std::to_string(arg) + " failed, return code: " + std::to_string(clret)));

	std::cout << "[htm/region] About to enqueue forget kernel, dimensions: " << _proximalDendritesDim.x << ", " << _proximalDendritesDim.y << std::endl;
	int ret = _cs->getQueue().enqueueNDRangeKernel(*_kernelInhibit, cl::NullRange, cl::NDRange(_proximalDendritesDim.x, _proximalDendritesDim.y), cl::NullRange);
	if (ret != CL_SUCCESS) std::cout << "[htm/region] Run kernel Inhibit failed, return code: " << std::to_string(ret) << std::endl;
}


void Region::stepOne()
{
	std::cout << "Running overlap" << std::endl;
	overlap();
	_cs->getQueue().finish();
	std::cout << "Running inhibit" << std::endl;
	inhibit();
	_cs->getQueue().finish();
	std::cout << "running learn" << std::endl;
	learn();
	_cs->getQueue().finish();
	std::cout << "learn done" << std::endl;
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
	std::cout << "[htm/region] Columns pixel 0-3 is: " << +sdr.data[0] << "," << +sdr.data[1] << "," << +sdr.data[2] << "," << +sdr.data[3] << std::endl;
	return sdr;
}

// getdendMat returns a cv::Mat of the Dendrite buffer
cv::Mat Region::getDendMat()
{
	// Create placeholder for returned grayscale image from opencl (8-bit Unsigned Char, 1 colour-channel)
	// Returned data will in reality be binary, 1 or 0.
	cv::Mat sdr = cv::Mat(_proximalDendritesDim.x, _proximalDendritesDim.y, CV_8UC1);

	// Read back SDR data from SDR buffer
	_cs->getQueue().enqueueReadBuffer(*_proximalDendritesBuff, CL_TRUE, 0,
			_proximalDendritesDim.x * _proximalDendritesDim.y,
			sdr.data, NULL, NULL);
	std::cout << "[htm/region] Dendrites pixel 0-3 is: " << +sdr.data[0] << "," << +sdr.data[1] << "," << +sdr.data[2] << "," << +sdr.data[3] << std::endl;
	return sdr;
}

};
