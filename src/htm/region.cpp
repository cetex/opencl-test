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
	
	// Set the Input-dimensions that each column sees (each column only sees a subset of the input SDR)
	_columnInputDim = Vec2i(columnInputDim.x, columnInputDim.y);
	
	// Create memory-buffer for columns
	_columnsBuff = new cl::Buffer(_cs->getContext(), CL_MEM_READ_WRITE,
			_columnsDim.x * _columnsDim.y * sizeof(uint8_t), NULL, NULL);
	std::cout << "Created Columns cl::Buffer buffer of size: " << _columnsDim.x * _columnsDim.y << std::endl;
	
	// Create memory-buffer for columns dimensions (x, y)
	_columnsBuffDim = new cl::Buffer(_cs->getContext(), CL_MEM_READ_WRITE,
			2 * sizeof(cl_uint), NULL, NULL);
	std::cout << "Created ColumnsDim cl::Buffer" << std::endl;

	cl_uint columnsBuffDim[2] = { _columnsDim.x, _columnsDim.y };

        // Write dimensions of columns to _columnBuffDim
        _cs->getQueue().enqueueWriteBuffer(*_columnsBuffDim, CL_TRUE, 0,
                        2 * sizeof(cl_uint),
                        columnsBuffDim, NULL, NULL);


	// Figure out how many proximal dendrites there should be (Connections from input-SDR to neurons)
	// Depends on how we want to calculate the field of view.
	// If it's an image, we may not care that much about what's going on towards the edges of the picture,
	// then we can "fade out" the number of columns the closer to the edge we get.
	// Assuming this now. Just remove half the size of the columns receptive field on each end of the image.
	//
	// Multiply size of SDR by the input each column sees. (input.size * columnInput)
	_distalDendritesDim = Vec2i(_input->getSDRDim().x * _columnInputDim.x, _input->getSDRDim().y * _columnInputDim.y);
	
	// Create buffer to store distal dendrites in
	_distalDendrites = new cl::Buffer(_cs->getContext(), CL_MEM_READ_WRITE,
			_distalDendritesDim.x * _distalDendritesDim.y * sizeof(uint8_t), NULL, NULL);
	std::cout << "Created distalDendrites cl::Buffer of size: " << _distalDendritesDim.x * _distalDendritesDim.y << std::endl;

	// Will need wrap-around on edges.
	
	
	// Create reference to the opencl kernel for overlap calculations
	int clret = 0;
	_kernelOverlap = new cl::Kernel(_cp->getProgram(), "CalculateOverlap", &clret);
	if (clret != CL_SUCCESS) {
		throw std::runtime_error(std::string("[htm/region] Setup kernel CalculateOverlap failed, return code: " + std::to_string(clret)));
	}

	// Set the second (first is index 0) kernel argument to _sdr
	// The first kernel argument is set in setInputData function
	//_kernelInput2SDR->setArg(1, *_sdr);

	// Only creating and setting _sdr buffer and kernel arguments here.
	// Expecting that _inputData is set and verified by caller
	// through setInputdata.
}

void Region::overlap()
{
	_kernelOverlap->setArg(0, _input->getSDRBuff());
	_kernelOverlap->setArg(1, _input->getSDRDim());
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
	return sdr;
}
};
