// ==============
// inputlayer.cpp
// ==============

#include "inputlayer.h"

InputLayer::InputLayer(ComputeSystem &cs, int rows, int cols)
{
	_inputSize.x = rows;
	_inputSize.y = cols;

	// SDR is in this setup statically set to 16 bits which means 16 uchars in this setup
	_sdrSize.x = rows;
	_sdrSize.y = cols * sizeof(cl_uchar16);

	// Store compute-system for later use
	_cs = &cs;

	// Create instance of compute-program with the opencl code
	_cp = new ComputeProgram(cs, std::string("camera.cl"));

	// Create reference to opencl kernel
	int clret = 0;
	_kernelInput2SDR = new cl::Kernel(_cp->getProgram(), "Input2SDR", &clret);
	if (clret != CL_SUCCESS) {
		throw std::runtime_error(std::string("[inputlayer/inputlayer] Setup kernel Input2SDR failed, return code: " + std::to_string(clret)));
	}

	// Create memory-buffer for the SDR (output from inputlayer)
	_sdr = new cl::Buffer(_cs->getContext(), CL_MEM_READ_WRITE,
			_sdrSize.x * _sdrSize.y * sizeof(uint8_t), NULL, NULL);
	std::cout << "Created sdr cl::Buffer buffer" << std::endl;

	// Set the second (first is index 0) kernel argument to _sdr
	// The first kernel argument is set in setInputData function
	_kernelInput2SDR->setArg(1, *_sdr);

	// Only creating and setting _sdr buffer and kernel arguments here.
	// Expecting that _inputData is set and verified by caller
	// through setInputdata.

}

void InputLayer::setInputData(cl::Buffer *inputData)
{
	// Update pointer to inputdata
	_inputData = inputData;
	// Update kernel arguments to use new pointer
	_kernelInput2SDR->setArg(0, *_inputData);
}

cl::Buffer* InputLayer::getSDR()
{
	// Run the input2SDR kernel (converts bytes to 16-bit (actually 16-byte) SDR's
	std::cout << "About to enqueue _kernelInput2SDR" << std::endl;
	int ret = _cs->getQueue().enqueueNDRangeKernel(*_kernelInput2SDR, cl::NullRange, cl::NDRange(_inputSize.x * _inputSize.y));
	std::cout << "getSDR Got returncode from enqueuendrangekernel: " << ret << std::endl;

	// Return the pointer to the SDR buffer
	return _sdr;
}

/*
 * getSDRMat returns a cv::Mat of the sdr buffer
 */
cv::Mat InputLayer::getSDRMat()
{
	// Fixme, this shouldn't be run here as we risk running it twice.
	getSDR();
	
	// Create placeholder for returned grayscale image from opencl (8-bit Unsigned Char, 1 colour-channel)
	// Returned data will in reality be binary, 1 or 0.
	cv::Mat sdr = cv::Mat(_sdrSize.x, _sdrSize.y, CV_8UC1);

	// Read back SDR data from SDR buffer
	_cs->getQueue().enqueueReadBuffer(*_sdr, CL_TRUE, 0,
			_sdrSize.x * _sdrSize.y,
			sdr.data, NULL, NULL);
	return sdr;
}
