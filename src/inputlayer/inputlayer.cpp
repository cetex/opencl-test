// ==============
// inputlayer.cpp
// ==============

#include "inputlayer.h"
namespace HTM {

InputLayer::InputLayer(ComputeSystem &cs, int rows, int cols)
{
	_inputDim.x = rows;
	_inputDim.y = cols;

	// SDR is in this setup statically set to 16 bits which means 16 uchars in this setup
	_sdrDim.x = rows;
	_sdrDim.y = cols * sizeof(cl_uchar16);

	// Store compute-system for later use
	_cs = &cs;

	// Create instance of compute-program with the opencl code
	_cp = new ComputeProgram(cs, std::string("inputlayer.cl"));

	// Create reference to opencl kernel
	int clret = 0;
	_kernelInput2SDR = new cl::Kernel(_cp->getProgram(), "Input2SDR", &clret);
	if (clret != CL_SUCCESS) {
		throw std::runtime_error(std::string("[inputlayer/inputlayer] Setup kernel Input2SDR failed, return code: " + std::to_string(clret)));
	}

	// Create memory-buffer for the SDR (output from inputlayer)
	_sdrBuff = new cl::Buffer(_cs->getContext(), CL_MEM_READ_WRITE,
			_sdrDim.x * _sdrDim.y * sizeof(uint8_t), NULL, NULL);
	std::cout << "[inputlayer/inputlayer] Created sdr cl::Buffer buffer of size: " << _sdrDim.x << ", " << _sdrDim.y << ", sum: " << _sdrDim.x * _sdrDim.y<< std::endl;

	// Create memory-buffer for the SDR Dimensions (CL::Buffer containing Size of output from inputlayer)
	_sdrBuffDim = new cl::Buffer(_cs->getContext(), CL_MEM_READ_WRITE,
			2 * sizeof(cl_uint), NULL, NULL);
	std::cout << "[inputlayer/inputlayer] Created sdr cl::Buffer buffer" << std::endl;

	cl_uint sdrBuffDim[2] = { (cl_uint)_sdrDim.x, (cl_uint)_sdrDim.y };

	// Write SDR size to _sdrBuffDim
	_cs->getQueue().enqueueWriteBuffer(*_sdrBuffDim, CL_TRUE, 0,
			2 * sizeof(cl_uint),
			sdrBuffDim, NULL, NULL);


	// Set the second (first is index 0) kernel argument to _sdr
	// The first kernel argument is set in setInputData function
	_kernelInput2SDR->setArg(1, *_sdrBuff);

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

void InputLayer::input2SDR() {
	// Run the input2SDR kernel (converts bytes to 16-bit (actually 16-byte) SDR's
	std::cout << "[inputlayer/inputlayer] About to enqueue _kernelInput2SDR" << std::endl;
	int ret = _cs->getQueue().enqueueNDRangeKernel(*_kernelInput2SDR, cl::NullRange, cl::NDRange(_inputDim.x * _inputDim.y));
	std::cout << "[inputlayer/inputlayer] input2SDR Got returncode from enqueuendrangekernel: " << ret << std::endl;
}

void InputLayer::stepOne() {
	input2SDR();
}

/*
 * getSDRMat returns a cv::Mat of the sdr buffer
 */
cv::Mat InputLayer::getSDRMat()
{
	// Create placeholder for returned grayscale image from opencl (8-bit Unsigned Char, 1 colour-channel)
	// Returned data will in reality be binary, 1 or 0.
	cv::Mat sdr = cv::Mat(_sdrDim.x, _sdrDim.y, CV_8UC1);

	// Read back SDR data from SDR buffer
	_cs->getQueue().enqueueReadBuffer(*_sdrBuff, CL_TRUE, 0,
			_sdrDim.x * _sdrDim.y,
			sdr.data, NULL, NULL);
	return sdr;
}
};
