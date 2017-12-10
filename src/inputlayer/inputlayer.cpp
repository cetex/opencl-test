// ==============
// inputlayer.cpp
// ==============

#include "inputlayer.h"

InputLayer::InputLayer(ComputeSystem &cs, int rows, int cols)
{
	std::cout << "Initializing inputlayer" << std::endl;
	_inputSize.x = rows;
	_inputSize.y = cols;
	_sdrSize.x = rows;
	_sdrSize.y = cols * sizeof(cl_uchar16);

	_cp = new ComputeProgram(cs, std::string("camera.cl"));
	_cs = &cs;
	int clret = 0;
	_kernelInput2SDR = new cl::Kernel(_cp->getProgram(), "Input2SDR", &clret);
	if (clret != CL_SUCCESS) {
		throw std::runtime_error(std::string("[inputlayer/inputlayer] Setup kernel Input2SDR failed, return code: " + std::to_string(clret)));
	}
	_sdr = new cl::Buffer(_cs->getContext(), CL_MEM_READ_WRITE,
			_sdrSize.x * _sdrSize.y * sizeof(uint8_t), NULL, NULL);
	std::cout << "Created sdr cl::Buffer buffer" << std::endl;
	_kernelInput2SDR->setArg(1, *_sdr);

	// Only creating and setting _sdr buffer and kernel arguments.
	// Expecting that _inputData is set and verified by caller.

}

void InputLayer::setInputData(cl::Buffer *inputData)
{
	// Update buffer for inputdata
	_inputData = inputData;
	// Update kernel arguments.
	_kernelInput2SDR->setArg(0, *_inputData);
}

cl::Buffer* InputLayer::getSDR()
{
	std::cout << "About to enqueue _kernelInput2SDR" << std::endl;
	std::cout << _cs->getQueue().getInfo<CL_QUEUE_REFERENCE_COUNT>() << std::endl;
	std::cout << "Enqueueing!" << std::endl;
	int ret = _cs->getQueue().enqueueNDRangeKernel(*_kernelInput2SDR, cl::NullRange, cl::NDRange(_inputSize.x * _inputSize.y));
	std::cout << "getSDR Got returncode from enqueuendrangekernel: " << ret << std::endl;
	_cs->getQueue().finish();
	return _sdr;
}

/*
 * getSDRMat returns a cv::Mat of the sdr buffer
 */
cv::Mat InputLayer::getSDRMat()
{
	getSDR();
	cv::Mat sdr = cv::Mat(_sdrSize.x, _sdrSize.y, CV_8UC1);
	_cs->getQueue().enqueueReadBuffer(*_sdr, CL_TRUE, 0,
			_sdrSize.x * _sdrSize.y,
			sdr.data, NULL, NULL);
	return sdr;
}
