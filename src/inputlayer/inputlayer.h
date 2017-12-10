// ===============
// inputlayer.h
// ===============


#ifndef INPUTLAYER_H
#define INPUTLAYER_H

#include <opencv2/videoio.hpp>
#include "../compute/compute-system.h"
#include "../compute/compute-program.h"
#include "../utils/utils.h"
#include <iostream>

class InputLayer
{
	public:
		InputLayer(ComputeSystem &cs, int _rows, int _cols);
		void setInputData(cl::Buffer *inputData);
		cl::Buffer* getSDR();
		cv::Mat getSDRMat();

		Vec2i getInputSize() {
			return _inputSize;
		}
		Vec2i getSdrSize() {
			return _sdrSize;
		}
	protected:
		ComputeSystem *_cs = NULL;
	private: 
		ComputeProgram *_cp = NULL;
		cl::Kernel *_kernelInput2SDR;
		cl::Buffer *_inputData;
		cl::Buffer *_sdr;
		Vec2i _inputSize;
		Vec2i _sdrSize;
};
#endif
