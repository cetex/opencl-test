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

namespace HTM {

class InputLayer
{
	public:
		InputLayer(ComputeSystem &cs, int rows, int cols);
		void setInputData(cl::Buffer *inputData);
		cv::Mat getSDRMat();
		void input2SDR();
		void stepOne();

		cl::Buffer* getSDRBuff() {
			return _sdrBuff;
		};
		cl::Buffer* getSDRBuffDim() {
			return _sdrBuffDim;
		}
		Vec2i getInputDim() {
			return _inputDim;
		}
		Vec2i getSDRDim() {
			return _sdrDim;
		}
		Vec2i getSDRBits() {
			return _sdrBits;
		}
	protected:
		ComputeSystem *_cs = NULL;
	private: 
		ComputeProgram *_cp = NULL;
		cl::Kernel *_kernelInput2SDR;
		cl::Buffer *_inputData;
		cl::Buffer *_sdrBuff;
		cl::Buffer *_sdrBuffDim;
		Vec2i _inputDim;
		Vec2i _sdrDim;
		Vec2i _sdrBits;
};
};
#endif
