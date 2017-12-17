// ===============
// region.h
// ===============


#ifndef REGION_H
#define REGION_H

#include <opencv2/videoio.hpp>
#include "../compute/compute-system.h"
#include "../compute/compute-program.h"
#include "../inputlayer/inputlayer.h"
#include "../utils/utils.h"
#include <iostream>
#include <random>
#include <functional>
namespace HTM {

class Region
{
	public:
		// Use 2d (rows & cols) or go for 1d?
		Region(ComputeSystem &cs, InputLayer &input, Vec2i columns, Vec2i columnInput, Vec2i columnDistalDim);
		void overlap();
		void inhibit();
		void learn();
		void stepOne();
                
		cv::Mat getSDRMat();
		cv::Mat getDendMat();
		cv::Mat getDistalMat();

		cl::Buffer* getSDRBuff() {
			return _columnsBuff[1];
		}
		Vec2i getColumnDim() {
			return _columnsDim;
		}
		Vec2i getColumnInputDim() {
			return _columnInputDim;
		}
	protected:
		ComputeSystem *_cs = NULL;
	private: 
		ComputeProgram *_cp = NULL;

		// Where we get our input
		InputLayer *_input;

		// The number of columns, x*y
		Vec2i _columnsDim;

		Vec2i _columnsSDRMult;

		// Buffer to store columns
		cl::Buffer *_columnsBuff[2];

		// Buffer to store boost (in reality, how often it is activated)
		cl::Buffer *_boostBuff;
		
		// The size of the input-data that each column sees, 2x2, 3x3, 10x10, 16x16, 32x32 and so on.
		Vec2i _columnInputDim;

		// The number of proximal inputs (feed-forward from input)
		Vec2i _proximalDendritesDim;

		// Storage of proximal dendrites, should be something like _input.getSdrDim()*_columnInput
		cl::Buffer *_proximalDendritesBuff;
		
		// The number of distal dendrites
		Vec2i _distalDendritesDim;

		// Storage of distal dendrites (column <-> column weights)
		cl::Buffer *_distalDendritesBuff;

		// Sparsity, for inhibition
		unsigned int _sparsity;
		// Stepsize, also for inhibition
		unsigned int _stepSize;

		// Kernels
		cl::Kernel *_kernelOverlap;
		cl::Kernel *_kernelInhibit;
		cl::Kernel *_kernelLearn;
		cl::Kernel *_kernelBoost;
};
};
#endif
