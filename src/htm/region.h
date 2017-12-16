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
		Region(ComputeSystem &cs, InputLayer &input, Vec2i columns, Vec2i columnInput);
		//cl::Buffer* getColumns();
		//cv::Mat getColumnsMat();
		void overlap();
		void inhibit();
		void learn();
		void forget();
		void stepOne();
		// Kernel: Calculate overlap
		// Kernel: Apply Inhibition
		// Kernel: Learn
		// Kernel: Boost
		//
                cl::Buffer* getSDRBuff();
                cl::Buffer* getSDRBuffDim();
		cv::Mat getSDRMat();
		cv::Mat getDendMat();


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

		// Buffer to store columns
		cl::Buffer *_columnsBuff;
		
		// Buffer to store columns size
		cl::Buffer *_columnsBuffDim;
		
		// The size of the input-data that each column sees, 2x2, 3x3, 10x10, 16x16, 32x32 and so on.
		Vec2i _columnInputDim;

		// Buffer to store the size of the input-data that each column sees.
		cl::Buffer *_columnInputBuffDim;

		// The number of proximal inputs (feed-forward from input)
		Vec2i _proximalDendritesDim;

		// Storage of proximal dendrites, should be something like _input.getSdrDim()*_columnInput
		cl::Buffer *_proximalDendritesBuff;

		// The size of the proximal dendrites buffer
		cl::Buffer *_proximalDendritesBuffDim;

		// Sparsity, for inhibition
		unsigned int _sparsity;
		// Stepsize, also for inhibition
		unsigned int _stepSize;

		// Inhibit StepSizeBuff
		cl::Buffer *_stepSizeBuff;

		// Kernels
		cl::Kernel *_kernelOverlap;
		cl::Kernel *_kernelInhibit;
		cl::Kernel *_kernelLearn;
		cl::Kernel *_kernelForget;
		cl::Kernel *_kernelBoost;


	       	// Skipping temporal pooler for now.
		// Temporal pooler needs to know which cells were previously activated
		// Maybe this is when we need to use a double-buffer of all cell activations
		// from this run and the previous run so we can see what cells are active now
		// and which ones were active the previous cycle, and then increase the 
		Vec2i _cell_synapse_connections_Dim; // Better name?, size of the grid of cell - cell connections, depends on the amount of overlap of cells. (10% 50%? 100%, size of grid varies from something like cells^(cells*0.1) to cells^(cells*1.0)
};

};
#endif
