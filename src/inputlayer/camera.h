// ===============
// camera.h
// ===============


#ifndef CAMERA_H
#define CAMERA_H

#include <opencv2/videoio.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "../compute/compute-system.h"
#include "../compute/compute-program.h"
#include "inputlayer.h"
#include "../utils/utils.h"
#include <iostream>

namespace HTM {
class Camera : public InputLayer
{
	public:
		Camera(ComputeSystem &cs, int rows, int cols);
		cv::Mat getNewImage();
		void convertToGray(cv::Mat&);
		cl::Buffer* getGrayBuffer();
		cv::Mat getGrayMat();

		HTM::Vec2i getDim() {
			return _camDim;
		}
		HTM::Vec2i getGrayDim() {
			return _grayDim;
		}
	private: 
		cv::VideoCapture device;
		ComputeProgram *_cp = NULL;
		cl::Kernel *_kernelBGR2Gray = NULL;
		cl::Buffer *_bgrImage;
		cl::Buffer *_grayImage;
		HTM::Vec2i _camDim;
		HTM::Vec2i _grayDim;
};
};
#endif
