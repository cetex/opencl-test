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

class Camera : public InputLayer
{
	public:
		Camera(ComputeSystem &cs, int rows, int cols);
		cv::Mat getNewImage();
		void convertToGray(cv::Mat&);
		cl::Buffer* getGrayBuffer();
		cv::Mat getGrayMat();

		Vec2i getSize() {
			return _camSize;
		}
		Vec2i getGraySize() {
			return _graySize;
		}
	private: 
		cv::VideoCapture device;
		ComputeProgram *_cp = NULL;
		cl::Kernel *_kernelBGR2Gray = NULL;
		cl::Buffer *_bgrImage;
		cl::Buffer *_grayImage;
		Vec2i _camSize;
		Vec2i _graySize;
};
#endif
