// ===============
// camera.h
// ===============


#ifndef CAMERA_H
#define CAMERA_H

#include <opencv2/videoio.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "../compute/compute-system.h"
#include "../compute/compute-program.h"
#include <iostream>

class Camera
{
	public:
		Camera(ComputeSystem cs, int _rows, int _cols);
		Camera(ComputeSystem cs, cv::VideoCapture *input);
		cv::Mat getNewImage();
		void convertToGray(cv::Mat&);
		cl::Buffer* getGrayBuffer();
		cv::Mat getGrayMat();
	private: 
		cv::VideoCapture device;
		ComputeSystem *_cs;
		ComputeProgram *_cp;
		cl::Kernel *_kernelBGR2Gray;
		cl::Buffer _bgrImage;
		cl::Buffer _grayImage;
		cl::Buffer _sdrImage;
		int _rows;
		int _cols;
};
#endif
