// ===============
// camera.h
// ===============


#ifndef CAMERA_H
#define CAMERA_H

#include <opencv2/videoio.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "../compute/compute-system.h"
#include "../compute/compute-program.h"
#include "../utils/utils.h"
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
		cl::Buffer* getSDR();
		cv::Mat getSDRMat();

		Vec2i getSize() {
			return _camSize;
		}
		Vec2i getGraySize() {
			return Vec2i(_camSize.x, _camSize.y*3);
		}
		Vec2i getSdrSize() {
			return Vec2i(_camSize.x, _camSize.y*16);
		}
	private: 
		cv::VideoCapture device;
		ComputeSystem *_cs;
		ComputeProgram *_cp;
		cl::Kernel *_kernelBGR2Gray;
		cl::Kernel *_kernelGray2SDR;
		cl::Buffer _bgrImage;
		cl::Buffer _grayImage;
		cl::Buffer _sdr;
		Vec2i _camSize;
};
#endif
