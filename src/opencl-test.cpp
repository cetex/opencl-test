//============================================================================
// Name        : opencl-test.cpp
// Author      : 
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <iostream>
#include <unistd.h>
#include "compute/compute-system.h"
#include "compute/compute-program.h"
#include "utils/utils.h"
#include "inputlayer/camera.h"
//#include "architect/architect.h"
#include <opencv2/highgui/highgui.hpp>

using namespace std;

int main( int argc, char** argv )
{
	// Setup ComputeSystem
	ComputeSystem cs = ComputeSystem(ComputeSystem::_gpu);
	cs.printCLInfo();
	// Setup ComputeProgram
	//const std::string fileName = getcwd_string() + std::string("/kernel.cl");
	//ComputeProgram cp = ComputeProgram(cs, fileName);

	Camera cap1 = Camera(cs, 120, 160);
	//Architect arch = Architect(cs, 120, 160);
	cv::namedWindow("cap1_orig", cv::WINDOW_NORMAL);
	cv::namedWindow("cap1_gray", cv::WINDOW_NORMAL);
	cv::namedWindow("cap1_sdr", cv::WINDOW_NORMAL);
	while (true) {
		cv::Mat image = cap1.getNewImage();
		std::cout << "Image Size: " << cap1.getSize() << ", Gray Size: " << cap1.getGraySize() << ", SDR size: " << cap1.getSdrSize() << std::endl;
		cv::Mat gray = cap1.getGrayMat();
		std::cout << "Got gray mat" << std::endl;
		cv::Mat sdr = cap1.getSDRMat();
		std::cout << "Got sdr mat" << std::endl;

		cv::imshow("cap1_orig", image);
		cv::resizeWindow("cap1_orig", image.cols, image.rows);
       		cv::imshow("cap1_gray", gray);
		cv::resizeWindow("cap1_gray", gray.cols, gray.rows);
		cv::imshow("cap1_sdr", sdr);
		cv::resizeWindow("cap1_sdr", sdr.cols, sdr.rows);

		usleep(10000);
                if(cv::waitKey(1) == 27) break; // check for esc key
	}
	return 0;
}
