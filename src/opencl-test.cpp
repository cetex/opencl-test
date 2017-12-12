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
#include "inputlayer/camera.h"
#include "htm/region.h"
#include "utils/utils.h"
//#include "architect/architect.h"
#include <opencv2/highgui/highgui.hpp>

using namespace std;

int main( int argc, char** argv )
{
	// Setup ComputeSystem
	ComputeSystem cs = ComputeSystem(ComputeSystem::_gpu);
	cs.printCLInfo();

	// Create inputlayer, camera
	HTM::Camera cap1 = HTM::Camera(cs, 120, 160);

	// Create region using camera
	HTM::Region region = HTM::Region(cs, cap1, HTM::Vec2i(240, 320), HTM::Vec2i(16, 16));

	// Create OpenCV windows for the different steps the data passes through.
	cv::namedWindow("cap1_orig", cv::WINDOW_NORMAL);
	cv::namedWindow("cap1_gray", cv::WINDOW_NORMAL);
	cv::namedWindow("cap1_sdr", cv::WINDOW_NORMAL);

	// Loop forever
	while (true) {
		// Grab new image from webcam (data from inputlayer)
		cap1.stepOne();
		cv::Mat image = cap1.getImageMat();
		std::cout << "Image Size: " << cap1.getDim() << ", Gray Size: " << cap1.getGrayDim() << ", SDR size: " << cap1.getSDRDim() << std::endl;
		
		// Grab the grayscale image
		cv::Mat gray = cap1.getGrayMat();
		std::cout << "Got gray mat" << std::endl;

		// Get the SDR for the Grayscale image
		cv::Mat sdr = cap1.getSDRMat();
		std::cout << "Got sdr mat" << std::endl;

		// Update the windows with the new images, rescale them as well
		cv::imshow("cap1_orig", image);
		cv::resizeWindow("cap1_orig", image.cols, image.rows);
       		cv::imshow("cap1_gray", gray);
		cv::resizeWindow("cap1_gray", gray.cols, gray.rows);
		cv::imshow("cap1_sdr", sdr);
		cv::resizeWindow("cap1_sdr", sdr.cols, sdr.rows);

		// Sleep for 10ms, not sure this is even needed as the framerate of the camera should slow us down significantly.
		usleep(10000);

		// Check for esc key
                if(cv::waitKey(1) == 27) break;
	}
	return 0;
}
