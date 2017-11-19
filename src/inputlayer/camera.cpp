// ==============
// camera.cpp
// ==============

#include "camera.h"

Camera::Camera(ComputeSystem cs, int rows, int cols)
{
	_rows = rows;
	_cols = cols;

	device = cv::VideoCapture(0);
	if (!device.isOpened())
	{
		throw std::runtime_error(std::string("[inputlayer/camera] Expected an opened camera device"));
	}

	_cp = new ComputeProgram(cs, std::string("camera.cl"));
	_cs = &cs;
	int clret = 0;
	_kernelBGR2Gray = new cl::Kernel(_cp->getProgram(), "BGR2Gray", &clret);
	if (clret != CL_SUCCESS) {
		throw std::runtime_error(std::string("[inputlayer/camera] Setup kernel BGR2Gray failed, return code: " + std::to_string(clret)));
	}

	_kernelGray2SDR = new cl::Kernel(_cp->getProgram(), "Gray2SDR", &clret);
	if (clret != CL_SUCCESS) {
		throw std::runtime_error(std::string("[inputlayer/camera] Setup kernel Gray2SDR failed, return code: " + std::to_string(clret)));
	}
	
	// Create buffers for original image, grayscale image, and SDR image.
	_bgrImage = cl::Buffer(_cs->getContext(), CL_MEM_READ_WRITE,
			rows * cols * 3 * sizeof(uint8_t), NULL, NULL);
	std::cout << "Created bgrImage cl::buffer buffer" << std::endl;
	_grayImage = cl::Buffer(_cs->getContext(), CL_MEM_READ_WRITE,
			rows * cols * sizeof(uint8_t), NULL, NULL);
	std::cout << "Created grayImage cl::Buffer buffer" << std::endl;
	_sdr = cl::Buffer(_cs->getContext(), CL_MEM_READ_WRITE,
			rows * cols * sizeof(cl_uint16), NULL, NULL);
	std::cout << "Created sdr cl::Buffer buffer" << std::endl;

	// Set BGR2Gray parameters to original image (which is expected to BGR)
	// and Grayscale image (which is 1x8bit unsigned characters per pixel)	
	_kernelBGR2Gray->setArg(0, _bgrImage);
	std::cout << "Set bgrImage as kernel arg 0" << std::endl;
	_kernelBGR2Gray->setArg(1, _grayImage);
	std::cout << "Set grayImage as kernel arg 1" << std::endl;

	_kernelGray2SDR->setArg(0, _grayImage);
	_kernelGray2SDR->setArg(1, _sdr);
	
}

cv::Mat Camera::getNewImage()
{
        // Read one image from camera to get dimensions and similar.
        cv::Mat tmpImage;
	std::cout << "Reading image" << std::endl;
        if (!device.read(tmpImage)) {
		throw std::runtime_error(std::string("[inputlayer/camera] Could not read an image from camera device"));
	}
	std::cout << "Image read, got: " << tmpImage.size() << ", " << tmpImage.channels() << std::endl;
	// Create new cv::Mat placeholder for new image of size rows x cols
        cv::Mat image(_rows, _cols, tmpImage.type());
	std::cout << "Create new scaled image-holder" << std::endl;
	// Resize original image to new image size
        cv::resize(tmpImage, image, image.size(), 0, 0, CV_INTER_LINEAR);
	std::cout << "resized image" << std::endl;
	std::cout << "Image type is: " << std::to_string(image.type()) << std::endl;
	return image;

}

void Camera::convertToGray(cv::Mat &image)
{
	_cs->getQueue().enqueueWriteBuffer(_bgrImage, CL_TRUE, 0,
			_rows * _cols * 3 * sizeof(uint8_t),
			image.data, NULL, NULL);
	std::cout << "Wrote bgrImage to cl device, first value: " << image.data[0] << std::endl;

	// Run the kernel
	int ret = _cs->getQueue().enqueueNDRangeKernel(*_kernelBGR2Gray, cl::NullRange, cl::NDRange(_rows * _cols));
	std::cout << "Got returncode from enqueuendrangekernel: " << ret << std::endl;

}

cl::Buffer* Camera::getGrayBuffer()
{
	// Grab image
	cv::Mat tmpImage = getNewImage();
	convertToGray(tmpImage);
	return &_grayImage;
}

cv::Mat Camera::getGrayMat()
{
	// Grab image
	cv::Mat tmpImage = getNewImage();
	convertToGray(tmpImage);
	// Write resized image to bgrImage buffer

	// Create placeholder for returned grayscale imge (8unsigned characters, 1 channel per pixel)
	cv::Mat gray = cv::Mat(_rows, _cols, CV_8UC1);
	// Read back grayscale image from buffer
	_cs->getQueue().enqueueReadBuffer(_grayImage, CL_TRUE, 0,
			_rows * _cols * sizeof(uint8_t),
			gray.data, NULL, NULL);
        //_cs->getQueue().enqueueReadImage(bgrImage, CL_TRUE, grayOrigin, grayRegion, 0, 0, image.data);
	std::cout << "Got image back from cl device, first value: " << gray.data[0] << std::endl;
	std::cout << "gray is of type: " << std::to_string(gray.type()) << std::endl;
	std::cout << "Read back grayImage" << std::endl;
	//return image;
	return gray;
}

cl::Buffer* Camera::getSDR()
{
	int ret = _cs->getQueue().enqueueNDRangeKernel(*_kernelGray2SDR, cl::NullRange, cl::NDRange(_rows * _cols));
	std::cout << "getSDR Got returncode from enqueuendrangekernel: " << ret << std::endl;
	_cs->getQueue().finish();
	return &_sdr;
}

cv::Mat Camera::getSDRMat()
{
	getSDR();
	cv::Mat sdr = cv::Mat(_rows, _cols * sizeof(cl_uchar16), CV_8UC1);
	_cs->getQueue().enqueueReadBuffer(_sdr, CL_TRUE, 0,
			_rows * _cols * sizeof(cl_uchar16),
			sdr.data, NULL, NULL);
	return sdr;
}
