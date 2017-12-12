// ==============
// camera.cpp
// ==============

#include "camera.h"
namespace HTM {

Camera::Camera(ComputeSystem &cs, int rows, int cols) : HTM::InputLayer(cs, rows, cols)
{
	_camDim.x = rows;
	_camDim.y = cols;
	_grayDim.x = rows;
	_grayDim.y = cols;

	// Setup video first found video-capture device	
	device = cv::VideoCapture(0);
	if (!device.isOpened())
	{
		throw std::runtime_error(std::string("[inputlayer/camera] Expected an opened camera device"));
	}

	// Create instance of compute-program with the opencl code
	_cp = new ComputeProgram(*_cs, std::string("camera.cl"));
	
	// Create a reference to opencl kernel
	int clret = 0;
	_kernelBGR2Gray = new cl::Kernel(_cp->getProgram(), "BGR2Gray", &clret);
	if (clret != CL_SUCCESS) {
		throw std::runtime_error(std::string("[inputlayer/camera] Setup kernel BGR2Gray failed, return code: " + std::to_string(clret)));
	}

	// Create opencl buffer for original image
	_bgrImage = new cl::Buffer(_cs->getContext(), CL_MEM_READ_WRITE,
			_camDim.x * _camDim.y * 3 * sizeof(uint8_t), NULL, NULL);
	std::cout << "Created bgrImage cl::buffer buffer" << std::endl;

	// Buffer for grayscale image
	_grayImage = new cl::Buffer(_cs->getContext(), CL_MEM_READ_WRITE,
			_camDim.x * _camDim.y * sizeof(uint8_t), NULL, NULL);
	setInputData(_grayImage);
	std::cout << "Created InputData (Grayscale cl::Buffer) for inputlayer" << std::endl;

	// Set BGR2Gray parameters to original image (which is expected to be BGR)
	// and Grayscale image (which is 1x8bit unsigned characters per pixel)	
	_kernelBGR2Gray->setArg(0, *_bgrImage);
	std::cout << "Set bgrImage as kernel arg 0" << std::endl;
	_kernelBGR2Gray->setArg(1, *_grayImage);
	std::cout << "Set grayImage as kernel arg 1" << std::endl;
}

cv::Mat Camera::getNewImage()
{
        // Create temporary image
        cv::Mat tmpImage;

	// Read one frame from camera into temporary image
	std::cout << "Reading image" << std::endl;
        if (!device.read(tmpImage)) {
		throw std::runtime_error(std::string("[inputlayer/camera] Could not read an image from camera device"));
	}
	std::cout << "Image read, got: " << tmpImage.size() << ", " << tmpImage.channels() << std::endl;

	// Create new cv::Mat (image) placeholder with correct size (rows (x) * cols (y))
        cv::Mat image(_camDim.x, _camDim.y, tmpImage.type());

	// Resize original image to new image size
        cv::resize(tmpImage, image, image.size(), 0, 0, CV_INTER_LINEAR);
	std::cout << "resized image" << std::endl;
	std::cout << "Image type is: " << std::to_string(image.type()) << std::endl;

	// Since we've captured one frame, may as well convert it to grayscale
	convertToGray(image);
	
	// Also convert it to SDR
	input2SDR();

	return image;
}

void Camera::convertToGray(cv::Mat &image)
{
	// Copy image into opencl memory buffer
	std::cout << "Writing image to _bgrImage cl::buffer" << std::endl;
	_cs->getQueue().enqueueWriteBuffer(*_bgrImage, CL_TRUE, 0,
			_camDim.x * _camDim.y * 3 * sizeof(uint8_t),
			image.data, NULL, NULL);
	std::cout << "Wrote bgrImage to cl device, first value: " << image.data[0] << std::endl;

	// Run the opencl kernel which converts it to grayscale for us
	int ret = _cs->getQueue().enqueueNDRangeKernel(*_kernelBGR2Gray, cl::NullRange, cl::NDRange(_camDim.x * _camDim.y));
	std::cout << "Got returncode from enqueuendrangekernel: " << ret << std::endl;
}

cl::Buffer* Camera::getGrayBuffer()
{
	return _grayImage;
}

cv::Mat Camera::getGrayMat()
{
	// Create placeholder for returned grayscale imge from opencl (8bit Unsigned Char, 1 color-channel per pixel)
	cv::Mat gray = cv::Mat(_camDim.x, _camDim.y, CV_8UC1);

	// Read back grayscale image from buffer
	_cs->getQueue().enqueueReadBuffer(*_grayImage, CL_TRUE, 0,
		_camDim.x * _camDim.y * sizeof(uint8_t),
		gray.data, NULL, NULL);
	std::cout << "Got image back from cl device, first value: " << gray.data[0] << std::endl;
	std::cout << "gray is of type: " << std::to_string(gray.type()) << std::endl;
	std::cout << "Read back grayImage" << std::endl;
	return gray;
}

};
