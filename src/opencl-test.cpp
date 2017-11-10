//============================================================================
// Name        : opencl-test.cpp
// Author      : 
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <iostream>
#include <unistd.h>
#include <stdlib.h>
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define CL_HPP_TARGET_OPENCL_VERSION 200
#pragma GCC diagnostic ignored "-Wignored-attributes" // To hide three warnings since compiler is picky
#include <CL/cl.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <chrono>
#include <cxxabi.h>
#include <bitset>

#define MAX_SOURCE_SIZE (0x100000)
#define COLS 160
#define ROWS 120

using namespace std;

std::string getcwd_string( void ) {
	char buff[PATH_MAX];
	getcwd( buff, PATH_MAX );
	std::string cwd( buff );
	return cwd;
}

std::string getImageType(int number)
{
    // find type
    int imgTypeInt = number%8;
    std::string imgTypeString;

    switch (imgTypeInt)
    {
        case 0:
            imgTypeString = "8U";
            break;
        case 1:
            imgTypeString = "8S";
            break;
        case 2:
            imgTypeString = "16U";
            break;
        case 3:
            imgTypeString = "16S";
            break;
        case 4:
            imgTypeString = "32S";
            break;
        case 5:
            imgTypeString = "32F";
            break;
        case 6:
            imgTypeString = "64F";
            break;
        default:
            break;
    }

    // find channel
    int channel = (number/8) + 1;

    std::stringstream type;
    type<<"CV_"<<imgTypeString<<"C"<<channel;

    return type.str();
}

struct nnDimensions {
	cl_int columns_size_r;
	cl_int columns_size_c;
	cl_int columns_spacing;
	cl_int sdr_size_r;
	cl_int sdr_size_c;
	cl_int sdr_per_column_r;
	cl_int sdr_per_column_c;
	cl_int dendrites_r;
	cl_int dendrites_c;
	cl_int inhibition_area_r;
	cl_int inhibition_area_c;
	cl_uchar synapse_threshold;
	cl_uchar column_threshold;
	cl_int numActiveColumnsPerInhArea;
};

bool clError(cl_int ret) {
	switch(ret) {
	case CL_SUCCESS:
		return false;
	case CL_INVALID_COMMAND_QUEUE:
		cout << "CL_INVALID_COMMAND_QUEUE" << endl;
		return true;
	case CL_INVALID_CONTEXT:
		cout << "CL_INVALID_CONTEXT" << endl;
		return true;
	case CL_INVALID_MEM_OBJECT:
		cout << "CL_INVALID_MEM_OBJECT" << endl;
		return true;
	case CL_INVALID_VALUE:
		cout << "CL_INVALID_VALUE" << endl;
		return true;
	case CL_INVALID_IMAGE_SIZE:
		cout << "CL_INVALID_IMAGE_SIZE" << endl;
		return true;
	case CL_INVALID_EVENT_WAIT_LIST:
		cout << "CL_INVALID_EVENT_WAIT_LIST" << endl;
		return true;
	case CL_MEM_OBJECT_ALLOCATION_FAILURE:
		cout << "CL_MEM_OBJECT_ALLOCATION_FAILURE" << endl;
		return true;
	case CL_OUT_OF_HOST_MEMORY:
		cout << "CL_OUT_OF_HOST_MEMORY" << endl;
		return true;
	case CL_INVALID_KERNEL:
		cout << "CL_INVALID_KERNEL" << endl;
		return true;
	case CL_INVALID_KERNEL_ARGS:
		cout << "CL_INVALID_KERNEL_ARGS" << endl;
		return true;
	case CL_INVALID_WORK_DIMENSION:
		cout << "CL_INVALID_WORK_DIMENSION" << endl;
		return true;
	case CL_INVALID_WORK_GROUP_SIZE:
		cout << "CL_INVALID_WORK_GROUP_SIZE" << endl;
		return true;
	case CL_INVALID_WORK_ITEM_SIZE:
		cout << "CL_INVALID_WORK_ITEM_SIZE" << endl;
		return true;
	case CL_INVALID_GLOBAL_OFFSET:
		cout << "CL_INVALID_GLOBAL_OFFSET" << endl;
		return true;
	case CL_OUT_OF_RESOURCES:
		cout << "CL_OUT_OF_RESOURCES" << endl;
		return true;
	case CL_INVALID_BUFFER_SIZE:
		cout << "CL_INVALID_BUFFER_SIZE" << endl;
		return true;
	case CL_INVALID_ARG_INDEX:
		cout << "CL_INVALID_ARG_INDEX" << endl;
		return true;
	case CL_INVALID_ARG_VALUE:
		cout << "CL_INVALID_ARG_VALUE" << endl;
		return true;
	case CL_INVALID_SAMPLER:
		cout << "CL_INVALID_SAMPLER" << endl;
		return true;
	case CL_INVALID_ARG_SIZE:
		cout << "CL_INVALID_ARG_SIZE" << endl;
		return true;
	case CL_INVALID_DEVICE:
		cout << "CL_INVALID_DEVICE" << endl;
		return true;
	case CL_INVALID_BINARY:
		cout << "CL_INVALID_BINARY" << endl;
		return true;
	case CL_INVALID_BUILD_OPTIONS:
		cout << "CL_INVALID_BUILD_OPTIONS" << endl;
		return true;
	case CL_INVALID_OPERATION:
		cout << "CL_INVALID_OPERATION" << endl;
		return true;
	case CL_COMPILER_NOT_AVAILABLE:
		cout << "CL_COMPILER_NOT_AVAILABLE" << endl;
		return true;
	case CL_BUILD_PROGRAM_FAILURE:
		cout << "CL_BUILD_PROGRAM_FAILURE" << endl;
		return true;
	case CL_INVALID_PROGRAM:
		cout << "CL_INVALID_PROGRAM" << endl;
		return true;
	case CL_INVALID_PROGRAM_EXECUTABLE:
		cout << "CL_INVALID_PROGRAM_EXECUTABLE" << endl;
		return true;
	case CL_INVALID_KERNEL_NAME:
		cout << "CL_INVALID_KERNEL_NAME" << endl;
		return true;
	case CL_INVALID_KERNEL_DEFINITION:
		cout << "CL_INVALID_KERNEL_DEFINITION" << endl;
		return true;
	default:
		cout << "Return code undefined: " << ret << endl;
		return true;
	}
}


int main( int argc, char** argv )
{
	cout << "!!!OpenCV + OpenCL Test!!!" << endl;
	cout << "Opening first camera device" << endl;
	cv::VideoCapture cap(0);
	if(!cap.isOpened())  {
		cout << "Failed to open camera!" << endl;
		return -1;
	}
	// Create window to show image(s)
	cv::namedWindow("Video", cv::WINDOW_NORMAL);
	cv::namedWindow("Columns", cv::WINDOW_NORMAL);
	// Check if we have opencl:
	//if (! cv::ocl: haveOpenCL()) {
	//	cout << "OpenCL Missing!: " << endl;
	//	return -1;
	//}

	// Load opencl kernel from file
	FILE *fp;
	char *source_str;
	size_t source_size;

	cout << "trying to load kernel from: " << getcwd_string() << "/kernel.c" << endl;

	fp = fopen("kernel.cl", "r");
	if (!fp) {
		fprintf(stderr, "Failed to load kernel.\n");
		exit(1);
	}
	// Allocate memory for file of MAX_SOURCE_SIZE size.
	source_str = (char*)malloc(MAX_SOURCE_SIZE);
	// Read the kernel source file
	source_size = fread( source_str, 1, MAX_SOURCE_SIZE, fp);
	fclose(fp);

	// Get platform and device information
	cl_platform_id platform_id = NULL;
	cl_device_id device_id = NULL;
	cl_uint ret_num_devices;
	cl_uint ret_num_platforms;
	cl_int ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
	ret = clGetDeviceIDs( platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, &ret_num_devices);
	if(clError(ret)) {
		cout << "Failed to query clGetDeviceID's" << endl;
		exit(1);
	}

	// Create an OpenCL context
	cl_context context = clCreateContext( NULL, 1, &device_id, NULL, NULL, &ret);
	cout << "Created OpenCL Context" << endl;

	// Create a command queue
	cl_command_queue command_queue = clCreateCommandQueue(context, device_id, 0, &ret);
	if(clError(ret)) {
		cout << "Failed to create OpenCL command queue" << endl;
		exit(1);
	}
	cout << "Created command_queue" << endl;

	// Read one image from camera to get dimensions and similar.
	cv::Mat tmpImage;
	if (!cap.read(tmpImage)) exit(-10);
	cv::Mat tmpImage2(ROWS, COLS, tmpImage.type());
	cv::resize(tmpImage, tmpImage2, tmpImage2.size(), 0, 0, CV_INTER_LINEAR);
	tmpImage = tmpImage2;
	cout << "Grabbed an example image through OpenCV of dimensions: " << tmpImage.rows << "," << tmpImage.cols << endl;

	// Create memory buffers on the device
	// cl_input is where we put the image we get from the webcam
	cl_mem cl_input = clCreateBuffer(context, CL_MEM_READ_ONLY,
			ROWS * COLS * tmpImage.channels() * sizeof(uint8_t), NULL, &ret);
	if(clError(ret)) {
		cout << "Failed to create input buffer on device" << endl;
		exit(1);
	}

	// cl_output is the image we get back from opencl / the image converted to grayscale
	cl_mem cl_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
			ROWS * COLS * sizeof(uint8_t), NULL, &ret);
	if(clError(ret)) {
		cout << "Failed to create output buffer on device" << endl;
		exit(1);
	}

	// cl_image_sdr is the SDR representation of the grayscale image
	cl_mem cl_image_sdr = clCreateBuffer(context, CL_MEM_READ_WRITE,
			ROWS * COLS * sizeof(cl_uchar16), NULL, &ret);
	if(clError(ret)) {
		cout << "Failed to create image_SDR buffer on device" << endl;
		exit(1);
	}

	// Calculate spatial pooler size of weight matrix:
	// Assume each neuron sees 128*128 bits = 8*8 pixels if SDR is 16bit's
	// input is: 320*240 pixels

	// neurons see 8*8 pixels each.
	int cols_per_neuron = 16;
	int rows_per_neuron = 16;
	// Each pixel (SDR) is 16 "bits" (in reality they're bytes)
	int bits_per_pixel = 16;

	// 320-2*(cols_per_neuron/2) = 312 "neurons" wide.
	int netwidth = COLS - 2*(cols_per_neuron/2);
	// 240-2*(rows_per_neuron/2) = 232 "neurons" high
	int netheight = ROWS - 2*(rows_per_neuron/2);

	cout << "columns wide: " << netwidth << ", columns high: " << netheight << endl;
	cout << "Each column sees: " << cols_per_neuron << "*" << rows_per_neuron << " pixels of: " << bits_per_pixel << " bits each" << endl;

	// 2 pixels spacing (2x16 bits spacing) between each neuron, so netwidth / neuron_pixel_spacing = number of neurons wide.

	// dendrites are the inputs to each neuron.
	// this calculates how many dendrites in total there will be for the input.
	// neuron_dendrites = (netwidth / spacing between pixels) * how many cols per neuron * bits per pixel =
	// (312 / 2) = 156 neurons wide
	// (232 / 2) = 116 neurons height
	// where each neuron for each dimension has:
	// cols_per_neuron * bits_per_pixel =
	// 8 * 16 = 128 inputs per dimension
	// for a total number per dimension of
	// (312 / 2) * 8 * 16 = 19968 width
	// (232 / 2) * 8 * 16 = 14848 height
	// and a grand total of:
	// width * height = 19968 × 14848 = 296484864 = ~300MB of data.

	int column_dendrites_width = COLS * cols_per_neuron;
	int column_dendrites_height = ROWS * rows_per_neuron;

	// To calculate the center of the pixel position in this matrix:
	// posWidth = (cols_per_neuron*bits_per_pixel / 2) + (neurons_dendrites_width / (netwidth * neuron_pixel_spacing)) * pixelW = (8*16/2) + (19968 / (312*2))  * pixelW == 64 + 32 * pixelW
	// posHeight = (rows_per_neuron*bits_per_pixel / 2) + (neurons_dendrites_height / (netheight * neuron_pixel_spacing)) * pixelH = (8*16/2) + (14848 / (232*2)) * pixelH == 64 + 32 * pixelH
	// To calculate the position of the top left pixel in a neuron area of the matrix:
	// pos = (pixelHeight * neurons_dendrites_width) + pixelWidth
	// To calculate the position of pixel at pixelHeight and pixelWidth in this area
	// pos = (pixelHeight * (neurons_dendrites_height / (netheight * neuron_pixel_spacing) * neurons_dendrites_width) + (pixelWidth * (neurons_dendrites_width / (netwidth*neuron_pixel_spacing))
	// This will give a start position which increments by 32 for each pixel in width and increments by 32 for each pixel in height.

	// To calculate the position of a pixel in a 1d matrix:
	// pos = (pixelHeight * neurons_dendrites_width) + (pixelWidth * 16)
	// First pixel in area:
	// pos = ((pixelHeight -4) * neurons_dendrites_width) + ((pixelWidth -4) * 16)


	// Important to divide pixels by two when working on this matrix as spacing between pixels is two.

	cout << "Number of dendrites wide: " << column_dendrites_width << ", where one pixel is 1 dendrite" << endl;
	cout << "Number of dendrites high: " << column_dendrites_height << ", where one pixel is 1 dendrite" << endl;

	int total_dendrites = column_dendrites_width * column_dendrites_height;
	cout << "Creating buffer for spatial pooler of size: " << total_dendrites << " bytes, / " << total_dendrites/1024/1024 << " MB ram" << endl;

	// cl_spatial_weights is the spatial pooler weights for the SDR
	cl_mem cl_spatial_weights = clCreateBuffer(context, CL_MEM_READ_WRITE,
			total_dendrites * sizeof(cl_uchar16), NULL, &ret);
	if(clError(ret)) {
		cout << "Failed to create cl_spatial_weights buffer on device" << endl;
		exit(1);
	}

	unsigned int total_columns = netwidth * netheight;
	// cl_columns is the column output from "and"ing (&) the SDR with cl_spatial_weights and comparing it to the threshold.
	cl_mem cl_columns = clCreateBuffer(context, CL_MEM_READ_WRITE,
			total_columns, NULL, &ret);
	if(clError(ret)) {
		cout << "Failed to create cl_columns buffer on device" << endl;
		exit(1);
	}

	// cl_columns_winners is the columns selected as winners from each inhibition area.
	cl_mem cl_columns_winners = clCreateBuffer(context, CL_MEM_READ_WRITE,
			total_columns, NULL, &ret);
	if(clError(ret)) {
		cout << "Failed to create cl_columns buffer on device" << endl;
		exit(1);
	}

	nnDimensions nnParams;
	nnParams.columns_size_r = netheight;
	nnParams.columns_size_c = netwidth;
	nnParams.columns_spacing = 1;
	nnParams.sdr_size_r = ROWS;
	nnParams.sdr_size_c = COLS;
	nnParams.sdr_per_column_r = rows_per_neuron;
	nnParams.sdr_per_column_c = cols_per_neuron;
	nnParams.dendrites_r = column_dendrites_height;
	nnParams.dendrites_c = column_dendrites_width;
	nnParams.inhibition_area_r = 16;
	nnParams.inhibition_area_c = 16;
	nnParams.synapse_threshold = 127;
	nnParams.column_threshold = 5;

	// cl_nn_params
	cl_mem cl_nn_params = clCreateBuffer(context, CL_MEM_READ_WRITE,
			sizeof(nnDimensions), NULL, &ret);
	if(clError(ret)) {
		cout << "Failed to create cl_nn_params buffer on device" << endl;
		exit(1);
	}
	cout << "Created memory buffers on the device" << endl;

	// Create a program from the kernel source
	cl_program program = clCreateProgramWithSource(context, 1,
			(const char **)&source_str, (const size_t *)&source_size, &ret);
	if(clError(ret)) {
		cout << "Failed to create program from loaded file" << endl;
		exit(1);
	}
	cout << "Created a program from kernel source" << endl;

	// Build the program
	const char *cloptions = ""; // "-Werror";
	ret = clBuildProgram(program, 1, &device_id, cloptions, NULL, NULL);
	if(clError(ret)) {
		cout << "Failed to compile the opencl kernel for the device" << endl;
		size_t len;
		char *buffer;
		clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &len);
		buffer = (char*)malloc(len);
		clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, len, buffer, NULL);
		cout << buffer << endl;
	}

	// Create the OpenCL kernel
	cl_kernel BGR2GRAY = clCreateKernel(program, "BGR2GRAY", &ret);
	if(clError(ret)) {
		cout << "Failed to create kernel from cl program" << endl;
		exit(1);
	}
	// Set args for BGR2GRAY
	ret = clSetKernelArg(BGR2GRAY, 0, sizeof(cl_mem), &cl_input);
	if (clError(ret)) {
		cout << "Failed to set cl_input" << endl;
		exit(1);
	}
	ret = clSetKernelArg(BGR2GRAY, 1, sizeof(cl_mem), &cl_output);
	if (clError(ret)) {
		cout << "Failed to set cl_output" << endl;
		exit(1);
	}

	// Create the OpenCL kernel
	cl_kernel BGR2SDR = clCreateKernel(program, "BGR2SDR", &ret);
	if(clError(ret)) {
		cout << "Failed to create kernel from cl program" << endl;
		exit(1);
	}
	// Set args for BGR2SDR
	ret = clSetKernelArg(BGR2SDR, 0, sizeof(cl_mem), &cl_output);
	if (clError(ret)) {
		cout << "Failed to set cl_input to BGR2SDR" << endl;
		exit(1);
	}
	ret = clSetKernelArg(BGR2SDR, 1, sizeof(cl_mem), &cl_image_sdr);
	if (clError(ret)) {
		cout << "Failed to set cl_image_sdr to BGR2SDR" << endl;
		exit(1);
	}
	cout << "Created BGR2SDR kernel" << endl;

	// Create the OpenCL kernel
	cl_kernel ColumnOverlap = clCreateKernel(program, "ColumnOverlap", &ret);
	if(clError(ret)) {
		cout << "Failed to create kernel from cl program" << endl;
		exit(1);
	}
	// Set args to ActiveColumns
	ret = clSetKernelArg(ColumnOverlap, 0, sizeof(cl_mem), &cl_image_sdr);
	if (clError(ret)) {
		cout << "Failed to set cl_image_sdr" << endl;
		exit(1);
	}
	ret = clSetKernelArg(ColumnOverlap, 1, sizeof(cl_mem), &cl_spatial_weights);
	if (clError(ret)) {
		cout << "Failed to set cl_spatial_weights" << endl;
		exit(1);
	}
	ret = clSetKernelArg(ColumnOverlap, 2, sizeof(cl_mem), &cl_columns);
	if (clError(ret)) {
		cout << "Failed to set cl_spatial_weights" << endl;
		exit(1);
	}
	ret = clSetKernelArg(ColumnOverlap, 3, sizeof(cl_mem), &cl_nn_params);
	if (clError(ret)) {
		cout << "Failed to set cl_columns" << endl;
		exit(1);
	}
	cout << "Created ActiveColumns kernel" << endl;

	uchar *columns = (uchar*)malloc(total_dendrites);
	for (int i = 0; i < total_dendrites; i++) {
		columns[i] = rand() % 30 + 112;
		//columns[i] = 250;
	}
	cout << "Generated random numbers for dendrites" << endl;
	ret = clEnqueueWriteBuffer(command_queue, cl_spatial_weights, CL_TRUE, 0,
			total_dendrites, columns, 0, NULL, NULL);
	if (clError(ret)) {
		cout << "Failed to enqueue write buffer" << endl;
		exit(1);
	}
	cout << "Wrote to buffer" << endl;
	free(columns);
	cout << "Freed" << endl;
	// Create the Inhibition kernel
	cl_kernel Inhibition = clCreateKernel(program, "Inhibition", &ret);
	if(clError(ret)) {
		cout << "Failed to create kernel from cl program" << endl;
		exit(1);
	}
	ret = clSetKernelArg(Inhibition, 0, sizeof(cl_mem), &cl_columns);
	if (clError(ret)) {
		cout << "Failed to set cl_spatial_weights" << endl;
		exit(1);
	}
	ret = clSetKernelArg(Inhibition, 1, sizeof(cl_mem), &cl_columns_winners);
	if (clError(ret)) {
		cout << "Failed to set cl_spatial_weights" << endl;
		exit(1);
	}

	//cv::ocl::setUseOpenCL(false);
	int count = 0;
	std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now(); // start-time for loop
	for (;;) { // Main loop, grab picture, convert to grayscale and show it
		// Read image from camera
		if (cap.read(tmpImage)) {
			if (!tmpImage.isContinuous()) {
				std::cout << "Data received from camera is not continuous, won't proceed!" << endl;
				exit(2);
			}
			cv::Mat image(ROWS, COLS, tmpImage.type());
			cv::resize(tmpImage, image, image.size(), 0, 0, CV_INTER_LINEAR);
			//image = tmpImage;

			// Copy the image to the opencl memory buffer
			ret = clEnqueueWriteBuffer(command_queue, cl_input, CL_TRUE, 0,
					image.rows * image.cols * image.channels() * sizeof(uint8_t), image.data, 0, NULL, NULL);
			if (clError(ret)) {
				cout << "Failed to enqueue write buffer" << endl;
				exit(1);
			}

			size_t global_item_size = image.rows * image.cols; // Process all pixels
			size_t local_item_size = 64; // Divide work items into groups of 64
			ret = clEnqueueNDRangeKernel(command_queue, BGR2GRAY, 1, NULL,
					&global_item_size, &local_item_size, 0, NULL, NULL);
			if (clError(ret)) {
				cout << "Failed to schedule cl BGR2GRAY kernel" << endl;
				exit(1);
			}

			size_t bgr2sdr_wi[1];
			bgr2sdr_wi[0] = image.rows * image.cols;
			ret = clEnqueueNDRangeKernel(command_queue, BGR2SDR, 1, NULL,
					bgr2sdr_wi, NULL, 0, NULL, NULL);
			if (clError(ret)) {
				cout << "Failed to schedule cl BGR2SDR kernel" << endl;
				exit(1);
			}

			// This is the real output format, although hard to visualize as it's so wide.
			//cv::Mat sdr(ROWS, COLS * sizeof(cl_uchar16), CV_8UC1);
			// This is a fake, switched ROWS and COLS.. still doesn't draw well..
			cv::Mat sdr(COLS * sizeof(cl_uchar4), ROWS * sizeof(cl_uchar4), CV_8UC1);
			cout << "Sizeof sdr: " << sizeof(sdr.data) << " rows: " << sdr.rows << " cols: " << sdr.cols << endl;
			ret = clEnqueueReadBuffer(command_queue, cl_image_sdr, CL_TRUE, 0,
					ROWS * COLS * sizeof(cl_uchar16), sdr.data, 0, NULL, NULL);
			if (clError(ret)) {
				cout << "Failed to read SDR data from cl device" << endl;
				exit(1);
			}

			ret = clEnqueueWriteBuffer(command_queue, cl_nn_params, CL_TRUE, 0,
					sizeof(nnParams), &nnParams, 0, NULL, NULL);
			if (clError(ret)) {
				cout << "Failed to enqueue write buffer for nnParams" << endl;
				exit(1);
			}

			// Compute active columns.
			size_t CheckActive_wi[2];
			CheckActive_wi[0] = netheight;
			CheckActive_wi[1] = netwidth;
			ret = clEnqueueNDRangeKernel(command_queue, ColumnOverlap, 2, NULL,
					CheckActive_wi, NULL, 0, NULL, NULL);
			if (clError(ret)) {
				cout << "Failed to schedule cl CheckActive kernel" << endl;
				exit(1);
			}

			cv::Mat output(image.rows, image.cols, CV_8UC1);
			ret = clEnqueueReadBuffer(command_queue, cl_output, CL_TRUE, 0,
					output.rows * output.cols * sizeof(uint8_t), output.data, 0, NULL, NULL);
			if (clError(ret)) {
				cout << "Failed to read data from cl device" << endl;
				exit(1);
			}

			cv::Mat columns(netheight, netwidth, CV_8UC1);
			//unsigned char *columns = (unsigned char*)malloc(total_columns);
			ret = clEnqueueReadBuffer(command_queue, cl_columns, CL_TRUE, 0,
					total_columns, columns.data, 0, NULL, NULL);
			if (clError(ret)) {
				cout << "Failed to read columns data from cl device" << endl;
				exit(1);
			}
			for (int i = 0; i < 1000; i++) {
				cout << "Column: " << +i << ", data: " << +columns.data[i] << endl;
			}

			cv::imshow("SDR", sdr);
			cv::resizeWindow("SDR", sdr.cols, sdr.rows);
			cv::imshow("Columns", columns);
			cv::resizeWindow("Columns", columns.cols, columns.rows); // Resize window to fit dimension of image
			cv::imshow("Video", output); // Show image
			cv::resizeWindow("Video", output.cols, output.rows); // Resize window to fit dimension of image
		} else {
			cout << "Failed to read image from camera" << endl;
		}
		if (count == 100) {
			std::chrono::steady_clock::time_point end= std::chrono::steady_clock::now();  // stop-time for for-loop
			std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << std::endl; // Time iteration took
			cout << "FPS: " << (float(1000000)/std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) << endl;
			count = 0;
		}
		count++;
		begin = std::chrono::steady_clock::now(); // (re)start-time for loop
		if(cv::waitKey(1) == 27) break; // check for esc key
	}

	return 0;
}
