//Udacity HW2 Driver

#include <iostream>
#include "timer.h"
#include "utils.h"
#include <string>
#include <stdio.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include "opencv2\imgproc\imgproc_c.h"
#include "cv.h"
#include "utils.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <string>

//#define USE_GPU 1

size_t numRows();  //return # of rows in the image
size_t numCols();  //return # of cols in the image

void preProcess(uchar4 **h_inputImageRGBA, uchar4 **h_outputImageRGBA,
                uchar4 **d_inputImageRGBA, uchar4 **d_outputImageRGBA,
                unsigned char **d_redBlurred,
                unsigned char **d_greenBlurred,
                unsigned char **d_blueBlurred,
                const std::string& filename);

void postProcess(const std::string& output_file);

void your_gaussian_blur(const uchar4 * const h_inputImageRGBA, uchar4 * const d_inputImageRGBA,
                        uchar4* const d_outputImageRGBA,
                        const size_t numRows, const size_t numCols,
                        unsigned char *d_redBlurred,
                        unsigned char *d_greenBlurred,
                        unsigned char *d_blueBlurred,
                        const int filterWidth);

void allocateMemoryAndCopyToGPU(const size_t numRowsImage, const size_t numColsImage,
                                const float* const h_filter, const size_t filterWidth);

void cleanup();

//include the definitions of the above functions for this homework
cv::Mat imageInputRGBA;
cv::Mat imageOutputRGBA;

uchar4 *d_inputImageRGBA__;
uchar4 *d_outputImageRGBA__;

float *h_filter__;

size_t numRows() { return imageInputRGBA.rows; }
size_t numCols() { return imageInputRGBA.cols; }

//return types are void since any internal error will be handled by quitting
//no point in returning error codes...
//returns a pointer to an RGBA version of the input image
//and a pointer to the single channel grey-scale output
//on both the host and device
void preProcess(uchar4 **h_inputImageRGBA, uchar4 **h_outputImageRGBA,
	uchar4 **d_inputImageRGBA, uchar4 **d_outputImageRGBA,
	unsigned char **d_redBlurred,
	unsigned char **d_greenBlurred,
	unsigned char **d_blueBlurred,
	float **h_filter, int *filterWidth,
	const std::string &filename) {

	//make sure the context initializes ok
	checkCudaErrors(cudaFree(0));

	cv::Mat image = cv::imread(filename.c_str(), CV_LOAD_IMAGE_COLOR);
	if (image.empty()) {
		std::cerr << "Couldn't open file: " << filename << std::endl;
		exit(1);
	}

	cv::cvtColor(image, imageInputRGBA, CV_BGR2RGBA);

	//allocate memory for the output
	imageOutputRGBA.create(image.rows, image.cols, CV_8UC4);

	//This shouldn't ever happen given the way the images are created
	//at least based upon my limited understanding of OpenCV, but better to check
	if (!imageInputRGBA.isContinuous() || !imageOutputRGBA.isContinuous()) {
		std::cerr << "Images aren't continuous!! Exiting." << std::endl;
		exit(1);
	}

	*h_inputImageRGBA = (uchar4 *)imageInputRGBA.ptr<unsigned char>(0);
	*h_outputImageRGBA = (uchar4 *)imageOutputRGBA.ptr<unsigned char>(0);

	const size_t numPixels = numRows() * numCols();
	//allocate memory on the device for both input and output
	checkCudaErrors(cudaMalloc(d_inputImageRGBA, sizeof(uchar4) * numPixels));
	checkCudaErrors(cudaMalloc(d_outputImageRGBA, sizeof(uchar4) * numPixels));
	checkCudaErrors(cudaMemset(*d_outputImageRGBA, 0, numPixels * sizeof(uchar4))); //make sure no memory is left laying around

																					//copy input array to the GPU
	checkCudaErrors(cudaMemcpy(*d_inputImageRGBA, *h_inputImageRGBA, sizeof(uchar4) * numPixels, cudaMemcpyHostToDevice));

	d_inputImageRGBA__ = *d_inputImageRGBA;
	d_outputImageRGBA__ = *d_outputImageRGBA;

	//now create the filter that they will use
	const int blurKernelWidth = 51;
	const float blurKernelSigma = 2.;

	*filterWidth = blurKernelWidth;

	//create and fill the filter we will convolve with
	*h_filter = new float[blurKernelWidth * blurKernelWidth];
	h_filter__ = *h_filter;

	float filterSum = 0.f; //for normalization

	for (int r = -blurKernelWidth / 2; r <= blurKernelWidth / 2; ++r) {
		for (int c = -blurKernelWidth / 2; c <= blurKernelWidth / 2; ++c) {
			float filterValue = expf(-(float)(c * c + r * r) / (2.f * blurKernelSigma * blurKernelSigma));
			(*h_filter)[(r + blurKernelWidth / 2) * blurKernelWidth + c + blurKernelWidth / 2] = filterValue;
			filterSum += filterValue;
		}
	}

	float normalizationFactor = 1.f / filterSum;

	for (int r = -blurKernelWidth / 2; r <= blurKernelWidth / 2; ++r) {
		for (int c = -blurKernelWidth / 2; c <= blurKernelWidth / 2; ++c) {
			(*h_filter)[(r + blurKernelWidth / 2) * blurKernelWidth + c + blurKernelWidth / 2] *= normalizationFactor;
		}
	}

	//blurred
	checkCudaErrors(cudaMalloc(d_redBlurred, sizeof(unsigned char) * numPixels));
	checkCudaErrors(cudaMalloc(d_greenBlurred, sizeof(unsigned char) * numPixels));
	checkCudaErrors(cudaMalloc(d_blueBlurred, sizeof(unsigned char) * numPixels));
	checkCudaErrors(cudaMemset(*d_redBlurred, 0, sizeof(unsigned char) * numPixels));
	checkCudaErrors(cudaMemset(*d_greenBlurred, 0, sizeof(unsigned char) * numPixels));
	checkCudaErrors(cudaMemset(*d_blueBlurred, 0, sizeof(unsigned char) * numPixels));
}

void postProcess(const std::string& output_file) {
	const int numPixels = numRows() * numCols();
	//copy the output back to the host
	checkCudaErrors(cudaMemcpy(imageOutputRGBA.ptr<unsigned char>(0), d_outputImageRGBA__, sizeof(uchar4) * numPixels, cudaMemcpyDeviceToHost));

	cv::Mat imageOutputBGR;
	cv::cvtColor(imageOutputRGBA, imageOutputBGR, CV_RGBA2BGR);
	//output the image
	cv::imwrite(output_file.c_str(), imageOutputBGR);

	//cleanup
	cudaFree(d_inputImageRGBA__);
	cudaFree(d_outputImageRGBA__);
	delete[] h_filter__;
}
//--------------------------------------------

int main(int argc, char **argv) {
  uchar4 *h_inputImageRGBA,  *d_inputImageRGBA;
  uchar4 *h_outputImageRGBA, *d_outputImageRGBA;
  unsigned char *d_redBlurred, *d_greenBlurred, *d_blueBlurred;

  float *h_filter;
  int    filterWidth;
  
  std::string input_file;
  std::string output_file;
  if (argc == 3) {
    input_file  = std::string("E:\\users\\shenca\\input.jpg");
    output_file = std::string("E:\\users\\shenca\\output.jpg");
  }
  else {
    std::cerr << "Usage: ./hw input_file output_file" << std::endl;
    exit(1);
  }
  //load the image and give us our input and output pointers
  preProcess(&h_inputImageRGBA, &h_outputImageRGBA, &d_inputImageRGBA, &d_outputImageRGBA,
             &d_redBlurred, &d_greenBlurred, &d_blueBlurred,
             &h_filter, &filterWidth, input_file);

  cv::Mat img;
  cv::cvtColor(imageInputRGBA,img,CV_RGBA2BGR);
  imshow("Original Image", img);

  allocateMemoryAndCopyToGPU(numRows(), numCols(), h_filter, filterWidth);
  //GpuTimer timer;
  int64 track_start, track_end;
  //call the students' code
#if defined(USE_GPU)
  //timer.Start();
  
  track_start = cvGetTickCount();
  your_gaussian_blur(h_inputImageRGBA, d_inputImageRGBA, d_outputImageRGBA, numRows(), numCols(),
                     d_redBlurred, d_greenBlurred, d_blueBlurred, filterWidth);
  //timer.Stop();
  track_end = cvGetTickCount();
  double tms = (track_end - track_start) / (double)cvGetTickFrequency() / 1000.0;
  printf("Processing 1 frame takes %8.3f ms \n", tms);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
  //int err = printf("%f msecs.\n", timer.Elapsed());
 
  //if (err < 0) {
  //  //Couldn't print! Probably the student closed stdout - bad news
  //  std::cerr << "Couldn't print timing information! STDOUT Closed!" << std::endl;
  //  exit(1);
  //}
  
  cleanup();
  //check results and output the blurred image
  postProcess(output_file);
  cv::cvtColor(imageOutputRGBA, img, CV_RGBA2BGR);
  imshow("output", img);
  cv::waitKey(0);

  checkCudaErrors(cudaFree(d_redBlurred));
  checkCudaErrors(cudaFree(d_greenBlurred));
  checkCudaErrors(cudaFree(d_blueBlurred));
#else
  cv::Mat imgCopy = img.clone();//cloning image
  //timer.Start();
  track_start = cvGetTickCount();
  GaussianBlur(img, imgCopy, cv::Size(51, 51), 2, 2);//applying Gaussian filter 
  //timer.Stop();
  track_end = cvGetTickCount();
  //int err = printf("%f msecs.\n", timer.Elapsed());
  double tms = (track_end - track_start) / (double)cvGetTickFrequency() / 1000.0;
  printf("Processing 1 frame takes %8.3f ms \n", tms);
  //if (err < 0) {
	 // //Couldn't print! Probably the student closed stdout - bad news
	 // std::cerr << "Couldn't print timing information! STDOUT Closed!" << std::endl;
	 // exit(1);
  //}
  imshow("output", imgCopy);
  cv::waitKey(0);
#endif
  return 0;
}
