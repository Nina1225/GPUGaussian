// Homework 2
// Image Blurring
//
// In this homework we are blurring an image. To do this, imagine that we have
// a square array of weight values. For each pixel in the image, imagine that we
// overlay this square array of weights on top of the image such that the center
// of the weight array is aligned with the current pixel. To compute a blurred
// pixel value, we multiply each pair of numbers that line up. In other words, we
// multiply each weight with the pixel underneath it. Finally, we add up all of the
// multiplied numbers and assign that value to our output for the current pixel.
// We repeat this process for all the pixels in the image.

// To help get you started, we have included some useful notes here.

//****************************************************************************

// For a color image that has multiple channels, we suggest separating
// the different color channels so that each color is stored contiguously
// instead of being interleaved. This will simplify your code.

// That is instead of RGBARGBARGBARGBA... we suggest transforming to three
// arrays (as in the previous homework we ignore the alpha channel again):
//  1) RRRRRRRR...
//  2) GGGGGGGG...
//  3) BBBBBBBB...
//
// The original layout is known an Array of Structures (AoS) whereas the
// format we are converting to is known as a Structure of Arrays (SoA).

// As a warm-up, we will ask you to write the kernel that performs this
// separation. You should then write the "meat" of the assignment,
// which is the kernel that performs the actual blur. We provide code that
// re-combines your blurred results for each color channel.

//****************************************************************************

// You must fill in the gaussian_blur kernel to perform the blurring of the
// inputChannel, using the array of weights, and put the result in the outputChannel.

// Here is an example of computing a blur, using a weighted average, for a single
// pixel in a small image.
//
// Array of weights:
//
//  0.0  0.2  0.0
//  0.2  0.2  0.2
//  0.0  0.2  0.0
//
// Image (note that we align the array of weights to the center of the box):
//
//    1  2  5  2  0  3
//       -------
//    3 |2  5  1| 6  0       0.0*2 + 0.2*5 + 0.0*1 +
//      |       |
//    4 |3  6  2| 1  4   ->  0.2*3 + 0.2*6 + 0.2*2 +   ->  3.2
//      |       |
//    0 |4  0  3| 4  2       0.0*4 + 0.2*0 + 0.0*3
//       -------
//    9  6  5  0  3  9
//
//         (1)                         (2)                 (3)
//
// A good starting place is to map each thread to a pixel as you have before.
// Then every thread can perform steps 2 and 3 in the diagram above
// completely independently of one another.

// Note that the array of weights is square, so its height is the same as its width.
// We refer to the array of weights as a filter, and we refer to its width with the
// variable filterWidth.

//****************************************************************************

// Your homework submission will be evaluated based on correctness and speed.
// We test each pixel against a reference solution. If any pixel differs by
// more than some small threshold value, the system will tell you that your
// solution is incorrect, and it will let you try again.

// Once you have gotten that working correctly, then you can think about using
// shared memory and having the threads cooperate to achieve better performance.

//****************************************************************************

// Also note that we've supplied a helpful debugging function called checkCudaErrors.
// You should wrap your allocation and copying statements like we've done in the
// code we're supplying you. Here is an example of the unsafe way to allocate
// memory on the GPU:
//
// cudaMalloc(&d_red, sizeof(unsigned char) * numRows * numCols);
//
// Here is an example of the safe way to do the same thing:
//
// checkCudaErrors(cudaMalloc(&d_red, sizeof(unsigned char) * numRows * numCols));
//
// Writing code the safe way requires slightly more typing, but is very helpful for
// catching mistakes. If you write code the unsafe way and you make a mistake, then
// any subsequent kernels won't compute anything, and it will be hard to figure out
// why. Writing code the safe way will inform you as soon as you make a mistake.

// Finally, remember to free the memory you allocate at the end of the function.

//****************************************************************************

#include "reference_calc.cpp"
#include "utils.h"

//This kernel runs the gaussian blur algorithm for one channel
__global__
void gaussian_blur(const unsigned char* const inputChannel,
                   unsigned char* const outputChannel,
                   int numRows, int numCols,
                   const float* const filter, const int filterWidth)
{
    const int2 thread_2D_pos = make_int2(blockIdx.x * blockDim.x + threadIdx.x,
                                         blockIdx.y * blockDim.y + threadIdx.y);

    const int thread_1D_pos = thread_2D_pos.y * numCols + thread_2D_pos.x;

    //Return if memory outside the bound of the image
    if (thread_2D_pos.x >= numCols || thread_2D_pos.y >= numRows)
        return;

    //Compute the new pixel value
    float weightedSum = 0;
    for (int fi = 0; fi < filterWidth; fi++) {
        for (int fj = 0; fj < filterWidth; fj++) {
            const int2 at_thread_2D_pos = make_int2(thread_2D_pos.x - filterWidth / 2 + fi,
                                                    thread_2D_pos.y - filterWidth / 2 + fj);
            const int at_thread_1D_pos = at_thread_2D_pos.y * numCols + at_thread_2D_pos.x;
            const int at_filter = fj * filterWidth + fi;
            weightedSum += inputChannel[at_thread_1D_pos] * filter[at_filter];
        }
    }

    //Store the new pixel value to outputChannel
    outputChannel[thread_1D_pos] = weightedSum;
}

//This kernel takes in an image represented as a uchar4 and splits
//it into three images consisting of only one color channel each
__global__
void separateChannels(const uchar4* const inputImageRGBA,
                      int numRows,
                      int numCols,
                      unsigned char* const redChannel,
                      unsigned char* const greenChannel,
                      unsigned char* const blueChannel)
{
    const int2 thread_2D_pos = make_int2(blockIdx.x * blockDim.x + threadIdx.x,
                                         blockIdx.y * blockDim.y + threadIdx.y);
    const int thread_1D_pos = thread_2D_pos.y * numCols + thread_2D_pos.x;

    //Return if memory outside the bound of the image
    if (thread_2D_pos.x >= numCols || thread_2D_pos.y >= numRows)
        return;

    //Store the three color channels in three variable
    redChannel[thread_1D_pos] = inputImageRGBA[thread_1D_pos].x;
    greenChannel[thread_1D_pos] = inputImageRGBA[thread_1D_pos].y;
    blueChannel[thread_1D_pos] = inputImageRGBA[thread_1D_pos].z;
}

//This kernel takes in three color channels and recombines them
//into one image.  The alpha channel is set to 255 to represent
//that this image has no transparency.
__global__
void recombineChannels(const unsigned char* const redChannel,
                       const unsigned char* const greenChannel,
                       const unsigned char* const blueChannel,
                       uchar4* const outputImageRGBA,
                       int numRows,
                       int numCols)
{
    const int2 thread_2D_pos = make_int2( blockIdx.x * blockDim.x + threadIdx.x,
                                        blockIdx.y * blockDim.y + threadIdx.y);

    const int thread_1D_pos = thread_2D_pos.y * numCols + thread_2D_pos.x;

    //Return if memory outside the bound of the image
    if (thread_2D_pos.x >= numCols || thread_2D_pos.y >= numRows)
        return;

    unsigned char red   = redChannel[thread_1D_pos];
    unsigned char green = greenChannel[thread_1D_pos];
    unsigned char blue  = blueChannel[thread_1D_pos];

    // Alpha should be 255 for no transparency
    uchar4 outputPixel = make_uchar4(red, green, blue, 255);

    // Store channels in outputChannel
    outputImageRGBA[thread_1D_pos] = outputPixel;
}

unsigned char *d_red, *d_green, *d_blue;
float         *d_filter;

//This kernel allocate memory in device for each color channel and
//for the filter
void allocateMemoryAndCopyToGPU(const size_t numRowsImage, const size_t numColsImage,
                                const float* const h_filter, const size_t filterWidth)
{

    //Allocate memory for the three different channels
    checkCudaErrors(cudaMalloc(&d_red,   sizeof(unsigned char) * numRowsImage * numColsImage));
    checkCudaErrors(cudaMalloc(&d_green, sizeof(unsigned char) * numRowsImage * numColsImage));
    checkCudaErrors(cudaMalloc(&d_blue,  sizeof(unsigned char) * numRowsImage * numColsImage));
  
    //Allocate memory for filter
    checkCudaErrors(cudaMalloc(&d_filter, sizeof(float) * filterWidth * filterWidth));

    //Copy filter from host to device
    checkCudaErrors(cudaMemcpy(d_filter, h_filter, sizeof(float) * filterWidth * filterWidth, cudaMemcpyHostToDevice));

}

void your_gaussian_blur(const uchar4 * const h_inputImageRGBA, uchar4 * const d_inputImageRGBA,
                        uchar4* const d_outputImageRGBA, const size_t numRows, const size_t numCols,
                        unsigned char *d_redBlurred, 
                        unsigned char *d_greenBlurred, 
                        unsigned char *d_blueBlurred,
                        const int filterWidth)
{
    const dim3 blockSize(32,8);
    const dim3 gridSize(numCols/32+1,numRows/8+1);

    //Separate the RGBA image into different color channels
    separateChannels<<<gridSize, blockSize>>>(d_inputImageRGBA,
                                              numRows,
                                              numCols,
                                              d_redBlurred,
                                              d_greenBlurred,
                                              d_blueBlurred);

    //Call cudaDeviceSynchronize(), then call checkCudaErrors() immediately after
    //launching your kernel to make sure that you didn't make any mistakes.
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    //Call the convolution kernel for each color channel
    gaussian_blur<<<gridSize, blockSize>>>(d_redBlurred,
                                           d_red,
                                           numRows,
                                           numCols,
                                           d_filter,
                                           filterWidth);
    gaussian_blur<<<gridSize, blockSize>>>(d_greenBlurred,
                                           d_green,
                                           numRows,
                                           numCols,
                                           d_filter,
                                           filterWidth);
    gaussian_blur<<<gridSize, blockSize>>>(d_blueBlurred,
                                           d_blue,
                                           numRows,
                                           numCols,
                                           d_filter,
                                           filterWidth);

    //Make sure that you didn't make any mistakes.
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    // Now we recombine your results. We take care of launching this kernel for you.
    recombineChannels<<<gridSize, blockSize>>>(d_red,
                                               d_green,
                                               d_blue,
                                               d_outputImageRGBA,
                                               numRows,
                                               numCols);

    //Make sure that you didn't make any mistakes.
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    ///****************************************************************************
    //* You can use the code below to help with debugging, but make sure to       *
    //* comment it out again before submitting your assignment for grading,       *
    //* otherwise this code will take too much time and make it seem like your    *
    //* GPU implementation isn't fast enough.                                     *
    //*                                                                           *
    //* This code generates a reference image on the host by running the          *
    //* reference calculation we have given you.  It then copies your GPU         *
    //* generated image back to the host and calls a function that compares the   *
    //* the two and will output the first location they differ by too much.       *
    //* ************************************************************************* */

    //uchar4 *h_outputImage     = new uchar4[numRows * numCols];
    //uchar4 *h_outputReference = new uchar4[numRows * numCols];
  
    //checkCudaErrors(cudaMemcpy(h_outputImage, d_outputImageRGBA, 
    //                           numRows * numCols * sizeof(uchar4), 
    //                           cudaMemcpyDeviceToHost));

    ////referenceCalculation(h_inputImageRGBA, h_outputReference, numRows, numCols,
    ////                     h_filter, filterWidth);

    ////the 4 is because there are 4 channels in the image
    ////checkResultsExact((unsigned char *)h_outputReference,
    ////                  (unsigned char *)h_outputImage,
    ////                  numRows * numCols * 4); 
 
    //delete [] h_outputImage;
    //delete [] h_outputReference;
}


//Free all the memory that we allocated
void cleanup() {
    checkCudaErrors(cudaFree(d_red));
    checkCudaErrors(cudaFree(d_green));
    checkCudaErrors(cudaFree(d_blue));
    checkCudaErrors(cudaFree(d_filter));
}
