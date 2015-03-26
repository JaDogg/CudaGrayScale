/*************************************************
 *	Part of cuda bitmap to grayscale converter   *
 *	- Bhathiya Perera                            *
 *************************************************/

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>

#include "utils.cuh"
#include "bitmap.cuh"
#define PIXEL_SIZE 3

cudaError_t turnGrayWithCuda(unsigned char* bitmapData, BitmapInfoHeader* header, unsigned int size);

// Turn given bitmap data to gray scale
__global__ void turnGray(unsigned char* bitmapData, unsigned long size, unsigned int width)
{
#define xIndex (blockIdx.x * blockDim.x + threadIdx.x)
#define yIndex (blockIdx.y * blockDim.y + threadIdx.y)
	unsigned long dataIndex = (xIndex + (yIndex * width)) * PIXEL_SIZE;
#define BLUE bitmapData[dataIndex]
#define GREEN bitmapData[dataIndex+1]
#define RED bitmapData[dataIndex+2]
	// Gray occurs when RED == GREEN == BLUE, so get average
	if(dataIndex < size) {
		unsigned char gray = (RED + GREEN + BLUE) / 3;
		// Convert all pixels to gray
		RED = gray;
		GREEN = gray;
		BLUE = gray;
	}
#undef BLUE
#undef GREEN
#undef RED
#undef i
#undef j
}

void printHelp(char* binary)
{
	printf("GrayScaleCuda\n");
	printf("----------------------------------");
	printf("\t-Bhathiya Perera\n");
	printf("Execute: %s <Bitmap>\n", binary);
}

int main(int argc, char** argv)
{
// Freeing data and calling cudaDeviceReset must be done
// All the time
#undef DO_FAILED_EXIT
#define DO_FAILED_EXIT()\
	free(header);\
	free(data);\
	cudaDeviceReset();\
	return EXIT_FAILURE;

	if (argc != 2) {
		printHelp(argv[0]);
		return EXIT_FAILURE;
	}

#ifdef DEBUG
#define bitmapFilename "C:\\Users\\Bhathiya\\Desktop\\img.bmp"
#else
#define bitmapFilename argv[1]
#endif
	
	puts("--------------------------------------------------");
	LOG("Welcome to grayscale with cuda.");

	LOG("Turning %s to grayscale...", bitmapFilename);

	BitmapInfoHeader* header = 0;
	header = (BitmapInfoHeader*)malloc(sizeof(BitmapInfoHeader));
	unsigned char* data = loadBitmapFile(bitmapFilename, header);
	if (data==NULL) {
		LOG_ERROR("Failed to load bitmap");
		DO_FAILED_EXIT();
	}

	cudaError_t cudaStatus = turnGrayWithCuda(data, header, header->sizeImage);
    REPORT_CUDA_ERROR(cudaStatus, "Unable to turn grayscale with cuda");

	int success = overwriteBitmapData(bitmapFilename, data);
	if(!success) {
		LOG_ERROR("Failed to overwrite bitmap");
		DO_FAILED_EXIT();
	}

	free(header);
	free(data);
	cudaDeviceReset();
    return EXIT_SUCCESS;
}

// Helper function for using CUDA to convert bitmap data to gray
cudaError_t turnGrayWithCuda(unsigned char* bitmapData, BitmapInfoHeader* header, unsigned int size)
{
#undef DO_FAILED_EXIT
#define DO_FAILED_EXIT() cudaFree(devBitmap); return cudaStatus;
	unsigned char* devBitmap = 0;
    cudaError_t cudaStatus;
	size_t dataSize = size * sizeof(unsigned char);
	unsigned long pixelCount = size / PIXEL_SIZE;
	LOG("size=%d, dataSize=%d, pixelCount=%d", size, dataSize, pixelCount);
	LOG("Image Width=%d Height=%d", header->width, header->height);
	cudaStatus = selectBestDevice();
	REPORT_CUDA_ERROR(cudaStatus, "Unable to select a cuda device! "
		"Do you have a CUDA-capable GPU installed?");

	// Allocate GPU buffer for bitmap data
    cudaStatus = cudaMalloc((void**)&devBitmap, dataSize);
	REPORT_CUDA_ERROR(cudaStatus, "Unable allocate device memory");

	// Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(devBitmap, bitmapData, dataSize,
		cudaMemcpyHostToDevice);
	REPORT_CUDA_ERROR(cudaStatus, "Copying memory failed!");

	dim3 threadsPerBlock(32, 32);
	dim3 blocksPerGrid((header->width + threadsPerBlock.x - 1) / threadsPerBlock.x,
		(header->height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    LOG("CUDA kernel launch with %dx%d blocks of %dx%d threads. Total threads=%d",
		blocksPerGrid.x, blocksPerGrid.y, threadsPerBlock.x, threadsPerBlock.y,
		blocksPerGrid.x * blocksPerGrid.y * threadsPerBlock.x * threadsPerBlock.y);

	CUDA_START_TIMER;
	// Launch a kernel on the GPU
	turnGray<<<blocksPerGrid, threadsPerBlock>>>(devBitmap, size, header->width);
	CUDA_STOP_TIMER;

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
	REPORT_CUDA_ERROR(cudaStatus, "Kernel launch failed: %s",
		cudaGetErrorString(cudaStatus));

    // Function cudaDeviceSynchronize waits for the kernel to finish, and returns
    // Any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
	REPORT_CUDA_ERROR(cudaStatus, "cudaDeviceSynchronize() returned error"
		" code %d after launching kernel!", cudaStatus);

	CUDA_LOG_TIME(size*2.0f/milliseconds/1e6f);

    // Copy bitmap data from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(bitmapData, devBitmap, dataSize,
		cudaMemcpyDeviceToHost);
	REPORT_CUDA_ERROR(cudaStatus, "Copying memory failed!");

    cudaFree(devBitmap);
    return cudaStatus;
}