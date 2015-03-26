#ifndef UTILS_CUH
#define UTILS_CUH

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#define LOG_ERROR(message, ...) fprintf(stderr, "[ERROR] " message "\n", ##__VA_ARGS__);
#define LOG(message, ...) printf("[INFO] " message "\n", ##__VA_ARGS__);

// Note: `do {} while (0)` is a best practice
// That prevents generating broken code

// Print an array
#define PRINT_ARRAY(data, size)\
	do {\
		LOG("Printing %s...", #data);\
		for(int i = 0; i < size; ++i) {\
			LOG("%s[%02d]=%d", #data, i, data[i]);\
		}\
	} while (0)

#define DO_FAILED_EXIT()

// Make sure that a custom DO_FAILED_EXIT is defined
#define REPORT_CUDA_ERROR(cudaStatus, message, ...)\
	do {\
		if(cudaStatus != cudaSuccess) {\
			fprintf(stderr, "[CUDA-ERROR] " message "\n", ##__VA_ARGS__);\
			DO_FAILED_EXIT();\
		}\
	} while (0)

#ifndef EXIT_SUCCESS
#define EXIT_SUCCESS 0
#endif

#ifndef EXIT_FAILURE
#define EXIT_FAILURE 1
#endif EXIT_FAILURE

#ifndef TRUE
#define TRUE 1
#endif

#ifndef FALSE
#define FALSE 0
#endif

// Note: these two cannot be put into a `do {} while(0)`
// Since `start` and `stop` should be accessible
#define CUDA_START_TIMER\
	cudaEvent_t start, stop;\
	cudaEventCreate(&start);\
	cudaEventCreate(&stop);\
	cudaEventRecord(start);

#define CUDA_STOP_TIMER	cudaEventRecord(stop);

#define CUDA_LOG_TIME(bandwidth)\
	do {\
		cudaEventSynchronize(stop);\
		float milliseconds = 0;\
		cudaEventElapsedTime(&milliseconds, start, stop);\
		LOG("Time taken: %f ms", milliseconds);\
		LOG("Effective Bandwidth: %f GB/s", bandwidth);\
	} while (0)


// Select best GPU for processing
cudaError_t selectBestDevice(void);

#endif // UTILS_CUH