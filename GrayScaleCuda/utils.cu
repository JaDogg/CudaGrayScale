#include "utils.cuh"

// Choose which Best Device to run on
// Based on number of multiProcessors available
cudaError_t selectBestDevice(void)
{
	int selectedDevice = 0;
	int numDevices;
	cudaGetDeviceCount(&numDevices);

	if(numDevices > 1) {
		int maxMultiProc = 0;
		int device;
		for(device=0; device < numDevices; ++device) {
			cudaDeviceProp prop;
			cudaGetDeviceProperties(&prop, device);
			if (maxMultiProc < prop.multiProcessorCount) {
				maxMultiProc = prop.multiProcessorCount;
				selectedDevice = device;
			}
		}
	}

	return cudaSetDevice(selectedDevice);
}