#pragma once
#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>
#include <string>

inline void _check_cuda_launch_and_execute_error(const char* file, int line) {
	// Check for kernel launch errors
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		throw std::runtime_error(std::string("CUDA kernel launch failed: ") + cudaGetErrorString(err) +
								 " at " + file + ":" + std::to_string(line));
	}

	// Check for kernel execution errors
	err = cudaDeviceSynchronize();
	if (err != cudaSuccess) {
		throw std::runtime_error(std::string("CUDA kernel execution failed: ") + cudaGetErrorString(err) +
								 " at " + file + ":" + std::to_string(line));
	}
}

#define CUDA_LAUNCH_AND_EXECUTE_CHECK() (_check_cuda_launch_and_execute_error(__FILE__, __LINE__))