#include "hip/hip_runtime.h"
/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

//
// DESCRIPTION:   Simple CUDA consumer rendering sample app
//

#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime.h>
#include <stdio.h>
#include <string.h>

#include "eglstrm_common.h"

extern bool isCrossDevice;

__device__ static unsigned int numErrors = 0, errorFound = 0;
__device__ void checkProducerDataGPU(char *data, int size, char expectedVal,
                                     int frameNumber) {
  if ((data[blockDim.x * blockIdx.x + threadIdx.x] != expectedVal) &&
      (!errorFound)) {
    printf("Producer FOUND:%d expected: %d at %d for trial %d %d\n",
           data[blockDim.x * blockIdx.x + threadIdx.x], expectedVal,
           (blockDim.x * blockIdx.x + threadIdx.x), frameNumber, numErrors);
    numErrors++;
    errorFound = 1;
    return;
  }
}

__device__ void checkConsumerDataGPU(char *data, int size, char expectedVal,
                                     int frameNumber) {
  if ((data[blockDim.x * blockIdx.x + threadIdx.x] != expectedVal) &&
      (!errorFound)) {
    printf("Consumer FOUND:%d expected: %d at %d for trial %d %d\n",
           data[blockDim.x * blockIdx.x + threadIdx.x], expectedVal,
           (blockDim.x * blockIdx.x + threadIdx.x), frameNumber, numErrors);
    numErrors++;
    errorFound = 1;
    return;
  }
}

__global__ void writeDataToBuffer(char *pSrc, char newVal) {
  pSrc[blockDim.x * blockIdx.x + threadIdx.x] = newVal;
}

__global__ void testKernelConsumer(char *pSrc, char size, char expectedVal,
                                   char newVal, int frameNumber) {
  checkConsumerDataGPU(pSrc, size, expectedVal, frameNumber);
}

__global__ void testKernelProducer(char *pSrc, char size, char expectedVal,
                                   char newVal, int frameNumber) {
  checkProducerDataGPU(pSrc, size, expectedVal, frameNumber);
}
__global__ void getNumErrors(int *numErr) { *numErr = numErrors; }

hipError_t cudaProducer_filter(hipStream_t pStream, char *pSrc, int width,
                                int height, char expectedVal, char newVal,
                                int frameNumber) {
  // in case where consumer is on dgpu and producer is on igpu when return is
  // called the frame is not copied back to igpu. So the consumer changes is not
  // visible to producer
  if (isCrossDevice == 0) {
    testKernelProducer<<<(width * height) / 1024, 1024, 1, pStream>>>(
        pSrc, width * height, expectedVal, newVal, frameNumber);
  }
  writeDataToBuffer<<<(width * height) / 1024, 1024, 1, pStream>>>(pSrc,
                                                                   newVal);
  return hipSuccess;
};

hipError_t cudaConsumer_filter(hipStream_t cStream, char *pSrc, int width,
                                int height, char expectedVal, char newVal,
                                int frameNumber) {
  testKernelConsumer<<<(width * height) / 1024, 1024, 1, cStream>>>(
      pSrc, width * height, expectedVal, newVal, frameNumber);
  writeDataToBuffer<<<(width * height) / 1024, 1024, 1, cStream>>>(pSrc,
                                                                   newVal);
  return hipSuccess;
};

hipError_t cudaGetValueMismatch() {
  int numErr_h;
  int *numErr_d = NULL;
  hipError_t err = hipSuccess;
  err = hipMalloc(&numErr_d, sizeof(int));
  if (err != hipSuccess) {
    printf("Cuda Main: hipMalloc failed with %s\n", hipGetErrorString(err));
    return err;
  }
  getNumErrors<<<1, 1>>>(numErr_d);
  err = hipDeviceSynchronize();
  if (err != hipSuccess) {
    printf("Cuda Main: hipDeviceSynchronize failed with %s\n",
           hipGetErrorString(err));
  }
  err = hipMemcpy(&numErr_h, numErr_d, sizeof(int), hipMemcpyDeviceToHost);
  if (err != hipSuccess) {
    printf("Cuda Main: hipMemcpy failed with %s\n", hipGetErrorString(err));
    hipFree(numErr_d);
    return err;
  }
  err = hipFree(numErr_d);
  if (err != hipSuccess) {
    printf("Cuda Main: hipFree failed with %s\n", hipGetErrorString(err));
    return err;
  }
  if (numErr_h > 0) {
    return hipErrorUnknown;
  }
  return hipSuccess;
}
