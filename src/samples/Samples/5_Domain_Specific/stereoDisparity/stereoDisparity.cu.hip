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

/* A CUDA program that demonstrates how to compute a stereo disparity map using
 * SIMD SAD (Sum of Absolute Difference) intrinsics
 */

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, kernels
#include <hip/hip_runtime.h>
#include "stereoDisparity_kernel.cuh"

// includes, project
#include "helper_functions.h"  // helper for shared that are common to CUDA Samples
#include "helper_cuda_hipified.h"  // helper for checking cuda initialization and error checking
#include <helper_string.h>  // helper functions for string parsing

static const char *sSDKsample = "[stereoDisparity]\0";

int iDivUp(int a, int b) { return ((a % b) != 0) ? (a / b + 1) : (a / b); }

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void runTest(int argc, char **argv);

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
  printf("%s Starting...\n\n", sSDKsample);
  runTest(argc, argv);
}

////////////////////////////////////////////////////////////////////////////////
//! CUDA Sample for calculating depth maps
////////////////////////////////////////////////////////////////////////////////
void runTest(int argc, char **argv) {
  hipDeviceProp_t deviceProp;
  deviceProp.major = 0;
  deviceProp.minor = 0;
  int dev = 0;

  // This will pick the best possible CUDA capable device
  dev = findCudaDevice(argc, (const char **)argv);

  HIPCHECK(hipGetDeviceProperties(&deviceProp, dev));

  // Statistics about the GPU device
  printf(
      "> GPU device has %d Multi-Processors, SM %d.%d compute capabilities\n\n",
      deviceProp.multiProcessorCount, deviceProp.major, deviceProp.minor);

  StopWatchInterface *timer;
  sdkCreateTimer(&timer);

  // Search parameters
  int minDisp = -16;
  int maxDisp = 0;

  // Load image data
  // allocate mem for the images on host side
  // initialize pointers to NULL to request lib call to allocate as needed
  // PPM images are loaded into 4 byte/pixel memory (RGBX)
  unsigned char *h_img0 = NULL;
  unsigned char *h_img1 = NULL;
  unsigned int w, h;
  char *fname0 = sdkFindFilePath("stereo.im0.640x533.ppm", argv[0]);
  char *fname1 = sdkFindFilePath("stereo.im1.640x533.ppm", argv[0]);

  printf("Loaded <%s> as image 0\n", fname0);

  if (!sdkLoadPPM4ub(fname0, &h_img0, &w, &h)) {
    fprintf(stderr, "Failed to load <%s>\n", fname0);
  }

  printf("Loaded <%s> as image 1\n", fname1);

  if (!sdkLoadPPM4ub(fname1, &h_img1, &w, &h)) {
    fprintf(stderr, "Failed to load <%s>\n", fname1);
  }

  dim3 numThreads = dim3(blockSize_x, blockSize_y, 1);
  dim3 numBlocks = dim3(iDivUp(w, numThreads.x), iDivUp(h, numThreads.y));
  unsigned int numData = w * h;
  unsigned int memSize = sizeof(int) * numData;

  // allocate mem for the result on host side
  unsigned int *h_odata = (unsigned int *)malloc(memSize);

  // initialize the memory
  for (unsigned int i = 0; i < numData; i++) h_odata[i] = 0;

  // allocate device memory for result
  unsigned int *d_odata, *d_img0, *d_img1;

  HIPCHECK(hipMalloc((void **)&d_odata, memSize));
  HIPCHECK(hipMalloc((void **)&d_img0, memSize));
  HIPCHECK(hipMalloc((void **)&d_img1, memSize));

  // copy host memory to device to initialize to zeros
  HIPCHECK(hipMemcpy(d_img0, h_img0, memSize, hipMemcpyHostToDevice));
  HIPCHECK(hipMemcpy(d_img1, h_img1, memSize, hipMemcpyHostToDevice));
  HIPCHECK(
      hipMemcpy(d_odata, h_odata, memSize, hipMemcpyHostToDevice));

  hipChannelFormatDesc ca_desc0 = hipCreateChannelDesc<unsigned int>();
  hipChannelFormatDesc ca_desc1 = hipCreateChannelDesc<unsigned int>();

  hipTextureObject_t tex2Dleft, tex2Dright;
  hipResourceDesc texRes;
  memset(&texRes, 0, sizeof(hipResourceDesc));

  texRes.resType = hipResourceTypePitch2D;
  texRes.res.pitch2D.devPtr = d_img0;
  texRes.res.pitch2D.desc = ca_desc0;
  texRes.res.pitch2D.width = w;
  texRes.res.pitch2D.height = h;
  texRes.res.pitch2D.pitchInBytes = w * 4;

  hipTextureDesc texDescr;
  memset(&texDescr, 0, sizeof(hipTextureDesc));

  texDescr.normalizedCoords = false;
  texDescr.filterMode = hipFilterModePoint;
  texDescr.addressMode[0] = hipAddressModeClamp;
  texDescr.addressMode[1] = hipAddressModeClamp;
  texDescr.readMode = hipReadModeElementType;

  HIPCHECK(
      hipCreateTextureObject(&tex2Dleft, &texRes, &texDescr, NULL));

  memset(&texRes, 0, sizeof(hipResourceDesc));

  texRes.resType = hipResourceTypePitch2D;
  texRes.res.pitch2D.devPtr = d_img1;
  texRes.res.pitch2D.desc = ca_desc1;
  texRes.res.pitch2D.width = w;
  texRes.res.pitch2D.height = h;
  texRes.res.pitch2D.pitchInBytes = w * 4;

  memset(&texDescr, 0, sizeof(hipTextureDesc));

  texDescr.normalizedCoords = false;
  texDescr.filterMode = hipFilterModePoint;
  texDescr.addressMode[0] = hipAddressModeClamp;
  texDescr.addressMode[1] = hipAddressModeClamp;
  texDescr.readMode = hipReadModeElementType;

  HIPCHECK(
      hipCreateTextureObject(&tex2Dright, &texRes, &texDescr, NULL));

  // First run the warmup kernel (which we'll use to get the GPU in the correct
  // max power state
  stereoDisparityKernel<<<numBlocks, numThreads>>>(
      d_img0, d_img1, d_odata, w, h, minDisp, maxDisp, tex2Dleft, tex2Dright);
  hipDeviceSynchronize();

  // Allocate CUDA events that we'll use for timing
  hipEvent_t start, stop;
  HIPCHECK(hipEventCreate(&start));
  HIPCHECK(hipEventCreate(&stop));

  printf("Launching CUDA stereoDisparityKernel()\n");

  // Record the start event
  HIPCHECK(hipEventRecord(start, NULL));

  // launch the stereoDisparity kernel
  stereoDisparityKernel<<<numBlocks, numThreads>>>(
      d_img0, d_img1, d_odata, w, h, minDisp, maxDisp, tex2Dleft, tex2Dright);

  // Record the stop event
  HIPCHECK(hipEventRecord(stop, NULL));

  // Wait for the stop event to complete
  HIPCHECK(hipEventSynchronize(stop));

  // Check to make sure the kernel didn't fail
  getLastCudaError("Kernel execution failed");

  float msecTotal = 0.0f;
  HIPCHECK(hipEventElapsedTime(&msecTotal, start, stop));

  // Copy result from device to host for verification
  HIPCHECK(
      hipMemcpy(h_odata, d_odata, memSize, hipMemcpyDeviceToHost));

  printf("Input Size  [%dx%d], ", w, h);
  printf("Kernel size [%dx%d], ", (2 * RAD + 1), (2 * RAD + 1));
  printf("Disparities [%d:%d]\n", minDisp, maxDisp);

  printf("GPU processing time : %.4f (ms)\n", msecTotal);
  printf("Pixel throughput    : %.3f Mpixels/sec\n",
         ((float)(w * h * 1000.f) / msecTotal) / 1000000);

  // calculate sum of resultant GPU image
  unsigned int checkSum = 0;

  for (unsigned int i = 0; i < w * h; i++) {
    checkSum += h_odata[i];
  }

  printf("GPU Checksum = %u, ", checkSum);

  // write out the resulting disparity image.
  unsigned char *dispOut = (unsigned char *)malloc(numData);
  int mult = 20;
  const char *fnameOut = "output_GPU.pgm";

  for (unsigned int i = 0; i < numData; i++) {
    dispOut[i] = (int)h_odata[i] * mult;
  }

  printf("GPU image: <%s>\n", fnameOut);
  sdkSavePGM(fnameOut, dispOut, w, h);

  // compute reference solution
  printf("Computing CPU reference...\n");
  cpu_gold_stereo((unsigned int *)h_img0, (unsigned int *)h_img1,
                  (unsigned int *)h_odata, w, h, minDisp, maxDisp);
  unsigned int cpuCheckSum = 0;

  for (unsigned int i = 0; i < w * h; i++) {
    cpuCheckSum += h_odata[i];
  }

  printf("CPU Checksum = %u, ", cpuCheckSum);
  const char *cpuFnameOut = "output_CPU.pgm";

  for (unsigned int i = 0; i < numData; i++) {
    dispOut[i] = (int)h_odata[i] * mult;
  }

  printf("CPU image: <%s>\n", cpuFnameOut);
  sdkSavePGM(cpuFnameOut, dispOut, w, h);

  // cleanup memory
  HIPCHECK(hipFree(d_odata));
  HIPCHECK(hipFree(d_img0));
  HIPCHECK(hipFree(d_img1));

  if (h_odata != NULL) free(h_odata);

  if (h_img0 != NULL) free(h_img0);

  if (h_img1 != NULL) free(h_img1);

  if (dispOut != NULL) free(dispOut);

  sdkDeleteTimer(&timer);

  exit((checkSum == cpuCheckSum) ? EXIT_SUCCESS : EXIT_FAILURE);
}
  if (dispOut != NULL) free(dispOut);

  sdkDeleteTimer(&timer);

  exit((checkSum == cpuCheckSum) ? EXIT_SUCCESS : EXIT_FAILURE);
}
  if (dispOut != NULL) free(dispOut);

  sdkDeleteTimer(&timer);

  exit((checkSum == cpuCheckSum) ? EXIT_SUCCESS : EXIT_FAILURE);
}
