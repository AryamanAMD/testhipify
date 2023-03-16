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

/*
 * This sample demonstrates how use texture fetches in CUDA
 *
 * This sample takes an input PGM image (imageFilename) and generates
 * an output PGM image (imageFilename_out).  This CUDA kernel performs
 * a simple 2D transform (rotation) on the texture coordinates (u,v).
 */

// Includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#ifdef _WIN32
#define WINDOWS_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include "rocprofiler.h"
#include "HIPCHECK.h"
#endif

// Includes CUDA
#include <hip/hip_runtime.h>

// Utilities and timing functions
#include "helper_functions.h"  // includes hip/hip_runtime.h and hip/hip_runtime_api.h

// CUDA helper functions
#include "helper_cuda_hipified.h"  // helper functions for CUDA error check

#define MIN_EPSILON_ERROR 5e-3f

////////////////////////////////////////////////////////////////////////////////
// Define the files that are to be save and the reference images for validation
const char *imageFilename = "teapot512.pgm";
const char *refFilename = "ref_rotated.pgm";
float angle = 0.5f;  // angle to rotate image by (in radians)

// Auto-Verification Code
bool testResult = true;

static const char *sampleName = "simpleSurfaceWrite";

////////////////////////////////////////////////////////////////////////////////
// Kernels
////////////////////////////////////////////////////////////////////////////////
//! Write to a cuArray (texture data source) using surface writes
//! @param gIData input data in global memory
////////////////////////////////////////////////////////////////////////////////
__global__ void surfaceWriteKernel(float *gIData, int width, int height,
                                   hipSurfaceObject_t outputSurface) {
  // calculate surface coordinates
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

  // read from global memory and write to cuarray (via surface reference)
  surf2Dwrite(gIData[y * width + x], outputSurface, x * 4, y);
             // hipBoundaryModeTrap);
}

////////////////////////////////////////////////////////////////////////////////
//! Transform an image using texture lookups
//! @param gOData  output data in global memory
////////////////////////////////////////////////////////////////////////////////
__global__ void transformKernel(float *gOData, int width, int height,
                                float theta, hipTextureObject_t tex) {
  // calculate normalized texture coordinates
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

  float u = x / (float)width;
  float v = y / (float)height;

  // transform coordinates
  u -= 0.5f;
  v -= 0.5f;
  float tu = u * cosf(theta) - v * sinf(theta) + 0.5f;
  float tv = v * cosf(theta) + u * sinf(theta) + 0.5f;

  // read from texture and write to global memory
  gOData[y * width + x] = tex2D<float>(tex, tu, tv);
}

////////////////////////////////////////////////////////////////////////////////
// Declaration, forward
void runTest(int argc, char **argv);

extern "C" void computeGold(float *reference, float *idata,
                            const unsigned int len);

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
  printf("%s starting...\n", sampleName);

  // Process command-line arguments
  if (argc > 1) {
    if (checkCmdLineFlag(argc, (const char **)argv, "input")) {
      getCmdLineArgumentString(argc, (const char **)argv, "input",
                               (char **)&imageFilename);

      if (checkCmdLineFlag(argc, (const char **)argv, "reference")) {
        getCmdLineArgumentString(argc, (const char **)argv, "reference",
                                 (char **)&refFilename);
      } else {
        printf("-input flag should be used with -reference flag");
        exit(EXIT_FAILURE);
      }
    } else if (checkCmdLineFlag(argc, (const char **)argv, "reference")) {
      printf("-reference flag should be used with -input flag");
      exit(EXIT_FAILURE);
    }
  }

  runTest(argc, argv);

  printf("%s completed, returned %s\n", sampleName,
         testResult ? "OK" : "ERROR!");
  exit(testResult ? EXIT_SUCCESS : EXIT_FAILURE);
}

////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
void runTest(int argc, char **argv) {
  // Use command-line specified CUDA device,
  // otherwise use device with highest Gflops/s
  int devID = findCudaDevice(argc, (const char **)argv);

  // Get number of SMs on this GPU
  hipDeviceProp_t deviceProps;

  HIPCHECK(hipGetDeviceProperties(&deviceProps, devID));
  printf("CUDA device [%s] has %d Multi-Processors, SM %d.%d\n",
         deviceProps.name, deviceProps.multiProcessorCount, deviceProps.major,
         deviceProps.minor);

  // Load image from disk
  float *hData = NULL;
  unsigned int width, height;
  char *imagePath = sdkFindFilePath(imageFilename, argv[0]);

  if (imagePath == NULL) {
    printf("Unable to source image input file: %s\n", imageFilename);
    exit(EXIT_FAILURE);
  }

  sdkLoadPGM(imagePath, &hData, &width, &height);

  unsigned int size = width * height * sizeof(float);
  printf("Loaded '%s', %d x %d pixels\n", imageFilename, width, height);

  // Load reference image from image (output)
  float *hDataRef = (float *)malloc(size);
  char *refPath = sdkFindFilePath(refFilename, argv[0]);

  if (refPath == NULL) {
    printf("Unable to find reference image file: %s\n", refFilename);
    exit(EXIT_FAILURE);
  }

  sdkLoadPGM(refPath, &hDataRef, &width, &height);

  // Allocate device memory for result
  float *dData = NULL;
  HIPCHECK(hipMalloc((void **)&dData, size));

  // Allocate array and copy image data
  hipChannelFormatDesc channelDesc =
      hipCreateChannelDesc(32, 0, 0, 0, hipChannelFormatKindFloat);
  hipArray *cuArray;
  HIPCHECK(hipMallocArray(&cuArray, &channelDesc, width, height,
                                  hipArraySurfaceLoadStore));

  dim3 dimBlock(8, 8, 1);
  dim3 dimGrid(width / dimBlock.x, height / dimBlock.y, 1);

  hipSurfaceObject_t outputSurface;
  hipResourceDesc surfRes;
  memset(&surfRes, 0, sizeof(hipResourceDesc));
  surfRes.resType = hipResourceTypeArray;
  surfRes.res.array.array = cuArray;

  HIPCHECK(hipCreateSurfaceObject(&outputSurface, &surfRes));
#if 1
  HIPCHECK(hipMemcpy(dData, hData, size, hipMemcpyHostToDevice));
  surfaceWriteKernel<<<dimGrid, dimBlock>>>(dData, width, height,
                                            outputSurface);
#else  // This is what differs from the example simpleTexture
  HIPCHECK(
      hipMemcpyToArray(cuArray, 0, 0, hData, size, hipMemcpyHostToDevice));
#endif

  hipTextureObject_t tex;
  hipResourceDesc texRes;
  memset(&texRes, 0, sizeof(hipResourceDesc));

  texRes.resType = hipResourceTypeArray;
  texRes.res.array.array = cuArray;

  hipTextureDesc texDescr;
  memset(&texDescr, 0, sizeof(hipTextureDesc));

  texDescr.normalizedCoords = true;
  texDescr.filterMode = hipFilterModeLinear;
  texDescr.addressMode[0] = hipAddressModeWrap;
  texDescr.addressMode[1] = hipAddressModeWrap;
  texDescr.readMode = hipReadModeElementType;

  HIPCHECK(hipCreateTextureObject(&tex, &texRes, &texDescr, NULL));

  // Warmup
  transformKernel<<<dimGrid, dimBlock, 0>>>(dData, width, height, angle, tex);

  HIPCHECK(hipDeviceSynchronize());

  StopWatchInterface *timer = NULL;
  sdkCreateTimer(&timer);
  sdkStartTimer(&timer);

  // Execute the kernel
  transformKernel<<<dimGrid, dimBlock, 0>>>(dData, width, height, angle, tex);

  // Check if kernel execution generated an error
  getLastCudaError("Kernel execution failed");

  hipDeviceSynchronize();
  sdkStopTimer(&timer);
  printf("Processing time: %f (ms)\n", sdkGetTimerValue(&timer));
  printf("%.2f Mpixels/sec\n",
         (width * height / (sdkGetTimerValue(&timer) / 1000.0f)) / 1e6);
  sdkDeleteTimer(&timer);

  // Allocate mem for the result on host side
  float *hOData = (float *)malloc(size);
  // copy result from device to host
  HIPCHECK(hipMemcpy(hOData, dData, size, hipMemcpyDeviceToHost));

  // Write result to file
  char outputFilename[1024];
  strcpy(outputFilename, "output.pgm");
  sdkSavePGM("output.pgm", hOData, width, height);
  printf("Wrote '%s'\n", outputFilename);

  // Write regression file if necessary
  if (checkCmdLineFlag(argc, (const char **)argv, "regression")) {
    // Write file for regression test
    sdkWriteFile<float>("./data/regression.dat", hOData, width * height, 0.0f,
                        false);
  } else {
    // We need to reload the data from disk,
    // because it is inverted upon output
    sdkLoadPGM(outputFilename, &hOData, &width, &height);

    printf("Comparing files\n");
    printf("\toutput:    <%s>\n", outputFilename);
    printf("\treference: <%s>\n", refPath);
    testResult =
        compareData(hOData, hDataRef, width * height, MIN_EPSILON_ERROR, 0.0f);
  }

  HIPCHECK(hipDestroySurfaceObject(outputSurface));
  HIPCHECK(hipDestroyTextureObject(tex));
  HIPCHECK(hipFree(dData));
  HIPCHECK(hipFreeArray(cuArray));
  free(imagePath);
  free(refPath);
}
