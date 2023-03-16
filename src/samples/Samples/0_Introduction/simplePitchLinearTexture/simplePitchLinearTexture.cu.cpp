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

/* pitchLinearTexture
*
* This example demonstrates how to use textures bound to pitch linear memory.
* It performs a shift of matrix elements using wrap addressing mode (aka
* periodic boundary conditions) on two arrays, a pitch linear and a CUDA array,
* in order to highlight the differences in using each.
*
* Textures binding to pitch linear memory is a new feature in CUDA 2.2,
* and allows use of texture features such as wrap addressing mode and
* filtering which are not possible with textures bound to regular linear memory
*/

// includes, system
#include <stdio.h>

#ifdef _WIN32
#define WINDOWS_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif

#include "rocprofiler.h"
#include "HIPCHECK.h"
// Includes CUDA
#include <hip/hip_runtime.h>

// Utilities and timing functions
#include "helper_functions.h"  // includes hip/hip_runtime.h and hip/hip_runtime_api.h

// CUDA helper functions
#include "helper_cuda_hipified.h"  // helper functions for CUDA error check

#define NUM_REPS 100  // number of repetitions performed
#define TILE_DIM 16   // tile/block size

const char *sSDKsample = "simplePitchLinearTexture";

// Auto-Verification Code
bool bTestResult = true;

////////////////////////////////////////////////////////////////////////////////
// NB: (1) The second argument "pitch" is in elements, not bytes
//     (2) normalized coordinates are used (required for wrap address mode)
////////////////////////////////////////////////////////////////////////////////
//! Shifts matrix elements using pitch linear array
//! @param odata  output data in global memory
////////////////////////////////////////////////////////////////////////////////
__global__ void shiftPitchLinear(float *odata, int pitch, int width, int height,
                                 int shiftX, int shiftY,
                                 hipTextureObject_t texRefPL) {
  int xid = blockIdx.x * blockDim.x + threadIdx.x;
  int yid = blockIdx.y * blockDim.y + threadIdx.y;

  odata[yid * pitch + xid] = tex2D<float>(
      texRefPL, (xid + shiftX) / (float)width, (yid + shiftY) / (float)height);
}

////////////////////////////////////////////////////////////////////////////////
//! Shifts matrix elements using regular array
//! @param odata  output data in global memory
////////////////////////////////////////////////////////////////////////////////
__global__ void shiftArray(float *odata, int pitch, int width, int height,
                           int shiftX, int shiftY,
                           hipTextureObject_t texRefArray) {
  int xid = blockIdx.x * blockDim.x + threadIdx.x;
  int yid = blockIdx.y * blockDim.y + threadIdx.y;

  odata[yid * pitch + xid] =
      tex2D<float>(texRefArray, (xid + shiftX) / (float)width,
                   (yid + shiftY) / (float)height);
}

////////////////////////////////////////////////////////////////////////////////
// Declaration, forward
void runTest(int argc, char **argv);

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
  printf("%s starting...\n\n", sSDKsample);

  runTest(argc, argv);

  printf("%s completed, returned %s\n", sSDKsample,
         bTestResult ? "OK" : "ERROR!");
  exit(bTestResult ? EXIT_SUCCESS : EXIT_FAILURE);
}

////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
void runTest(int argc, char **argv) {
  // Set array size
  const int nx = 2048;
  const int ny = 2048;

  // Setup shifts applied to x and y data
  const int x_shift = 5;
  const int y_shift = 7;

  if ((nx % TILE_DIM != 0) || (ny % TILE_DIM != 0)) {
    printf("nx and ny must be multiples of TILE_DIM\n");
    exit(EXIT_FAILURE);
  }

  // Setup execution configuration parameters
  dim3 dimGrid(nx / TILE_DIM, ny / TILE_DIM), dimBlock(TILE_DIM, TILE_DIM);

  // This will pick the best possible CUDA capable device
  int devID = findCudaDevice(argc, (const char **)argv);

  // CUDA events for timing
  hipEvent_t start, stop;
  hipEventCreate(&start);
  hipEventCreate(&stop);

  // Host allocation and initialization
  float *h_idata = (float *)malloc(sizeof(float) * nx * ny);
  float *h_odata = (float *)malloc(sizeof(float) * nx * ny);
  float *gold = (float *)malloc(sizeof(float) * nx * ny);

  for (int i = 0; i < nx * ny; ++i) {
    h_idata[i] = (float)i;
  }

  // Device memory allocation
  // Pitch linear input data
  float *d_idataPL;
  size_t d_pitchBytes;

  HIPCHECK(hipMallocPitch((void **)&d_idataPL, &d_pitchBytes,
                                  nx * sizeof(float), ny));

  // Array input data
  hipArray *d_idataArray;
  hipChannelFormatDesc channelDesc = hipCreateChannelDesc<float>();

  HIPCHECK(hipMallocArray(&d_idataArray, &channelDesc, nx, ny));

  // Pitch linear output data
  float *d_odata;
  HIPCHECK(hipMallocPitch((void **)&d_odata, &d_pitchBytes,
                                  nx * sizeof(float), ny));

  // Copy host data to device
  // Pitch linear
  size_t h_pitchBytes = nx * sizeof(float);

  HIPCHECK(hipMemcpy2D(d_idataPL, d_pitchBytes, h_idata, h_pitchBytes,
                               nx * sizeof(float), ny, hipMemcpyHostToDevice));

  // Array
  HIPCHECK(hipMemcpyToArray(d_idataArray, 0, 0, h_idata,
                                    nx * ny * sizeof(float),
                                    hipMemcpyHostToDevice));

  hipTextureObject_t texRefPL;
  hipTextureObject_t texRefArray;
  hipResourceDesc texRes;
  memset(&texRes, 0, sizeof(hipResourceDesc));

  texRes.resType = hipResourceTypePitch2D;
  texRes.res.pitch2D.devPtr = d_idataPL;
  texRes.res.pitch2D.desc = channelDesc;
  texRes.res.pitch2D.width = nx;
  texRes.res.pitch2D.height = ny;
  texRes.res.pitch2D.pitchInBytes = h_pitchBytes;
  hipTextureDesc texDescr;
  memset(&texDescr, 0, sizeof(hipTextureDesc));

  texDescr.normalizedCoords = true;
  texDescr.filterMode = hipFilterModePoint;
  texDescr.addressMode[0] = hipAddressModeWrap;
  texDescr.addressMode[1] = hipAddressModeWrap;
  texDescr.readMode = hipReadModeElementType;

  HIPCHECK(hipCreateTextureObject(&texRefPL, &texRes, &texDescr, NULL));
  memset(&texRes, 0, sizeof(hipResourceDesc));
  memset(&texDescr, 0, sizeof(hipTextureDesc));
  texRes.resType = hipResourceTypeArray;
  texRes.res.array.array = d_idataArray;
  texDescr.normalizedCoords = true;
  texDescr.filterMode = hipFilterModePoint;
  texDescr.addressMode[0] = hipAddressModeWrap;
  texDescr.addressMode[1] = hipAddressModeWrap;
  texDescr.readMode = hipReadModeElementType;
  HIPCHECK(
      hipCreateTextureObject(&texRefArray, &texRes, &texDescr, NULL));

  // Reference calculation
  for (int j = 0; j < ny; ++j) {
    int jshift = (j + y_shift) % ny;

    for (int i = 0; i < nx; ++i) {
      int ishift = (i + x_shift) % nx;
      gold[j * nx + i] = h_idata[jshift * nx + ishift];
    }
  }

  // Run ShiftPitchLinear kernel
  HIPCHECK(
      hipMemset2D(d_odata, d_pitchBytes, 0, nx * sizeof(float), ny));

  HIPCHECK(hipEventRecord(start, 0));

  for (int i = 0; i < NUM_REPS; ++i) {
    shiftPitchLinear<<<dimGrid, dimBlock>>>(d_odata,
                                            (int)(d_pitchBytes / sizeof(float)),
                                            nx, ny, x_shift, y_shift, texRefPL);
  }

  HIPCHECK(hipEventRecord(stop, 0));
  HIPCHECK(hipEventSynchronize(stop));
  float timePL;
  HIPCHECK(hipEventElapsedTime(&timePL, start, stop));

  // Check results
  HIPCHECK(hipMemcpy2D(h_odata, h_pitchBytes, d_odata, d_pitchBytes,
                               nx * sizeof(float), ny, hipMemcpyDeviceToHost));

  bool res = compareData(gold, h_odata, nx * ny, 0.0f, 0.15f);

  bTestResult = true;

  if (res == false) {
    printf("*** shiftPitchLinear failed ***\n");
    bTestResult = false;
  }

  // Run ShiftArray kernel
  HIPCHECK(
      hipMemset2D(d_odata, d_pitchBytes, 0, nx * sizeof(float), ny));
  HIPCHECK(hipEventRecord(start, 0));

  for (int i = 0; i < NUM_REPS; ++i) {
    shiftArray<<<dimGrid, dimBlock>>>(d_odata,
                                      (int)(d_pitchBytes / sizeof(float)), nx,
                                      ny, x_shift, y_shift, texRefArray);
  }

  HIPCHECK(hipEventRecord(stop, 0));
  HIPCHECK(hipEventSynchronize(stop));
  float timeArray;
  HIPCHECK(hipEventElapsedTime(&timeArray, start, stop));

  // Check results
  HIPCHECK(hipMemcpy2D(h_odata, h_pitchBytes, d_odata, d_pitchBytes,
                               nx * sizeof(float), ny, hipMemcpyDeviceToHost));
  res = compareData(gold, h_odata, nx * ny, 0.0f, 0.15f);

  if (res == false) {
    printf("*** shiftArray failed ***\n");
    bTestResult = false;
  }

  float bandwidthPL =
      2.f * 1000.f * nx * ny * sizeof(float) / (1.e+9f) / (timePL / NUM_REPS);
  float bandwidthArray = 2.f * 1000.f * nx * ny * sizeof(float) / (1.e+9f) /
                         (timeArray / NUM_REPS);

  printf("\nBandwidth (GB/s) for pitch linear: %.2e; for array: %.2e\n",
         bandwidthPL, bandwidthArray);

  float fetchRatePL = nx * ny / 1.e+6f / (timePL / (1000.0f * NUM_REPS));
  float fetchRateArray = nx * ny / 1.e+6f / (timeArray / (1000.0f * NUM_REPS));

  printf(
      "\nTexture fetch rate (Mpix/s) for pitch linear: "
      "%.2e; for array: %.2e\n\n",
      fetchRatePL, fetchRateArray);

  // Cleanup
  free(h_idata);
  free(h_odata);
  free(gold);

  HIPCHECK(hipDestroyTextureObject(texRefPL));
  HIPCHECK(hipDestroyTextureObject(texRefArray));
  HIPCHECK(hipFree(d_idataPL));
  HIPCHECK(hipFreeArray(d_idataArray));
  HIPCHECK(hipFree(d_odata));

  HIPCHECK(hipEventDestroy(start));
  HIPCHECK(hipEventDestroy(stop));
}
