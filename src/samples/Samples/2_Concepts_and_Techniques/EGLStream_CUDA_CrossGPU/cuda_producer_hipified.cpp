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
// DESCRIPTION:   Simple cuda EGL stream producer app
//

#include "cudaEGL.h"
#include "cuda_producer.h"
#include "eglstrm_common.h"
#include <hip/hip_runtime.h>
#if defined(EXTENSION_LIST)
EXTENSION_LIST(EXTLST_EXTERN)
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string.h>
#include "hip/hip_runtime.h"
#include "math.h"

int cudaPresentReturnData = INIT_DATA;
int fakePresent = 0;
CUeglFrame fakeFrame;
hipDeviceptr_t cudaPtrFake;
extern bool isCrossDevice;

void cudaProducerPrepareFrame(CUeglFrame *cudaEgl, hipDeviceptr_t cudaPtr,
                              int bufferSize) {
  cudaEgl->frame.pPitch[0] = (void *)cudaPtr;
  cudaEgl->width = WIDTH;
  cudaEgl->depth = 0;
  cudaEgl->height = HEIGHT;
  cudaEgl->pitch = WIDTH * 4;
  cudaEgl->frameType = CU_EGL_FRAME_TYPE_PITCH;
  cudaEgl->planeCount = 1;
  cudaEgl->numChannels = 4;
  cudaEgl->eglColorFormat = CU_EGL_COLOR_FORMAT_ARGB;
  cudaEgl->cuFormat = HIP_AD_FORMAT_UNSIGNED_INT8;
}

static int count_present = 0, count_return = 0;
static double present_time[25000] = {0}, total_time_present = 0;
static double return_time[25000] = {0}, total_time_return = 0;

void presentApiStat(void);
void presentApiStat(void) {
  int i = 0;
  double min = 10000000, max = 0;
  double average_launch_time = 0, standard_deviation = 0;
  if (count_present == 0) return;
  // lets compute the standard deviation
  min = max = present_time[1];
  average_launch_time = (total_time_present) / count_present;
  for (i = 1; i < count_present; i++) {
    standard_deviation += (present_time[i] - average_launch_time) *
                          (present_time[i] - average_launch_time);
    if (present_time[i] < min) min = present_time[i];
    if (present_time[i] > max) max = present_time[i];
  }
  standard_deviation = sqrt(standard_deviation / count_present);
  printf("present Avg: %lf\n", average_launch_time);
  printf("present  SD: %lf\n", standard_deviation);
  printf("present min: %lf\n", min);
  printf("present max: %lf\n", max);

  min = max = return_time[1];
  average_launch_time = (total_time_return - return_time[0]) / count_return;
  for (i = 1; i < count_return; i++) {
    standard_deviation += (return_time[i] - average_launch_time) *
                          (return_time[i] - average_launch_time);
    if (return_time[i] < min) min = return_time[i];
    if (return_time[i] > max) max = return_time[i];
  }
  standard_deviation = sqrt(standard_deviation / count_return);
  printf("return  Avg: %lf\n", average_launch_time);
  printf("return   SD: %lf\n", standard_deviation);
  printf("return  min: %lf\n", min);
  printf("return  max: %lf\n", max);
}
hipError_t cudaProducerPresentFrame(test_cuda_producer_s *cudaProducer,
                                  CUeglFrame cudaEgl, int t) {
  static int flag = 0;
  hipError_t status = hipSuccess;
  struct timespec start, end;
  double curTime;
  hipDeviceptr_t pDevPtr = (hipDeviceptr_t)cudaEgl.frame.pPitch[0];
  cudaProducer_filter(cudaProducer->prodCudaStream, (char *)pDevPtr, WIDTH * 4,
                      HEIGHT, cudaPresentReturnData, PROD_DATA + t, t);
  if (cudaProducer->profileAPI) {
    getTime(&start);
  }
  status = cuEGLStreamProducerPresentFrame(&cudaProducer->cudaConn, cudaEgl,
                                           &cudaProducer->prodCudaStream);
  if (status != hipSuccess) {
    printf("Cuda Producer: Present frame failed, status:%d\n", status);
    goto done;
  }
  flag++;
  if (cudaProducer->profileAPI && flag > 10) {
    getTime(&end);
    curTime = TIME_DIFF(end, start);
    present_time[count_present++] = curTime;
    if (count_present == 25000) count_present = 0;
    total_time_present += curTime;
  }
done:
  return status;
}

int flag = 0;
hipError_t cudaProducerReturnFrame(test_cuda_producer_s *cudaProducer,
                                 CUeglFrame cudaEgl, int t) {
  hipError_t status = hipSuccess;
  struct timespec start, end;
  double curTime;
  hipDeviceptr_t pDevPtr = 0;

  pDevPtr = (hipDeviceptr_t)cudaEgl.frame.pPitch[0];
  if (cudaProducer->profileAPI) {
    getTime(&start);
  }

  while (1) {
    status = cuEGLStreamProducerReturnFrame(&cudaProducer->cudaConn, &cudaEgl,
                                            &cudaProducer->prodCudaStream);
    if (status == hipErrorLaunchTimeOut) {
      continue;
    } else if (status != hipSuccess) {
      printf("Cuda Producer: Return frame failed, status:%d\n", status);
      goto done;
    }
    break;
  }
  if (cudaProducer->profileAPI) {
    getTime(&end);
    curTime = TIME_DIFF(end, start);
    return_time[count_return++] = curTime;
    if (count_return == 25000) count_return = 0;
    total_time_return += curTime;
  }
  if (flag % 2 == 0) {
    cudaPresentReturnData++;
  }
  cudaProducer_filter(cudaProducer->prodCudaStream, (char *)pDevPtr, WIDTH * 4,
                      HEIGHT, CONS_DATA + t, cudaPresentReturnData, t);
  flag++;
done:
  return status;
}

hipError_t cudaDeviceCreateProducer(test_cuda_producer_s *cudaProducer) {
  hipDevice_t device;
  hipError_t status = hipSuccess;

  if (hipSuccess != (status = hipInit(0))) {
    printf("Failed to initialize CUDA\n");
    return status;
  }

  if (hipSuccess !=
      (status = hipDeviceGet(&device, cudaProducer->cudaDevId))) {
    printf("failed to get CUDA device\n");
    return status;
  }

  if (hipSuccess !=
      (status = hipCtxCreate(&cudaProducer->context, 0, device))) {
    printf("failed to create CUDA context\n");
    return status;
  }

  int major = 0, minor = 0;
  char deviceName[256];
  hipDeviceGetAttribute(&major, hipDeviceAttributeComputeCapabilityMajor,
                       device);
  hipDeviceGetAttribute(&minor, hipDeviceAttributeComputeCapabilityMinor,
                       device);
  hipDeviceGetName(deviceName, 256, device);
  printf(
      "CUDA Producer on GPU Device %d: \"%s\" with compute capability "
      "%d.%d\n\n",
      device, deviceName, major, minor);

  hipCtxPopCurrent(&cudaProducer->context);

  if (major < 6) {
    printf(
        "EGLStream_CUDA_CrossGPU requires SM 6.0 or higher arch GPU.  "
        "Exiting...\n");
    exit(2);  // EXIT_WAIVED
  }

  return status;
}

hipError_t cudaProducerInit(test_cuda_producer_s *cudaProducer, TestArgs *args) {
  hipError_t status = hipSuccess;
  int bufferSize;

  cudaProducer->charCnt = args->charCnt;
  bufferSize = cudaProducer->charCnt;

  cudaProducer->tempBuff = (char *)malloc(bufferSize);
  if (!cudaProducer->tempBuff) {
    printf("Cuda Producer: Failed to allocate image buffer\n");
    status = hipErrorUnknown;
    goto done;
  }
  memset((void *)cudaProducer->tempBuff, INIT_DATA, cudaProducer->charCnt);

  // Fill this init data
  status = hipMalloc(&cudaProducer->cudaPtr, bufferSize);
  if (status != hipSuccess) {
    printf("Cuda Producer: cuda Malloc failed, status:%d\n", status);
    goto done;
  }
  status = hipMemcpyHtoD(cudaProducer->cudaPtr, (void *)(cudaProducer->tempBuff),
                        bufferSize);
  if (status != hipSuccess) {
    printf("Cuda Producer: cuMemCpy failed, status:%d\n", status);
    goto done;
  }

  // Fill this init data
  status = hipMalloc(&cudaProducer->cudaPtr1, bufferSize);
  if (status != hipSuccess) {
    printf("Cuda Producer: cuda Malloc failed, status:%d\n", status);
    goto done;
  }
  status = hipMemcpyHtoD(cudaProducer->cudaPtr1,
                        (void *)(cudaProducer->tempBuff), bufferSize);
  if (status != hipSuccess) {
    printf("Cuda Producer: cuMemCpy failed, status:%d\n", status);
    goto done;
  }

  status = hipStreamCreateWithFlags(&cudaProducer->prodCudaStream, 0);
  if (status != hipSuccess) {
    printf("Cuda Producer: hipStreamCreateWithFlags failed, status:%d\n", status);
    goto done;
  }

  // Fill this init data
  status = hipMalloc(&cudaPtrFake, 100);
  if (status != hipSuccess) {
    printf("Cuda Producer: cuda Malloc failed, status:%d\n", status);
    goto done;
  }

  atexit(presentApiStat);
done:
  return status;
}

hipError_t cudaProducerDeinit(test_cuda_producer_s *cudaProducer) {
  if (cudaProducer->tempBuff) {
    free(cudaProducer->tempBuff);
  }
  if (cudaProducer->cudaPtr) {
    hipFree(cudaProducer->cudaPtr);
  }
  return cuEGLStreamProducerDisconnect(&cudaProducer->cudaConn);
}
