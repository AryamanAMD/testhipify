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
// DESCRIPTION:   Simple cuda producer header file
//

#ifndef _CUDA_PRODUCER_H_
#define _CUDA_PRODUCER_H_
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include "cudaEGL.h"
#include "eglstrm_common.h"
#include <hip/hip_runtime.h>
#include <hip/hip_runtime.h>

typedef struct _test_cuda_producer_s {
  //  Stream params
  hipCtx_t context;
  CUeglStreamConnection cudaConn;
  int cudaDevId;
  EGLStreamKHR eglStream;
  EGLDisplay eglDisplay;
  unsigned int charCnt;
  bool profileAPI;
  char *tempBuff;
  hipDeviceptr_t cudaPtr;
  hipDeviceptr_t cudaPtr1;
  hipStream_t prodCudaStream;
} test_cuda_producer_s;

hipError_t cudaProducerInit(test_cuda_producer_s *cudaProducer, TestArgs *args);
hipError_t cudaProducerPresentFrame(test_cuda_producer_s *parserArg,
                                  CUeglFrame cudaEgl, int t);
hipError_t cudaProducerReturnFrame(test_cuda_producer_s *parserArg,
                                 CUeglFrame cudaEgl, int t);
hipError_t cudaProducerDeinit(test_cuda_producer_s *cudaProducer);
hipError_t cudaDeviceCreateProducer(test_cuda_producer_s *cudaProducer);
hipError_t cudaProducer_filter(hipStream_t cStream, char *pSrc, int width,
                                int height, char expectedVal, char newVal,
                                int frameNumber);
void cudaProducerPrepareFrame(CUeglFrame *cudaEgl, hipDeviceptr_t cudaPtr,
                              int bufferSize);
#endif
