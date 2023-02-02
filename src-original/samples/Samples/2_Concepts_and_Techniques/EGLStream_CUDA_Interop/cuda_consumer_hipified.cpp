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

#include "cuda_consumer.h"
#include <helper_cuda_drvapi.h>
#include "eglstrm_common.h"

#if defined(EXTENSION_LIST)
EXTENSION_LIST(EXTLST_EXTERN)
#endif

int checkbuf(FILE *fp1, FILE *fp2);

hipError_t cudaConsumerTest(test_cuda_consumer_s *data, char *fileName) {
  hipError_t cuStatus = hipSuccess;
  hipArray_t cudaArr = NULL;
  CUeglFrame cudaEgl;
  hipGraphicsResource_t cudaResource;
  unsigned int i;
  int check_result;
  FILE *pInFile1 = NULL, *pInFile2 = NULL, *file_p = NULL;
  EGLint streamState = 0;

  if (!data) {
    printf("%s: Bad parameter\n", __func__);
    goto done;
  }

  if (!eglQueryStreamKHR(g_display, eglStream, EGL_STREAM_STATE_KHR,
                         &streamState)) {
    printf("Cuda consumer, eglQueryStreamKHR EGL_STREAM_STATE_KHR failed\n");
  }
  if (streamState == EGL_STREAM_STATE_DISCONNECTED_KHR) {
    printf("CUDA Consumer: - EGL_STREAM_STATE_DISCONNECTED_KHR received\n");
  }

  if (streamState == EGL_STREAM_STATE_NEW_FRAME_AVAILABLE_KHR) {
    cuStatus = cuEGLStreamConsumerAcquireFrame(&(data->cudaConn), &cudaResource,
                                               NULL, 16000);

    if (cuStatus == hipSuccess) {
      hipDeviceptr_t pDevPtr = 0;
      int bufferSize;
      unsigned char *pCudaCopyMem = NULL;
      unsigned int copyWidthInBytes = 0, copyHeight = 0;

      file_p = fopen(fileName, "wb+");
      if (!file_p) {
        printf("WriteFrame: file open failed %s\n", fileName);
        cuStatus = hipErrorUnknown;
        goto done;
      }
      cuStatus =
          cuGraphicsResourceGetMappedEglFrame(&cudaEgl, cudaResource, 0, 0);
      if (cuStatus != hipSuccess) {
        printf("Cuda get resource failed with %d\n", cuStatus);
        goto done;
      }
      cuStatus = hipCtxSynchronize();
      if (cuStatus != hipSuccess) {
        printf("hipCtxSynchronize failed \n");
        goto done;
      }
      if (!(cudaEgl.planeCount >= 1 && cudaEgl.planeCount <= 3)) {
        printf("Plane count is invalid\nExiting\n");
        goto done;
      }

      for (i = 0; i < cudaEgl.planeCount; i++) {
        if (cudaEgl.frameType == CU_EGL_FRAME_TYPE_PITCH) {
          pDevPtr = (hipDeviceptr_t)cudaEgl.frame.pPitch[i];
          if (cudaEgl.planeCount == 1) {
            bufferSize = cudaEgl.pitch * cudaEgl.height;
            copyWidthInBytes = cudaEgl.pitch;
            copyHeight = data->height;
          } else if (i == 1 && cudaEgl.planeCount == 2) {  // YUV 420
                                                           // semi-planar
            bufferSize = cudaEgl.pitch * cudaEgl.height / 2;
            copyWidthInBytes = cudaEgl.pitch;
            copyHeight = data->height / 2;
          } else {
            bufferSize = data->width * data->height;
            copyWidthInBytes = data->width;
            copyHeight = data->height;
            if (i > 0) {
              bufferSize >>= 2;
              copyWidthInBytes >>= 1;
              copyHeight >>= 1;
            }
          }
        } else {
          cudaArr = cudaEgl.frame.pArray[i];
          if (cudaEgl.planeCount == 1) {
            bufferSize = data->width * data->height * 4;
            copyWidthInBytes = data->width * 4;
            copyHeight = data->height;
          } else if (i == 1 && cudaEgl.planeCount == 2) {  // YUV 420
                                                           // semi-planar
            bufferSize = data->width * data->height / 2;
            copyWidthInBytes = data->width;
            copyHeight = data->height / 2;
          } else {
            bufferSize = data->width * data->height;
            copyWidthInBytes = data->width;
            copyHeight = data->height;
            if (i > 0) {
              bufferSize >>= 2;
              copyWidthInBytes >>= 1;
              copyHeight >>= 1;
            }
          }
        }
        if (i == 0) {
          pCudaCopyMem = (unsigned char *)malloc(bufferSize);
          if (pCudaCopyMem == NULL) {
            printf("pCudaCopyMem malloc failed\n");
            goto done;
          }
        }
        memset(pCudaCopyMem, 0, bufferSize);
        if (data->pitchLinearOutput) {
          cuStatus = hipMemcpyDtoH(pCudaCopyMem, pDevPtr, bufferSize);
          if (cuStatus != hipSuccess) {
            printf(
                "cuda_consumer: pitch linear Memcpy failed, bufferSize =%d\n",
                bufferSize);
            goto done;
          }
          cuStatus = hipCtxSynchronize();
          if (cuStatus != hipSuccess) {
            printf("cuda_consumer: hipCtxSynchronize failed after memcpy \n");
            goto done;
          }
        } else {
          HIP_MEMCPY3D cpdesc;
          memset(&cpdesc, 0, sizeof(cpdesc));
          cpdesc.srcXInBytes = cpdesc.srcY = cpdesc.srcZ = cpdesc.srcLOD = 0;
          cpdesc.srcMemoryType = hipMemoryTypeArray;
          cpdesc.srcArray = cudaArr;
          cpdesc.dstXInBytes = cpdesc.dstY = cpdesc.dstZ = cpdesc.dstLOD = 0;
          cpdesc.dstMemoryType = hipMemoryTypeHost;
          cpdesc.dstHost = (void *)pCudaCopyMem;
          cpdesc.WidthInBytes = copyWidthInBytes;  // data->width * 4;
          cpdesc.Height = copyHeight;              // data->height;
          cpdesc.Depth = 1;

          cuStatus = hipDrvMemcpy3D(&cpdesc);
          if (cuStatus != hipSuccess) {
            printf(
                "Cuda consumer: cuMemCpy3D failed,  copyWidthInBytes=%d, "
                "copyHight=%d\n",
                copyWidthInBytes, copyHeight);
          }
          cuStatus = hipCtxSynchronize();
          if (cuStatus != hipSuccess) {
            printf("hipCtxSynchronize failed after memcpy \n");
          }
        }
        if (cuStatus == hipSuccess) {
          if (fwrite(pCudaCopyMem, bufferSize, 1, file_p) != 1) {
            printf("Cuda consumer: output file write failed\n");
            cuStatus = hipErrorUnknown;
            goto done;
          }
        }
      }
      pInFile1 = fopen(data->fileName1, "rb");
      if (!pInFile1) {
        printf("Failed to open file :%s\n", data->fileName1);
        goto done;
      }
      pInFile2 = fopen(data->fileName2, "rb");
      if (!pInFile2) {
        printf("Failed to open file :%s\n", data->fileName2);
        goto done;
      }
      rewind(file_p);
      check_result = checkbuf(file_p, pInFile1);
      if (check_result == -1) {
        rewind(file_p);
        check_result = checkbuf(file_p, pInFile2);
        if (check_result == -1) {
          printf("Frame received does not match any valid image: FAILED\n");
        } else {
          printf("Frame check Passed\n");
        }
      } else {
        printf("Frame check Passed\n");
      }
      if (pCudaCopyMem) {
        free(pCudaCopyMem);
        pCudaCopyMem = NULL;
      }
      cuStatus =
          cuEGLStreamConsumerReleaseFrame(&data->cudaConn, cudaResource, NULL);
      if (cuStatus != hipSuccess) {
        printf("cuEGLStreamConsumerReleaseFrame failed with cuStatus = %d\n",
               cuStatus);
        goto done;
      }
    } else {
      printf("cuda AcquireFrame FAILED with  cuStatus=%d\n", cuStatus);
      goto done;
    }
  }

done:
  if (file_p) {
    fclose(file_p);
    file_p = NULL;
  }
  if (pInFile1) {
    fclose(pInFile1);
    pInFile1 = NULL;
  }
  if (pInFile1) {
    fclose(pInFile2);
    pInFile2 = NULL;
  }
  return cuStatus;
}

int checkbuf(FILE *fp1, FILE *fp2) {
  int match = 0;
  int ch1, ch2;
  if (fp1 == NULL) {
    printf("Invalid file pointer for first file\n");
    return -1;
  } else if (fp2 == NULL) {
    printf("Invalid file pointer for second file\n");
    return -1;
  } else {
    ch1 = getc(fp1);
    ch2 = getc(fp2);
    while ((ch1 != EOF) && (ch2 != EOF) && (ch1 == ch2)) {
      ch1 = getc(fp1);
      ch2 = getc(fp2);
    }
    if (ch1 == ch2) {
      match = 1;
    } else if (ch1 != ch2) {
      match = -1;
    }
  }
  return match;
}

hipError_t cudaDeviceCreateConsumer(test_cuda_consumer_s *cudaConsumer,
                                  hipDevice_t device) {
  hipError_t status = hipSuccess;
  if (hipSuccess != (status = hipInit(0))) {
    printf("Failed to initialize CUDA\n");
    return status;
  }

  int major = 0, minor = 0;
  char deviceName[256];
  checkCudaErrors(hipDeviceGetAttribute(
      &major, hipDeviceAttributeComputeCapabilityMajor, device));
  checkCudaErrors(hipDeviceGetAttribute(
      &minor, hipDeviceAttributeComputeCapabilityMinor, device));
  checkCudaErrors(hipDeviceGetName(deviceName, 256, device));
  printf(
      "CUDA Consumer on GPU Device %d: \"%s\" with compute capability "
      "%d.%d\n\n",
      device, deviceName, major, minor);

  if (hipSuccess !=
      (status = hipCtxCreate(&cudaConsumer->context, 0, device))) {
    printf("failed to create CUDA context\n");
    return status;
  }
  checkCudaErrors(hipCtxPopCurrent(&cudaConsumer->context));
  return status;
}

void cuda_consumer_init(test_cuda_consumer_s *cudaConsumer, TestArgs *args) {
  cudaConsumer->pitchLinearOutput = args->pitchLinearOutput;
  cudaConsumer->width = args->inputWidth;
  cudaConsumer->height = args->inputHeight;
  cudaConsumer->fileName1 = args->infile1;
  cudaConsumer->fileName2 = args->infile2;

  cudaConsumer->outFile1 = "cuda_out1.yuv";
  cudaConsumer->outFile2 = "cuda_out2.yuv";
}

hipError_t cuda_consumer_deinit(test_cuda_consumer_s *cudaConsumer) {
  return cuEGLStreamConsumerDisconnect(&cudaConsumer->cudaConn);
}
