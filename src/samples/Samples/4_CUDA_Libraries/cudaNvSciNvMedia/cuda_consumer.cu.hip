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

#include <iostream>

#include <hip/hip_runtime.h>
#include "cuda_consumer.h"
#include <helper_image.h>
#include "nvmedia_image_nvscibuf.h"
#include "nvmedia_utils/cmdline.h"

// Enable this to 1 if require cuda processed output to ppm file.
#define WRITE_OUTPUT_IMAGE 0

#define checkNvSciErrors(call)                              \
  do {                                                      \
    NvSciError _status = call;                              \
    if (NvSciError_Success != _status) {                    \
      printf(                                               \
          "NVSCI call in file '%s' in line %i returned"     \
          " %d, expected %d\n",                             \
          __FILE__, __LINE__, _status, NvSciError_Success); \
      fflush(stdout);                                       \
      exit(EXIT_FAILURE);                                   \
    }                                                       \
  } while (0)

__global__ static void yuvToGrayscale(hipSurfaceObject_t surfaceObject,
                                      unsigned int *dstImage,
                                      int32_t imageWidth, int32_t imageHeight) {
  size_t x = blockIdx.x * blockDim.x + threadIdx.x;
  size_t y = blockIdx.y * blockDim.y + threadIdx.y;

  uchar4 *dstImageUchar4 = (uchar4 *)dstImage;
  for (; x < imageWidth && y < imageHeight;
       x += gridDim.x * blockDim.x, y += gridDim.y * blockDim.y) {
    int colInBytes = x * sizeof(unsigned char);
    unsigned char luma =
        surf2Dread<unsigned char>(surfaceObject, colInBytes, y);
    uchar4 grayscalePix = make_uchar4(luma, luma, luma, 0);

    dstImageUchar4[y * imageWidth + x] = grayscalePix;
  }
}

static void cudaImportNvSciSync(hipExternalSemaphore_t &extSem,
                                NvSciSyncObj &syncObj) {
  hipExternalSemaphoreHandleDesc extSemDesc;
  memset(&extSemDesc, 0, sizeof(extSemDesc));
  extSemDesc.type = cudaExternalSemaphoreHandleTypeNvSciSync;
  extSemDesc.handle.nvSciSyncObj = (void *)syncObj;

  HIPCHECK(hipImportExternalSemaphore(&extSem, &extSemDesc));
}

static void waitExternalSemaphore(hipExternalSemaphore_t &waitSem,
                                  NvSciSyncFence *fence, hipStream_t stream) {
  hipExternalSemaphoreWaitParams waitParams;
  memset(&waitParams, 0, sizeof(waitParams));
  // For cross-process signaler-waiter applications need to use NvSciIpc
  // and NvSciSync[Export|Import] utilities to share the NvSciSyncFence
  // across process. This step is optional in single-process.
  waitParams.params.nvSciSync.fence = (void *)fence;
  waitParams.flags = 0;

  HIPCHECK(
      hipWaitExternalSemaphoresAsync(&waitSem, &waitParams, 1, stream));
}

static void signalExternalSemaphore(hipExternalSemaphore_t &signalSem,
                                    NvSciSyncFence *fence,
                                    hipStream_t stream) {
  hipExternalSemaphoreSignalParams signalParams;
  memset(&signalParams, 0, sizeof(signalParams));
  // For cross-process signaler-waiter applications need to use NvSciIpc
  // and NvSciSync[Export|Import] utilities to share the NvSciSyncFence
  // across process. This step is optional in single-process.
  signalParams.params.nvSciSync.fence = (void *)fence;
  signalParams.flags = 0;

  HIPCHECK(
      hipSignalExternalSemaphoresAsync(&signalSem, &signalParams, 1, stream));
}

static void yuvToGrayscaleCudaKernel(cudaExternalResInterop &cudaExtResObj,
                                     int32_t imageWidth, int32_t imageHeight) {
#if WRITE_OUTPUT_IMAGE
  unsigned int *h_dstImage;
  HIPCHECK(hipHostMalloc(
      &h_dstImage, sizeof(unsigned int) * imageHeight * imageWidth));
#endif
  dim3 block(16, 16, 1);
  dim3 grid((imageWidth / block.x) + 1, (imageHeight / block.y) + 1, 1);

  yuvToGrayscale<<<grid, block, 0, cudaExtResObj.stream>>>(
      cudaExtResObj.cudaSurfaceNvmediaBuf[0], cudaExtResObj.d_outputImage,
      imageWidth, imageHeight);

#if WRITE_OUTPUT_IMAGE
  HIPCHECK(
      hipMemcpyAsync(h_dstImage, cudaExtResObj.d_outputImage,
                      sizeof(unsigned int) * imageHeight * imageWidth,
                      hipMemcpyDeviceToHost, cudaExtResObj.stream));
  HIPCHECK(hipStreamSynchronize(cudaExtResObj.stream));
  char outputFilename[1024];
  std::string image_filename = "Grayscale";
  strcpy(outputFilename, image_filename.c_str());
  strcpy(outputFilename + image_filename.length(), "_nvsci_out.ppm");
  sdkSavePPM4ub(outputFilename, (unsigned char *)h_dstImage, imageWidth,
                imageHeight);
  printf("Wrote '%s'\n", outputFilename);
  HIPCHECK(hipHostFree(h_dstImage));
#endif
}

static void cudaImportNvSciImage(cudaExternalResInterop &cudaExtResObj,
                                 NvSciBufObj &inputBufObj) {
  NvSciBufModule module = NULL;
  NvSciBufAttrList attrlist = NULL;
  NvSciBufAttrKeyValuePair pairArrayOut[10];

  checkNvSciErrors(NvSciBufModuleOpen(&module));
  checkNvSciErrors(NvSciBufAttrListCreate(module, &attrlist));
  checkNvSciErrors(NvSciBufObjGetAttrList(inputBufObj, &attrlist));

  memset(pairArrayOut, 0, sizeof(NvSciBufAttrKeyValuePair) * 10);

  int numAttrs = 0;
  pairArrayOut[numAttrs++].key = NvSciBufImageAttrKey_Size;
  pairArrayOut[numAttrs++].key = NvSciBufImageAttrKey_PlaneChannelCount;
  pairArrayOut[numAttrs++].key = NvSciBufImageAttrKey_PlaneCount;
  pairArrayOut[numAttrs++].key = NvSciBufImageAttrKey_PlaneWidth;
  pairArrayOut[numAttrs++].key = NvSciBufImageAttrKey_PlaneHeight;
  pairArrayOut[numAttrs++].key = NvSciBufImageAttrKey_Layout;
  pairArrayOut[numAttrs++].key = NvSciBufImageAttrKey_PlaneBitsPerPixel;
  pairArrayOut[numAttrs++].key = NvSciBufImageAttrKey_PlaneOffset;

  checkNvSciErrors(NvSciBufAttrListGetAttrs(attrlist, pairArrayOut, numAttrs));

  uint64_t size = *(uint64_t *)pairArrayOut[0].value;
  uint8_t channelCount = *(uint8_t *)pairArrayOut[1].value;
  cudaExtResObj.planeCount = *(int32_t *)pairArrayOut[2].value;
  cudaExtResObj.imageWidth =
      (int32_t *)malloc(sizeof(int32_t) * cudaExtResObj.planeCount);
  cudaExtResObj.imageHeight =
      (int32_t *)malloc(sizeof(int32_t) * cudaExtResObj.planeCount);
  cudaExtResObj.planeOffset =
      (uint64_t *)malloc(sizeof(uint64_t) * cudaExtResObj.planeCount);

  memcpy(cudaExtResObj.imageWidth, (int32_t *)pairArrayOut[3].value,
         cudaExtResObj.planeCount * sizeof(int32_t));
  memcpy(cudaExtResObj.imageHeight, (int32_t *)pairArrayOut[4].value,
         cudaExtResObj.planeCount * sizeof(int32_t));
  memcpy(cudaExtResObj.planeOffset, (uint64_t *)pairArrayOut[7].value,
         cudaExtResObj.planeCount * sizeof(uint64_t));

  NvSciBufAttrValImageLayoutType layout =
      *(NvSciBufAttrValImageLayoutType *)pairArrayOut[5].value;
  uint32_t bitsPerPixel = *(uint32_t *)pairArrayOut[6].value;

  if (layout != NvSciBufImage_BlockLinearType) {
    printf("Image layout is not block linear.. waiving execution\n");
    exit(EXIT_WAIVED);
  }

  hipExternalMemoryHandleDesc memHandleDesc;
  memset(&memHandleDesc, 0, sizeof(memHandleDesc));
  memHandleDesc.type = cudaExternalMemoryHandleTypeNvSciBuf;
  memHandleDesc.handle.nvSciBufObject = inputBufObj;
  memHandleDesc.size = size;
  HIPCHECK(
      hipImportExternalMemory(&cudaExtResObj.extMemImageBuf, &memHandleDesc));

  cudaExtResObj.d_mipmapArray = (hipMipmappedArray_t *)malloc(
      sizeof(hipMipmappedArray_t) * cudaExtResObj.planeCount);

  for (int i = 0; i < cudaExtResObj.planeCount; i++) {
    hipExtent extent = {};
    memset(&extent, 0, sizeof(extent));
    extent.width = cudaExtResObj.imageWidth[i];
    extent.height = cudaExtResObj.imageHeight[i];
    extent.depth = 0;
    hipChannelFormatDesc desc;
    switch (channelCount) {
      case 1:
      default:
        desc = hipCreateChannelDesc(bitsPerPixel, 0, 0, 0,
                                     hipChannelFormatKindUnsigned);
        break;
      case 2:
        desc = hipCreateChannelDesc(bitsPerPixel, bitsPerPixel, 0, 0,
                                     hipChannelFormatKindUnsigned);
        break;
      case 3:
        desc = hipCreateChannelDesc(bitsPerPixel, bitsPerPixel, bitsPerPixel,
                                     0, hipChannelFormatKindUnsigned);
        break;
      case 4:
        desc =
            hipCreateChannelDesc(bitsPerPixel, bitsPerPixel, bitsPerPixel,
                                  bitsPerPixel, hipChannelFormatKindUnsigned);
        break;
    }

    cudaExternalMemoryMipmappedArrayDesc mipmapDesc = {0};
    mipmapDesc.offset = cudaExtResObj.planeOffset[i];
    mipmapDesc.formatDesc = desc;
    mipmapDesc.extent = extent;
    mipmapDesc.flags = 0;
    mipmapDesc.numLevels = 1;
    HIPCHECK(cudaExternalMemoryGetMappedMipmappedArray(
        &cudaExtResObj.d_mipmapArray[i], cudaExtResObj.extMemImageBuf,
        &mipmapDesc));
  }
}

static hipSurfaceObject_t createCudaSurface(hipArray_t &d_mipLevelArray) {
  hipResourceDesc resourceDesc;
  memset(&resourceDesc, 0, sizeof(resourceDesc));
  resourceDesc.resType = hipResourceTypeArray;
  resourceDesc.res.array.array = d_mipLevelArray;

  hipSurfaceObject_t surfaceObject;
  HIPCHECK(hipCreateSurfaceObject(&surfaceObject, &resourceDesc));
  return surfaceObject;
}

static hipStream_t createCudaStream(int deviceId) {
  HIPCHECK(hipSetDevice(deviceId));
  hipStream_t stream;
  HIPCHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
  return stream;
}

// CUDA setup buffers/synchronization objects for interop via NvSci API.
void setupCuda(cudaExternalResInterop &cudaExtResObj, NvSciBufObj &inputBufObj,
               NvSciSyncObj &syncObj, NvSciSyncObj &cudaSignalerSyncObj,
               int deviceId) {
  HIPCHECK(hipSetDevice(deviceId));
  cudaImportNvSciSync(cudaExtResObj.waitSem, syncObj);
  cudaImportNvSciSync(cudaExtResObj.signalSem, cudaSignalerSyncObj);

  cudaImportNvSciImage(cudaExtResObj, inputBufObj);
  cudaExtResObj.d_mipLevelArray =
      (hipArray_t *)malloc(sizeof(hipArray_t) * cudaExtResObj.planeCount);
  cudaExtResObj.cudaSurfaceNvmediaBuf = (hipSurfaceObject_t *)malloc(
      sizeof(hipSurfaceObject_t) * cudaExtResObj.planeCount);

  for (int i = 0; i < cudaExtResObj.planeCount; ++i) {
    uint32_t mipLevelId = 0;
    HIPCHECK(
        hipGetMipmappedArrayLevel(&cudaExtResObj.d_mipLevelArray[i],
                                   cudaExtResObj.d_mipmapArray[i], mipLevelId));
    cudaExtResObj.cudaSurfaceNvmediaBuf[i] =
        createCudaSurface(cudaExtResObj.d_mipLevelArray[i]);
  }

  cudaExtResObj.stream = createCudaStream(deviceId);
  HIPCHECK(hipMalloc(&cudaExtResObj.d_outputImage,
                             sizeof(unsigned int) *
                                 cudaExtResObj.imageWidth[0] *
                                 cudaExtResObj.imageHeight[0]));
}

// CUDA clean up buffers used **with** NvSci API.
void cleanupCuda(cudaExternalResInterop &cudaExtResObj) {
  for (int i = 0; i < cudaExtResObj.planeCount; i++) {
    HIPCHECK(
        hipDestroySurfaceObject(cudaExtResObj.cudaSurfaceNvmediaBuf[i]));
    HIPCHECK(hipFreeMipmappedArray(cudaExtResObj.d_mipmapArray[i]));
  }
  free(cudaExtResObj.d_mipmapArray);
  free(cudaExtResObj.d_mipLevelArray);
  free(cudaExtResObj.cudaSurfaceNvmediaBuf);
  free(cudaExtResObj.imageWidth);
  free(cudaExtResObj.imageHeight);
  HIPCHECK(hipDestroyExternalSemaphore(cudaExtResObj.waitSem));
  HIPCHECK(hipDestroyExternalSemaphore(cudaExtResObj.signalSem));
  HIPCHECK(hipDestroyExternalMemory(cudaExtResObj.extMemImageBuf));
  HIPCHECK(hipStreamDestroy(cudaExtResObj.stream));
  HIPCHECK(hipFree(cudaExtResObj.d_outputImage));
}

void runCudaOperation(cudaExternalResInterop &cudaExtResObj,
                      NvSciSyncFence *cudaWaitFence,
                      NvSciSyncFence *cudaSignalFence, int deviceId,
                      int iterations) {
  HIPCHECK(hipSetDevice(deviceId));
  static int64_t launch = 0;

  waitExternalSemaphore(cudaExtResObj.waitSem, cudaWaitFence,
                        cudaExtResObj.stream);

  // run cuda kernel over surface object of the LUMA surface part to extract
  // grayscale.
  yuvToGrayscaleCudaKernel(cudaExtResObj, cudaExtResObj.imageWidth[0],
                           cudaExtResObj.imageHeight[0]);

  // signal fence till the second last iterations for NvMedia2DBlit to wait for
  // cuda signal and for final iteration as there is no corresponding NvMedia
  // operation pending therefore we end with hipStreamSynchronize()
  if (launch < iterations - 1) {
    signalExternalSemaphore(cudaExtResObj.signalSem, cudaSignalFence,
                            cudaExtResObj.stream);
  } else {
    HIPCHECK(hipStreamSynchronize(cudaExtResObj.stream));
  }
  launch++;
}

// CUDA imports and operates on NvSci buffer/synchronization objects
void setupCuda(Blit2DTest *ctx, cudaResources &cudaResObj, int deviceId) {
  HIPCHECK(hipSetDevice(deviceId));
  cudaResObj.d_yuvArray =
      (hipArray_t *)malloc(sizeof(hipArray_t) * ctx->numSurfaces);
  cudaResObj.cudaSurfaceNvmediaBuf = (hipSurfaceObject_t *)malloc(
      sizeof(hipSurfaceObject_t) * ctx->numSurfaces);
  hipChannelFormatDesc channelDesc;
  switch (ctx->bytesPerPixel) {
    case 1:
    default:
      channelDesc =
          hipCreateChannelDesc(8, 0, 0, 0, hipChannelFormatKindUnsigned);
      break;
  }

  for (int k = 0; k < ctx->numSurfaces; k++) {
    HIPCHECK(hipMallocArray(
        &cudaResObj.d_yuvArray[k], &channelDesc,
        ctx->widthSurface * ctx->xScalePtr[k] * ctx->bytesPerPixel,
        ctx->heightSurface * ctx->yScalePtr[k]));
    cudaResObj.cudaSurfaceNvmediaBuf[k] =
        createCudaSurface(cudaResObj.d_yuvArray[k]);
  }
  HIPCHECK(hipMalloc(
      &cudaResObj.d_outputImage,
      sizeof(unsigned int) * ctx->widthSurface * ctx->heightSurface));

  cudaResObj.stream = createCudaStream(deviceId);
}

// CUDA clean up buffers used **without** NvSci API.
void cleanupCuda(Blit2DTest *ctx, cudaResources &cudaResObj) {
  for (int k = 0; k < ctx->numSurfaces; k++) {
    HIPCHECK(
        hipDestroySurfaceObject(cudaResObj.cudaSurfaceNvmediaBuf[k]));
    HIPCHECK(hipFreeArray(cudaResObj.d_yuvArray[k]));
  }

  free(cudaResObj.cudaSurfaceNvmediaBuf);

  HIPCHECK(hipStreamDestroy(cudaResObj.stream));
  HIPCHECK(hipFree(cudaResObj.d_outputImage));
}

static void yuvToGrayscaleCudaKernelNonNvSci(cudaResources &cudaResObj,
                                             int deviceId, int32_t imageWidth,
                                             int32_t imageHeight) {
#if WRITE_OUTPUT_IMAGE
  unsigned int *h_dstImage;
  HIPCHECK(hipHostMalloc(
      &h_dstImage, sizeof(unsigned int) * imageHeight * imageWidth));
#endif
  dim3 block(16, 16, 1);
  dim3 grid((imageWidth / block.x) + 1, (imageHeight / block.y) + 1, 1);

  yuvToGrayscale<<<grid, block, 0, cudaResObj.stream>>>(
      cudaResObj.cudaSurfaceNvmediaBuf[0], cudaResObj.d_outputImage, imageWidth,
      imageHeight);

#if WRITE_OUTPUT_IMAGE
  HIPCHECK(
      hipMemcpyAsync(h_dstImage, cudaResObj.d_outputImage,
                      sizeof(unsigned int) * imageHeight * imageWidth,
                      hipMemcpyDeviceToHost, cudaResObj.stream));
  HIPCHECK(hipStreamSynchronize(cudaResObj.stream));
  char outputFilename[1024];
  std::string image_filename = "Grayscale";
  strcpy(outputFilename, image_filename.c_str());
  strcpy(outputFilename + image_filename.length(), "_non-nvsci_out.ppm");
  sdkSavePPM4ub(outputFilename, (unsigned char *)h_dstImage, imageWidth,
                imageHeight);
  printf("Wrote '%s'\n", outputFilename);
  HIPCHECK(hipHostFree(h_dstImage));
#else
  HIPCHECK(hipStreamSynchronize(cudaResObj.stream));
#endif
}

// CUDA operates **without** NvSci APIs buffer/synchronization objects.
void runCudaOperation(Blit2DTest *ctx, cudaResources &cudaResObj,
                      int deviceId) {
  for (int k = 0; k < ctx->numSurfaces; k++) {
    HIPCHECK(hipMemcpy2DToArray(
        cudaResObj.d_yuvArray[k], 0, 0, ctx->dstBuff[k],
        ctx->widthSurface * ctx->xScalePtr[k] * ctx->bytesPerPixel,
        ctx->widthSurface * ctx->xScalePtr[k] * ctx->bytesPerPixel,
        ctx->heightSurface * ctx->yScalePtr[k], hipMemcpyHostToDevice));
  }
  // run cuda kernel over surface object of the LUMA surface part to extract
  // grayscale.
  yuvToGrayscaleCudaKernelNonNvSci(cudaResObj, deviceId, ctx->widthSurface,
                                   ctx->heightSurface);
}
emcpyHostToDevice));
  }
  // run cuda kernel over surface object of the LUMA surface part to extract
  // grayscale.
  yuvToGrayscaleCudaKernelNonNvSci(cudaResObj, deviceId, ctx->widthSurface,
                                   ctx->heightSurface);
}
