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

#include "cudaNvSci.h"
#include <hip/hip_runtime.h>
#include <condition_variable>
#include <iostream>
#include <thread>

std::mutex m_mutex;
std::condition_variable m_condVar;
bool workSubmitted = false;

class cudaNvSciSignal {
 private:
  NvSciSyncModule m_syncModule;
  NvSciBufModule m_bufModule;

  NvSciSyncAttrList m_syncAttrList;
  NvSciSyncFence *m_fence;

  NvSciBufAttrList m_rawBufAttrList;
  NvSciBufAttrList m_imageBufAttrList;
  NvSciBufAttrList m_buffAttrListOut[2];
  NvSciBufAttrKeyValuePair pairArrayOut[10];

  hipExternalMemory_t extMemRawBuf, extMemImageBuf;
  hipMipmappedArray_t d_mipmapArray;
  hipArray_t d_mipLevelArray;
  hipTextureObject_t texObject;
  hipExternalSemaphore_t signalSem;

  hipStream_t streamToRun;
  int m_cudaDeviceId;
  hipUUID m_devUUID;
  uint64_t m_imageWidth;
  uint64_t m_imageHeight;
  void *d_outputBuf;
  size_t m_bufSize;

 public:
  cudaNvSciSignal(NvSciBufModule bufModule, NvSciSyncModule syncModule,
                  int cudaDeviceId, int bufSize, uint64_t imageWidth,
                  uint64_t imageHeight, NvSciSyncFence *fence)
      : m_syncModule(syncModule),
        m_bufModule(bufModule),
        m_cudaDeviceId(cudaDeviceId),
        d_outputBuf(NULL),
        m_bufSize(bufSize),
        m_imageWidth(imageWidth),
        m_imageHeight(imageHeight),
        m_fence(fence) {
    initCuda();

    checkNvSciErrors(NvSciSyncAttrListCreate(m_syncModule, &m_syncAttrList));
    checkNvSciErrors(NvSciBufAttrListCreate(m_bufModule, &m_rawBufAttrList));
    checkNvSciErrors(NvSciBufAttrListCreate(m_bufModule, &m_imageBufAttrList));

    setRawBufAttrList(m_bufSize);
    setImageBufAttrList(m_imageWidth, m_imageHeight);

    checkCudaErrors(cudaDeviceGetNvSciSyncAttributes(
        m_syncAttrList, m_cudaDeviceId, cudaNvSciSyncAttrSignal));
  }

  ~cudaNvSciSignal() {
    checkCudaErrors(hipSetDevice(m_cudaDeviceId));
    checkCudaErrors(hipFreeMipmappedArray(d_mipmapArray));
    checkCudaErrors(hipFree(d_outputBuf));
    checkCudaErrors(hipDestroyExternalSemaphore(signalSem));
    checkCudaErrors(hipDestroyExternalMemory(extMemRawBuf));
    checkCudaErrors(hipDestroyExternalMemory(extMemImageBuf));
    checkCudaErrors(hipDestroyTextureObject(texObject));
    checkCudaErrors(hipStreamDestroy(streamToRun));
  }

  void initCuda() {
    checkCudaErrors(hipSetDevice(m_cudaDeviceId));
    checkCudaErrors(
        hipStreamCreateWithFlags(&streamToRun, hipStreamNonBlocking));

    int major = 0, minor = 0;
    checkCudaErrors(hipDeviceGetAttribute(
        &major, hipDeviceAttributeComputeCapabilityMajor, m_cudaDeviceId));
    checkCudaErrors(hipDeviceGetAttribute(
        &minor, hipDeviceAttributeComputeCapabilityMinor, m_cudaDeviceId));
    printf(
        "[cudaNvSciSignal] GPU Device %d: \"%s\" with compute capability "
        "%d.%d\n\n",
        m_cudaDeviceId, _ConvertSMVer2ArchName(major, minor), major, minor);

#ifdef hipDeviceGetUuid
    hipError_t res = hipDeviceGetUuid(&m_devUUID, m_cudaDeviceId);
#else
    hipError_t res = hipDeviceGetUuid(&m_devUUID, m_cudaDeviceId);
#endif

    if (res != hipSuccess) {
      fprintf(stderr, "Driver API error = %04d \n", res);
      exit(EXIT_FAILURE);
    }
  }

  void setRawBufAttrList(uint64_t size) {
    NvSciBufType bufType = NvSciBufType_RawBuffer;
    bool cpuAccess = false;
    NvSciBufAttrValAccessPerm perm = NvSciBufAccessPerm_ReadWrite;
    NvSciBufAttrKeyValuePair rawBufAttrs[] = {
        {NvSciBufRawBufferAttrKey_Size, &size, sizeof(size)},
        {NvSciBufGeneralAttrKey_Types, &bufType, sizeof(bufType)},
        {NvSciBufGeneralAttrKey_NeedCpuAccess, &cpuAccess, sizeof(cpuAccess)},
        {NvSciBufGeneralAttrKey_RequiredPerm, &perm, sizeof(perm)},
        {NvSciBufGeneralAttrKey_GpuId, &m_devUUID, sizeof(m_devUUID)},
    };

    checkNvSciErrors(NvSciBufAttrListSetAttrs(
        m_rawBufAttrList, rawBufAttrs,
        sizeof(rawBufAttrs) / sizeof(NvSciBufAttrKeyValuePair)));
  }

  void setImageBufAttrList(uint32_t width, uint32_t height) {
    NvSciBufType bufType = NvSciBufType_Image;
    NvSciBufAttrValImageLayoutType layout = NvSciBufImage_BlockLinearType;
    NvSciBufAttrValAccessPerm perm = NvSciBufAccessPerm_ReadWrite;

    uint32_t planeCount = 1;
    uint32_t planeWidths[] = {width};
    uint32_t planeHeights[] = {height};
    uint64_t lrpad = 0, tbpad = 100;

    bool cpuAccessFlag = false;

    NvSciBufAttrValColorFmt planecolorfmts[] = {NvSciColor_B8G8R8A8};
    NvSciBufAttrValColorStd planecolorstds[] = {NvSciColorStd_SRGB};
    NvSciBufAttrValImageScanType planescantype[] = {NvSciBufScan_InterlaceType};

    NvSciBufAttrKeyValuePair imgBufAttrs[] = {
        {NvSciBufGeneralAttrKey_Types, &bufType, sizeof(bufType)},
        {NvSciBufImageAttrKey_PlaneCount, &planeCount, sizeof(planeCount)},
        {NvSciBufImageAttrKey_Layout, &layout, sizeof(layout)},
        {NvSciBufImageAttrKey_TopPadding, &tbpad, sizeof(tbpad)},
        {NvSciBufImageAttrKey_BottomPadding, &tbpad, sizeof(tbpad)},
        {NvSciBufImageAttrKey_LeftPadding, &lrpad, sizeof(lrpad)},
        {NvSciBufImageAttrKey_RightPadding, &lrpad, sizeof(lrpad)},
        {NvSciBufImageAttrKey_PlaneColorFormat, planecolorfmts,
         sizeof(planecolorfmts)},
        {NvSciBufImageAttrKey_PlaneColorStd, planecolorstds,
         sizeof(planecolorstds)},
        {NvSciBufImageAttrKey_PlaneWidth, planeWidths, sizeof(planeWidths)},
        {NvSciBufImageAttrKey_PlaneHeight, planeHeights, sizeof(planeHeights)},
        {NvSciBufGeneralAttrKey_NeedCpuAccess, &cpuAccessFlag,
         sizeof(cpuAccessFlag)},
        {NvSciBufGeneralAttrKey_RequiredPerm, &perm, sizeof(perm)},
        {NvSciBufImageAttrKey_PlaneScanType, planescantype,
         sizeof(planescantype)},
        {NvSciBufGeneralAttrKey_GpuId, &m_devUUID, sizeof(m_devUUID)},
    };

    checkNvSciErrors(NvSciBufAttrListSetAttrs(
        m_imageBufAttrList, imgBufAttrs,
        sizeof(imgBufAttrs) / sizeof(NvSciBufAttrKeyValuePair)));
  }

  NvSciSyncAttrList getNvSciSyncAttrList() { return m_syncAttrList; }

  NvSciBufAttrList getNvSciRawBufAttrList() { return m_rawBufAttrList; }

  NvSciBufAttrList getNvSciImageBufAttrList() { return m_imageBufAttrList; }

  void runRotateImageAndSignal(unsigned char *imageData) {
    int numOfGPUs = 0;
    checkCudaErrors(hipGetDeviceCount(&numOfGPUs));  // For cuda init purpose
    checkCudaErrors(hipSetDevice(m_cudaDeviceId));

    copyDataToImageArray(imageData);
    createTexture();

    float angle = 0.5f;  // angle to rotate image by (in radians)
    rotateKernel(texObject, angle, (unsigned int *)d_outputBuf, m_imageWidth,
                 m_imageHeight, streamToRun);

    signalExternalSemaphore();
  }

  void cudaImportNvSciSemaphore(NvSciSyncObj syncObj) {
    checkCudaErrors(hipSetDevice(m_cudaDeviceId));

    hipExternalSemaphoreHandleDesc extSemDesc;
    memset(&extSemDesc, 0, sizeof(extSemDesc));
    extSemDesc.type = cudaExternalSemaphoreHandleTypeNvSciSync;
    extSemDesc.handle.nvSciSyncObj = (void *)syncObj;

    checkCudaErrors(hipImportExternalSemaphore(&signalSem, &extSemDesc));
  }

  void signalExternalSemaphore() {
    hipExternalSemaphoreSignalParams signalParams;
    memset(&signalParams, 0, sizeof(signalParams));
    // For cross-process signaler-waiter applications need to use NvSciIpc
    // and NvSciSync[Export|Import] utilities to share the NvSciSyncFence
    // across process. This step is optional in single-process.
    signalParams.params.nvSciSync.fence = (void *)m_fence;
    signalParams.flags = 0;

    checkCudaErrors(hipSignalExternalSemaphoresAsync(&signalSem, &signalParams,
                                                      1, streamToRun));
  }

  void cudaImportNvSciRawBuf(NvSciBufObj inputBufObj) {
    checkCudaErrors(hipSetDevice(m_cudaDeviceId));
    checkNvSciErrors(
        NvSciBufObjGetAttrList(inputBufObj, &m_buffAttrListOut[0]));

    memset(pairArrayOut, 0, sizeof(NvSciBufAttrKeyValuePair) * 10);
    pairArrayOut[0].key = NvSciBufRawBufferAttrKey_Size;

    checkNvSciErrors(
        NvSciBufAttrListGetAttrs(m_buffAttrListOut[0], pairArrayOut, 1));

    uint64_t size = *(uint64_t *)pairArrayOut[0].value;

    hipExternalMemoryHandleDesc memHandleDesc;
    memset(&memHandleDesc, 0, sizeof(memHandleDesc));
    memHandleDesc.type = cudaExternalMemoryHandleTypeNvSciBuf;
    memHandleDesc.handle.nvSciBufObject = inputBufObj;
    memHandleDesc.size = size;
    checkCudaErrors(hipImportExternalMemory(&extMemRawBuf, &memHandleDesc));

    hipExternalMemoryBufferDesc bufferDesc;
    memset(&bufferDesc, 0, sizeof(bufferDesc));
    bufferDesc.offset = 0;
    bufferDesc.size = size;
    m_bufSize = size;
    checkCudaErrors(hipExternalMemoryGetMappedBuffer(
        &d_outputBuf, extMemRawBuf, &bufferDesc));
  }

  void cudaImportNvSciImage(NvSciBufObj inputBufObj) {
    checkCudaErrors(hipSetDevice(m_cudaDeviceId));
    checkNvSciErrors(
        NvSciBufObjGetAttrList(inputBufObj, &m_buffAttrListOut[1]));

    memset(pairArrayOut, 0, sizeof(NvSciBufAttrKeyValuePair) * 10);
    pairArrayOut[0].key = NvSciBufImageAttrKey_Size;
    pairArrayOut[1].key = NvSciBufImageAttrKey_Alignment;
    pairArrayOut[2].key = NvSciBufImageAttrKey_PlaneCount;
    pairArrayOut[3].key = NvSciBufImageAttrKey_PlaneWidth;
    pairArrayOut[4].key = NvSciBufImageAttrKey_PlaneHeight;

    checkNvSciErrors(
        NvSciBufAttrListGetAttrs(m_buffAttrListOut[1], pairArrayOut, 5));

    uint64_t size = *(uint64_t *)pairArrayOut[0].value;
    uint64_t alignment = *(uint64_t *)pairArrayOut[1].value;
    uint64_t planeCount = *(uint64_t *)pairArrayOut[2].value;
    uint64_t imageWidth = *(uint64_t *)pairArrayOut[3].value;
    uint64_t imageHeight = *(uint64_t *)pairArrayOut[4].value;

    hipExternalMemoryHandleDesc memHandleDesc;
    memset(&memHandleDesc, 0, sizeof(memHandleDesc));
    memHandleDesc.type = cudaExternalMemoryHandleTypeNvSciBuf;
    memHandleDesc.handle.nvSciBufObject = inputBufObj;
    memHandleDesc.size = size;
    checkCudaErrors(hipImportExternalMemory(&extMemImageBuf, &memHandleDesc));

    hipExtent extent = {};
    memset(&extent, 0, sizeof(extent));
    extent.width = imageWidth;
    extent.height = imageHeight;
    extent.depth = 0;

    hipChannelFormatDesc desc;
    desc.x = 8;
    desc.y = 8;
    desc.z = 8;
    desc.w = 8;
    desc.f = hipChannelFormatKindUnsigned;

    cudaExternalMemoryMipmappedArrayDesc mipmapDesc = {0};
    mipmapDesc.offset = 0;
    mipmapDesc.formatDesc = desc;
    mipmapDesc.extent = extent;
    mipmapDesc.flags = 0;

    mipmapDesc.numLevels = 1;
    checkCudaErrors(cudaExternalMemoryGetMappedMipmappedArray(
        &d_mipmapArray, extMemImageBuf, &mipmapDesc));
  }

  void copyDataToImageArray(unsigned char *imageData) {
    uint32_t mipLevelId = 0;
    checkCudaErrors(hipGetMipmappedArrayLevel(&d_mipLevelArray, d_mipmapArray,
                                               mipLevelId));

    checkCudaErrors(hipMemcpy2DToArrayAsync(
        d_mipLevelArray, 0, 0, imageData, m_imageWidth * sizeof(unsigned int),
        m_imageWidth * sizeof(unsigned int), m_imageHeight,
        hipMemcpyHostToDevice, streamToRun));
  }

  void createTexture() {
    hipResourceDesc texRes;
    memset(&texRes, 0, sizeof(hipResourceDesc));

    texRes.resType = hipResourceTypeArray;
    texRes.res.array.array = d_mipLevelArray;

    hipTextureDesc texDescr;
    memset(&texDescr, 0, sizeof(hipTextureDesc));

    texDescr.normalizedCoords = true;
    texDescr.filterMode = hipFilterModeLinear;
    texDescr.addressMode[0] = hipAddressModeWrap;
    texDescr.addressMode[1] = hipAddressModeWrap;
    texDescr.readMode = hipReadModeNormalizedFloat;

    checkCudaErrors(
        hipCreateTextureObject(&texObject, &texRes, &texDescr, NULL));
  }
};

class cudaNvSciWait {
 private:
  NvSciSyncModule m_syncModule;
  NvSciBufModule m_bufModule;

  NvSciSyncAttrList m_syncAttrList;
  NvSciBufAttrList m_rawBufAttrList;
  NvSciBufAttrList m_buffAttrListOut;
  NvSciBufAttrKeyValuePair pairArrayOut[10];
  NvSciSyncFence *m_fence;

  hipExternalMemory_t extMemRawBuf;
  hipExternalSemaphore_t waitSem;
  hipStream_t streamToRun;
  int m_cudaDeviceId;
  hipUUID m_devUUID;
  void *d_outputBuf;
  size_t m_bufSize;
  size_t imageWidth;
  size_t imageHeight;

 public:
  cudaNvSciWait(NvSciBufModule bufModule, NvSciSyncModule syncModule,
                int cudaDeviceId, int bufSize, NvSciSyncFence *fence)
      : m_bufModule(bufModule),
        m_syncModule(syncModule),
        m_cudaDeviceId(cudaDeviceId),
        m_bufSize(bufSize),
        m_fence(fence) {
    initCuda();
    checkNvSciErrors(NvSciSyncAttrListCreate(m_syncModule, &m_syncAttrList));
    checkNvSciErrors(NvSciBufAttrListCreate(m_bufModule, &m_rawBufAttrList));

    setRawBufAttrList(m_bufSize);
    checkCudaErrors(cudaDeviceGetNvSciSyncAttributes(
        m_syncAttrList, m_cudaDeviceId, cudaNvSciSyncAttrWait));
  }

  ~cudaNvSciWait() {
    checkCudaErrors(hipStreamDestroy(streamToRun));
    checkCudaErrors(hipDestroyExternalSemaphore(waitSem));
    checkCudaErrors(hipDestroyExternalMemory(extMemRawBuf));
    checkCudaErrors(hipFree(d_outputBuf));
  }

  void initCuda() {
    checkCudaErrors(hipSetDevice(m_cudaDeviceId));
    checkCudaErrors(
        hipStreamCreateWithFlags(&streamToRun, hipStreamNonBlocking));
#ifdef hipDeviceGetUuid
    hipError_t res = hipDeviceGetUuid(&m_devUUID, m_cudaDeviceId);
#else
    hipError_t res = hipDeviceGetUuid(&m_devUUID, m_cudaDeviceId);
#endif
    if (res != hipSuccess) {
      fprintf(stderr, "Driver API error = %04d \n", res);
      exit(EXIT_FAILURE);
    }

    int major = 0, minor = 0;
    checkCudaErrors(hipDeviceGetAttribute(
        &major, hipDeviceAttributeComputeCapabilityMajor, m_cudaDeviceId));
    checkCudaErrors(hipDeviceGetAttribute(
        &minor, hipDeviceAttributeComputeCapabilityMinor, m_cudaDeviceId));
    printf(
        "[cudaNvSciWait] GPU Device %d: \"%s\" with compute capability "
        "%d.%d\n\n",
        m_cudaDeviceId, _ConvertSMVer2ArchName(major, minor), major, minor);
  }

  void setRawBufAttrList(uint64_t size) {
    NvSciBufType bufType = NvSciBufType_RawBuffer;
    bool cpuAccess = false;
    NvSciBufAttrValAccessPerm perm = NvSciBufAccessPerm_ReadWrite;
    NvSciBufAttrKeyValuePair rawBufAttrs[] = {
        {NvSciBufRawBufferAttrKey_Size, &size, sizeof(size)},
        {NvSciBufGeneralAttrKey_Types, &bufType, sizeof(bufType)},
        {NvSciBufGeneralAttrKey_NeedCpuAccess, &cpuAccess, sizeof(cpuAccess)},
        {NvSciBufGeneralAttrKey_RequiredPerm, &perm, sizeof(perm)},
        {NvSciBufGeneralAttrKey_GpuId, &m_devUUID, sizeof(m_devUUID)},
    };

    checkNvSciErrors(NvSciBufAttrListSetAttrs(
        m_rawBufAttrList, rawBufAttrs,
        sizeof(rawBufAttrs) / sizeof(NvSciBufAttrKeyValuePair)));
  }

  NvSciSyncAttrList getNvSciSyncAttrList() { return m_syncAttrList; }

  NvSciBufAttrList getNvSciRawBufAttrList() { return m_rawBufAttrList; }

  void runImageGrayscale(std::string image_filename, size_t imageWidth,
                         size_t imageHeight) {
    int numOfGPUs = 0;
    checkCudaErrors(hipGetDeviceCount(&numOfGPUs));  // For cuda init purpose
    checkCudaErrors(hipSetDevice(m_cudaDeviceId));

    waitExternalSemaphore();
    launchGrayScaleKernel((unsigned int *)d_outputBuf, image_filename,
                          imageWidth, imageHeight, streamToRun);
  }

  void cudaImportNvSciSemaphore(NvSciSyncObj syncObj) {
    checkCudaErrors(hipSetDevice(m_cudaDeviceId));

    hipExternalSemaphoreHandleDesc extSemDesc;
    memset(&extSemDesc, 0, sizeof(extSemDesc));
    extSemDesc.type = cudaExternalSemaphoreHandleTypeNvSciSync;
    extSemDesc.handle.nvSciSyncObj = (void *)syncObj;

    checkCudaErrors(hipImportExternalSemaphore(&waitSem, &extSemDesc));
  }

  void waitExternalSemaphore() {
    checkCudaErrors(hipSetDevice(m_cudaDeviceId));

    hipExternalSemaphoreWaitParams waitParams;
    memset(&waitParams, 0, sizeof(waitParams));
    // For cross-process signaler-waiter applications need to use NvSciIpc
    // and NvSciSync[Export|Import] utilities to share the NvSciSyncFence
    // across process. This step is optional in single-process.
    waitParams.params.nvSciSync.fence = (void *)m_fence;
    waitParams.flags = 0;

    checkCudaErrors(
        hipWaitExternalSemaphoresAsync(&waitSem, &waitParams, 1, streamToRun));
  }

  void cudaImportNvSciRawBuf(NvSciBufObj inputBufObj) {
    checkCudaErrors(hipSetDevice(m_cudaDeviceId));

    checkNvSciErrors(NvSciBufObjGetAttrList(inputBufObj, &m_buffAttrListOut));

    memset(pairArrayOut, 0, sizeof(NvSciBufAttrKeyValuePair) * 10);
    pairArrayOut[0].key = NvSciBufRawBufferAttrKey_Size;

    checkNvSciErrors(
        NvSciBufAttrListGetAttrs(m_buffAttrListOut, pairArrayOut, 1));

    uint64_t size = *(uint64_t *)pairArrayOut[0].value;

    hipExternalMemoryHandleDesc memHandleDesc;
    memset(&memHandleDesc, 0, sizeof(memHandleDesc));
    memHandleDesc.type = cudaExternalMemoryHandleTypeNvSciBuf;
    memHandleDesc.handle.nvSciBufObject = inputBufObj;
    memHandleDesc.size = size;
    checkCudaErrors(hipImportExternalMemory(&extMemRawBuf, &memHandleDesc));

    hipExternalMemoryBufferDesc bufferDesc;
    memset(&bufferDesc, 0, sizeof(bufferDesc));
    bufferDesc.offset = 0;
    bufferDesc.size = size;
    m_bufSize = size;

    checkCudaErrors(hipExternalMemoryGetMappedBuffer(
        &d_outputBuf, extMemRawBuf, &bufferDesc));
  }
};

void thread_rotateAndSignal(cudaNvSciSignal *cudaNvSciSignalObj,
                            unsigned char *imageData) {
  std::lock_guard<std::mutex> guard(m_mutex);
  cudaNvSciSignalObj->runRotateImageAndSignal(imageData);
  workSubmitted = true;
  m_condVar.notify_one();
}

void thread_waitAndGrayscale(cudaNvSciWait *cudaNvSciWaitObj,
                             std::string image_filename, size_t imageWidth,
                             size_t imageHeight) {
  // Acquire the lock
  std::unique_lock<std::mutex> mlock(m_mutex);
  m_condVar.wait(mlock, [] { return workSubmitted; });
  cudaNvSciWaitObj->runImageGrayscale(image_filename, imageWidth, imageHeight);
}

cudaNvSci::cudaNvSci(int isMultiGPU, std::vector<int> &deviceIds,
                     unsigned char *imageData, size_t width, size_t height)
    : m_isMultiGPU(isMultiGPU),
      image_data(imageData),
      imageWidth(width),
      imageHeight(height) {
  if (isMultiGPU) {
    m_cudaNvSciSignalDeviceId = deviceIds[0];
    m_cudaNvSciWaitDeviceId = deviceIds[1];
  } else {
    m_cudaNvSciSignalDeviceId = m_cudaNvSciWaitDeviceId = deviceIds[0];
  }

  m_bufSize = imageWidth * imageHeight * sizeof(unsigned int);
}

void cudaNvSci::initNvSci() {
  checkNvSciErrors(NvSciSyncModuleOpen(&syncModule));
  checkNvSciErrors(NvSciBufModuleOpen(&buffModule));
  fence = (NvSciSyncFence *)calloc(1, sizeof(NvSciSyncFence));
}

void cudaNvSci::runCudaNvSci(std::string &image_filename) {
  initNvSci();

  cudaNvSciSignal rotateAndSignal(buffModule, syncModule,
                                  m_cudaNvSciSignalDeviceId, m_bufSize,
                                  imageWidth, imageHeight, fence);
  cudaNvSciWait waitAndGrayscale(buffModule, syncModule,
                                 m_cudaNvSciWaitDeviceId, m_bufSize, fence);

  rawBufUnreconciledList[0] = rotateAndSignal.getNvSciRawBufAttrList();
  rawBufUnreconciledList[1] = waitAndGrayscale.getNvSciRawBufAttrList();

  createNvSciRawBufObj();

  imageBufUnreconciledList[0] = rotateAndSignal.getNvSciImageBufAttrList();

  createNvSciBufImageObj();

  rotateAndSignal.cudaImportNvSciRawBuf(rawBufObj);
  rotateAndSignal.cudaImportNvSciImage(imageBufObj);

  waitAndGrayscale.cudaImportNvSciRawBuf(rawBufObj);

  syncUnreconciledList[0] = rotateAndSignal.getNvSciSyncAttrList();
  syncUnreconciledList[1] = waitAndGrayscale.getNvSciSyncAttrList();

  createNvSciSyncObj();

  rotateAndSignal.cudaImportNvSciSemaphore(syncObj);
  waitAndGrayscale.cudaImportNvSciSemaphore(syncObj);

  std::thread rotateThread(&thread_rotateAndSignal, &rotateAndSignal,
                           image_data);

  std::thread grayscaleThread(&thread_waitAndGrayscale, &waitAndGrayscale,
                              image_filename, imageWidth, imageHeight);

  rotateThread.join();
  grayscaleThread.join();
}

void cudaNvSci::createNvSciRawBufObj() {
  int numAttrList = 2;
  checkNvSciErrors(NvSciBufAttrListReconcile(rawBufUnreconciledList,
                                             numAttrList, &rawBufReconciledList,
                                             &buffConflictList));
  checkNvSciErrors(NvSciBufObjAlloc(rawBufReconciledList, &rawBufObj));
  printf("created NvSciBufObj\n");
}

void cudaNvSci::createNvSciBufImageObj() {
  int numAttrList = 1;
  checkNvSciErrors(NvSciBufAttrListReconcile(
      imageBufUnreconciledList, numAttrList, &imageBufReconciledList,
      &imageBufConflictList));
  checkNvSciErrors(NvSciBufObjAlloc(imageBufReconciledList, &imageBufObj));
  printf("created NvSciBufImageObj\n");
}

void cudaNvSci::createNvSciSyncObj() {
  int numAttrList = 2;
  checkNvSciErrors(NvSciSyncAttrListReconcile(syncUnreconciledList, numAttrList,
                                              &syncReconciledList,
                                              &syncConflictList));
  checkNvSciErrors(NvSciSyncObjAlloc(syncReconciledList, &syncObj));
  printf("created NvSciSyncObj\n");
}
