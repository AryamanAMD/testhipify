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

// CUDA Runtime
#include <hip/hip_runtime.h>

// Helper functions
#include "helper_cuda_hipified.h"
#include <helper_math.h>
#include "volume.h"

void Volume_init(Volume *vol, hipExtent dataSize, void *h_data,
                 int allowStore) {
  // create 3D array
  vol->channelDesc = hipCreateChannelDesc<VolumeType>();
  checkCudaErrors(
      hipMalloc3DArray(&vol->content, &vol->channelDesc, dataSize,
                        allowStore ? hipArraySurfaceLoadStore : 0));
  vol->size = dataSize;

  if (h_data) {
    // copy data to 3D array
    hipMemcpy3DParms copyParams = {0};
    copyParams.srcPtr =
        make_hipPitchedPtr(h_data, dataSize.width * sizeof(VolumeType),
                            dataSize.width, dataSize.height);
    copyParams.dstArray = vol->content;
    copyParams.extent = dataSize;
    copyParams.kind = hipMemcpyHostToDevice;
    checkCudaErrors(hipMemcpy3D(&copyParams));
  }

  if (allowStore) {
    hipResourceDesc surfRes;
    memset(&surfRes, 0, sizeof(hipResourceDesc));
    surfRes.resType = hipResourceTypeArray;
    surfRes.res.array.array = vol->content;

    checkCudaErrors(hipCreateSurfaceObject(&vol->volumeSurf, &surfRes));
  }

  hipResourceDesc texRes;
  memset(&texRes, 0, sizeof(hipResourceDesc));

  texRes.resType = hipResourceTypeArray;
  texRes.res.array.array = vol->content;

  hipTextureDesc texDescr;
  memset(&texDescr, 0, sizeof(hipTextureDesc));

  texDescr.normalizedCoords = true;
  texDescr.filterMode = hipFilterModeLinear;
  texDescr.addressMode[0] = hipAddressModeWrap;
  texDescr.addressMode[1] = hipAddressModeWrap;
  texDescr.addressMode[2] = hipAddressModeWrap;
  texDescr.readMode =
      hipReadModeNormalizedFloat;  // VolumeTypeInfo<VolumeType>::readMode;

  checkCudaErrors(
      hipCreateTextureObject(&vol->volumeTex, &texRes, &texDescr, NULL));
}

void Volume_deinit(Volume *vol) {
  checkCudaErrors(hipDestroyTextureObject(vol->volumeTex));
  checkCudaErrors(hipDestroySurfaceObject(vol->volumeSurf));
  checkCudaErrors(hipFreeArray(vol->content));
  vol->content = 0;
}
