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
 * This sample demonstrates peer-to-peer access of stream ordered memory
 * allocated with hipMallocAsync and cudaMemPool family of APIs through simple
 * kernel which does peer-to-peer to access & scales vector elements.
 */

// System includes
#include <assert.h>
#include <stdio.h>
#include <iostream>
#include <map>
#include <set>
#include <utility>

// CUDA runtime
#include <hip/hip_runtime.h>
#include "rocprofiler.h"
#include "HIPCHECK.h"

// helper functions and utilities to work with CUDA
#include "helper_cuda_hipified.h"
#include "helper_functions.h"

// Simple kernel to demonstrate copying hipMallocAsync memory via P2P to peer
// device
__global__ void copyP2PAndScale(const int *src, int *dst, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < N) {
    // scale & store src vector.
    dst[idx] = 2 * src[idx];
  }
}

// Map of device version to device number
std::multimap<std::pair<int, int>, int> getIdenticalGPUs() {
  int numGpus = 0;
  HIPCHECK(hipGetDeviceCount(&numGpus));

  std::multimap<std::pair<int, int>, int> identicalGpus;

  for (int i = 0; i < numGpus; i++) {
    int isMemPoolSupported = 0;
    HIPCHECK(hipDeviceGetAttribute(&isMemPoolSupported,
                                           hipDeviceAttributeMemoryPoolsSupported, i));

    // Filter unsupported devices
    if (isMemPoolSupported) {
      int major = 0, minor = 0;
      HIPCHECK(
          hipDeviceGetAttribute(&major, hipDeviceAttributeComputeCapabilityMajor, i));
      HIPCHECK(
          hipDeviceGetAttribute(&minor, hipDeviceAttributeComputeCapabilityMinor, i));
      identicalGpus.emplace(std::make_pair(major, minor), i);
    }
  }

  return identicalGpus;
}

std::pair<int, int> getP2PCapableGpuPair() {
  constexpr size_t kNumGpusRequired = 2;

  auto gpusByArch = getIdenticalGPUs();

  auto it = gpusByArch.begin();
  auto end = gpusByArch.end();

  auto bestFit = std::make_pair(it, it);
  // use std::distance to find the largest number of GPUs amongst architectures
  auto distance = [](decltype(bestFit) p) {
    return std::distance(p.first, p.second);
  };

  // Read each unique key/pair element in order
  for (; it != end; it = gpusByArch.upper_bound(it->first)) {
    // first and second are iterators bounded within the architecture group
    auto testFit = gpusByArch.equal_range(it->first);
    // Always use devices with highest architecture version or whichever has the
    // most devices available
    if (distance(bestFit) <= distance(testFit)) bestFit = testFit;
  }

  if (distance(bestFit) < kNumGpusRequired) {
    printf(
        "No Two or more GPUs with same architecture capable of cuda Memory "
        "Pools found."
        "\nWaiving the sample\n");
    exit(EXIT_WAIVED);
  }

  std::set<int> bestFitDeviceIds;

  // check & select peer-to-peer access capable GPU devices.
  int devIds[2];
  for (auto itr = bestFit.first; itr != bestFit.second; itr++) {
    int deviceId = itr->second;
    HIPCHECK(hipSetDevice(deviceId));

    std::for_each(itr, bestFit.second, [&deviceId, &bestFitDeviceIds,
                                        &kNumGpusRequired](
                                           decltype(*itr) mapPair) {
      if (deviceId != mapPair.second) {
        int access = 0;
        HIPCHECK(
            hipDeviceCanAccessPeer(&access, deviceId, mapPair.second));
        printf("Device=%d %s Access Peer Device=%d\n", deviceId,
               access ? "CAN" : "CANNOT", mapPair.second);
        if (access && bestFitDeviceIds.size() < kNumGpusRequired) {
          bestFitDeviceIds.emplace(deviceId);
          bestFitDeviceIds.emplace(mapPair.second);
        } else {
          printf("Ignoring device %i (max devices exceeded)\n", mapPair.second);
        }
      }
    });

    if (bestFitDeviceIds.size() >= kNumGpusRequired) {
      printf("Selected p2p capable devices - ");
      int i = 0;
      for (auto devicesItr = bestFitDeviceIds.begin();
           devicesItr != bestFitDeviceIds.end(); devicesItr++) {
        devIds[i++] = *devicesItr;
        printf("deviceId = %d  ", *devicesItr);
      }
      printf("\n");
      break;
    }
  }

  // if bestFitDeviceIds.size() == 0 it means the GPUs in system are not p2p
  // capable, hence we add it without p2p capability check.
  if (!bestFitDeviceIds.size()) {
    printf("No Two or more Devices p2p capable found.. exiting..\n");
    exit(EXIT_WAIVED);
  }

  auto p2pGpuPair = std::make_pair(devIds[0], devIds[1]);

  return p2pGpuPair;
}

int memPoolP2PCopy() {
  int *dev0_srcVec, *dev1_dstVec;  // Device buffers
  hipStream_t stream1, stream2;
  hipMemPool_t memPool;
  hipEvent_t waitOnStream1;

  // Allocate CPU memory.
  size_t nelem = 1048576;
  size_t bytes = nelem * sizeof(int);

  int *a = (int *)malloc(bytes);
  int *output = (int *)malloc(bytes);

  /* Initialize the vectors. */
  for (int n = 0; n < nelem; n++) {
    a[n] = rand() / (int)RAND_MAX;
  }

  auto p2pDevices = getP2PCapableGpuPair();
  printf("selected devices = %d & %d\n", p2pDevices.first, p2pDevices.second);
  HIPCHECK(hipSetDevice(p2pDevices.first));
  HIPCHECK(hipEventCreate(&waitOnStream1));

  HIPCHECK(hipStreamCreateWithFlags(&stream1, hipStreamNonBlocking));

  // Get the default mempool for device p2pDevices.first from the pair
  HIPCHECK(hipDeviceGetDefaultMemPool(&memPool, p2pDevices.first));

  // Allocate memory in a stream from the pool set above.
  HIPCHECK(hipMallocAsync(&dev0_srcVec, bytes, stream1));

  HIPCHECK(
      hipMemcpyAsync(dev0_srcVec, a, bytes, hipMemcpyHostToDevice, stream1));
  HIPCHECK(hipEventRecord(waitOnStream1, stream1));

  HIPCHECK(hipSetDevice(p2pDevices.second));
  HIPCHECK(hipStreamCreateWithFlags(&stream2, hipStreamNonBlocking));

  // Allocate memory in p2pDevices.second device
  HIPCHECK(hipMallocAsync(&dev1_dstVec, bytes, stream2));

  // Setup peer mappings for p2pDevices.second device
  hipMemAccessDesc desc;
  memset(&desc, 0, sizeof(hipMemAccessDesc));
  desc.location.type = hipMemLocationTypeDevice;
  desc.location.id = p2pDevices.second;
  desc.flags = hipMemAccessFlagsProtReadWrite;
  HIPCHECK(hipMemPoolSetAccess(memPool, &desc, 1));

  printf("> copyP2PAndScale kernel running ...\n");
  dim3 block(256);
  dim3 grid((unsigned int)ceil(nelem / (int)block.x));
  HIPCHECK(hipStreamWaitEvent(stream2, waitOnStream1,3));
  copyP2PAndScale<<<grid, block, 0, stream2>>>(dev0_srcVec, dev1_dstVec, nelem);

  HIPCHECK(hipMemcpyAsync(output, dev1_dstVec, bytes,
                                  hipMemcpyDeviceToHost, stream2));
  HIPCHECK(hipFreeAsync(dev0_srcVec, stream2));
  HIPCHECK(hipFreeAsync(dev1_dstVec, stream2));
  HIPCHECK(hipStreamSynchronize(stream2));

  /* Compare the results */
  printf("> Checking the results from copyP2PAndScale() ...\n");

  for (int n = 0; n < nelem; n++) {
    if ((2 * a[n]) != output[n]) {
      printf("mismatch i = %d expected = %d val = %d\n", n, 2 * a[n],
             output[n]);
      return EXIT_FAILURE;
    }
  }

  free(a);
  free(output);
  HIPCHECK(hipStreamDestroy(stream1));
  HIPCHECK(hipStreamDestroy(stream2));
  printf("PASSED\n");

  return EXIT_SUCCESS;
}

int main(int argc, char **argv) {
  int ret = memPoolP2PCopy();
  return ret;
}
