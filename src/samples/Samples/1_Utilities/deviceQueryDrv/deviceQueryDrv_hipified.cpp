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

/* This sample queries the properties of the CUDA devices present
 * in the system.
 */

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <hip/hip_runtime.h>
#include <helper_cuda_drvapi.h>

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
  hipDevice_t dev;
  int major = 0, minor = 0;
  int deviceCount = 0;
  char deviceName[256];

  printf("%s Starting...\n\n", argv[0]);

  // note your project will need to link with cuda.lib files on windows
  printf("CUDA Device Query (Driver API) statically linked version \n");

  checkCudaErrors(hipInit(0));

  checkCudaErrors(hipGetDeviceCount(&deviceCount));

  // This function call returns 0 if there are no CUDA capable devices.
  if (deviceCount == 0) {
    printf("There are no available device(s) that support CUDA\n");
  } else {
    printf("Detected %d CUDA Capable device(s)\n", deviceCount);
  }

  for (dev = 0; dev < deviceCount; ++dev) {
    checkCudaErrors(hipDeviceGetAttribute(
        &major, hipDeviceAttributeComputeCapabilityMajor, dev));
    checkCudaErrors(hipDeviceGetAttribute(
        &minor, hipDeviceAttributeComputeCapabilityMinor, dev));

    checkCudaErrors(hipDeviceGetName(deviceName, 256, dev));

    printf("\nDevice %d: \"%s\"\n", dev, deviceName);

    int driverVersion = 0;
    checkCudaErrors(hipDriverGetVersion(&driverVersion));
    printf("  CUDA Driver Version:                           %d.%d\n",
           driverVersion / 1000, (driverVersion % 100) / 10);
    printf("  CUDA Capability Major/Minor version number:    %d.%d\n", major,
           minor);

    size_t totalGlobalMem;
    checkCudaErrors(hipDeviceTotalMem(&totalGlobalMem, dev));

    char msg[256];
    SPRINTF(msg,
            "  Total amount of global memory:                 %.0f MBytes "
            "(%llu bytes)\n",
            (float)totalGlobalMem / 1048576.0f,
            (unsigned long long)totalGlobalMem);
    printf("%s", msg);

    int multiProcessorCount;
    getCudaAttribute<int>(&multiProcessorCount,
                          hipDeviceAttributeMultiprocessorCount, dev);

    printf("  (%2d) Multiprocessors, (%3d) CUDA Cores/MP:     %d CUDA Cores\n",
           multiProcessorCount, _ConvertSMVer2CoresDRV(major, minor),
           _ConvertSMVer2CoresDRV(major, minor) * multiProcessorCount);

    int clockRate;
    getCudaAttribute<int>(&clockRate, hipDeviceAttributeClockRate, dev);
    printf(
        "  GPU Max Clock rate:                            %.0f MHz (%0.2f "
        "GHz)\n",
        clockRate * 1e-3f, clockRate * 1e-6f);
    int memoryClock;
    getCudaAttribute<int>(&memoryClock, hipDeviceAttributeMemoryClockRate,
                          dev);
    printf("  Memory Clock rate:                             %.0f Mhz\n",
           memoryClock * 1e-3f);
    int memBusWidth;
    getCudaAttribute<int>(&memBusWidth,
                          hipDeviceAttributeMemoryBusWidth, dev);
    printf("  Memory Bus Width:                              %d-bit\n",
           memBusWidth);
    int L2CacheSize;
    getCudaAttribute<int>(&L2CacheSize, hipDeviceAttributeL2CacheSize, dev);

    if (L2CacheSize) {
      printf("  L2 Cache Size:                                 %d bytes\n",
             L2CacheSize);
    }

    int maxTex1D, maxTex2D[2], maxTex3D[3];
    getCudaAttribute<int>(&maxTex1D,
                          hipDeviceAttributeMaxTexture1DWidth, dev);
    getCudaAttribute<int>(&maxTex2D[0],
                          hipDeviceAttributeMaxTexture2DWidth, dev);
    getCudaAttribute<int>(&maxTex2D[1],
                          hipDeviceAttributeMaxTexture2DHeight, dev);
    getCudaAttribute<int>(&maxTex3D[0],
                          hipDeviceAttributeMaxTexture3DWidth, dev);
    getCudaAttribute<int>(&maxTex3D[1],
                          hipDeviceAttributeMaxTexture3DHeight, dev);
    getCudaAttribute<int>(&maxTex3D[2],
                          hipDeviceAttributeMaxTexture3DDepth, dev);
    printf(
        "  Max Texture Dimension Sizes                    1D=(%d) 2D=(%d, %d) "
        "3D=(%d, %d, %d)\n",
        maxTex1D, maxTex2D[0], maxTex2D[1], maxTex3D[0], maxTex3D[1],
        maxTex3D[2]);

    int maxTex1DLayered[2];
    getCudaAttribute<int>(&maxTex1DLayered[0],
                          hipDeviceAttributeMaxTexture1DLayered,
                          dev);
    getCudaAttribute<int>(&maxTex1DLayered[1],
                          CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_LAYERS,
                          dev);
    printf(
        "  Maximum Layered 1D Texture Size, (num) layers  1D=(%d), %d layers\n",
        maxTex1DLayered[0], maxTex1DLayered[1]);

    int maxTex2DLayered[3];
    getCudaAttribute<int>(&maxTex2DLayered[0],
                          hipDeviceAttributeMaxTexture2DLayered,
                          dev);
    getCudaAttribute<int>(&maxTex2DLayered[1],
                          hipDeviceAttributeMaxTexture2DLayered,
                          dev);
    getCudaAttribute<int>(&maxTex2DLayered[2],
                          CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS,
                          dev);
    printf(
        "  Maximum Layered 2D Texture Size, (num) layers  2D=(%d, %d), %d "
        "layers\n",
        maxTex2DLayered[0], maxTex2DLayered[1], maxTex2DLayered[2]);

    int totalConstantMemory;
    getCudaAttribute<int>(&totalConstantMemory,
                          hipDeviceAttributeTotalConstantMemory, dev);
    printf("  Total amount of constant memory:               %u bytes\n",
           totalConstantMemory);
    int sharedMemPerBlock;
    getCudaAttribute<int>(&sharedMemPerBlock,
                          hipDeviceAttributeMaxSharedMemoryPerBlock, dev);
    printf("  Total amount of shared memory per block:       %u bytes\n",
           sharedMemPerBlock);
    int regsPerBlock;
    getCudaAttribute<int>(&regsPerBlock,
                          hipDeviceAttributeMaxRegistersPerBlock, dev);
    printf("  Total number of registers available per block: %d\n",
           regsPerBlock);
    int warpSize;
    getCudaAttribute<int>(&warpSize, hipDeviceAttributeWarpSize, dev);
    printf("  Warp size:                                     %d\n", warpSize);
    int maxThreadsPerMultiProcessor;
    getCudaAttribute<int>(&maxThreadsPerMultiProcessor,
                          hipDeviceAttributeMaxThreadsPerMultiProcessor,
                          dev);
    printf("  Maximum number of threads per multiprocessor:  %d\n",
           maxThreadsPerMultiProcessor);
    int maxThreadsPerBlock;
    getCudaAttribute<int>(&maxThreadsPerBlock,
                          hipDeviceAttributeMaxThreadsPerBlock, dev);
    printf("  Maximum number of threads per block:           %d\n",
           maxThreadsPerBlock);

    int blockDim[3];
    getCudaAttribute<int>(&blockDim[0], hipDeviceAttributeMaxBlockDimX,
                          dev);
    getCudaAttribute<int>(&blockDim[1], hipDeviceAttributeMaxBlockDimY,
                          dev);
    getCudaAttribute<int>(&blockDim[2], hipDeviceAttributeMaxBlockDimZ,
                          dev);
    printf("  Max dimension size of a thread block (x,y,z): (%d, %d, %d)\n",
           blockDim[0], blockDim[1], blockDim[2]);
    int gridDim[3];
    getCudaAttribute<int>(&gridDim[0], hipDeviceAttributeMaxGridDimX, dev);
    getCudaAttribute<int>(&gridDim[1], hipDeviceAttributeMaxGridDimY, dev);
    getCudaAttribute<int>(&gridDim[2], hipDeviceAttributeMaxGridDimZ, dev);
    printf("  Max dimension size of a grid size (x,y,z):    (%d, %d, %d)\n",
           gridDim[0], gridDim[1], gridDim[2]);

    int textureAlign;
    getCudaAttribute<int>(&textureAlign, hipDeviceAttributeTextureAlignment,
                          dev);
    printf("  Texture alignment:                             %u bytes\n",
           textureAlign);

    int memPitch;
    getCudaAttribute<int>(&memPitch, hipDeviceAttributeMaxPitch, dev);
    printf("  Maximum memory pitch:                          %u bytes\n",
           memPitch);

    int gpuOverlap;
    getCudaAttribute<int>(&gpuOverlap, hipDeviceAttributeAsyncEngineCount, dev);

    int asyncEngineCount;
    getCudaAttribute<int>(&asyncEngineCount,
                          hipDeviceAttributeAsyncEngineCount, dev);
    printf(
        "  Concurrent copy and kernel execution:          %s with %d copy "
        "engine(s)\n",
        (gpuOverlap ? "Yes" : "No"), asyncEngineCount);

    int kernelExecTimeoutEnabled;
    getCudaAttribute<int>(&kernelExecTimeoutEnabled,
                          hipDeviceAttributeKernelExecTimeout, dev);
    printf("  Run time limit on kernels:                     %s\n",
           kernelExecTimeoutEnabled ? "Yes" : "No");
    int integrated;
    getCudaAttribute<int>(&integrated, hipDeviceAttributeIntegrated, dev);
    printf("  Integrated GPU sharing Host Memory:            %s\n",
           integrated ? "Yes" : "No");
    int canMapHostMemory;
    getCudaAttribute<int>(&canMapHostMemory,
                          hipDeviceAttributeCanMapHostMemory, dev);
    printf("  Support host page-locked memory mapping:       %s\n",
           canMapHostMemory ? "Yes" : "No");

    int concurrentKernels;
    getCudaAttribute<int>(&concurrentKernels,
                          hipDeviceAttributeConcurrentKernels, dev);
    printf("  Concurrent kernel execution:                   %s\n",
           concurrentKernels ? "Yes" : "No");

    int surfaceAlignment;
    getCudaAttribute<int>(&surfaceAlignment,
                          hipDeviceAttributeSurfaceAlignment, dev);
    printf("  Alignment requirement for Surfaces:            %s\n",
           surfaceAlignment ? "Yes" : "No");

    int eccEnabled;
    getCudaAttribute<int>(&eccEnabled, hipDeviceAttributeEccEnabled, dev);
    printf("  Device has ECC support:                        %s\n",
           eccEnabled ? "Enabled" : "Disabled");

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
    int tccDriver;
    getCudaAttribute<int>(&tccDriver, hipDeviceAttributeTccDriver, dev);
    printf("  CUDA Device Driver Mode (TCC or WDDM):         %s\n",
           tccDriver ? "TCC (Tesla Compute Cluster Driver)"
                     : "WDDM (Windows Display Driver Model)");
#endif

    int unifiedAddressing;
    getCudaAttribute<int>(&unifiedAddressing,
                          hipDeviceAttributeUnifiedAddressing, dev);
    printf("  Device supports Unified Addressing (UVA):      %s\n",
           unifiedAddressing ? "Yes" : "No");

    int managedMemory;
    getCudaAttribute<int>(&managedMemory, hipDeviceAttributeManagedMemory,
                          dev);
    printf("  Device supports Managed Memory:                %s\n",
           managedMemory ? "Yes" : "No");

    int computePreemption;
    getCudaAttribute<int>(&computePreemption,
                          hipDeviceAttributeComputePreemptionSupported,
                          dev);
    printf("  Device supports Compute Preemption:            %s\n",
           computePreemption ? "Yes" : "No");

    int cooperativeLaunch;
    getCudaAttribute<int>(&cooperativeLaunch,
                          hipDeviceAttributeCooperativeLaunch, dev);
    printf("  Supports Cooperative Kernel Launch:            %s\n",
           cooperativeLaunch ? "Yes" : "No");

    int cooperativeMultiDevLaunch;
    getCudaAttribute<int>(&cooperativeMultiDevLaunch,
                          hipDeviceAttributeCooperativeMultiDeviceLaunch,
                          dev);
    printf("  Supports MultiDevice Co-op Kernel Launch:      %s\n",
           cooperativeMultiDevLaunch ? "Yes" : "No");

    int pciDomainID, pciBusID, pciDeviceID;
    getCudaAttribute<int>(&pciDomainID, hipDeviceAttributePciDomainID, dev);
    getCudaAttribute<int>(&pciBusID, hipDeviceAttributePciBusId, dev);
    getCudaAttribute<int>(&pciDeviceID, hipDeviceAttributePciDeviceId, dev);
    printf("  Device PCI Domain ID / Bus ID / location ID:   %d / %d / %d\n",
           pciDomainID, pciBusID, pciDeviceID);

    const char *sComputeMode[] = {
        "Default (multiple host threads can use ::hipSetDevice() with device "
        "simultaneously)",
        "Exclusive (only one host thread in one process is able to use "
        "::hipSetDevice() with this device)",
        "Prohibited (no host thread can use ::hipSetDevice() with this "
        "device)",
        "Exclusive Process (many threads in one process is able to use "
        "::hipSetDevice() with this device)",
        "Unknown", NULL};

    int computeMode;
    getCudaAttribute<int>(&computeMode, hipDeviceAttributeComputeMode, dev);
    printf("  Compute Mode:\n");
    printf("     < %s >\n", sComputeMode[computeMode]);
  }

  // If there are 2 or more GPUs, query to determine whether RDMA is supported
  if (deviceCount >= 2) {
    int gpuid[64];  // we want to find the first two GPUs that can support P2P
    int gpu_p2p_count = 0;
    int tccDriver = 0;

    for (int i = 0; i < deviceCount; i++) {
      checkCudaErrors(hipDeviceGetAttribute(
          &major, hipDeviceAttributeComputeCapabilityMajor, i));
      checkCudaErrors(hipDeviceGetAttribute(
          &minor, hipDeviceAttributeComputeCapabilityMinor, i));
      getCudaAttribute<int>(&tccDriver, hipDeviceAttributeTccDriver, i);

      // Only boards based on Fermi or later can support P2P
      if ((major >= 2)
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
          // on Windows (64-bit), the Tesla Compute Cluster driver for windows
          // must be enabled to support this
          && tccDriver
#endif
          ) {
        // This is an array of P2P capable GPUs
        gpuid[gpu_p2p_count++] = i;
      }
    }

    // Show all the combinations of support P2P GPUs
    int can_access_peer;
    char deviceName0[256], deviceName1[256];

    if (gpu_p2p_count >= 2) {
      for (int i = 0; i < gpu_p2p_count; i++) {
        for (int j = 0; j < gpu_p2p_count; j++) {
          if (gpuid[i] == gpuid[j]) {
            continue;
          }
          checkCudaErrors(
              hipDeviceCanAccessPeer(&can_access_peer, gpuid[i], gpuid[j]));
          checkCudaErrors(hipDeviceGetName(deviceName0, 256, gpuid[i]));
          checkCudaErrors(hipDeviceGetName(deviceName1, 256, gpuid[j]));
          printf(
              "> Peer-to-Peer (P2P) access from %s (GPU%d) -> %s (GPU%d) : "
              "%s\n",
              deviceName0, gpuid[i], deviceName1, gpuid[j],
              can_access_peer ? "Yes" : "No");
        }
      }
    }
  }

  printf("Result = PASS\n");

  exit(EXIT_SUCCESS);
}
