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

#include <stdio.h>
#include <string.h>
#include "helper_cuda_hipified.h"
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

hipError_t setProp(hipMemAllocationProp *prop, bool UseCompressibleMemory)
{
    hipDevice_t currentDevice;
    if (hipCtxGetDevice(&currentDevice) != hipSuccess)
        return hipErrorOutOfMemory;

    memset(prop, 0, sizeof(hipMemAllocationProp));
    prop->type = hipMemAllocationTypePinned;
    prop->location.type = hipMemLocationTypeDevice;
    prop->location.id = currentDevice;

    if (UseCompressibleMemory)
       // prop->allocFlags.compressionType = CU_MEM_ALLOCATION_COMP_GENERIC;

    return hipSuccess;
}

hipError_t allocateCompressible(void **adr, size_t size, bool UseCompressibleMemory)
{
    hipMemAllocationProp prop = {};
    hipError_t err = setProp(&prop, UseCompressibleMemory);
    if (err != hipSuccess)
        return err;

    size_t granularity = 0;
    if (hipMemGetAllocationGranularity(&granularity, &prop,
                                      hipMemAllocationGranularityMinimum) != hipSuccess)
        return hipErrorOutOfMemory;
    size = ((size - 1) / granularity + 1) * granularity;
    hipDeviceptr_t dptr;
    if (hipMemAddressReserve(&dptr, size, 0, 0, 0) != hipSuccess)
        return hipErrorOutOfMemory;

    hipMemGenericAllocationHandle_t allocationHandle;
    if (hipMemCreate(&allocationHandle, size, &prop, 0) != hipSuccess)
        return hipErrorOutOfMemory;

    // Check if hipMemCreate was able to allocate compressible memory.
    if (UseCompressibleMemory) {
        hipMemAllocationProp allocationProp = {};
        hipMemGetAllocationPropertiesFromHandle(&allocationProp, allocationHandle);
       // if (allocationProp.allocFlags.compressionType != CU_MEM_ALLOCATION_COMP_GENERIC) {
       //     printf("Could not allocate compressible memory... so waiving execution\n");
       //     exit(EXIT_WAIVED);
       // }
    }

    if (hipMemMap(dptr, size, 0, allocationHandle, 0) != hipSuccess)
        return hipErrorOutOfMemory;

    if (hipMemRelease(allocationHandle) != hipSuccess)
        return hipErrorOutOfMemory;

    hipMemAccessDesc accessDescriptor;
    accessDescriptor.location.id = prop.location.id;
    accessDescriptor.location.type = prop.location.type;
    accessDescriptor.flags = hipMemAccessFlagsProtReadWrite;

    if (hipMemSetAccess(dptr, size, &accessDescriptor, 1) != hipSuccess)
        return hipErrorOutOfMemory;

    *adr = (void *)dptr;
    return hipSuccess;
}

hipError_t freeCompressible(void *ptr, size_t size, bool UseCompressibleMemory)
{
    hipMemAllocationProp prop = {};
    hipError_t err = setProp(&prop, UseCompressibleMemory);
    if (err != hipSuccess)
        return err;

    size_t granularity = 0;
    if (hipMemGetAllocationGranularity(&granularity, &prop,
                                      hipMemAllocationGranularityMinimum) != hipSuccess)
        return hipErrorOutOfMemory;
    size = ((size - 1) / granularity + 1) * granularity;

    if (ptr == NULL)
        return hipSuccess;
    if (hipMemUnmap((hipDeviceptr_t)ptr, size) != hipSuccess ||
        hipMemAddressFree((hipDeviceptr_t)ptr, size) != hipSuccess)
        return hipErrorInvalidValue;
    return hipSuccess;
}
