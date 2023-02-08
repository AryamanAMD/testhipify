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

// This sample demonstrates how to use the CUDA hook library to receive
// callbacks

#include <hip/hip_runtime.h>
#include <stdio.h>
#include <dlfcn.h>

#include "libcuhook.h"

#define ASSERT_COND(x, msg)                                                    \
  do {                                                                         \
    if (!(x)) {                                                                \
      fprintf(stderr, "Error: Condition (%s) failed at %s:%d\n", #x, __FILE__, \
              __LINE__);                                                       \
      fprintf(stderr, "cuHook sample failed (%s)\n", msg);                     \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

/*
** Example of how to use the CUDA Interception Library, libcuhook.so
** The library has to be loaded via LD_PRELOAD, e.g.
*LD_PRELOAD=<full_path>/libcuhook.so.1 ./cuHook
*/

static int allocation_cb = 0;
static int free_cb = 0;
static int destroy_ctx_cb = 0;

hipError_t device_allocation_callback(hipDeviceptr_t *dptr, size_t bytesize) {
  fprintf(stdout, "Received memory allocation callback!\n");
  allocation_cb++;
  return hipSuccess;
}

hipError_t device_free_callback(hipDeviceptr_t dptr) {
  fprintf(stdout, "Received memory de-allocation callback!\n");
  free_cb++;
  return hipSuccess;
}

hipError_t destroy_context_callback(hipCtx_t ctx) {
  fprintf(stdout, "Received context destroy event!\n");
  destroy_ctx_cb++;
  return hipSuccess;
}

int main() {
  int count;
  hipCtx_t ctx;

  count = 0;

  hipInit(0);
  hipGetDeviceCount(&count);
  ASSERT_COND(count > 0, "No suitable devices found");

  // Load the cudaHookRegisterCallback symbol using the default library search
  // order. If we found the symbol, then the hooking library has been loaded
  fnCuHookRegisterCallback cuHook =
      (fnCuHookRegisterCallback)dlsym(RTLD_DEFAULT, "cuHookRegisterCallback");
  //    ASSERT_COND(cuHook, dlerror());
  if (cuHook) {
    // CUDA Runtime symbols cannot be hooked but the underlying driver ones
    // _can_. Example:
    // - hipFree() will trigger hipFree
    // - hipDeviceReset() will trigger a context change and you would need to
    // intercept hipCtxGetCurrent/hipCtxSetCurrent
    cuHook(CU_HOOK_MEM_ALLOC, POST_CALL_HOOK,
           (void *)device_allocation_callback);
    cuHook(CU_HOOK_MEM_FREE, PRE_CALL_HOOK, (void *)device_free_callback);
    cuHook(CU_HOOK_CTX_DESTROY, POST_CALL_HOOK,
           (void *)destroy_context_callback);
  }

  hipCtxCreate(&ctx, 0, 0);
  {
    hipError_t status;
    hipDeviceptr_t dptr;

    status = hipMalloc(&dptr, 1024);
    ASSERT_COND(status == hipSuccess, "hipMalloc call failed");

    status = hipFree(dptr);
    ASSERT_COND(status == hipSuccess, "hipFree call failed");
  }
  hipCtxDestroy(ctx);

  ASSERT_COND(allocation_cb == 1, "Didn't receive the allocation callback");
  ASSERT_COND(free_cb == 1, "Didn't receive the free callback");
  ASSERT_COND(destroy_ctx_cb == 1,
              "Didn't receive the destroy context callback");

  fprintf(stdout, "Sample finished successfully.\n");
  return (0);
}
