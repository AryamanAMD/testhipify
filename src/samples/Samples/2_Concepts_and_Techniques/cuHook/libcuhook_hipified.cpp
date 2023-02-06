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

// This sample demonstrates a simple library to interpose CUDA symbols

#define __USE_GNU
#include <dlfcn.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <stdlib.h>

#include <hip/hip_runtime.h>
#include "libcuhook.h"

// For interposing dlsym(). See elf/dl-libc.c for the internal dlsym interface
// function For interposing dlopen(). Sell elf/dl-lib.c for the internal
// dlopen_mode interface function
extern "C" {
void* __libc_dlsym(void* map, const char* name);
}
extern "C" {
void* __libc_dlopen_mode(const char* name, int mode);
}

// We need to give the pre-processor a chance to replace a function, such as:
// hipMalloc => hipMalloc
#define STRINGIFY(x) #x
#define CUDA_SYMBOL_STRING(x) STRINGIFY(x)

// We need to interpose dlsym since anyone using dlopen+dlsym to get the CUDA
// driver symbols will bypass the hooking mechanism (this includes the CUDA
// runtime). Its tricky though, since if we replace the real dlsym with ours, we
// can't dlsym() the real dlsym. To get around that, call the 'private' libc
// interface called __libc_dlsym to get the real dlsym.
typedef void* (*fnDlsym)(void*, const char*);

static void* real_dlsym(void* handle, const char* symbol) {
  static fnDlsym internal_dlsym = (fnDlsym)__libc_dlsym(
      __libc_dlopen_mode("libdl.so.2", RTLD_LAZY), "dlsym");
  return (*internal_dlsym)(handle, symbol);
}

// Main structure that gets initialized at library load time
// Choose a unique name, or it can clash with other preloaded libraries.
struct cuHookInfo {
  void* handle;
  void* preHooks[CU_HOOK_SYMBOLS];
  void* postHooks[CU_HOOK_SYMBOLS];

  // Debugging/Stats Info
  int bDebugEnabled;
  int hookedFunctionCalls[CU_HOOK_SYMBOLS];

  cuHookInfo() {
    const char* envHookDebug;

    // Check environment for CU_HOOK_DEBUG to facilitate debugging
    envHookDebug = getenv("CU_HOOK_DEBUG");
    if (envHookDebug && envHookDebug[0] == '1') {
      bDebugEnabled = 1;
      fprintf(stderr, "* %6d >> CUDA HOOK Library loaded.\n", getpid());
    }
  }

  ~cuHookInfo() {
    if (bDebugEnabled) {
      pid_t pid = getpid();
      // You can gather statistics, timings, etc.
      fprintf(stderr, "* %6d >> CUDA HOOK Library Unloaded - Statistics:\n",
              pid);
      fprintf(stderr, "* %6d >> %20s ... %d\n", pid,
              CUDA_SYMBOL_STRING(hipMalloc),
              hookedFunctionCalls[CU_HOOK_MEM_ALLOC]);
      fprintf(stderr, "* %6d >> %20s ... %d\n", pid,
              CUDA_SYMBOL_STRING(hipFree),
              hookedFunctionCalls[CU_HOOK_MEM_FREE]);
      fprintf(stderr, "* %6d >> %20s ... %d\n", pid,
              CUDA_SYMBOL_STRING(hipCtxGetCurrent),
              hookedFunctionCalls[CU_HOOK_CTX_GET_CURRENT]);
      fprintf(stderr, "* %6d >> %20s ... %d\n", pid,
              CUDA_SYMBOL_STRING(hipCtxSetCurrent),
              hookedFunctionCalls[CU_HOOK_CTX_SET_CURRENT]);
      fprintf(stderr, "* %6d >> %20s ... %d\n", pid,
              CUDA_SYMBOL_STRING(hipCtxDestroy),
              hookedFunctionCalls[CU_HOOK_CTX_DESTROY]);
    }
    if (handle) {
      dlclose(handle);
    }
  }
};

static struct cuHookInfo cuhl;

// Exposed API
void cuHookRegisterCallback(HookSymbols symbol, HookTypes type,
                            void* callback) {
  if (type == PRE_CALL_HOOK) {
    cuhl.preHooks[symbol] = callback;
  } else if (type == POST_CALL_HOOK) {
    cuhl.postHooks[symbol] = callback;
  }
}

/*
 ** Interposed Functions
 */
void* dlsym(void* handle, const char* symbol) {
  // Early out if not a CUDA driver symbol
  if (strncmp(symbol, "cu", 2) != 0) {
    return (real_dlsym(handle, symbol));
  }

  if (strcmp(symbol, CUDA_SYMBOL_STRING(hipMalloc)) == 0) {
    return (void*)(&hipMalloc);
  } else if (strcmp(symbol, CUDA_SYMBOL_STRING(hipFree)) == 0) {
    return (void*)(&hipFree);
  } else if (strcmp(symbol, CUDA_SYMBOL_STRING(hipCtxGetCurrent)) == 0) {
    return (void*)(&hipCtxGetCurrent);
  } else if (strcmp(symbol, CUDA_SYMBOL_STRING(hipCtxSetCurrent)) == 0) {
    return (void*)(&hipCtxSetCurrent);
  } else if (strcmp(symbol, CUDA_SYMBOL_STRING(hipCtxDestroy)) == 0) {
    return (void*)(&hipCtxDestroy);
  }
  return (real_dlsym(handle, symbol));
}

/*
** If the user of this library does not wish to include CUDA specific
*code/headers in the code,
** then all the parameters can be changed and/or simply casted before calling
*the callback.
*/
#define CU_HOOK_GENERATE_INTERCEPT(hooksymbol, funcname, params, ...)        \
  hipError_t CUDAAPI funcname params {                                         \
    static void* real_func =                                                 \
        (void*)real_dlsym(RTLD_NEXT, CUDA_SYMBOL_STRING(funcname));          \
    hipError_t result = hipSuccess;                                          \
                                                                             \
    if (cuhl.bDebugEnabled) {                                                \
      cuhl.hookedFunctionCalls[hooksymbol]++;                                \
    }                                                                        \
    if (cuhl.preHooks[hooksymbol]) {                                         \
      ((hipError_t CUDAAPI(*) params)cuhl.preHooks[hooksymbol])(__VA_ARGS__);  \
    }                                                                        \
    result = ((hipError_t CUDAAPI(*) params)real_func)(__VA_ARGS__);           \
    if (cuhl.postHooks[hooksymbol] && result == hipSuccess) {              \
      ((hipError_t CUDAAPI(*) params)cuhl.postHooks[hooksymbol])(__VA_ARGS__); \
    }                                                                        \
    return (result);                                                         \
  }

CU_HOOK_GENERATE_INTERCEPT(CU_HOOK_MEM_ALLOC, hipMalloc,
                           (hipDeviceptr_t * dptr, size_t bytesize), dptr,
                           bytesize)
CU_HOOK_GENERATE_INTERCEPT(CU_HOOK_MEM_FREE, hipFree, (hipDeviceptr_t dptr),
                           dptr)
CU_HOOK_GENERATE_INTERCEPT(CU_HOOK_CTX_GET_CURRENT, hipCtxGetCurrent,
                           (hipCtx_t * pctx), pctx)
CU_HOOK_GENERATE_INTERCEPT(CU_HOOK_CTX_SET_CURRENT, hipCtxSetCurrent,
                           (hipCtx_t ctx), ctx)
CU_HOOK_GENERATE_INTERCEPT(CU_HOOK_CTX_DESTROY, hipCtxDestroy, (hipCtx_t ctx),
                           ctx)
