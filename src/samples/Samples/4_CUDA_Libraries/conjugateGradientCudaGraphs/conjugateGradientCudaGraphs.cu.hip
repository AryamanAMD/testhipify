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
 * This sample implements a conjugate gradient solver on GPU
 * using CUBLAS and CUSPARSE with CUDA Graphs
 *
 */

// includes, system
#include <stdio.h>
#include "rocprofiler.h"
#include "HIPCHECK.h"
#include <stdlib.h>
#include <string.h>

/* Using updated (v2) interfaces to cublas */
#include <hipblas.h>
#include <hip/hip_runtime.h>
#include <hipsparse.h>

// Utilities and system includes
#include "helper_cuda_hipified.h"  // helper function CUDA error checking and initialization
#include "helper_functions.h"  // helper for shared functions common to CUDA Samples

const char *sSDKname = "conjugateGradientCudaGraphs";

#ifndef WITH_GRAPH
#define WITH_GRAPH 1
#endif

/* genTridiag: generate a random tridiagonal symmetric matrix */
void genTridiag(int *I, int *J, float *val, int N, int nz) {
  I[0] = 0, J[0] = 0, J[1] = 1;
  val[0] = (float)rand() / RAND_MAX + 10.0f;
  val[1] = (float)rand() / RAND_MAX;
  int start;

  for (int i = 1; i < N; i++) {
    if (i > 1) {
      I[i] = I[i - 1] + 3;
    } else {
      I[1] = 2;
    }

    start = (i - 1) * 3 + 2;
    J[start] = i - 1;
    J[start + 1] = i;

    if (i < N - 1) {
      J[start + 2] = i + 1;
    }

    val[start] = val[start - 1];
    val[start + 1] = (float)rand() / RAND_MAX + 10.0f;

    if (i < N - 1) {
      val[start + 2] = (float)rand() / RAND_MAX;
    }
  }

  I[N] = nz;
}

__global__ void initVectors(float *rhs, float *x, int N) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;

  for (size_t i = gid; i < N; i += gridDim.x * blockDim.x) {
    rhs[i] = 1.0;
    x[i] = 0.0;
  }
}

__global__ void r1_div_x(float *r1, float *r0, float *b) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid == 0) {
    b[0] = r1[0] / r0[0];
  }
}

__global__ void a_minus(float *a, float *na) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid == 0) {
    na[0] = -(a[0]);
  }
}

int main(int argc, char **argv) {
  int N = 0, nz = 0, *I = NULL, *J = NULL;
  float *val = NULL;
  const float tol = 1e-5f;
  const int max_iter = 10000;
  float *x;
  float *rhs;
  float r1;

  int *d_col, *d_row;
  float *d_val, *d_x;
  float *d_r, *d_p, *d_Ax;
  int k;
  float alpha, beta, alpham1;

  hipStream_t stream1, streamForGraph;

  // This will pick the best possible CUDA capable device
  hipDeviceProp_t deviceProp;
  int devID = findCudaDevice(argc, (const char **)argv);

  if (devID < 0) {
    printf("exiting...\n");
    exit(EXIT_SUCCESS);
  }

  HIPCHECK(hipGetDeviceProperties(&deviceProp, devID));

  // Statistics about the GPU device
  printf(
      "> GPU device has %d Multi-Processors, SM %d.%d compute capabilities\n\n",
      deviceProp.multiProcessorCount, deviceProp.major, deviceProp.minor);

  /* Generate a random tridiagonal symmetric matrix in CSR format */
  N = 1048576;
  nz = (N - 2) * 3 + 4;
  HIPCHECK(hipHostMalloc(&I, sizeof(int) * (N + 1)));
  HIPCHECK(hipHostMalloc(&J, sizeof(int) * nz));
  HIPCHECK(hipHostMalloc(&val, sizeof(float) * nz));
  genTridiag(I, J, val, N, nz);

  HIPCHECK(hipHostMalloc(&x, sizeof(float) * N));
  rhs = (float *)malloc(sizeof(float) * N);

  for (int i = 0; i < N; i++) {
    rhs[i] = 1.0;
    x[i] = 0.0;
  }

  /* Get handle to the CUBLAS context */
  hipblasHandle_t cublasHandle = 0;
  hipblasStatus_t hipblasStatus_t;
  hipblasStatus_t = hipblasCreate(&cublasHandle);

  HIPCHECK(hipblasStatus_t);

  /* Get handle to the CUSPARSE context */
  hipsparseHandle_t cusparseHandle = 0;
  hipsparseStatus_t cusparseStatus;
  cusparseStatus = hipsparseCreate(&cusparseHandle);

  HIPCHECK(cusparseStatus);

  HIPCHECK(hipStreamCreate(&stream1));

  HIPCHECK(hipMalloc((void **)&d_col, nz * sizeof(int)));
  HIPCHECK(hipMalloc((void **)&d_row, (N + 1) * sizeof(int)));
  HIPCHECK(hipMalloc((void **)&d_val, nz * sizeof(float)));
  HIPCHECK(hipMalloc((void **)&d_x, N * sizeof(float)));
  HIPCHECK(hipMalloc((void **)&d_r, N * sizeof(float)));
  HIPCHECK(hipMalloc((void **)&d_p, N * sizeof(float)));
  HIPCHECK(hipMalloc((void **)&d_Ax, N * sizeof(float)));

  float *d_r1, *d_r0, *d_dot, *d_a, *d_na, *d_b;
  HIPCHECK(hipMalloc((void **)&d_r1, sizeof(float)));
  HIPCHECK(hipMalloc((void **)&d_r0, sizeof(float)));
  HIPCHECK(hipMalloc((void **)&d_dot, sizeof(float)));
  HIPCHECK(hipMalloc((void **)&d_a, sizeof(float)));
  HIPCHECK(hipMalloc((void **)&d_na, sizeof(float)));
  HIPCHECK(hipMalloc((void **)&d_b, sizeof(float)));

  /* Wrap raw data into cuSPARSE generic API objects */
  hipsparseSpMatDescr_t matA = NULL;
  HIPCHECK(hipsparseCreateCsr(&matA, N, N, nz, d_row, d_col, d_val,
                                    HIPSPARSE_INDEX_32I, HIPSPARSE_INDEX_32I,
                                    HIPSPARSE_INDEX_BASE_ZERO, HIPBLAS_R_32F));
  hipsparseDnVecDescr_t vecx = NULL;
  HIPCHECK(hipsparseCreateDnVec(&vecx, N, d_x, HIPBLAS_R_32F));
  hipsparseDnVecDescr_t vecp = NULL;
  HIPCHECK(hipsparseCreateDnVec(&vecp, N, d_p, HIPBLAS_R_32F));
  hipsparseDnVecDescr_t vecAx = NULL;
  HIPCHECK(hipsparseCreateDnVec(&vecAx, N, d_Ax, HIPBLAS_R_32F));

  /* Allocate workspace for cuSPARSE */
  size_t bufferSize = 0;
  HIPCHECK(hipsparseSpMV_bufferSize(
      cusparseHandle, HIPSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vecx,
      &beta, vecAx, HIPBLAS_R_32F, HIPSPARSE_SPMV_ALG_DEFAULT, &bufferSize));
  void *buffer = NULL;
  HIPCHECK(hipMalloc(&buffer, bufferSize));

  hipsparseMatDescr_t descr = 0;
  HIPCHECK(hipsparseCreateMatDescr(&descr));

  HIPCHECK(hipsparseSetMatType(descr, HIPSPARSE_MATRIX_TYPE_GENERAL));
  HIPCHECK(hipsparseSetMatIndexBase(descr, HIPSPARSE_INDEX_BASE_ZERO));

  int numBlocks = 0, blockSize = 0;
  HIPCHECK(
      hipOccupancyMaxPotentialBlockSize(&numBlocks, &blockSize, initVectors));

  HIPCHECK(hipMemcpyAsync(d_col, J, nz * sizeof(int),
                                  hipMemcpyHostToDevice, stream1));
  HIPCHECK(hipMemcpyAsync(d_row, I, (N + 1) * sizeof(int),
                                  hipMemcpyHostToDevice, stream1));
  HIPCHECK(hipMemcpyAsync(d_val, val, nz * sizeof(float),
                                  hipMemcpyHostToDevice, stream1));

  initVectors<<<numBlocks, blockSize, 0, stream1>>>(d_r, d_x, N);

  alpha = 1.0;
  alpham1 = -1.0;
  beta = 0.0;

  HIPCHECK(hipsparseSetStream(cusparseHandle, stream1));
  HIPCHECK(hipsparseSpMV(cusparseHandle, HIPSPARSE_OPERATION_NON_TRANSPOSE,
                               &alpha, matA, vecx, &beta, vecAx, HIPBLAS_R_32F,
                               HIPSPARSE_SPMV_ALG_DEFAULT, buffer));

  HIPCHECK(hipblasSetStream(cublasHandle, stream1));
  HIPCHECK(hipblasSaxpy(cublasHandle, N, &alpham1, d_Ax, 1, d_r, 1));

  HIPCHECK(
      hipblasSetPointerMode(cublasHandle, HIPBLAS_POINTER_MODE_DEVICE));
  HIPCHECK(hipblasSdot(cublasHandle, N, d_r, 1, d_r, 1, d_r1));

  k = 1;
  // First Iteration when k=1 starts
  HIPCHECK(hipblasScopy(cublasHandle, N, d_r, 1, d_p, 1));
  HIPCHECK(hipsparseSpMV(cusparseHandle, HIPSPARSE_OPERATION_NON_TRANSPOSE,
                               &alpha, matA, vecp, &beta, vecAx, HIPBLAS_R_32F,
                               HIPSPARSE_SPMV_ALG_DEFAULT, buffer));

  HIPCHECK(hipblasSdot(cublasHandle, N, d_p, 1, d_Ax, 1, d_dot));

  r1_div_x<<<1, 1, 0, stream1>>>(d_r1, d_dot, d_a);

  HIPCHECK(hipblasSaxpy(cublasHandle, N, d_a, d_p, 1, d_x, 1));

  a_minus<<<1, 1, 0, stream1>>>(d_a, d_na);

  HIPCHECK(hipblasSaxpy(cublasHandle, N, d_na, d_Ax, 1, d_r, 1));

  HIPCHECK(hipMemcpyAsync(d_r0, d_r1, sizeof(float),
                                  hipMemcpyDeviceToDevice, stream1));

  HIPCHECK(hipblasSdot(cublasHandle, N, d_r, 1, d_r, 1, d_r1));

  HIPCHECK(hipMemcpyAsync(&r1, d_r1, sizeof(float),
                                  hipMemcpyDeviceToHost, stream1));
  HIPCHECK(hipStreamSynchronize(stream1));
  printf("iteration = %3d, residual = %e\n", k, sqrt(r1));
  // First Iteration when k=1 ends
  k++;

#if WITH_GRAPH
  hipGraph_t initGraph;
  HIPCHECK(hipStreamCreate(&streamForGraph));
  HIPCHECK(hipblasSetStream(cublasHandle, stream1));
  HIPCHECK(hipsparseSetStream(cusparseHandle, stream1));
  HIPCHECK(hipStreamBeginCapture(stream1, hipStreamCaptureModeGlobal));

  r1_div_x<<<1, 1, 0, stream1>>>(d_r1, d_r0, d_b);
  hipblasSetPointerMode(cublasHandle, HIPBLAS_POINTER_MODE_DEVICE);
  HIPCHECK(hipblasSscal(cublasHandle, N, d_b, d_p, 1));
  hipblasSetPointerMode(cublasHandle, HIPBLAS_POINTER_MODE_HOST);
  HIPCHECK(hipblasSaxpy(cublasHandle, N, &alpha, d_r, 1, d_p, 1));
  hipblasSetPointerMode(cublasHandle, HIPBLAS_POINTER_MODE_DEVICE);

  HIPCHECK(
      hipsparseSetPointerMode(cusparseHandle, HIPSPARSE_POINTER_MODE_HOST));
  HIPCHECK(hipsparseSpMV(cusparseHandle, HIPSPARSE_OPERATION_NON_TRANSPOSE,
                               &alpha, matA, vecp, &beta, vecAx, HIPBLAS_R_32F,
                               HIPSPARSE_SPMV_ALG_DEFAULT, buffer));

  HIPCHECK(hipMemsetAsync(d_dot, 0, sizeof(float), stream1));
  HIPCHECK(hipblasSdot(cublasHandle, N, d_p, 1, d_Ax, 1, d_dot));

  r1_div_x<<<1, 1, 0, stream1>>>(d_r1, d_dot, d_a);

  HIPCHECK(hipblasSaxpy(cublasHandle, N, d_a, d_p, 1, d_x, 1));

  a_minus<<<1, 1, 0, stream1>>>(d_a, d_na);

  HIPCHECK(hipblasSaxpy(cublasHandle, N, d_na, d_Ax, 1, d_r, 1));

  HIPCHECK(hipMemcpyAsync(d_r0, d_r1, sizeof(float),
                                  hipMemcpyDeviceToDevice, stream1));
  HIPCHECK(hipMemsetAsync(d_r1, 0, sizeof(float), stream1));

  HIPCHECK(hipblasSdot(cublasHandle, N, d_r, 1, d_r, 1, d_r1));

  HIPCHECK(hipMemcpyAsync((float *)&r1, d_r1, sizeof(float),
                                  hipMemcpyDeviceToHost, stream1));

  HIPCHECK(hipStreamEndCapture(stream1, &initGraph));
  hipGraphExec_t graphExec;
  HIPCHECK(hipGraphInstantiate(&graphExec, initGraph, NULL, NULL, 0));
#endif

  HIPCHECK(hipblasSetStream(cublasHandle, stream1));
  HIPCHECK(hipsparseSetStream(cusparseHandle, stream1));

  while (r1 > tol * tol && k <= max_iter) {
#if WITH_GRAPH
    HIPCHECK(hipGraphLaunch(graphExec, streamForGraph));
    HIPCHECK(hipStreamSynchronize(streamForGraph));
#else
    r1_div_x<<<1, 1, 0, stream1>>>(d_r1, d_r0, d_b);
    hipblasSetPointerMode(cublasHandle, HIPBLAS_POINTER_MODE_DEVICE);
    HIPCHECK(hipblasSscal(cublasHandle, N, d_b, d_p, 1));

    hipblasSetPointerMode(cublasHandle, HIPBLAS_POINTER_MODE_HOST);
    HIPCHECK(hipblasSaxpy(cublasHandle, N, &alpha, d_r, 1, d_p, 1));

    HIPCHECK(hipsparseSpMV(
        cusparseHandle, HIPSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vecp,
        &beta, vecAx, HIPBLAS_R_32F, HIPSPARSE_SPMV_ALG_DEFAULT, buffer));

    hipblasSetPointerMode(cublasHandle, HIPBLAS_POINTER_MODE_DEVICE);
    HIPCHECK(hipblasSdot(cublasHandle, N, d_p, 1, d_Ax, 1, d_dot));

    r1_div_x<<<1, 1, 0, stream1>>>(d_r1, d_dot, d_a);

    HIPCHECK(hipblasSaxpy(cublasHandle, N, d_a, d_p, 1, d_x, 1));

    a_minus<<<1, 1, 0, stream1>>>(d_a, d_na);
    HIPCHECK(hipblasSaxpy(cublasHandle, N, d_na, d_Ax, 1, d_r, 1));

    HIPCHECK(hipMemcpyAsync(d_r0, d_r1, sizeof(float),
                                    hipMemcpyDeviceToDevice, stream1));

    HIPCHECK(hipblasSdot(cublasHandle, N, d_r, 1, d_r, 1, d_r1));
    HIPCHECK(hipMemcpyAsync((float *)&r1, d_r1, sizeof(float),
                                    hipMemcpyDeviceToHost, stream1));
    HIPCHECK(hipStreamSynchronize(stream1));
#endif
    printf("iteration = %3d, residual = %e\n", k, sqrt(r1));
    k++;
  }

#if WITH_GRAPH
  HIPCHECK(hipMemcpyAsync(x, d_x, N * sizeof(float),
                                  hipMemcpyDeviceToHost, streamForGraph));
  HIPCHECK(hipStreamSynchronize(streamForGraph));
#else
  HIPCHECK(hipMemcpyAsync(x, d_x, N * sizeof(float),
                                  hipMemcpyDeviceToHost, stream1));
  HIPCHECK(hipStreamSynchronize(stream1));
#endif

  float rsum, diff, err = 0.0;

  for (int i = 0; i < N; i++) {
    rsum = 0.0;

    for (int j = I[i]; j < I[i + 1]; j++) {
      rsum += val[j] * x[J[j]];
    }

    diff = fabs(rsum - rhs[i]);

    if (diff > err) {
      err = diff;
    }
  }

#if WITH_GRAPH
  HIPCHECK(hipGraphExecDestroy(graphExec));
  HIPCHECK(hipGraphDestroy(initGraph));
  HIPCHECK(hipStreamDestroy(streamForGraph));
#endif
  HIPCHECK(hipStreamDestroy(stream1));
  hipsparseDestroy(cusparseHandle);
  hipblasDestroy(cublasHandle);

  if (matA) {
    HIPCHECK(hipsparseDestroySpMat(matA));
  }
  if (vecx) {
    HIPCHECK(hipsparseDestroyDnVec(vecx));
  }
  if (vecAx) {
    HIPCHECK(hipsparseDestroyDnVec(vecAx));
  }
  if (vecp) {
    HIPCHECK(hipsparseDestroyDnVec(vecp));
  }

  HIPCHECK(hipHostFree(I));
  HIPCHECK(hipHostFree(J));
  HIPCHECK(hipHostFree(val));
  HIPCHECK(hipHostFree(x));
  free(rhs);
  HIPCHECK(hipFree(d_col));
  HIPCHECK(hipFree(d_row));
  HIPCHECK(hipFree(d_val));
  HIPCHECK(hipFree(d_x));
  HIPCHECK(hipFree(d_r));
  HIPCHECK(hipFree(d_p));
  HIPCHECK(hipFree(d_Ax));

  printf("Test Summary:  Error amount = %f\n", err);
  exit((k <= max_iter) ? 0 : 1);
}
