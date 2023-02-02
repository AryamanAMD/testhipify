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
 * using CUBLAS and CUSPARSE
 *
 */

// includes, system
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Using updated (v2) interfaces to cublas and cusparse */
#include <hipblas.h>
#include <hip/hip_runtime.h>
#include <hipsparse.h>

// Utilities and system includes
#include "helper_cuda_hipified.h"  // helper function CUDA error checking and initialization
#include "helper_functions.h"  // helper for shared functions common to CUDA Samples

const char *sSDKname = "conjugateGradientUM";

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

int main(int argc, char **argv) {
  int N = 0, nz = 0, *I = NULL, *J = NULL;
  float *val = NULL;
  const float tol = 1e-5f;
  const int max_iter = 10000;
  float *x;
  float *rhs;
  float a, b, na, r0, r1;
  float dot;
  float *r, *p, *Ax;
  int k;
  float alpha, beta, alpham1;

  printf("Starting [%s]...\n", sSDKname);

  // This will pick the best possible CUDA capable device
  hipDeviceProp_t deviceProp;
  int devID = findCudaDevice(argc, (const char **)argv);
  checkCudaErrors(hipGetDeviceProperties(&deviceProp, devID));

  if (!deviceProp.managedMemory) {
    // This samples requires being run on a device that supports Unified Memory
    fprintf(stderr, "Unified Memory not supported on this device\n");
    exit(EXIT_WAIVED);
  }

  // Statistics about the GPU device
  printf(
      "> GPU device has %d Multi-Processors, SM %d.%d compute capabilities\n\n",
      deviceProp.multiProcessorCount, deviceProp.major, deviceProp.minor);

  /* Generate a random tridiagonal symmetric matrix in CSR format */
  N = 1048576;
  nz = (N - 2) * 3 + 4;

  hipMallocManaged((void **)&I, sizeof(int) * (N + 1));
  hipMallocManaged((void **)&J, sizeof(int) * nz);
  hipMallocManaged((void **)&val, sizeof(float) * nz);

  genTridiag(I, J, val, N, nz);

  hipMallocManaged((void **)&x, sizeof(float) * N);
  hipMallocManaged((void **)&rhs, sizeof(float) * N);

  for (int i = 0; i < N; i++) {
    rhs[i] = 1.0;
    x[i] = 0.0;
  }

  /* Get handle to the CUBLAS context */
  hipblasHandle_t cublasHandle = 0;
  hipblasStatus_t hipblasStatus_t;
  hipblasStatus_t = hipblasCreate(&cublasHandle);

  checkCudaErrors(hipblasStatus_t);

  /* Get handle to the CUSPARSE context */
  hipsparseHandle_t cusparseHandle = 0;
  hipsparseStatus_t cusparseStatus;
  cusparseStatus = hipsparseCreate(&cusparseHandle);

  checkCudaErrors(cusparseStatus);

  hipsparseMatDescr_t descr = 0;
  cusparseStatus = hipsparseCreateMatDescr(&descr);

  checkCudaErrors(cusparseStatus);

  hipsparseSetMatType(descr, HIPSPARSE_MATRIX_TYPE_GENERAL);
  hipsparseSetMatIndexBase(descr, HIPSPARSE_INDEX_BASE_ZERO);

  // temp memory for CG
  checkCudaErrors(hipMallocManaged((void **)&r, N * sizeof(float)));
  checkCudaErrors(hipMallocManaged((void **)&p, N * sizeof(float)));
  checkCudaErrors(hipMallocManaged((void **)&Ax, N * sizeof(float)));

  /* Wrap raw data into cuSPARSE generic API objects */
  hipsparseSpMatDescr_t matA = NULL;
  checkCudaErrors(hipsparseCreateCsr(&matA, N, N, nz, I, J, val,
                                    HIPSPARSE_INDEX_32I, HIPSPARSE_INDEX_32I,
                                    HIPSPARSE_INDEX_BASE_ZERO, HIPBLAS_R_32F));
  hipsparseDnVecDescr_t vecx = NULL;
  checkCudaErrors(hipsparseCreateDnVec(&vecx, N, x, HIPBLAS_R_32F));
  hipsparseDnVecDescr_t vecp = NULL;
  checkCudaErrors(hipsparseCreateDnVec(&vecp, N, p, HIPBLAS_R_32F));
  hipsparseDnVecDescr_t vecAx = NULL;
  checkCudaErrors(hipsparseCreateDnVec(&vecAx, N, Ax, HIPBLAS_R_32F));

  hipDeviceSynchronize();

  for (int i = 0; i < N; i++) {
    r[i] = rhs[i];
  }

  alpha = 1.0;
  alpham1 = -1.0;
  beta = 0.0;
  r0 = 0.;

  /* Allocate workspace for cuSPARSE */
  size_t bufferSize = 0;
  checkCudaErrors(hipsparseSpMV_bufferSize(
      cusparseHandle, HIPSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vecx,
      &beta, vecAx, HIPBLAS_R_32F, HIPSPARSE_SPMV_ALG_DEFAULT, &bufferSize));
  void *buffer = NULL;
  checkCudaErrors(hipMalloc(&buffer, bufferSize));

  checkCudaErrors(hipsparseSpMV(cusparseHandle, HIPSPARSE_OPERATION_NON_TRANSPOSE,
                               &alpha, matA, vecx, &beta, vecAx, HIPBLAS_R_32F,
                               HIPSPARSE_SPMV_ALG_DEFAULT, buffer));

  hipblasSaxpy(cublasHandle, N, &alpham1, Ax, 1, r, 1);
  hipblasStatus_t = hipblasSdot(cublasHandle, N, r, 1, r, 1, &r1);

  k = 1;

  while (r1 > tol * tol && k <= max_iter) {
    if (k > 1) {
      b = r1 / r0;
      hipblasStatus_t = hipblasSscal(cublasHandle, N, &b, p, 1);
      hipblasStatus_t = hipblasSaxpy(cublasHandle, N, &alpha, r, 1, p, 1);
    } else {
      hipblasStatus_t = hipblasScopy(cublasHandle, N, r, 1, p, 1);
    }

    checkCudaErrors(hipsparseSpMV(
        cusparseHandle, HIPSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vecp,
        &beta, vecAx, HIPBLAS_R_32F, HIPSPARSE_SPMV_ALG_DEFAULT, buffer));
    hipblasStatus_t = hipblasSdot(cublasHandle, N, p, 1, Ax, 1, &dot);
    a = r1 / dot;

    hipblasStatus_t = hipblasSaxpy(cublasHandle, N, &a, p, 1, x, 1);
    na = -a;
    hipblasStatus_t = hipblasSaxpy(cublasHandle, N, &na, Ax, 1, r, 1);

    r0 = r1;
    hipblasStatus_t = hipblasSdot(cublasHandle, N, r, 1, r, 1, &r1);
    hipDeviceSynchronize();
    printf("iteration = %3d, residual = %e\n", k, sqrt(r1));
    k++;
  }

  printf("Final residual: %e\n", sqrt(r1));

  fprintf(stdout, "&&&& conjugateGradientUM %s\n",
          (sqrt(r1) < tol) ? "PASSED" : "FAILED");

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

  hipsparseDestroy(cusparseHandle);
  hipblasDestroy(cublasHandle);
  if (matA) {
    checkCudaErrors(hipsparseDestroySpMat(matA));
  }
  if (vecx) {
    checkCudaErrors(hipsparseDestroyDnVec(vecx));
  }
  if (vecAx) {
    checkCudaErrors(hipsparseDestroyDnVec(vecAx));
  }
  if (vecp) {
    checkCudaErrors(hipsparseDestroyDnVec(vecp));
  }

  hipFree(I);
  hipFree(J);
  hipFree(val);
  hipFree(x);
  hipFree(rhs);
  hipFree(r);
  hipFree(p);
  hipFree(Ax);

  printf("Test Summary:  Error amount = %f, result = %s\n", err,
         (k <= max_iter) ? "SUCCESS" : "FAILURE");
  exit((k <= max_iter) ? EXIT_SUCCESS : EXIT_FAILURE);
}
