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
 * This sample implements a preconditioned conjugate gradient solver on
 * the GPU using CUBLAS and CUSPARSE.  Relative to the conjugateGradient
 * SDK example, this demonstrates the use of hipsparseScsrilu02() for
 * computing the incompute-LU preconditioner and hipsparseScsrsv2_solve()
 * for solving triangular systems.  Specifically, the preconditioned
 * conjugate gradient method with an incomplete LU preconditioner is
 * used to solve the Laplacian operator in 2D on a uniform mesh.
 *
 * Note that the code in this example and the specific matrices used here
 * were chosen to demonstrate the use of the CUSPARSE library as simply
 * and as clearly as possible.  This is not optimized code and the input
 * matrices have been chosen for simplicity rather than performance.
 * These should not be used either as a performance guide or for
 * benchmarking purposes.
 */


// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// CUDA Runtime
#include <hip/hip_runtime.h>

// Using updated (v2) interfaces for CUBLAS and CUSPARSE
#include <hipsparse.h>
#include <hipblas.h>

// Utilities and system includes
#include "helper_functions.h"  // shared functions common to CUDA Samples
#include "helper_cuda_hipified.h"       // CUDA error checking

const char *sSDKname     = "conjugateGradientPrecond";

/*
 * Generate a matrix representing a second order regular Laplacian operator
 * on a 2D domain in Compressed Sparse Row format.
 */
void genLaplace(int *row_ptr, int *col_ind, float *val, int M, int N, int nz,
                float *rhs)
{
    assert(M==N);
    int n=(int)sqrt((double)N);
    assert(n*n==N);
    printf("laplace dimension = %d\n", n);
    int idx = 0;

    // loop over degrees of freedom
    for (int i = 0; i < N; i++)
    {
        int ix = i % n;
        int iy = i / n;

        row_ptr[i] = idx;

        // up
        if (iy > 0)
        {
            val[idx] = 1.0;
            col_ind[idx] = i - n;
            idx++;
        }
        else
        {
            rhs[i] -= 1.0;
        }

        // left
        if (ix > 0) {
            val[idx] = 1.0;
            col_ind[idx] = i - 1;
            idx++;
        } else {
            rhs[i] -= 0.0;
        }

        // center
        val[idx] = -4.0;
        col_ind[idx] = i;
        idx++;

        //right
        if (ix  < n - 1)
        {
            val[idx] = 1.0;
            col_ind[idx] = i + 1;
            idx++;
        }
        else
        {
            rhs[i] -= 0.0;
        }

        // down
        if (iy  < n - 1)
        {
            val[idx] = 1.0;
            col_ind[idx] = i + n;
            idx++;
        }
        else
        {
            rhs[i] -= 0.0;
        }

    }

    row_ptr[N] = idx;

}

/*
 * Solve Ax=b using the conjugate gradient method
 * a) without any preconditioning,
 * b) using an Incomplete Cholesky preconditioner, and
 * c) using an ILU0 preconditioner.
 */
int main(int argc, char **argv){
    const int max_iter = 1000;
    int k, M = 0, N = 0, nz = 0, *I = NULL, *J = NULL;
    int *d_col, *d_row;
    int qatest = 0;
    const float tol = 1e-12f;
    float *x, *rhs;
    float r0, r1, alpha, beta;
    float *d_val, *d_x;
    float *d_zm1, *d_zm2, *d_rm2;
    float *d_r, *d_p, *d_omega, *d_y;
    float *val = NULL;
    float *d_valsILU0;
    float rsum, diff, err = 0.0;
    float qaerr1, qaerr2 = 0.0;
    float dot, numerator, denominator, nalpha;
    const float floatone = 1.0;
    const float floatzero = 0.0;

    int nErrors = 0;

    printf("conjugateGradientPrecond starting...\n");

      /* QA testing mode */
  if (checkCmdLineFlag(argc, (const char **)argv, "qatest")) {
    qatest = 1;
  }

    /* This will pick the best possible CUDA capable device */
    hipDeviceProp_t deviceProp;
    int devID = findCudaDevice(argc, (const char **)argv);
    printf("GPU selected Device ID = %d \n", devID);

    if (devID < 0)
    {
        printf("Invalid GPU device %d selected,  exiting...\n", devID);
        exit(EXIT_SUCCESS);
    }

    checkCudaErrors(hipGetDeviceProperties(&deviceProp, devID));

    /* Statistics about the GPU device */
    printf("> GPU device has %d Multi-Processors, SM %d.%d compute capabilities\n\n",
           deviceProp.multiProcessorCount, deviceProp.major, deviceProp.minor);

    /* Generate a Laplace matrix in CSR (Compressed Sparse Row) format */
    M = N = 16384;
    nz = 5 * N - 4 * (int)sqrt((double)N);
    I = (int *)malloc(sizeof(int) * (N + 1));   // csr row pointers for matrix A
    J = (int *)malloc(sizeof(int) * nz);       // csr column indices for matrix A
    val = (float *)malloc(sizeof(float) * nz); // csr values for matrix A
    x = (float *)malloc(sizeof(float) * N);
    rhs = (float *)malloc(sizeof(float) * N);

    for (int i = 0; i < N; i++)
    {
        rhs[i] = 0.0;  // Initialize RHS
        x[i] = 0.0;    // Initial solution approximation
    }

    genLaplace(I, J, val, M, N, nz, rhs);

    /* Create CUBLAS context */
    hipblasHandle_t cublasHandle = NULL;
    checkCudaErrors(hipblasCreate(&cublasHandle));

    /* Create CUSPARSE context */
    hipsparseHandle_t cusparseHandle = NULL;
    checkCudaErrors(hipsparseCreate(&cusparseHandle));

    /* Description of the A matrix */
    hipsparseMatDescr_t descr = 0;
    checkCudaErrors(hipsparseCreateMatDescr(&descr));
    checkCudaErrors(hipsparseSetMatType(descr, HIPSPARSE_MATRIX_TYPE_GENERAL));
    checkCudaErrors(hipsparseSetMatIndexBase(descr, HIPSPARSE_INDEX_BASE_ZERO));

    /* Allocate required memory */
    checkCudaErrors(hipMalloc((void **)&d_col, nz * sizeof(int)));
    checkCudaErrors(hipMalloc((void **)&d_row, (N + 1) * sizeof(int)));
    checkCudaErrors(hipMalloc((void **)&d_val, nz * sizeof(float)));
    checkCudaErrors(hipMalloc((void **)&d_x, N * sizeof(float)));
    checkCudaErrors(hipMalloc((void **)&d_y, N * sizeof(float)));
    checkCudaErrors(hipMalloc((void **)&d_r, N * sizeof(float)));
    checkCudaErrors(hipMalloc((void **)&d_p, N * sizeof(float)));
    checkCudaErrors(hipMalloc((void **)&d_omega, N * sizeof(float)));
    checkCudaErrors(hipMalloc((void **)&d_valsILU0, nz * sizeof(float)));
    checkCudaErrors(hipMalloc((void **)&d_zm1, (N) * sizeof(float)));
    checkCudaErrors(hipMalloc((void **)&d_zm2, (N) * sizeof(float)));
    checkCudaErrors(hipMalloc((void **)&d_rm2, (N) * sizeof(float)));

    /* Wrap raw data into cuSPARSE generic API objects */
    hipsparseDnVecDescr_t vecp = NULL, vecX=NULL, vecY = NULL, vecR = NULL, vecZM1=NULL;
    checkCudaErrors(hipsparseCreateDnVec(&vecp, N, d_p, HIPBLAS_R_32F));
    checkCudaErrors(hipsparseCreateDnVec(&vecX, N, d_x, HIPBLAS_R_32F));
    checkCudaErrors(hipsparseCreateDnVec(&vecY, N, d_y, HIPBLAS_R_32F));
    checkCudaErrors(hipsparseCreateDnVec(&vecR, N, d_r, HIPBLAS_R_32F));
    checkCudaErrors(hipsparseCreateDnVec(&vecZM1, N, d_zm1, HIPBLAS_R_32F));
    hipsparseDnVecDescr_t vecomega = NULL;
    checkCudaErrors(hipsparseCreateDnVec(&vecomega, N, d_omega, HIPBLAS_R_32F));

    /* Initialize problem data */
    checkCudaErrors(hipMemcpy(
        d_col, J, nz * sizeof(int), hipMemcpyHostToDevice));
    checkCudaErrors(hipMemcpy(
        d_row, I, (N + 1) * sizeof(int), hipMemcpyHostToDevice));
    checkCudaErrors(hipMemcpy(
        d_val, val, nz * sizeof(float), hipMemcpyHostToDevice));
    checkCudaErrors(hipMemcpy(
        d_val, val, nz * sizeof(float), hipMemcpyHostToDevice));
    checkCudaErrors(hipMemcpy(
        d_x, x, N*sizeof(float), hipMemcpyHostToDevice));
    checkCudaErrors(hipMemcpy(
        d_r, rhs, N * sizeof(float), hipMemcpyHostToDevice));

    hipsparseSpMatDescr_t matA = NULL;
    hipsparseSpMatDescr_t matM_lower, matM_upper;
    hipsparseFillMode_t   fill_lower    = HIPSPARSE_FILL_MODE_LOWER;
    hipsparseDiagType_t   diag_unit     = HIPSPARSE_DIAG_TYPE_UNIT;
    hipsparseFillMode_t   fill_upper    = HIPSPARSE_FILL_MODE_UPPER;
    hipsparseDiagType_t   diag_non_unit = HIPSPARSE_DIAG_TYPE_NON_UNIT;

    checkCudaErrors(hipsparseCreateCsr(
        &matA, N, N, nz, d_row, d_col, d_val, HIPSPARSE_INDEX_32I,
        HIPSPARSE_INDEX_32I, HIPSPARSE_INDEX_BASE_ZERO, HIPBLAS_R_32F));

    /* Copy A data to ILU(0) vals as input*/
    checkCudaErrors(hipMemcpy(
        d_valsILU0, d_val, nz*sizeof(float), hipMemcpyDeviceToDevice));
    
    //Lower Part 
     checkCudaErrors( hipsparseCreateCsr(&matM_lower, N, N, nz, d_row, d_col, d_valsILU0,
                                      HIPSPARSE_INDEX_32I, HIPSPARSE_INDEX_32I,
                                      HIPSPARSE_INDEX_BASE_ZERO, HIPBLAS_R_32F) );

    checkCudaErrors( hipsparseSpMatSetAttribute(matM_lower,
                                              HIPSPARSE_SPMAT_FILL_MODE,
                                              &fill_lower, sizeof(fill_lower)) );
    checkCudaErrors( hipsparseSpMatSetAttribute(matM_lower,
                                              HIPSPARSE_SPMAT_DIAG_TYPE,
                                              &diag_unit, sizeof(diag_unit)) );
    // M_upper
    checkCudaErrors( hipsparseCreateCsr(&matM_upper, N, N, nz, d_row, d_col, d_valsILU0,
                                      HIPSPARSE_INDEX_32I, HIPSPARSE_INDEX_32I,
                                      HIPSPARSE_INDEX_BASE_ZERO, HIPBLAS_R_32F) );
    checkCudaErrors( hipsparseSpMatSetAttribute(matM_upper,
                                              HIPSPARSE_SPMAT_FILL_MODE,
                                              &fill_upper, sizeof(fill_upper)) );
    checkCudaErrors( hipsparseSpMatSetAttribute(matM_upper,
                                              HIPSPARSE_SPMAT_DIAG_TYPE,
                                              &diag_non_unit,
                                              sizeof(diag_non_unit)) );


    /* Create ILU(0) info object */
    int                 bufferSizeLU = 0;
    size_t              bufferSizeMV, bufferSizeL, bufferSizeU;
    void*               d_bufferLU, *d_bufferMV,  *d_bufferL, *d_bufferU;
    hipsparseSpSVDescr_t spsvDescrL, spsvDescrU;
    hipsparseMatDescr_t   matLU;
    csrilu02Info_t      infoILU = NULL;

    checkCudaErrors(hipsparseCreateCsrilu02Info(&infoILU));
    checkCudaErrors( hipsparseCreateMatDescr(&matLU) );
    checkCudaErrors( hipsparseSetMatType(matLU, HIPSPARSE_MATRIX_TYPE_GENERAL) );
    checkCudaErrors( hipsparseSetMatIndexBase(matLU, HIPSPARSE_INDEX_BASE_ZERO) );

    /* Allocate workspace for cuSPARSE */
    checkCudaErrors(hipsparseSpMV_bufferSize(
        cusparseHandle, HIPSPARSE_OPERATION_NON_TRANSPOSE, &floatone, matA,
        vecp, &floatzero, vecomega, HIPBLAS_R_32F, HIPSPARSE_SPMV_ALG_DEFAULT,
        &bufferSizeMV));
    checkCudaErrors( hipMalloc(&d_bufferMV, bufferSizeMV) );

    checkCudaErrors(hipsparseScsrilu02_bufferSize(
        cusparseHandle, N, nz, matLU, d_val, d_row, d_col, infoILU, &bufferSizeLU));
    checkCudaErrors( hipMalloc(&d_bufferLU, bufferSizeLU) );

    checkCudaErrors( hipsparseSpSV_createDescr(&spsvDescrL) );
    checkCudaErrors(hipsparseSpSV_bufferSize(
        cusparseHandle, HIPSPARSE_OPERATION_NON_TRANSPOSE, &floatone, matM_lower, vecR, vecX, HIPBLAS_R_32F,
        HIPSPARSE_SPSV_ALG_DEFAULT, spsvDescrL, &bufferSizeL));
    checkCudaErrors( hipMalloc(&d_bufferL, bufferSizeL) );

    checkCudaErrors( hipsparseSpSV_createDescr(&spsvDescrU) );
    checkCudaErrors( hipsparseSpSV_bufferSize(
        cusparseHandle, HIPSPARSE_OPERATION_NON_TRANSPOSE, &floatone, matM_upper, vecR, vecX, HIPBLAS_R_32F,
        HIPSPARSE_SPSV_ALG_DEFAULT, spsvDescrU, &bufferSizeU));
    checkCudaErrors( hipMalloc(&d_bufferU, bufferSizeU) );

    /* Conjugate gradient without preconditioning.
       ------------------------------------------

       Follows the description by Golub & Van Loan,
       "Matrix Computations 3rd ed.", Section 10.2.6  */

    printf("Convergence of CG without preconditioning: \n");
    k = 0;
    r0 = 0;
    checkCudaErrors(hipblasSdot(cublasHandle, N, d_r, 1, d_r, 1, &r1));

    while (r1 > tol * tol && k <= max_iter)
    {
        k++;

        if (k == 1)
        {
            checkCudaErrors(hipblasScopy(cublasHandle, N, d_r, 1, d_p, 1));
        }
        else
        {
            beta = r1 / r0;
            checkCudaErrors(hipblasSscal(cublasHandle, N, &beta, d_p, 1));
            checkCudaErrors(hipblasSaxpy(
                cublasHandle, N, &floatone, d_r, 1, d_p, 1));
        }

        checkCudaErrors(hipsparseSpMV(
            cusparseHandle, HIPSPARSE_OPERATION_NON_TRANSPOSE, &floatone, matA,
            vecp, &floatzero, vecomega, HIPBLAS_R_32F, HIPSPARSE_SPMV_ALG_DEFAULT,
            d_bufferMV));
        checkCudaErrors(hipblasSdot(cublasHandle, N, d_p, 1, d_omega, 1, &dot));
        alpha = r1 / dot;
        checkCudaErrors(hipblasSaxpy(cublasHandle, N, &alpha, d_p, 1, d_x, 1));
        nalpha = -alpha;
        checkCudaErrors(hipblasSaxpy(
            cublasHandle, N, &nalpha, d_omega, 1, d_r, 1));
        r0 = r1;
        checkCudaErrors(hipblasSdot(cublasHandle, N, d_r, 1, d_r, 1, &r1));
    }

    printf("  iteration = %3d, residual = %e \n", k, sqrt(r1));

    checkCudaErrors(hipMemcpy(
        x, d_x, N * sizeof(float), hipMemcpyDeviceToHost));

    /* check result */
    err = 0.0;

    for (int i = 0; i < N; i++)
    {
        rsum = 0.0;

        for (int j = I[i]; j < I[i + 1]; j++)
        {
            rsum += val[j] * x[J[j]];
        }

        diff = fabs(rsum - rhs[i]);

        if (diff > err)
        {
            err = diff;
        }
    }

    printf("  Convergence Test: %s \n", (k <= max_iter) ? "OK" : "FAIL");
    nErrors += (k > max_iter) ? 1 : 0;
    qaerr1 = err;

    if (0)
    {
        // output result in matlab-style array
        int n = (int)sqrt((double)N);
        printf("a = [  ");

        for (int iy = 0; iy < n; iy++)
        {
            for (int ix = 0; ix < n; ix++)
            {
                printf(" %f ", x[iy * n + ix]);
            }

            if (iy == n - 1)
            {
                printf(" ]");
            }

            printf("\n");
        }
    }


    /* Preconditioned Conjugate Gradient using ILU.
       --------------------------------------------
       Follows the description by Golub & Van Loan,
       "Matrix Computations 3rd ed.", Algorithm 10.3.1  */

    printf("\nConvergence of CG using ILU(0) preconditioning: \n");

    /* Perform analysis for ILU(0) */
    checkCudaErrors(hipsparseScsrilu02_analysis(
        cusparseHandle, N, nz, descr, d_valsILU0, d_row, d_col, infoILU,
        HIPSPARSE_SOLVE_POLICY_USE_LEVEL, d_bufferLU));

    /* generate the ILU(0) factors */
    checkCudaErrors(hipsparseScsrilu02(
        cusparseHandle, N, nz, matLU, d_valsILU0, d_row, d_col, infoILU,
        HIPSPARSE_SOLVE_POLICY_USE_LEVEL, d_bufferLU));

    /* perform triangular solve analysis */
    checkCudaErrors(hipsparseSpSV_analysis(
        cusparseHandle, HIPSPARSE_OPERATION_NON_TRANSPOSE, &floatone,
        matM_lower, vecR, vecX, HIPBLAS_R_32F,
        HIPSPARSE_SPSV_ALG_DEFAULT, spsvDescrL, d_bufferL));

    checkCudaErrors(hipsparseSpSV_analysis(
        cusparseHandle, HIPSPARSE_OPERATION_NON_TRANSPOSE, &floatone,
        matM_upper, vecR, vecX, HIPBLAS_R_32F,
        HIPSPARSE_SPSV_ALG_DEFAULT, spsvDescrU, d_bufferU));

    /* reset the initial guess of the solution to zero */
    for (int i = 0; i < N; i++)
    {
        x[i] = 0.0;
    }
    checkCudaErrors(hipMemcpy(
        d_r, rhs, N * sizeof(float), hipMemcpyHostToDevice));
    checkCudaErrors(hipMemcpy(
        d_x, x, N * sizeof(float), hipMemcpyHostToDevice));

    k = 0;
    checkCudaErrors(hipblasSdot(cublasHandle, N, d_r, 1, d_r, 1, &r1));

    while (r1 > tol * tol && k <= max_iter)
    {
        // preconditioner application: d_zm1 = U^-1 L^-1 d_r
        checkCudaErrors(hipsparseSpSV_solve(cusparseHandle,
            HIPSPARSE_OPERATION_NON_TRANSPOSE, &floatone,
            matM_lower, vecR, vecY, HIPBLAS_R_32F,
            HIPSPARSE_SPSV_ALG_DEFAULT,
            spsvDescrL) );
            
        checkCudaErrors(hipsparseSpSV_solve(cusparseHandle,
            HIPSPARSE_OPERATION_NON_TRANSPOSE, &floatone, matM_upper,
            vecY, vecZM1,
            HIPBLAS_R_32F,
            HIPSPARSE_SPSV_ALG_DEFAULT,
            spsvDescrU));
        k++;

        if (k == 1)
        {
            checkCudaErrors(hipblasScopy(cublasHandle, N, d_zm1, 1, d_p, 1));
        }
        else
        {
            checkCudaErrors(hipblasSdot(
                cublasHandle, N, d_r, 1, d_zm1, 1, &numerator));
            checkCudaErrors(hipblasSdot(
                cublasHandle, N, d_rm2, 1, d_zm2, 1, &denominator));
            beta = numerator / denominator;
            checkCudaErrors(hipblasSscal(cublasHandle, N, &beta, d_p, 1));
            checkCudaErrors(hipblasSaxpy(
                cublasHandle, N, &floatone, d_zm1, 1, d_p, 1));
        }

        checkCudaErrors(hipsparseSpMV(
            cusparseHandle, HIPSPARSE_OPERATION_NON_TRANSPOSE, &floatone, matA,
            vecp, &floatzero, vecomega, HIPBLAS_R_32F, HIPSPARSE_SPMV_ALG_DEFAULT,
            d_bufferMV));
        checkCudaErrors(hipblasSdot(
            cublasHandle, N, d_r, 1, d_zm1, 1, &numerator));
        checkCudaErrors(hipblasSdot(
            cublasHandle, N, d_p, 1, d_omega, 1, &denominator));
        alpha = numerator / denominator;
        checkCudaErrors(hipblasSaxpy(cublasHandle, N, &alpha, d_p, 1, d_x, 1));
        checkCudaErrors(hipblasScopy(cublasHandle, N, d_r, 1, d_rm2, 1));
        checkCudaErrors(hipblasScopy(cublasHandle, N, d_zm1, 1, d_zm2, 1));
        nalpha = -alpha;
        checkCudaErrors(hipblasSaxpy(
            cublasHandle, N, &nalpha, d_omega, 1, d_r, 1));
        checkCudaErrors(hipblasSdot(cublasHandle, N, d_r, 1, d_r, 1, &r1));
    }

    printf("  iteration = %3d, residual = %e \n", k, sqrt(r1));

    checkCudaErrors(hipMemcpy(
        x, d_x, N * sizeof(float), hipMemcpyDeviceToHost));

    /* check result */
    err = 0.0;

    for (int i = 0; i < N; i++)
    {
        rsum = 0.0;

        for (int j = I[i]; j < I[i + 1]; j++)
        {
            rsum += val[j] * x[J[j]];
        }

        diff = fabs(rsum - rhs[i]);

        if (diff > err)
        {
            err = diff;
        }
    }

    printf("  Convergence Test: %s \n", (k <= max_iter) ? "OK" : "FAIL");
    nErrors += (k > max_iter) ? 1 : 0;
    qaerr2 = err;

    /* Destroy descriptors */
    checkCudaErrors(hipsparseDestroyCsrilu02Info(infoILU));
    checkCudaErrors(hipsparseDestroyMatDescr(matLU));
    checkCudaErrors(hipsparseSpSV_destroyDescr(spsvDescrL));
    checkCudaErrors(hipsparseSpSV_destroyDescr(spsvDescrU));
    checkCudaErrors(hipsparseDestroySpMat(matM_lower));
    checkCudaErrors(hipsparseDestroySpMat(matM_upper));
    checkCudaErrors(hipsparseDestroySpMat(matA));
    checkCudaErrors(hipsparseDestroyDnVec(vecp));
    checkCudaErrors(hipsparseDestroyDnVec(vecomega));
    checkCudaErrors(hipsparseDestroyDnVec(vecR));
    checkCudaErrors(hipsparseDestroyDnVec(vecX));
    checkCudaErrors(hipsparseDestroyDnVec(vecY));
    checkCudaErrors(hipsparseDestroyDnVec(vecZM1));

    /* Destroy contexts */
    checkCudaErrors(hipsparseDestroy(cusparseHandle));
    checkCudaErrors(hipblasDestroy(cublasHandle));

    /* Free device memory */
    free(I);
    free(J);
    free(val);
    free(x);
    free(rhs);
    checkCudaErrors(hipFree(d_bufferMV));
    checkCudaErrors(hipFree(d_bufferLU));
    checkCudaErrors(hipFree(d_bufferL));
    checkCudaErrors(hipFree(d_bufferU));
    checkCudaErrors(hipFree(d_col));
    checkCudaErrors(hipFree(d_row));
    checkCudaErrors(hipFree(d_val));
    checkCudaErrors(hipFree(d_x));
    checkCudaErrors(hipFree(d_y));
    checkCudaErrors(hipFree(d_r));
    checkCudaErrors(hipFree(d_p));
    checkCudaErrors(hipFree(d_omega));
    checkCudaErrors(hipFree(d_valsILU0));
    checkCudaErrors(hipFree(d_zm1));
    checkCudaErrors(hipFree(d_zm2));
    checkCudaErrors(hipFree(d_rm2));

    // hipDeviceReset causes the driver to clean up all state. While
    // not mandatory in normal operation, it is good practice.  It is also
    // needed to ensure correct operation when the application is being
    // profiled. Calling hipDeviceReset causes all profile data to be
    // flushed before the application exits
    hipDeviceReset();

    printf("\n");
    printf("Test Summary:\n");
    printf("   Counted total of %d errors\n", nErrors);
    printf("   qaerr1 = %f qaerr2 = %f\n\n", fabs(qaerr1), fabs(qaerr2));
    exit((nErrors == 0 &&fabs(qaerr1) < 1e-5 && fabs(qaerr2) < 1e-5
        ? EXIT_SUCCESS
        : EXIT_FAILURE));
}

