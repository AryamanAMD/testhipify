/*
 * Copyright 2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <assert.h>

#include "cusolverSp.h"

#include "cusolverSp_LOWLEVEL_PREVIEW.h"

#include <hip/hip_runtime.h>

#include "helper_cuda.h"
#include "helper_cusolver.h"

template <typename T_ELEM>
int loadMMSparseMatrix(
    char *filename,
    char elem_type,
    bool csrFormat,
    int *m,
    int *n,
    int *nnz,
    T_ELEM **aVal,
    int **aRowInd,
    int **aColInd,
    int extendSymMatrix);

void UsageSP(void)
{
    printf( "<options>\n");
    printf( "-h          : display this help\n");
    printf( "-file=<filename> : filename containing a matrix in MM format\n");
    printf( "-device=<device_id> : <device_id> if want to run on specific GPU\n");

    exit( 0 );
}

void parseCommandLineArguments(int argc, char *argv[], struct testOpts &opts)
{
    memset(&opts, 0, sizeof(opts));

    if (checkCmdLineFlag(argc, (const char **)argv, "-h"))
    {
        UsageSP();
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "file"))
    {
        char *fileName = 0;
        getCmdLineArgumentString(argc, (const char **)argv, "file", &fileName);

        if (fileName)
        {
            opts.sparse_mat_filename = fileName;
        }
        else
        {
            printf("\nIncorrect filename passed to -file \n ");
            UsageSP();
        }
    }
}


int main (int argc, char *argv[])
{
    struct testOpts opts;
    cusolverSpHandle_t cusolverSpH = NULL; // reordering, permutation and 1st LU factorization
    hipsparseHandle_t   cusparseH = NULL;   // residual evaluation
    hipStream_t stream = NULL;
    hipsparseMatDescr_t descrA = NULL; // A is a base-0 general matrix

    csrcholInfoHost_t h_info = NULL; // opaque info structure for LU with parital pivoting
    csrcholInfo_t d_info = NULL; // opaque info structure for LU with parital pivoting

    int rowsA = 0; // number of rows of A
    int colsA = 0; // number of columns of A
    int nnzA  = 0; // number of nonzeros of A
    int baseA = 0; // base index in CSR format

    // CSR(A) from I/O
    int *h_csrRowPtrA = NULL; // <int> n+1 
    int *h_csrColIndA = NULL; // <int> nnzA 
    double *h_csrValA = NULL; // <double> nnzA 

    double *h_x = NULL; // <double> n,  x = A \ b
    double *h_b = NULL; // <double> n, b = ones(m,1)
    double *h_r = NULL; // <double> n, r = b - A*x

    size_t size_internal = 0; 
    size_t size_chol = 0; // size of working space for csrlu
    void *buffer_cpu = NULL; // working space for Cholesky
    void *buffer_gpu = NULL; // working space for Cholesky

    int *d_csrRowPtrA = NULL; // <int> n+1
    int *d_csrColIndA = NULL; // <int> nnzA
    double *d_csrValA = NULL; // <double> nnzA
    double *d_x = NULL; // <double> n, x = A \ b 
    double *d_b = NULL; // <double> n, a copy of h_b
    double *d_r = NULL; // <double> n, r = b - A*x

    // the constants used in residual evaluation, r = b - A*x
    const double minus_one = -1.0;
    const double one = 1.0;
    // the constant used in cusolverSp
    // singularity is -1 if A is invertible under tol
    // tol determines the condition of singularity
    int singularity = 0; 
    const double tol = 1.e-14;

    double x_inf = 0.0; // |x|
    double r_inf = 0.0; // |r|
    double A_inf = 0.0; // |A|
    int errors = 0;

    parseCommandLineArguments(argc, argv, opts);

    findCudaDevice(argc, (const char **)argv);

    if (opts.sparse_mat_filename == NULL)
    {
        opts.sparse_mat_filename =  sdkFindFilePath("lap2D_5pt_n100.mtx", argv[0]);
        if (opts.sparse_mat_filename != NULL)
            printf("Using default input file [%s]\n", opts.sparse_mat_filename);
        else
            printf("Could not find lap2D_5pt_n100.mtx\n");
    }
    else
    {
        printf("Using input file [%s]\n", opts.sparse_mat_filename);
    }

    printf("step 1: read matrix market format\n");

    if (opts.sparse_mat_filename)
    {
        if (loadMMSparseMatrix<double>(opts.sparse_mat_filename, 'd', true , &rowsA, &colsA,
               &nnzA, &h_csrValA, &h_csrRowPtrA, &h_csrColIndA, true)) 
        {
            return 1;
        }
        baseA = h_csrRowPtrA[0]; // baseA = {0,1}
    }
    else
    {
        fprintf(stderr, "Error: input matrix is not provided\n");
        return 1;
    }

    if ( rowsA != colsA ) 
    {
        fprintf(stderr, "Error: only support square matrix\n");
        return 1;
    }

    printf("sparse matrix A is %d x %d with %d nonzeros, base=%d\n", rowsA, colsA, nnzA, baseA);

    checkCudaErrors(cusolverSpCreate(&cusolverSpH));

    checkCudaErrors(hipsparseCreate(&cusparseH));

    checkCudaErrors(hipStreamCreate(&stream));

    checkCudaErrors(cusolverSpSetStream(cusolverSpH, stream));

    checkCudaErrors(hipsparseSetStream(cusparseH, stream));

    checkCudaErrors(hipsparseCreateMatDescr(&descrA));

    checkCudaErrors(hipsparseSetMatType(descrA, HIPSPARSE_MATRIX_TYPE_GENERAL));
    if (baseA)
    {
        checkCudaErrors(hipsparseSetMatIndexBase(descrA, HIPSPARSE_INDEX_BASE_ONE));
    }
    else
    {
        checkCudaErrors(hipsparseSetMatIndexBase(descrA, HIPSPARSE_INDEX_BASE_ZERO));
    }

    h_x    = (double*)malloc(sizeof(double)*colsA);
    h_b    = (double*)malloc(sizeof(double)*rowsA);
    h_r    = (double*)malloc(sizeof(double)*rowsA);

    assert(NULL != h_x);
    assert(NULL != h_b);
    assert(NULL != h_r);

    checkCudaErrors(hipMalloc((void **)&d_csrRowPtrA, sizeof(int)*(rowsA+1)));
    checkCudaErrors(hipMalloc((void **)&d_csrColIndA, sizeof(int)*nnzA));
    checkCudaErrors(hipMalloc((void **)&d_csrValA   , sizeof(double)*nnzA));
    checkCudaErrors(hipMalloc((void **)&d_x, sizeof(double)*colsA));
    checkCudaErrors(hipMalloc((void **)&d_b, sizeof(double)*rowsA));
    checkCudaErrors(hipMalloc((void **)&d_r, sizeof(double)*rowsA));

    for(int row = 0 ; row < rowsA ; row++)
    {
        h_b[row] = 1.0;
    }

    printf("step 2: create opaque info structure\n");
    checkCudaErrors(cusolverSpCreateCsrcholInfoHost(&h_info));

    printf("step 3: analyze chol(A) to know structure of L\n");
    checkCudaErrors(cusolverSpXcsrcholAnalysisHost(
        cusolverSpH, rowsA, nnzA,
        descrA, h_csrRowPtrA, h_csrColIndA,
        h_info));

    printf("step 4: workspace for chol(A)\n");
    checkCudaErrors(cusolverSpDcsrcholBufferInfoHost(
        cusolverSpH, rowsA, nnzA,
        descrA, h_csrValA, h_csrRowPtrA, h_csrColIndA,
        h_info,
        &size_internal,
        &size_chol));

    if (buffer_cpu) 
    {
        free(buffer_cpu); 
    }
    buffer_cpu = (void*)malloc(sizeof(char)*size_chol);
    assert(NULL != buffer_cpu);

    printf("step 5: compute A = L*L^T \n");
    checkCudaErrors(cusolverSpDcsrcholFactorHost(
        cusolverSpH, rowsA, nnzA,
        descrA, h_csrValA, h_csrRowPtrA, h_csrColIndA,
        h_info,
        buffer_cpu));

    printf("step 6: check if the matrix is singular \n");
    checkCudaErrors(cusolverSpDcsrcholZeroPivotHost(
        cusolverSpH, h_info, tol, &singularity));

    if ( 0 <= singularity)
    {
        fprintf(stderr, "Error: A is not invertible, singularity=%d\n", singularity);
        return 1;
    }

    printf("step 7: solve A*x = b \n");
    checkCudaErrors(cusolverSpDcsrcholSolveHost(
        cusolverSpH, rowsA, h_b, h_x, h_info, buffer_cpu));

    printf("step 8: evaluate residual r = b - A*x (result on CPU)\n");
    // use GPU gemv to compute r = b - A*x
    checkCudaErrors(hipMemcpy(d_csrRowPtrA, h_csrRowPtrA, sizeof(int)*(rowsA+1), hipMemcpyHostToDevice));
    checkCudaErrors(hipMemcpy(d_csrColIndA, h_csrColIndA, sizeof(int)*nnzA     , hipMemcpyHostToDevice));
    checkCudaErrors(hipMemcpy(d_csrValA   , h_csrValA   , sizeof(double)*nnzA  , hipMemcpyHostToDevice));

    checkCudaErrors(hipMemcpy(d_r, h_b, sizeof(double)*rowsA, hipMemcpyHostToDevice));
    checkCudaErrors(hipMemcpy(d_x, h_x, sizeof(double)*colsA, hipMemcpyHostToDevice));

    /* Wrap raw data into cuSPARSE generic API objects */
    hipsparseSpMatDescr_t matA = NULL;
    if (baseA)
    {
        checkCudaErrors(hipsparseCreateCsr(
        &matA, rowsA, colsA, nnzA, d_csrRowPtrA, d_csrColIndA, d_csrValA, HIPSPARSE_INDEX_32I,
        HIPSPARSE_INDEX_32I, HIPSPARSE_INDEX_BASE_ONE, HIPBLAS_R_64F));
    }
    else
    {
        checkCudaErrors(hipsparseCreateCsr(
        &matA, rowsA, colsA, nnzA, d_csrRowPtrA, d_csrColIndA, d_csrValA, HIPSPARSE_INDEX_32I,
        HIPSPARSE_INDEX_32I, HIPSPARSE_INDEX_BASE_ZERO, HIPBLAS_R_64F));
    }

    hipsparseDnVecDescr_t vecx = NULL;
    checkCudaErrors(hipsparseCreateDnVec(&vecx, colsA, d_x, HIPBLAS_R_64F));
    hipsparseDnVecDescr_t vecAx = NULL;
    checkCudaErrors(hipsparseCreateDnVec(&vecAx, rowsA, d_r, HIPBLAS_R_64F));

    /* Allocate workspace for cuSPARSE */
    size_t bufferSize = 0;
    checkCudaErrors(hipsparseSpMV_bufferSize(
        cusparseH, HIPSPARSE_OPERATION_NON_TRANSPOSE, &minus_one, matA, vecx,
        &one, vecAx, HIPBLAS_R_64F, HIPSPARSE_SPMV_ALG_DEFAULT, &bufferSize));
    void *buffer = NULL;
    checkCudaErrors(hipMalloc(&buffer, bufferSize));

    checkCudaErrors(hipsparseSpMV(
        cusparseH, HIPSPARSE_OPERATION_NON_TRANSPOSE, &minus_one, matA, vecx,
        &one, vecAx, HIPBLAS_R_64F, HIPSPARSE_SPMV_ALG_DEFAULT, buffer));

    checkCudaErrors(hipMemcpy(h_r, d_r, sizeof(double)*rowsA, hipMemcpyDeviceToHost));

    x_inf = vec_norminf(colsA, h_x);
    r_inf = vec_norminf(rowsA, h_r);
    A_inf = csr_mat_norminf(rowsA, colsA, nnzA, descrA, h_csrValA, h_csrRowPtrA, h_csrColIndA);

    printf("(CPU) |b - A*x| = %E \n", r_inf);
    printf("(CPU) |A| = %E \n", A_inf);
    printf("(CPU) |x| = %E \n", x_inf);
    printf("(CPU) |b - A*x|/(|A|*|x|) = %E \n", r_inf/(A_inf * x_inf));

    printf("step 9: create opaque info structure\n");
    checkCudaErrors(cusolverSpCreateCsrcholInfo(&d_info));

    checkCudaErrors(hipMemcpy(d_csrRowPtrA, h_csrRowPtrA, sizeof(int)*(rowsA+1), hipMemcpyHostToDevice));
    checkCudaErrors(hipMemcpy(d_csrColIndA, h_csrColIndA, sizeof(int)*nnzA     , hipMemcpyHostToDevice));
    checkCudaErrors(hipMemcpy(d_csrValA   , h_csrValA   , sizeof(double)*nnzA  , hipMemcpyHostToDevice));
    checkCudaErrors(hipMemcpy(d_b, h_b, sizeof(double)*rowsA, hipMemcpyHostToDevice));

    printf("step 10: analyze chol(A) to know structure of L\n");
    checkCudaErrors(cusolverSpXcsrcholAnalysis(
        cusolverSpH, rowsA, nnzA,
        descrA, d_csrRowPtrA, d_csrColIndA,
        d_info));

    printf("step 11: workspace for chol(A)\n");
    checkCudaErrors(cusolverSpDcsrcholBufferInfo(
        cusolverSpH, rowsA, nnzA,
        descrA, d_csrValA, d_csrRowPtrA, d_csrColIndA,
        d_info,
        &size_internal,
        &size_chol));

    if (buffer_gpu) {
        checkCudaErrors(hipFree(buffer_gpu));
    }
    checkCudaErrors(hipMalloc(&buffer_gpu, sizeof(char)*size_chol));

    printf("step 12: compute A = L*L^T \n");
    checkCudaErrors(cusolverSpDcsrcholFactor(
        cusolverSpH, rowsA, nnzA,
        descrA, d_csrValA, d_csrRowPtrA, d_csrColIndA,
        d_info,
        buffer_gpu));

    printf("step 13: check if the matrix is singular \n");
    checkCudaErrors(cusolverSpDcsrcholZeroPivot(
        cusolverSpH, d_info, tol, &singularity));

    if ( 0 <= singularity){
        fprintf(stderr, "Error: A is not invertible, singularity=%d\n", singularity);
        return 1;
    }

    printf("step 14: solve A*x = b \n");
    checkCudaErrors(cusolverSpDcsrcholSolve(
        cusolverSpH, rowsA, d_b, d_x, d_info, buffer_gpu));

    checkCudaErrors(hipMemcpy(d_r, h_b, sizeof(double)*rowsA, hipMemcpyHostToDevice));

    checkCudaErrors(hipsparseSpMV(
        cusparseH, HIPSPARSE_OPERATION_NON_TRANSPOSE, &minus_one, matA, vecx,
        &one, vecAx, HIPBLAS_R_64F, HIPSPARSE_SPMV_ALG_DEFAULT, buffer));

    checkCudaErrors(hipMemcpy(h_r, d_r, sizeof(double)*rowsA, hipMemcpyDeviceToHost));

    r_inf = vec_norminf(rowsA, h_r);

    printf("(GPU) |b - A*x| = %E \n", r_inf);
    printf("(GPU) |b - A*x|/(|A|*|x|) = %E \n", r_inf/(A_inf * x_inf));


    if (cusolverSpH) { checkCudaErrors(cusolverSpDestroy(cusolverSpH)); }
    if (cusparseH  ) { checkCudaErrors(hipsparseDestroy(cusparseH)); }
    if (stream     ) { checkCudaErrors(hipStreamDestroy(stream)); }
    if (descrA     ) { checkCudaErrors(hipsparseDestroyMatDescr(descrA)); }
    if (h_info     ) { checkCudaErrors(cusolverSpDestroyCsrcholInfoHost(h_info)); }
    if (d_info     ) { checkCudaErrors(cusolverSpDestroyCsrcholInfo(d_info)); }
    if (matA       ) { checkCudaErrors(hipsparseDestroySpMat(matA)); }
    if (vecx       ) { checkCudaErrors(hipsparseDestroyDnVec(vecx)); }
    if (vecAx      ) { checkCudaErrors(hipsparseDestroyDnVec(vecAx)); }

    if (h_csrValA   ) { free(h_csrValA); }
    if (h_csrRowPtrA) { free(h_csrRowPtrA); }
    if (h_csrColIndA) { free(h_csrColIndA); }

    if (h_x   ) { free(h_x); }
    if (h_b   ) { free(h_b); }
    if (h_r   ) { free(h_r); }

    if (buffer_cpu) { free(buffer_cpu); }
    if (buffer_gpu) { checkCudaErrors(hipFree(buffer_gpu)); }

    if (d_csrValA   ) { checkCudaErrors(hipFree(d_csrValA)); }
    if (d_csrRowPtrA) { checkCudaErrors(hipFree(d_csrRowPtrA)); }
    if (d_csrColIndA) { checkCudaErrors(hipFree(d_csrColIndA)); }
    if (d_x) { checkCudaErrors(hipFree(d_x)); }
    if (d_b) { checkCudaErrors(hipFree(d_b)); }
    if (d_r) { checkCudaErrors(hipFree(d_r)); }

    return 0;
}

