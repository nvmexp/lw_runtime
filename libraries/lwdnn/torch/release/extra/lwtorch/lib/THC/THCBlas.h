#ifndef THC_BLAS_INC
#define THC_BLAS_INC

#include "THCGeneral.h"
#include "THCHalf.h"

/* Level 1 */
THC_API float THLwdaBlas_Sdot(THCState *state, long n, float *x, long incx, float *y, long incy);
THC_API double THLwdaBlas_Ddot(THCState *state, long n, double *x, long incx, double *y, long incy);
#ifdef LWDA_HALF_TENSOR
THC_API float THLwdaBlas_Hdot(THCState *state, long n, half *x, long incx, half *y, long incy);
#endif

/* Level 2 */
THC_API void THLwdaBlas_Sgemv(THCState *state, char trans, long m, long n, float alpha, float *a, long lda, float *x, long incx, float beta, float *y, long incy);
THC_API void THLwdaBlas_Dgemv(THCState *state, char trans, long m, long n, double alpha, double *a, long lda, double *x, long incx, double beta, double *y, long incy);
THC_API void THLwdaBlas_Sger(THCState *state, long m, long n, float alpha, float *x, long incx, float *y, long incy, float *a, long lda);
THC_API void THLwdaBlas_Dger(THCState *state, long m, long n, double alpha, double *x, long incx, double *y, long incy, double *a, long lda);

/* Level 3 */
THC_API void THLwdaBlas_Sgemm(THCState *state, char transa, char transb, long m, long n, long k, float alpha, float *a, long lda, float *b, long ldb, float beta, float *c, long ldc);
THC_API void THLwdaBlas_Dgemm(THCState *state, char transa, char transb, long m, long n, long k, double alpha, double *a, long lda, double *b, long ldb, double beta, double *c, long ldc);

#ifdef LWDA_HALF_TENSOR
THC_API void THLwdaBlas_Hgemm(THCState *state, char transa, char transb, long m, long n, long k, half alpha, half *a, long lda, half *b, long ldb, half beta, half *c, long ldc);
#endif

THC_API void THLwdaBlas_SgemmBatched(THCState *state, char transa, char transb, long m, long n, long k,
                                     float alpha, const float *a[], long lda, const float *b[], long ldb,
                                     float beta, float *c[], long ldc, long batchCount);
THC_API void THLwdaBlas_DgemmBatched(THCState *state, char transa, char transb, long m, long n, long k,
                                     double alpha, const double *a[], long lda, const double *b[], long ldb,
                                     double beta, double *c[], long ldc, long batchCount);
#if LWDA_VERSION >= 8000
THC_API void THLwdaBlas_SgemmStridedBatched(THCState *state, char transa, char transb, long m, long n, long k,
                                     float alpha, const float *a, long lda, long strideA, const float *b, long ldb, long strideB,
                                     float beta, float *c, long ldc, long strideC, long batchCount);
THC_API void THLwdaBlas_DgemmStridedBatched(THCState *state, char transa, char transb, long m, long n, long k,
                                     double alpha, const double *a, long lda, long strideA, const double *b, long ldb, long strideB, 
                                     double beta, double *c, long ldc, long strideC, long batchCount);
#endif
/* Ilwerse */
THC_API void THLwdaBlas_Sgetrf(THCState *state, int n, float **a, int lda, int *pivot, int *info, int batchSize);
THC_API void THLwdaBlas_Dgetrf(THCState *state, int n, double **a, int lda, int *pivot, int *info, int batchSize);

THC_API void THLwdaBlas_Sgetrs(THCState *state, char transa, int n, int nrhs, const float **a, int lda, int *pivot, float **b, int ldb, int *info, int batchSize);
THC_API void THLwdaBlas_Dgetrs(THCState *state, char transa, int n, int nrhs, const double **a, int lda, int *pivot, double **b, int ldb, int *info, int batchSize);

THC_API void THLwdaBlas_Sgetri(THCState *state, int n, const float **a, int lda, int *pivot, float **c, int ldc, int *info, int batchSize);
THC_API void THLwdaBlas_Dgetri(THCState *state, int n, const double **a, int lda, int *pivot, double **c, int ldc, int *info, int batchSize);

#endif
