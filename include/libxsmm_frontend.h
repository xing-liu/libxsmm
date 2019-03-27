/******************************************************************************
** Copyright (c) 2015-2019, Intel Corporation                                **
** All rights reserved.                                                      **
**                                                                           **
** Redistribution and use in source and binary forms, with or without        **
** modification, are permitted provided that the following conditions        **
** are met:                                                                  **
** 1. Redistributions of source code must retain the above copyright         **
**    notice, this list of conditions and the following disclaimer.          **
** 2. Redistributions in binary form must reproduce the above copyright      **
**    notice, this list of conditions and the following disclaimer in the    **
**    documentation and/or other materials provided with the distribution.   **
** 3. Neither the name of the copyright holder nor the names of its          **
**    contributors may be used to endorse or promote products derived        **
**    from this software without specific prior written permission.          **
**                                                                           **
** THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS       **
** "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT         **
** LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR     **
** A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT      **
** HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,    **
** SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED  **
** TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR    **
** PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF    **
** LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING      **
** NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS        **
** SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.              **
******************************************************************************/
/* Hans Pabst (Intel Corp.)
******************************************************************************/
#ifndef LIBXSMM_FRONTEND_H
#define LIBXSMM_FRONTEND_H

#include "libxsmm_typedefs.h"

/** Helper macros for eliding prefetch address calculations depending on prefetch scheme. */
#if !defined(_WIN32) && !defined(__CYGWIN__) /* TODO: fully support calling convention */
#if 0 != ((LIBXSMM_PREFETCH) & 2/*AL2*/) \
 || 0 != ((LIBXSMM_PREFETCH) & 4/*AL2_JPST*/) \
 || 0 != ((LIBXSMM_PREFETCH) & 16/*AL2_AHEAD*/) \
 || 0 != ((LIBXSMM_PREFETCH) & 32/*AL1*/)
# define LIBXSMM_GEMM_PREFETCH_A(EXPR) (EXPR)
#endif
#if 0 != ((LIBXSMM_PREFETCH) & 8/*BL2_VIA_C*/) \
 || 0 != ((LIBXSMM_PREFETCH) & 64/*BL1*/)
# define LIBXSMM_GEMM_PREFETCH_B(EXPR) (EXPR)
#endif
#if 0 != ((LIBXSMM_PREFETCH) & 128/*CL1*/)
# define LIBXSMM_GEMM_PREFETCH_C(EXPR) (EXPR)
#endif
#endif
/** Secondary helper macros derived from the above group. */
#if defined(LIBXSMM_GEMM_PREFETCH_A)
# define LIBXSMM_NOPREFETCH_A(EXPR)
#else
# define LIBXSMM_NOPREFETCH_A(EXPR) EXPR
# define LIBXSMM_GEMM_PREFETCH_A(EXPR) 0
#endif
#if defined(LIBXSMM_GEMM_PREFETCH_B)
# define LIBXSMM_NOPREFETCH_B(EXPR)
#else
# define LIBXSMM_NOPREFETCH_B(EXPR) EXPR
# define LIBXSMM_GEMM_PREFETCH_B(EXPR) 0
#endif
#if defined(LIBXSMM_GEMM_PREFETCH_C)
# define LIBXSMM_NOPREFETCH_C(EXPR)
#else
# define LIBXSMM_NOPREFETCH_C(EXPR) EXPR
# define LIBXSMM_GEMM_PREFETCH_C(EXPR) 0
#endif

/** MKL_DIRECT_CALL requires to include the MKL interface. */
#if defined(MKL_DIRECT_CALL_SEQ) || defined(MKL_DIRECT_CALL)
# if (0 != LIBXSMM_ILP64 && !defined(MKL_ILP64))
#   error "Inconsistent ILP64 configuration detected!"
# endif
# if defined(LIBXSMM_OFFLOAD_BUILD)
#   pragma offload_attribute(push,target(LIBXSMM_OFFLOAD_TARGET))
#   include <mkl.h>
#   pragma offload_attribute(pop)
# else
#   include <mkl.h>
# endif
#endif

/** Automatically select a prefetch-strategy (libxsmm_get_gemm_xprefetch, etc.). */
#define LIBXSMM_PREFETCH_AUTO -1

/** Helper macro for BLAS-style prefixes. */
#define LIBXSMM_TPREFIX_NAME(TYPE) LIBXSMM_CONCATENATE(LIBXSMM_TPREFIX_, TYPE)
#define LIBXSMM_TPREFIX(TYPE, FUNCTION) LIBXSMM_CONCATENATE(LIBXSMM_TPREFIX_NAME(TYPE), FUNCTION)
#define LIBXSMM_TPREFIX_doubledouble d
#define LIBXSMM_TPREFIX_floatfloat s
#define LIBXSMM_TPREFIX_shortfloat ws
#define LIBXSMM_TPREFIX_shortint wi
#define LIBXSMM_TPREFIX_libxsmm_bfloat16float bs
/** Defaults if only the input type is specified. */
#define LIBXSMM_TPREFIX_double LIBXSMM_TPREFIX_doubledouble
#define LIBXSMM_TPREFIX_float LIBXSMM_TPREFIX_floatfloat
#define LIBXSMM_TPREFIX_short LIBXSMM_TPREFIX_shortint

/** Construct symbol name from a given real type name (float, double and short). */
#define LIBXSMM_BLAS_FNTYPE(TYPE, KIND) LIBXSMM_CONCATENATE2(libxsmm_, LIBXSMM_TPREFIX(TYPE, KIND), _function)
#define LIBXSMM_MMFUNCTION_TYPE(TYPE)   LIBXSMM_CONCATENATE(libxsmm_, LIBXSMM_TPREFIX(TYPE, mmfunction))
#define LIBXSMM_MMDISPATCH_SYMBOL(TYPE) LIBXSMM_CONCATENATE(libxsmm_, LIBXSMM_TPREFIX(TYPE, mmdispatch))
#define LIBXSMM_XBLAS_SYMBOL(TYPE)      LIBXSMM_CONCATENATE(libxsmm_blas_, LIBXSMM_TPREFIX(TYPE, gemm))
#define LIBXSMM_XGEMM_SYMBOL(TYPE)      LIBXSMM_CONCATENATE(libxsmm_, LIBXSMM_TPREFIX(TYPE, gemm))
#define LIBXSMM_YGEMM_SYMBOL(TYPE)      LIBXSMM_USEOMP(LIBXSMM_XGEMM_SYMBOL(TYPE))
#define LIBXSMM_USEOMP(FUNCTION)        LIBXSMM_CONCATENATE(FUNCTION, _omp)
#define LIBXSMM_BLAS_SYMBOL(TYPE, KIND) LIBXSMM_FSYMBOL(LIBXSMM_TPREFIX(TYPE, KIND))
#define LIBXSMM_GEMM_SYMBOL_NAME(TYPE)  LIBXSMM_BLAS_SYMBOL(TYPE, gemm)
#define LIBXSMM_GEMV_SYMBOL_NAME(TYPE)  LIBXSMM_BLAS_SYMBOL(TYPE, gemv)

/* Construct prefix names, function type or dispatch function from given input and output types. */
#define LIBXSMM_MMFUNCTION_TYPE2(ITYPE, OTYPE)    LIBXSMM_MMFUNCTION_TYPE(LIBXSMM_CONCATENATE(ITYPE, OTYPE))
#define LIBXSMM_MMDISPATCH_SYMBOL2(ITYPE, OTYPE)  LIBXSMM_MMDISPATCH_SYMBOL(LIBXSMM_CONCATENATE(ITYPE, OTYPE))
#define LIBXSMM_TPREFIX_NAME2(ITYPE, OTYPE)       LIBXSMM_TPREFIX_NAME(LIBXSMM_CONCATENATE(ITYPE, OTYPE))
#define LIBXSMM_TPREFIX2(ITYPE, OTYPE, FUNCTION)  LIBXSMM_TPREFIX(LIBXSMM_CONCATENATE(ITYPE, OTYPE), FUNCTION)

/** Helper macro for comparing selected types. */
#define LIBXSMM_EQUAL(T1, T2) LIBXSMM_CONCATENATE2(LIBXSMM_EQUAL_, T1, T2)
#define LIBXSMM_EQUAL_floatfloat 1
#define LIBXSMM_EQUAL_doubledouble 1
#define LIBXSMM_EQUAL_floatdouble 0
#define LIBXSMM_EQUAL_doublefloat 0

#if defined(LIBXSMM_GEMM_CONST)
# undef LIBXSMM_GEMM_CONST
# define LIBXSMM_GEMM_CONST const
#elif defined(LIBXSMM_GEMM_NONCONST) || defined(__OPENBLAS)
# define LIBXSMM_GEMM_CONST
#else
# define LIBXSMM_GEMM_CONST const
#endif

#if !defined(LIBXSMM_NO_BLAS)
# if (!defined(__BLAS) || (0 != __BLAS))
#   define LIBXSMM_NO_BLAS 0
# else
#   define LIBXSMM_NO_BLAS 1
# endif
#endif

#if defined(LIBXSMM_BUILD)
# if defined(LIBXSMM_BUILD_EXT) && !defined(__STATIC)
#   define LIBXSMM_GEMM_SYMBOL_VISIBILITY LIBXSMM_APIEXT
# elif defined(LIBXSMM_NO_BLAS) && (1 == LIBXSMM_NO_BLAS)
#   define LIBXSMM_GEMM_SYMBOL_VISIBILITY LIBXSMM_API
# endif
#endif
#if !defined(LIBXSMM_GEMM_SYMBOL_VISIBILITY)
# define LIBXSMM_GEMM_SYMBOL_VISIBILITY LIBXSMM_VISIBILITY_IMPORT LIBXSMM_RETARGETABLE
#endif

#define LIBXSMM_BLAS_SYMBOL_SIGNATURE_gemm(CONST, TYPE) CONST char*, CONST char*, \
  CONST libxsmm_blasint*, CONST libxsmm_blasint*, CONST libxsmm_blasint*, CONST TYPE*, CONST TYPE*, CONST libxsmm_blasint*, \
  CONST TYPE*, CONST libxsmm_blasint*, CONST TYPE*, TYPE*, CONST libxsmm_blasint*
#define LIBXSMM_BLAS_SYMBOL_SIGNATURE_gemv(CONST, TYPE) CONST char*, CONST libxsmm_blasint*, CONST libxsmm_blasint*, \
  CONST TYPE*, CONST TYPE*, CONST libxsmm_blasint*, CONST TYPE*, CONST libxsmm_blasint*, \
  CONST TYPE*, TYPE*, CONST libxsmm_blasint*
#define LIBXSMM_BLAS_SYMBOL_SIGNATURE(CONST, TYPE, KIND) LIBXSMM_CONCATENATE(LIBXSMM_BLAS_SYMBOL_SIGNATURE_, KIND)(CONST, TYPE)
#define LIBXSMM_BLAS_SYMBOL_DECL(CONST, TYPE, KIND) LIBXSMM_GEMM_SYMBOL_VISIBILITY \
  void LIBXSMM_BLAS_SYMBOL(TYPE, KIND)(LIBXSMM_BLAS_SYMBOL_SIGNATURE(CONST, TYPE, KIND));
#if (0 == LIBXSMM_NO_BLAS) /* BLAS available */
# define LIBXSMM_GEMM_SYMBOL_DECL(CONST, TYPE) LIBXSMM_BLAS_SYMBOL_DECL(CONST, TYPE, gemm)
#else
# define LIBXSMM_GEMM_SYMBOL_DECL(CONST, TYPE)
#endif

/** Helper macro consolidating the transpose requests into a set of flags. */
#define LIBXSMM_GEMM_FLAGS(TRANSA, TRANSB) /* check for N/n rather than T/t since C/c is also valid! */ \
   ((('n' == (TRANSA) || *"N" == (TRANSA)) ? LIBXSMM_GEMM_FLAG_NONE : LIBXSMM_GEMM_FLAG_TRANS_A) \
  | (('n' == (TRANSB) || *"N" == (TRANSB)) ? LIBXSMM_GEMM_FLAG_NONE : LIBXSMM_GEMM_FLAG_TRANS_B))

/** Helper macro allowing NULL-requests (transposes) supplied by some default. */
#define LIBXSMM_GEMM_PFLAGS(TRANSA, TRANSB, DEFAULT) LIBXSMM_GEMM_FLAGS( \
  NULL != ((const void*)(TRANSA)) ? (*(const char*)(TRANSA)) : (0 == (LIBXSMM_GEMM_FLAG_TRANS_A & (DEFAULT)) ? 'n' : 't'), \
  NULL != ((const void*)(TRANSB)) ? (*(const char*)(TRANSB)) : (0 == (LIBXSMM_GEMM_FLAG_TRANS_B & (DEFAULT)) ? 'n' : 't')) \
  | (~(LIBXSMM_GEMM_FLAG_TRANS_A | LIBXSMM_GEMM_FLAG_TRANS_B) & (DEFAULT))

/** Inlinable GEMM exercising the compiler's code generation (macro template). TODO: only NN is supported and SP/DP matrices. */
#define LIBXSMM_INLINE_XGEMM(ITYPE, OTYPE, TRANSA, TRANSB, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC) { \
  /* Use 'n' (instead of 'N') avoids warning about "no macro replacement within a character constant". */ \
  const char libxsmm_inline_xgemm_transa_ = (char)(NULL != ((void*)(TRANSA)) ? (*(const char*)(TRANSA)) : \
    (0 == (LIBXSMM_GEMM_FLAG_TRANS_A & LIBXSMM_FLAGS) ? 'n' : 't')); \
  const char libxsmm_inline_xgemm_transb_ = (char)(NULL != ((void*)(TRANSB)) ? (*(const char*)(TRANSB)) : \
    (0 == (LIBXSMM_GEMM_FLAG_TRANS_B & LIBXSMM_FLAGS) ? 'n' : 't')); \
  const libxsmm_blasint libxsmm_inline_xgemm_m_ = *(const libxsmm_blasint*)(M); /* must be specified */ \
  const libxsmm_blasint libxsmm_inline_xgemm_k_ = (NULL != ((void*)(K)) ? (*(const libxsmm_blasint*)(K)) : libxsmm_inline_xgemm_m_); \
  const libxsmm_blasint libxsmm_inline_xgemm_n_ = (NULL != ((void*)(N)) ? (*(const libxsmm_blasint*)(N)) : libxsmm_inline_xgemm_k_); \
  const libxsmm_blasint libxsmm_inline_xgemm_lda_ = (NULL != ((void*)(LDA)) ? (*(const libxsmm_blasint*)(LDA)) : \
    (('n' == libxsmm_inline_xgemm_transa_ || *"N" == libxsmm_inline_xgemm_transa_) ? libxsmm_inline_xgemm_m_ : libxsmm_inline_xgemm_k_)); \
  const libxsmm_blasint libxsmm_inline_xgemm_ldb_ = (NULL != ((void*)(LDB)) ? (*(const libxsmm_blasint*)(LDB)) : \
    (('n' == libxsmm_inline_xgemm_transb_ || *"N" == libxsmm_inline_xgemm_transb_) ? libxsmm_inline_xgemm_k_ : libxsmm_inline_xgemm_n_)); \
  const libxsmm_blasint libxsmm_inline_xgemm_ldc_ = (NULL != ((void*)(LDC)) ? (*(const libxsmm_blasint*)(LDC)) : libxsmm_inline_xgemm_m_); \
  const OTYPE libxsmm_inline_xgemm_alpha_ = (NULL != ((void*)(ALPHA)) ? (*(const OTYPE*)(ALPHA)) : ((OTYPE)LIBXSMM_ALPHA)); \
  const OTYPE libxsmm_inline_xgemm_beta_  = (NULL != ((void*)(BETA))  ? (*(const OTYPE*)(BETA))  : ((OTYPE)LIBXSMM_BETA)); \
  libxsmm_blasint libxsmm_inline_xgemm_ni_, libxsmm_inline_xgemm_mi_, libxsmm_inline_xgemm_ki_; /* loop induction variables */ \
  LIBXSMM_ASSERT('n' == libxsmm_inline_xgemm_transa_ || *"N" == libxsmm_inline_xgemm_transa_); \
  LIBXSMM_ASSERT('n' == libxsmm_inline_xgemm_transb_ || *"N" == libxsmm_inline_xgemm_transb_); \
  LIBXSMM_PRAGMA_SIMD \
  for (libxsmm_inline_xgemm_mi_ = 0; libxsmm_inline_xgemm_mi_ < libxsmm_inline_xgemm_m_; ++libxsmm_inline_xgemm_mi_) { \
    LIBXSMM_PRAGMA_LOOP_COUNT(1, LIBXSMM_CONFIG_MAX_DIM, LIBXSMM_CONFIG_AVG_DIM) \
    for (libxsmm_inline_xgemm_ki_ = 0; libxsmm_inline_xgemm_ki_ < libxsmm_inline_xgemm_k_; ++libxsmm_inline_xgemm_ki_) { \
      LIBXSMM_PRAGMA_UNROLL \
      for (libxsmm_inline_xgemm_ni_ = 0; libxsmm_inline_xgemm_ni_ < libxsmm_inline_xgemm_n_; ++libxsmm_inline_xgemm_ni_) { \
        ((OTYPE*)(C))[libxsmm_inline_xgemm_ni_*libxsmm_inline_xgemm_ldc_+libxsmm_inline_xgemm_mi_] \
          = ((const ITYPE*)(B))[libxsmm_inline_xgemm_ni_*libxsmm_inline_xgemm_ldb_+libxsmm_inline_xgemm_ki_] * \
           (((const ITYPE*)(A))[libxsmm_inline_xgemm_ki_*libxsmm_inline_xgemm_lda_+libxsmm_inline_xgemm_mi_] * libxsmm_inline_xgemm_alpha_) \
          + ((const OTYPE*)(C))[libxsmm_inline_xgemm_ni_*libxsmm_inline_xgemm_ldc_+libxsmm_inline_xgemm_mi_] * libxsmm_inline_xgemm_beta_; \
      } \
    } \
  } \
}

/** Map to appropriate BLAS function (or fall-back). The mapping is used e.g., inside of LIBXSMM_BLAS_XGEMM. */
#define LIBXSMM_BLAS_FUNCTION(ITYPE, OTYPE, FUNCTION) LIBXSMM_CONCATENATE(LIBXSMM_BLAS_FUNCTION_, LIBXSMM_TPREFIX2(ITYPE, OTYPE, FUNCTION))
#if (0 == LIBXSMM_NO_BLAS)
# define LIBXSMM_BLAS_FUNCTION_dgemm libxsmm_original_dgemm()
# define LIBXSMM_BLAS_FUNCTION_sgemm libxsmm_original_sgemm()
#else /* no BLAS */
# define LIBXSMM_BLAS_FUNCTION_dgemm(TRANSA, TRANSB, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC) \
    LIBXSMM_INLINE_XGEMM(double, double, TRANSA, TRANSB, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC)
# define LIBXSMM_BLAS_FUNCTION_sgemm(TRANSA, TRANSB, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC) \
    LIBXSMM_INLINE_XGEMM(float, float, TRANSA, TRANSB, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC)
#endif
#define LIBXSMM_BLAS_FUNCTION_wigemm(TRANSA, TRANSB, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC) \
  LIBXSMM_INLINE_XGEMM(short, int, TRANSA, TRANSB, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC)
#define LIBXSMM_BLAS_FUNCTION_wsgemm(TRANSA, TRANSB, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC) \
  LIBXSMM_INLINE_XGEMM(short, float, TRANSA, TRANSB, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC)
#define LIBXSMM_BLAS_FUNCTION_bsgemm(TRANSA, TRANSB, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC) \
  LIBXSMM_INLINE_XGEMM(libxsmm_bfloat16, float, TRANSA, TRANSB, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC)

/** BLAS-based GEMM supplied by the linked LAPACK/BLAS library (macro template). */
#define LIBXSMM_BLAS_XGEMM(ITYPE, OTYPE, TRANSA, TRANSB, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC) { \
  /* Use 'n' (instead of 'N') avoids warning about "no macro replacement within a character constant". */ \
  const char libxsmm_blas_xgemm_transa_ = (char)(NULL != ((void*)(TRANSA)) ? (*(const char*)(TRANSA)) : \
    (0 == (LIBXSMM_GEMM_FLAG_TRANS_A & LIBXSMM_FLAGS) ? 'n' : 't')); \
  const char libxsmm_blas_xgemm_transb_ = (char)(NULL != ((void*)(TRANSB)) ? (*(const char*)(TRANSB)) : \
    (0 == (LIBXSMM_GEMM_FLAG_TRANS_B & LIBXSMM_FLAGS) ? 'n' : 't')); \
  const libxsmm_blasint *const libxsmm_blas_xgemm_k_ = (NULL != ((void*)(K)) ? (K) : (M)); \
  const libxsmm_blasint *const libxsmm_blas_xgemm_n_ = (NULL != ((void*)(N)) ? (N) : libxsmm_blas_xgemm_k_); \
  const libxsmm_blasint libxsmm_blas_xgemm_lda_ = LIBXSMM_MAX(NULL != ((void*)(LDA)) ? *(LDA) : \
    *(('n' == libxsmm_blas_xgemm_transa_ || *"N" == libxsmm_blas_xgemm_transa_) ? (M) : libxsmm_blas_xgemm_k_), 1); \
  const libxsmm_blasint libxsmm_blas_xgemm_ldb_ = LIBXSMM_MAX(NULL != ((void*)(LDB)) ? *(LDB) : \
    *(('n' == libxsmm_blas_xgemm_transb_ || *"N" == libxsmm_blas_xgemm_transb_) ? libxsmm_blas_xgemm_k_ : libxsmm_blas_xgemm_n_), 1); \
  const libxsmm_blasint libxsmm_blas_xgemm_ldc_ = LIBXSMM_MAX(NULL != ((void*)(LDC)) ? *(LDC) : *(M), 1); \
  const OTYPE libxsmm_blas_xgemm_alpha_ = (NULL != ((void*)(ALPHA)) ? (*(const OTYPE*)(ALPHA)) : ((OTYPE)LIBXSMM_ALPHA)); \
  const OTYPE libxsmm_blas_xgemm_beta_  = (NULL != ((void*)(BETA))  ? (*(const OTYPE*)(BETA))  : ((OTYPE)LIBXSMM_BETA)); \
  LIBXSMM_BLAS_FUNCTION(ITYPE, OTYPE, gemm)(&libxsmm_blas_xgemm_transa_, &libxsmm_blas_xgemm_transb_, \
    M, libxsmm_blas_xgemm_n_, libxsmm_blas_xgemm_k_, \
    &libxsmm_blas_xgemm_alpha_, (const ITYPE*)(A), &libxsmm_blas_xgemm_lda_, \
                                (const ITYPE*)(B), &libxsmm_blas_xgemm_ldb_, \
     &libxsmm_blas_xgemm_beta_,       (ITYPE*)(C), &libxsmm_blas_xgemm_ldc_); \
}

/** Helper macros for calling a dispatched function in a row/column-major aware fashion. */
#define LIBXSMM_MMCALL_ABC(FN, A, B, C) \
  LIBXSMM_ASSERT(FN); FN(A, B, C)
#define LIBXSMM_MMCALL_PRF(FN, A, B, C, PA, PB, PC) { \
  LIBXSMM_NOPREFETCH_A(LIBXSMM_UNUSED(PA)); \
  LIBXSMM_NOPREFETCH_B(LIBXSMM_UNUSED(PB)); \
  LIBXSMM_NOPREFETCH_C(LIBXSMM_UNUSED(PC)); \
  LIBXSMM_ASSERT(FN); FN(A, B, C, \
    LIBXSMM_GEMM_PREFETCH_A(PA), \
    LIBXSMM_GEMM_PREFETCH_B(PB), \
    LIBXSMM_GEMM_PREFETCH_C(PC)); \
}

#if (0/*LIBXSMM_GEMM_PREFETCH_NONE*/ == LIBXSMM_PREFETCH)
# define LIBXSMM_MMCALL_LDX(FN, A, B, C, M, N, K, LDA, LDB, LDC) \
  LIBXSMM_MMCALL_ABC(FN, A, B, C)
#else
# define LIBXSMM_MMCALL_LDX(FN, A, B, C, M, N, K, LDA, LDB, LDC) \
  LIBXSMM_MMCALL_PRF(FN, A, B, C, (A) + ((size_t)LDA) * (K), (B) + ((size_t)LDB) * (N), (C) + ((size_t)LDC) * (N))
#endif
#define LIBXSMM_MMCALL(FN, A, B, C, M, N, K) LIBXSMM_MMCALL_LDX(FN, A, B, C, M, N, K, M, K, M)

/** Calculate problem size from M, N, and K using the correct integer type in order to cover the general case. */
#define LIBXSMM_MNK_SIZE(M, N, K) (((size_t)(M)) * ((size_t)(N)) * ((size_t)(K)))
/** Calculate total number of matrix-elements; matrices A, B, C are given per M, N, K, and emphasize (S) the C-size. */
#define LIBXSMM_SIZE(M, N, K, S) (((size_t)(M) * (K)) + ((size_t)(K) * (N)) + ((size_t)(S) * (M) * (N)))
/** Condition based on arithmetic intensity (AI) */
#define LIBXSMM_SMM_AI(M, N, K, S, TYPESIZE) ((LIBXSMM_MNK_SIZE(M, N, K) * 2) <= ((size_t)(TYPESIZE) * 3/*AI*/ * LIBXSMM_SIZE(M, N, K, S)))
/** Determine whether an SMM is suitable i.e., small enough. */
#if !defined(LIBXSMM_THRESHOLD_AI) /* traditional MNK-threshold */
# define LIBXSMM_SMM(M, N, K, S, TYPESIZE) (LIBXSMM_MNK_SIZE(M, N, K) <= (LIBXSMM_MAX_MNK))
#else /* threshold based on arithmetic intensity */
# define LIBXSMM_SMM LIBXSMM_SMM_AI
#endif

/** Fall-back code paths: LIBXSMM_XGEMM_FALLBACK0, and LIBXSMM_XGEMM_FALLBACK1 (macro template). */
#if !defined(LIBXSMM_XGEMM_FALLBACK0)
# define LIBXSMM_XGEMM_FALLBACK0(ITYPE, OTYPE, TRANSA, TRANSB, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC) \
     LIBXSMM_BLAS_FUNCTION(ITYPE, OTYPE, gemm)(TRANSA, TRANSB, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC)
#endif
#if !defined(LIBXSMM_XGEMM_FALLBACK1)
# define LIBXSMM_XGEMM_FALLBACK1(ITYPE, OTYPE, TRANSA, TRANSB, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC) \
     LIBXSMM_BLAS_FUNCTION(ITYPE, OTYPE, gemm)(TRANSA, TRANSB, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC)
#endif

/**
 * Execute a specialized function, or use a fall-back code path depending on threshold (macro template).
 * LIBXSMM_XGEMM_FALLBACK0 or specialized function: below LIBXSMM_MAX_MNK
 * LIBXSMM_XGEMM_FALLBACK1: above LIBXSMM_MAX_MNK
 */
#define LIBXSMM_XGEMM(ITYPE, OTYPE, TRANSA, TRANSB, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC) { \
  const int libxsmm_xgemm_flags_ = LIBXSMM_GEMM_PFLAGS(TRANSA, TRANSB, LIBXSMM_FLAGS); \
  const libxsmm_blasint *const libxsmm_xgemm_k_ = (NULL != (K) ? (K) : (M)); \
  const libxsmm_blasint *const libxsmm_xgemm_n_ = (NULL != (N) ? (N) : libxsmm_xgemm_k_); \
  const libxsmm_blasint libxsmm_xgemm_lda_ = LIBXSMM_MAX(NULL != ((void*)(LDA)) ? *(LDA) : \
    *(0 == (LIBXSMM_GEMM_FLAG_TRANS_A & libxsmm_xgemm_flags_) ? (M) : libxsmm_xgemm_k_), 1); \
  const libxsmm_blasint libxsmm_xgemm_ldb_ = LIBXSMM_MAX(NULL != ((void*)(LDB)) ? *(LDB) : \
    *(0 == (LIBXSMM_GEMM_FLAG_TRANS_B & libxsmm_xgemm_flags_) ? libxsmm_xgemm_k_ : libxsmm_xgemm_n_), 1); \
  const libxsmm_blasint libxsmm_xgemm_ldc_ = LIBXSMM_MAX(NULL != (LDC) ? *(LDC) : *(M), 1); \
  if (LIBXSMM_SMM(*(M), *libxsmm_xgemm_n_, *libxsmm_xgemm_k_, 2/*RFO*/, sizeof(OTYPE))) { \
    const LIBXSMM_MMFUNCTION_TYPE2(ITYPE, OTYPE) libxsmm_mmfunction_ = LIBXSMM_MMDISPATCH_SYMBOL2(ITYPE, OTYPE)( \
      *(M), *libxsmm_xgemm_n_, *libxsmm_xgemm_k_, &libxsmm_xgemm_lda_, &libxsmm_xgemm_ldb_, &libxsmm_xgemm_ldc_, \
      (const OTYPE*)(ALPHA), (const OTYPE*)(BETA), &libxsmm_xgemm_flags_, NULL); \
    if (NULL != libxsmm_mmfunction_) { \
      LIBXSMM_MMCALL_LDX(libxsmm_mmfunction_, (const ITYPE*)(A), (const ITYPE*)(B), (OTYPE*)(C), \
        *(M), *libxsmm_xgemm_n_, *libxsmm_xgemm_k_, libxsmm_xgemm_lda_, libxsmm_xgemm_ldb_, libxsmm_xgemm_ldc_); \
    } \
    else { \
      const char libxsmm_xgemm_transa_ = (char)(0 == (LIBXSMM_GEMM_FLAG_TRANS_A & libxsmm_xgemm_flags_) ? 'n' : 't'); \
      const char libxsmm_xgemm_transb_ = (char)(0 == (LIBXSMM_GEMM_FLAG_TRANS_B & libxsmm_xgemm_flags_) ? 'n' : 't'); \
      const OTYPE libxsmm_xgemm_alpha_ = (NULL != ((void*)(ALPHA)) ? (*(const OTYPE*)(ALPHA)) : ((OTYPE)LIBXSMM_ALPHA)); \
      const OTYPE libxsmm_xgemm_beta_  = (NULL != ((void*)(BETA))  ? (*(const OTYPE*)(BETA))  : ((OTYPE)LIBXSMM_BETA)); \
      LIBXSMM_XGEMM_FALLBACK0(ITYPE, OTYPE, &libxsmm_xgemm_transa_, &libxsmm_xgemm_transb_, \
        M, libxsmm_xgemm_n_, libxsmm_xgemm_k_, \
        &libxsmm_xgemm_alpha_, A, &libxsmm_xgemm_lda_, \
                               B, &libxsmm_xgemm_ldb_, \
         &libxsmm_xgemm_beta_, C, &libxsmm_xgemm_ldc_); \
    } \
  } \
  else { \
    const char libxsmm_xgemm_transa_ = (char)(0 == (LIBXSMM_GEMM_FLAG_TRANS_A & libxsmm_xgemm_flags_) ? 'n' : 't'); \
    const char libxsmm_xgemm_transb_ = (char)(0 == (LIBXSMM_GEMM_FLAG_TRANS_B & libxsmm_xgemm_flags_) ? 'n' : 't'); \
    const OTYPE libxsmm_xgemm_alpha_ = (NULL != ((void*)(ALPHA)) ? (*(const OTYPE*)(ALPHA)) : ((OTYPE)LIBXSMM_ALPHA)); \
    const OTYPE libxsmm_xgemm_beta_  = (NULL != ((void*)(BETA))  ? (*(const OTYPE*)(BETA))  : ((OTYPE)LIBXSMM_BETA)); \
    LIBXSMM_XGEMM_FALLBACK1(ITYPE, OTYPE, &libxsmm_xgemm_transa_, &libxsmm_xgemm_transb_, \
      M, libxsmm_xgemm_n_, libxsmm_xgemm_k_, \
      &libxsmm_xgemm_alpha_, A, &libxsmm_xgemm_lda_, \
                             B, &libxsmm_xgemm_ldb_, \
       &libxsmm_xgemm_beta_, C, &libxsmm_xgemm_ldc_); \
  } \
}

/** Helper macro to setup a matrix with some initial values. */
#define LIBXSMM_MATINIT_AUX(OMP, TYPE, SEED, DST, NROWS, NCOLS, LD, SCALE) { \
  /*const*/ double libxsmm_matinit_seed_ = (double)SEED; /* avoid constant conditional */ \
  const double libxsmm_matinit_scale_ = (SCALE) * libxsmm_matinit_seed_ + (SCALE); \
  const libxsmm_blasint libxsmm_matinit_ld_ = (libxsmm_blasint)LD; \
  libxsmm_blasint libxsmm_matinit_i_, libxsmm_matinit_j_; \
  if (0 != libxsmm_matinit_seed_) { \
    OMP(parallel for private(libxsmm_matinit_i_, libxsmm_matinit_j_)) \
    for (libxsmm_matinit_i_ = 0; libxsmm_matinit_i_ < ((libxsmm_blasint)NCOLS); ++libxsmm_matinit_i_) { \
      for (libxsmm_matinit_j_ = 0; libxsmm_matinit_j_ < ((libxsmm_blasint)NROWS); ++libxsmm_matinit_j_) { \
        const libxsmm_blasint libxsmm_matinit_k_ = libxsmm_matinit_i_ * libxsmm_matinit_ld_ + libxsmm_matinit_j_; \
        (DST)[libxsmm_matinit_k_] = (TYPE)(libxsmm_matinit_scale_ / (1.0 + libxsmm_matinit_k_)); \
      } \
      for (; libxsmm_matinit_j_ < libxsmm_matinit_ld_; ++libxsmm_matinit_j_) { \
        const libxsmm_blasint libxsmm_matinit_k_ = libxsmm_matinit_i_ * libxsmm_matinit_ld_ + libxsmm_matinit_j_; \
        (DST)[libxsmm_matinit_k_] = (TYPE)SEED; \
      } \
    } \
  } \
  else { /* shuffle based initialization */ \
    const unsigned int libxsmm_matinit_maxval_ = ((unsigned int)NCOLS) * ((unsigned int)libxsmm_matinit_ld_); \
    const TYPE libxsmm_matinit_maxval2_ = (TYPE)(libxsmm_matinit_maxval_ / 2), libxsmm_matinit_inv_ = (TYPE)((SCALE) / libxsmm_matinit_maxval2_); \
    const size_t libxsmm_matinit_shuffle_ = libxsmm_shuffle(libxsmm_matinit_maxval_); \
    LIBXSMM_OMP_VAR(libxsmm_matinit_j_); OMP(parallel for private(libxsmm_matinit_i_, libxsmm_matinit_j_)) \
    for (libxsmm_matinit_i_ = 0; libxsmm_matinit_i_ < ((libxsmm_blasint)NCOLS); ++libxsmm_matinit_i_) { \
      for (libxsmm_matinit_j_ = 0; libxsmm_matinit_j_ < libxsmm_matinit_ld_; ++libxsmm_matinit_j_) { \
        const libxsmm_blasint libxsmm_matinit_k_ = libxsmm_matinit_i_ * libxsmm_matinit_ld_ + libxsmm_matinit_j_; \
        (DST)[libxsmm_matinit_k_] = libxsmm_matinit_inv_ * /* normalize values to an interval of [-1, +1] */ \
          ((TYPE)(libxsmm_matinit_shuffle_ * libxsmm_matinit_k_ % libxsmm_matinit_maxval_) - libxsmm_matinit_maxval2_); \
      } \
    } \
  } \
}

#define LIBXSMM_MATINIT(TYPE, SEED, DST, NROWS, NCOLS, LD, SCALE) \
  LIBXSMM_MATINIT_AUX(LIBXSMM_ELIDE, TYPE, SEED, DST, NROWS, NCOLS, LD, SCALE)
#define LIBXSMM_MATINIT_SEQ(TYPE, SEED, DST, NROWS, NCOLS, LD, SCALE) \
  LIBXSMM_MATINIT(TYPE, SEED, DST, NROWS, NCOLS, LD, SCALE)
#define LIBXSMM_MATINIT_OMP(TYPE, SEED, DST, NROWS, NCOLS, LD, SCALE) \
  LIBXSMM_MATINIT_AUX(LIBXSMM_PRAGMA_OMP, TYPE, SEED, DST, NROWS, NCOLS, LD, SCALE)

/** Call libxsmm_gemm_print using LIBXSMM's GEMM-flags. */
#define LIBXSMM_GEMM_PRINT(OSTREAM, PRECISION, FLAGS, M, N, K, DALPHA, A, LDA, B, LDB, DBETA, C, LDC) \
  LIBXSMM_GEMM_PRINT2(OSTREAM, PRECISION, PRECISION, FLAGS, M, N, K, DALPHA, A, LDA, B, LDB, DBETA, C, LDC)
#define LIBXSMM_GEMM_PRINT2(OSTREAM, IPREC, OPREC, FLAGS, M, N, K, DALPHA, A, LDA, B, LDB, DBETA, C, LDC) \
  libxsmm_gemm_dprint2(OSTREAM, (libxsmm_gemm_precision)(IPREC), (libxsmm_gemm_precision)(OPREC), \
    /* Use 'n' (instead of 'N') avoids warning about "no macro replacement within a character constant". */ \
    (char)(0 == (LIBXSMM_GEMM_FLAG_TRANS_A & (FLAGS)) ? 'n' : 't'), \
    (char)(0 == (LIBXSMM_GEMM_FLAG_TRANS_B & (FLAGS)) ? 'n' : 't'), \
    M, N, K, DALPHA, A, LDA, B, LDB, DBETA, C, LDC)

/**
 * Utility function, which either prints information about the GEMM call
 * or dumps (FILE/ostream=0) all input and output data into MHD files.
 * The Meta Image Format (MHD) is suitable for visual inspection using e.g.,
 * ITK-SNAP or ParaView.
 */
LIBXSMM_API void libxsmm_gemm_print(void* ostream,
  libxsmm_gemm_precision precision, const char* transa, const char* transb,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const void* alpha, const void* a, const libxsmm_blasint* lda,
  const void* b, const libxsmm_blasint* ldb,
  const void* beta, void* c, const libxsmm_blasint* ldc);
LIBXSMM_API void libxsmm_gemm_print2(void* ostream,
  libxsmm_gemm_precision iprec, libxsmm_gemm_precision oprec, const char* transa, const char* transb,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const void* alpha, const void* a, const libxsmm_blasint* lda,
  const void* b, const libxsmm_blasint* ldb,
  const void* beta, void* c, const libxsmm_blasint* ldc);
LIBXSMM_API void libxsmm_gemm_dprint(void* ostream,
  libxsmm_gemm_precision precision, char transa, char transb,
  libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k,
  double dalpha, const void* a, libxsmm_blasint lda,
  const void* b, libxsmm_blasint ldb,
  double dbeta, void* c, libxsmm_blasint ldc);
LIBXSMM_API void libxsmm_gemm_dprint2(void* ostream,
  libxsmm_gemm_precision iprec, libxsmm_gemm_precision oprec, char transa, char transb,
  libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k,
  double dalpha, const void* a, libxsmm_blasint lda,
  const void* b, libxsmm_blasint ldb,
  double dbeta, void* c, libxsmm_blasint ldc);
LIBXSMM_API void libxsmm_gemm_xprint(void* ostream,
  libxsmm_xmmfunction kernel, const void* a, const void* b, void* c);

/** GEMM: fall-back prototype functions served by any compliant LAPACK/BLAS. */
LIBXSMM_EXTERN_C typedef LIBXSMM_RETARGETABLE void (*libxsmm_dgemm_function)(LIBXSMM_BLAS_SYMBOL_SIGNATURE(const, double, gemm));
LIBXSMM_EXTERN_C typedef LIBXSMM_RETARGETABLE void (*libxsmm_sgemm_function)(LIBXSMM_BLAS_SYMBOL_SIGNATURE(const, float,  gemm));
/** GEMV: fall-back prototype functions served by any compliant LAPACK/BLAS. */
LIBXSMM_EXTERN_C typedef LIBXSMM_RETARGETABLE void (*libxsmm_dgemv_function)(LIBXSMM_BLAS_SYMBOL_SIGNATURE(const, double, gemv));
LIBXSMM_EXTERN_C typedef LIBXSMM_RETARGETABLE void (*libxsmm_sgemv_function)(LIBXSMM_BLAS_SYMBOL_SIGNATURE(const, float,  gemv));

/** The original BLAS functions. */
LIBXSMM_APIVAR_ALIGNED(/*volatile*/libxsmm_dgemm_function libxsmm_original_dgemm_function);
LIBXSMM_APIVAR_ALIGNED(/*volatile*/libxsmm_sgemm_function libxsmm_original_sgemm_function);
LIBXSMM_APIVAR_ALIGNED(/*volatile*/libxsmm_dgemv_function libxsmm_original_dgemv_function);
LIBXSMM_APIVAR_ALIGNED(/*volatile*/libxsmm_sgemv_function libxsmm_original_sgemv_function);
LIBXSMM_API_EXPORT libxsmm_dgemm_function libxsmm_original_dgemm(void);
LIBXSMM_API_EXPORT libxsmm_sgemm_function libxsmm_original_sgemm(void);

/* Helper macro to eventually (if defined) call libxsmm_init */
#if defined(LIBXSMM_INIT) || defined(LIBXSMM_CTOR)
# define LIBXSMM_GEMM_SYMBOL(TYPE) LIBXSMM_CONCATENATE(libxsmm_original_, LIBXSMM_TPREFIX(TYPE, gemm_function))
# define LIBXSMM_GEMV_SYMBOL(TYPE) LIBXSMM_CONCATENATE(libxsmm_original_, LIBXSMM_TPREFIX(TYPE, gemv_function))
# undef LIBXSMM_INIT
# define LIBXSMM_INIT LIBXSMM_ASSERT_MSG(0 != libxsmm_ninit, "LIBXSMM is not initialized");
# define LIBXSMM_INIT_COMPLETED
#else
# define LIBXSMM_INIT if (0 == libxsmm_ninit) libxsmm_init();
# define LIBXSMM_GEMM_SYMBOL(TYPE) LIBXSMM_BLAS_FUNCTION(TYPE, TYPE, gemm)
# define LIBXSMM_GEMV_SYMBOL(TYPE) LIBXSMM_BLAS_FUNCTION(TYPE, TYPE, gemv)
#endif

/**
 * General dense matrix multiplication, which re-exposes LAPACK/BLAS
 * but allows to rely on LIBXSMM's defaults (libxsmm_config.h)
 * when supplying NULL-arguments in certain places.
 */
LIBXSMM_API void libxsmm_blas_xgemm(libxsmm_gemm_precision iprec, libxsmm_gemm_precision oprec,
  const char* transa, const char* transb, const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const void* alpha, const void* a, const libxsmm_blasint* lda,
  const void* b, const libxsmm_blasint* ldb,
  const void* beta, void* c, const libxsmm_blasint* ldc);

#define libxsmm_blas_dgemm(TRANSA, TRANSB, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC) \
  libxsmm_blas_xgemm(LIBXSMM_GEMM_PRECISION_F64, LIBXSMM_GEMM_PRECISION_F64, \
    TRANSA, TRANSB, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC)
#define libxsmm_blas_sgemm(TRANSA, TRANSB, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC) \
  libxsmm_blas_xgemm(LIBXSMM_GEMM_PRECISION_F32, LIBXSMM_GEMM_PRECISION_F32, \
    TRANSA, TRANSB, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC)

#define libxsmm_dgemm_omp(TRANSA, TRANSB, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC) \
  libxsmm_xgemm_omp(LIBXSMM_GEMM_PRECISION_F64, LIBXSMM_GEMM_PRECISION_F64, \
    TRANSA, TRANSB, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC)
#define libxsmm_sgemm_omp(TRANSA, TRANSB, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC) \
  libxsmm_xgemm_omp(LIBXSMM_GEMM_PRECISION_F32, LIBXSMM_GEMM_PRECISION_F32, \
    TRANSA, TRANSB, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC)

/** Translates GEMM prefetch request into prefetch-enumeration (incl. FE's auto-prefetch). */
LIBXSMM_API libxsmm_gemm_prefetch_type libxsmm_get_gemm_xprefetch(const int* prefetch);
LIBXSMM_API libxsmm_gemm_prefetch_type libxsmm_get_gemm_prefetch(int prefetch);

#endif /*LIBXSMM_FRONTEND_H*/

