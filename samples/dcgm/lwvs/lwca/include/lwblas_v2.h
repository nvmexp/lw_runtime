/*
 * Copyright 1993-2014 LWPU Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to LWPU intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to LWPU and is being provided under the terms and
 * conditions of a form of LWPU software license agreement by and
 * between LWPU and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of LWPU is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, LWPU MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
 * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
 * LWPU DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
 * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, IN NO EVENT SHALL LWPU BE LIABLE FOR ANY
 * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THESE LICENSED DELIVERABLES.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include, in the user documentation and internal
 * comments to the code, the above Disclaimer and U.S. Government End
 * Users Notice.
 */
 
/*
 * This is the public header file for the new LWBLAS library API, it mapped the generic 
 * Lwblas name functions to the actual _v2 implementations.
 */

#if !defined(LWBLAS_V2_H_)
#define LWBLAS_V2_H_

#undef LWBLASAPI
#ifdef __LWDACC__
#define LWBLASAPI __host__ __device__
#else
#define LWBLASAPI
#endif

#include "lwblas_api.h"

#define lwblasCreate         lwblasCreate_v2
#define lwblasDestroy        lwblasDestroy_v2
#define lwblasGetVersion     lwblasGetVersion_v2
#define lwblasSetStream      lwblasSetStream_v2
#define lwblasGetStream      lwblasGetStream_v2
#define lwblasGetPointerMode lwblasGetPointerMode_v2
#define lwblasSetPointerMode lwblasSetPointerMode_v2

/* Blas3 Routines   */

#define lwblasSnrm2          lwblasSnrm2_v2
#define lwblasDnrm2          lwblasDnrm2_v2 
#define lwblasScnrm2         lwblasScnrm2_v2
#define lwblasDznrm2         lwblasDznrm2_v2

#define lwblasSdot           lwblasSdot_v2
#define lwblasDdot           lwblasDdot_v2
#define lwblasCdotu          lwblasCdotu_v2
#define lwblasCdotc          lwblasCdotc_v2
#define lwblasZdotu          lwblasZdotu_v2
#define lwblasZdotc          lwblasZdotc_v2

#define lwblasSscal          lwblasSscal_v2
#define lwblasDscal          lwblasDscal_v2
#define lwblasCscal          lwblasCscal_v2
#define lwblasCsscal         lwblasCsscal_v2
#define lwblasZscal          lwblasZscal_v2
#define lwblasZdscal         lwblasZdscal_v2

#define lwblasSaxpy          lwblasSaxpy_v2
#define lwblasDaxpy          lwblasDaxpy_v2
#define lwblasCaxpy          lwblasCaxpy_v2
#define lwblasZaxpy          lwblasZaxpy_v2

#define lwblasScopy          lwblasScopy_v2
#define lwblasDcopy          lwblasDcopy_v2
#define lwblasCcopy          lwblasCcopy_v2
#define lwblasZcopy          lwblasZcopy_v2

#define lwblasSswap          lwblasSswap_v2
#define lwblasDswap          lwblasDswap_v2
#define lwblasCswap          lwblasCswap_v2
#define lwblasZswap          lwblasZswap_v2

#define lwblasIsamax         lwblasIsamax_v2
#define lwblasIdamax         lwblasIdamax_v2
#define lwblasIcamax         lwblasIcamax_v2
#define lwblasIzamax         lwblasIzamax_v2
 
#define lwblasIsamin         lwblasIsamin_v2
#define lwblasIdamin         lwblasIdamin_v2
#define lwblasIcamin         lwblasIcamin_v2
#define lwblasIzamin         lwblasIzamin_v2
                         
#define lwblasSasum          lwblasSasum_v2
#define lwblasDasum          lwblasDasum_v2
#define lwblasScasum         lwblasScasum_v2
#define lwblasDzasum         lwblasDzasum_v2

#define lwblasSrot           lwblasSrot_v2 
#define lwblasDrot           lwblasDrot_v2 
#define lwblasCrot           lwblasCrot_v2 
#define lwblasCsrot          lwblasCsrot_v2
#define lwblasZrot           lwblasZrot_v2 
#define lwblasZdrot          lwblasZdrot_v2

#define lwblasSrotg          lwblasSrotg_v2
#define lwblasDrotg          lwblasDrotg_v2
#define lwblasCrotg          lwblasCrotg_v2
#define lwblasZrotg          lwblasZrotg_v2

#define lwblasSrotm          lwblasSrotm_v2 
#define lwblasDrotm          lwblasDrotm_v2 
                                
#define lwblasSrotmg         lwblasSrotmg_v2 
#define lwblasDrotmg         lwblasDrotmg_v2 


/* Blas2 Routines */

#define lwblasSgemv          lwblasSgemv_v2
#define lwblasDgemv          lwblasDgemv_v2
#define lwblasCgemv          lwblasCgemv_v2
#define lwblasZgemv          lwblasZgemv_v2

#define lwblasSgbmv          lwblasSgbmv_v2
#define lwblasDgbmv          lwblasDgbmv_v2
#define lwblasCgbmv          lwblasCgbmv_v2
#define lwblasZgbmv          lwblasZgbmv_v2

#define lwblasStrmv          lwblasStrmv_v2
#define lwblasDtrmv          lwblasDtrmv_v2
#define lwblasCtrmv          lwblasCtrmv_v2
#define lwblasZtrmv          lwblasZtrmv_v2

#define lwblasStbmv          lwblasStbmv_v2
#define lwblasDtbmv          lwblasDtbmv_v2
#define lwblasCtbmv          lwblasCtbmv_v2
#define lwblasZtbmv          lwblasZtbmv_v2

#define lwblasStpmv          lwblasStpmv_v2
#define lwblasDtpmv          lwblasDtpmv_v2
#define lwblasCtpmv          lwblasCtpmv_v2
#define lwblasZtpmv          lwblasZtpmv_v2

#define lwblasStrsv          lwblasStrsv_v2
#define lwblasDtrsv          lwblasDtrsv_v2
#define lwblasCtrsv          lwblasCtrsv_v2
#define lwblasZtrsv          lwblasZtrsv_v2

#define lwblasStpsv          lwblasStpsv_v2
#define lwblasDtpsv          lwblasDtpsv_v2
#define lwblasCtpsv          lwblasCtpsv_v2
#define lwblasZtpsv          lwblasZtpsv_v2

#define lwblasStbsv          lwblasStbsv_v2
#define lwblasDtbsv          lwblasDtbsv_v2
#define lwblasCtbsv          lwblasCtbsv_v2
#define lwblasZtbsv          lwblasZtbsv_v2

#define lwblasSsymv          lwblasSsymv_v2
#define lwblasDsymv          lwblasDsymv_v2
#define lwblasCsymv          lwblasCsymv_v2
#define lwblasZsymv          lwblasZsymv_v2
#define lwblasChemv          lwblasChemv_v2
#define lwblasZhemv          lwblasZhemv_v2

#define lwblasSsbmv          lwblasSsbmv_v2
#define lwblasDsbmv          lwblasDsbmv_v2
#define lwblasChbmv          lwblasChbmv_v2
#define lwblasZhbmv          lwblasZhbmv_v2

#define lwblasSspmv          lwblasSspmv_v2
#define lwblasDspmv          lwblasDspmv_v2
#define lwblasChpmv          lwblasChpmv_v2
#define lwblasZhpmv          lwblasZhpmv_v2


#define lwblasSger           lwblasSger_v2
#define lwblasDger           lwblasDger_v2
#define lwblasCgeru          lwblasCgeru_v2
#define lwblasCgerc          lwblasCgerc_v2
#define lwblasZgeru          lwblasZgeru_v2
#define lwblasZgerc          lwblasZgerc_v2

#define lwblasSsyr           lwblasSsyr_v2
#define lwblasDsyr           lwblasDsyr_v2
#define lwblasCsyr           lwblasCsyr_v2
#define lwblasZsyr           lwblasZsyr_v2
#define lwblasCher           lwblasCher_v2
#define lwblasZher           lwblasZher_v2

#define lwblasSspr           lwblasSspr_v2
#define lwblasDspr           lwblasDspr_v2
#define lwblasChpr           lwblasChpr_v2
#define lwblasZhpr           lwblasZhpr_v2

#define lwblasSsyr2          lwblasSsyr2_v2
#define lwblasDsyr2          lwblasDsyr2_v2
#define lwblasCsyr2          lwblasCsyr2_v2
#define lwblasZsyr2          lwblasZsyr2_v2
#define lwblasCher2          lwblasCher2_v2
#define lwblasZher2          lwblasZher2_v2

#define lwblasSspr2          lwblasSspr2_v2
#define lwblasDspr2          lwblasDspr2_v2
#define lwblasChpr2          lwblasChpr2_v2
#define lwblasZhpr2          lwblasZhpr2_v2

/* Blas3 Routines   */

#define lwblasSgemm          lwblasSgemm_v2
#define lwblasDgemm          lwblasDgemm_v2
#define lwblasCgemm          lwblasCgemm_v2
#define lwblasZgemm          lwblasZgemm_v2

#define lwblasSsyrk          lwblasSsyrk_v2
#define lwblasDsyrk          lwblasDsyrk_v2
#define lwblasCsyrk          lwblasCsyrk_v2
#define lwblasZsyrk          lwblasZsyrk_v2
#define lwblasCherk          lwblasCherk_v2
#define lwblasZherk          lwblasZherk_v2

#define lwblasSsyr2k         lwblasSsyr2k_v2
#define lwblasDsyr2k         lwblasDsyr2k_v2
#define lwblasCsyr2k         lwblasCsyr2k_v2
#define lwblasZsyr2k         lwblasZsyr2k_v2
#define lwblasCher2k         lwblasCher2k_v2
#define lwblasZher2k         lwblasZher2k_v2

#define lwblasSsymm          lwblasSsymm_v2
#define lwblasDsymm          lwblasDsymm_v2
#define lwblasCsymm          lwblasCsymm_v2
#define lwblasZsymm          lwblasZsymm_v2
#define lwblasChemm          lwblasChemm_v2
#define lwblasZhemm          lwblasZhemm_v2

#define lwblasStrsm          lwblasStrsm_v2
#define lwblasDtrsm          lwblasDtrsm_v2
#define lwblasCtrsm          lwblasCtrsm_v2
#define lwblasZtrsm          lwblasZtrsm_v2

#define lwblasStrmm          lwblasStrmm_v2
#define lwblasDtrmm          lwblasDtrmm_v2
#define lwblasCtrmm          lwblasCtrmm_v2
#define lwblasZtrmm          lwblasZtrmm_v2

#endif /* !defined(LWBLAS_V2_H_) */
