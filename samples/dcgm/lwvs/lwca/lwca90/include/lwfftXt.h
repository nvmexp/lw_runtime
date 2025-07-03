
 /* Copyright 2005-2014 LWPU Corporation.  All rights reserved.
  *
  * NOTICE TO LICENSEE:
  *
  * The source code and/or documentation ("Licensed Deliverables") are
  * subject to LWPU intellectual property rights under U.S. and
  * international Copyright laws.
  *
  * The Licensed Deliverables contained herein are PROPRIETARY and
  * CONFIDENTIAL to LWPU and are being provided under the terms and
  * conditions of a form of LWPU software license agreement by and
  * between LWPU and Licensee ("License Agreement") or electronically
  * accepted by Licensee.  Notwithstanding any terms or conditions to
  * the contrary in the License Agreement, reproduction or disclosure
  * of the Licensed Deliverables to any third party without the express
  * written consent of LWPU is prohibited.
  *
  * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
  * LICENSE AGREEMENT, LWPU MAKES NO REPRESENTATION ABOUT THE
  * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  THEY ARE
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
  * C.F.R. 12.212 (SEPT 1995) and are provided to the U.S. Government
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

/*!
* \file lwfftXt.h  
* \brief Public header file for the LWPU LWCA FFT library (LWFFT)  
*/ 

#ifndef _LWFFTXT_H_
#define _LWFFTXT_H_
#include "lwdalibxt.h"
#include "lwfft.h"


#ifndef LWFFTAPI
#ifdef _WIN32
#define LWFFTAPI __stdcall
#else
#define LWFFTAPI 
#endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

//
// lwfftXtSubFormat identifies the data layout of 
// a memory descriptor owned by lwfft.
// note that multi GPU lwfft does not yet support out-of-place transforms
//

typedef enum lwfftXtSubFormat_t {
    LWFFT_XT_FORMAT_INPUT = 0x00,              //by default input is in linear order across GPUs
    LWFFT_XT_FORMAT_OUTPUT = 0x01,             //by default output is in scrambled order depending on transform
    LWFFT_XT_FORMAT_INPLACE = 0x02,            //by default inplace is input order, which is linear across GPUs
    LWFFT_XT_FORMAT_INPLACE_SHUFFLED = 0x03,   //shuffled output order after exelwtion of the transform
    LWFFT_XT_FORMAT_1D_INPUT_SHUFFLED = 0x04,  //shuffled input order prior to exelwtion of 1D transforms
    LWFFT_FORMAT_UNDEFINED = 0x05
} lwfftXtSubFormat;

//
// lwfftXtCopyType specifies the type of copy for lwfftXtMemcpy
//
typedef enum lwfftXtCopyType_t {
    LWFFT_COPY_HOST_TO_DEVICE = 0x00,
    LWFFT_COPY_DEVICE_TO_HOST = 0x01,
    LWFFT_COPY_DEVICE_TO_DEVICE = 0x02,
    LWFFT_COPY_UNDEFINED = 0x03
} lwfftXtCopyType;

//
// lwfftXtQueryType specifies the type of query for lwfftXtQueryPlan
//
typedef enum lwfftXtQueryType_t {
    LWFFT_QUERY_1D_FACTORS = 0x00,
    LWFFT_QUERY_UNDEFINED = 0x01
} lwfftXtQueryType;

typedef struct lwfftXt1dFactors_t {
    long long int size;
    long long int stringCount;
    long long int stringLength;
    long long int substringLength;
    long long int factor1;
    long long int factor2;
    long long int stringMask;
    long long int substringMask;
    long long int factor1Mask;
    long long int factor2Mask;
    int stringShift;
    int substringShift;
    int factor1Shift;
    int factor2Shift;
} lwfftXt1dFactors;

// multi-GPU routines
lwfftResult LWFFTAPI lwfftXtSetGPUs(lwfftHandle handle, int nGPUs, int *whichGPUs);

lwfftResult LWFFTAPI lwfftXtMalloc(lwfftHandle plan, 
                                   lwdaLibXtDesc ** descriptor,
                                   lwfftXtSubFormat format);

lwfftResult LWFFTAPI lwfftXtMemcpy(lwfftHandle plan, 
                                   void *dstPointer, 
                                   void *srcPointer,
                                   lwfftXtCopyType type);
                                   
lwfftResult LWFFTAPI lwfftXtFree(lwdaLibXtDesc *descriptor);

lwfftResult LWFFTAPI lwfftXtSetWorkArea(lwfftHandle plan, void **workArea);

lwfftResult LWFFTAPI lwfftXtExecDescriptorC2C(lwfftHandle plan, 
                                              lwdaLibXtDesc *input,
                                              lwdaLibXtDesc *output, 
                                              int direction);

lwfftResult LWFFTAPI lwfftXtExecDescriptorR2C(lwfftHandle plan, 
                                              lwdaLibXtDesc *input,
                                              lwdaLibXtDesc *output);     
                                        
lwfftResult LWFFTAPI lwfftXtExecDescriptorC2R(lwfftHandle plan, 
                                              lwdaLibXtDesc *input,
                                              lwdaLibXtDesc *output);
                                  
lwfftResult LWFFTAPI lwfftXtExecDescriptorZ2Z(lwfftHandle plan, 
                                              lwdaLibXtDesc *input,
                                              lwdaLibXtDesc *output, 
                                              int direction);

lwfftResult LWFFTAPI lwfftXtExecDescriptorD2Z(lwfftHandle plan, 
                                              lwdaLibXtDesc *input,
                                              lwdaLibXtDesc *output);     
                                        
lwfftResult LWFFTAPI lwfftXtExecDescriptorZ2D(lwfftHandle plan, 
                                              lwdaLibXtDesc *input,
                                              lwdaLibXtDesc *output);
   
// Utility functions
                                              
lwfftResult LWFFTAPI lwfftXtQueryPlan(lwfftHandle plan, void *queryStruct, lwfftXtQueryType queryType);


// callbacks


typedef enum lwfftXtCallbackType_t {
    LWFFT_CB_LD_COMPLEX = 0x0,
    LWFFT_CB_LD_COMPLEX_DOUBLE = 0x1,
    LWFFT_CB_LD_REAL = 0x2,
    LWFFT_CB_LD_REAL_DOUBLE = 0x3,
    LWFFT_CB_ST_COMPLEX = 0x4,
    LWFFT_CB_ST_COMPLEX_DOUBLE = 0x5,
    LWFFT_CB_ST_REAL = 0x6,
    LWFFT_CB_ST_REAL_DOUBLE = 0x7,
    LWFFT_CB_UNDEFINED = 0x8

} lwfftXtCallbackType;

typedef lwfftComplex (*lwfftCallbackLoadC)(void *dataIn, size_t offset, void *callerInfo, void *sharedPointer);
typedef lwfftDoubleComplex (*lwfftCallbackLoadZ)(void *dataIn, size_t offset, void *callerInfo, void *sharedPointer);
typedef lwfftReal (*lwfftCallbackLoadR)(void *dataIn, size_t offset, void *callerInfo, void *sharedPointer);
typedef lwfftDoubleReal(*lwfftCallbackLoadD)(void *dataIn, size_t offset, void *callerInfo, void *sharedPointer);

typedef void (*lwfftCallbackStoreC)(void *dataOut, size_t offset, lwfftComplex element, void *callerInfo, void *sharedPointer);
typedef void (*lwfftCallbackStoreZ)(void *dataOut, size_t offset, lwfftDoubleComplex element, void *callerInfo, void *sharedPointer);
typedef void (*lwfftCallbackStoreR)(void *dataOut, size_t offset, lwfftReal element, void *callerInfo, void *sharedPointer);
typedef void (*lwfftCallbackStoreD)(void *dataOut, size_t offset, lwfftDoubleReal element, void *callerInfo, void *sharedPointer);


lwfftResult LWFFTAPI lwfftXtSetCallback(lwfftHandle plan, void **callback_routine, lwfftXtCallbackType cbType, void **caller_info);
lwfftResult LWFFTAPI lwfftXtClearCallback(lwfftHandle plan, lwfftXtCallbackType cbType);
lwfftResult LWFFTAPI lwfftXtSetCallbackSharedSize(lwfftHandle plan, lwfftXtCallbackType cbType, size_t sharedSize);

lwfftResult LWFFTAPI lwfftXtMakePlanMany(lwfftHandle plan,
                                         int rank,
                                         long long int *n,
                                         long long int *inembed,
                                         long long int istride,
                                         long long int idist,
                                         lwdaDataType inputtype,
                                         long long int *onembed,
                                         long long int ostride,
                                         long long int odist,
                                         lwdaDataType outputtype,
                                         long long int batch,
                                         size_t *workSize,
                                       	 lwdaDataType exelwtiontype);

lwfftResult LWFFTAPI lwfftXtGetSizeMany(lwfftHandle plan,
                                        int rank,
                                        long long int *n,
                                        long long int *inembed, 
                                        long long int istride, 
                                        long long int idist,
                                        lwdaDataType inputtype,
                                        long long int *onembed, 
                                        long long int ostride, 
                                        long long int odist,
                                        lwdaDataType outputtype,
                                        long long int batch,
                                        size_t *workSize,
                                        lwdaDataType exelwtiontype);

lwfftResult LWFFTAPI lwfftXtExec(lwfftHandle plan,
                                 void *input,
                                 void *output,
                                 int direction);

lwfftResult LWFFTAPI lwfftXtExecDescriptor(lwfftHandle plan,
                                           lwdaLibXtDesc *input,
                                           lwdaLibXtDesc *output,
                                           int direction);

#ifdef __cplusplus
}
#endif

#endif
