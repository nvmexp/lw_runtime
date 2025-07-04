 /* Copyright 2010-2016 LWPU Corporation.  All rights reserved. 
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
#ifndef LW_NPPS_COLWERSION_FUNCTIONS_H
#define LW_NPPS_COLWERSION_FUNCTIONS_H
 
/**
 * \file npps_colwersion_functions.h
 * NPP Signal Processing Functionality.
 */
 
#include "nppdefs.h"


#ifdef __cplusplus
extern "C" {
#endif


/** @defgroup signal_colwersion_functions Colwersion Functions
 *  @ingroup npps
 *
 * @{
 *
 */

/** @defgroup signal_colwert Colwert
 *
 * @{
 *
 */

/** @name Colwert
 * Routines for colwerting the sample-data type of signals.
 *
 * @{
 *
 */

NppStatus 
nppsColwert_8s16s(const Npp8s * pSrc, Npp16s * pDst, int nLength);

NppStatus 
nppsColwert_8s32f(const Npp8s * pSrc, Npp32f * pDst, int nLength);

NppStatus 
nppsColwert_8u32f(const Npp8u * pSrc, Npp32f * pDst, int nLength);

NppStatus 
nppsColwert_16s8s_Sfs(const Npp16s * pSrc, Npp8s * pDst, Npp32u nLength, NppRoundMode eRoundMode, int nScaleFactor);

NppStatus 
nppsColwert_16s32s(const Npp16s * pSrc, Npp32s * pDst, int nLength);

NppStatus 
nppsColwert_16s32f(const Npp16s * pSrc, Npp32f * pDst, int nLength);

NppStatus 
nppsColwert_16u32f(const Npp16u * pSrc, Npp32f * pDst, int nLength);

NppStatus 
nppsColwert_32s16s(const Npp32s * pSrc, Npp16s * pDst, int nLength);

NppStatus 
nppsColwert_32s32f(const Npp32s * pSrc, Npp32f * pDst, int nLength);

NppStatus 
nppsColwert_32s64f(const Npp32s * pSrc, Npp64f * pDst, int nLength);

NppStatus 
nppsColwert_32f64f(const Npp32f * pSrc, Npp64f * pDst, int nLength);

NppStatus 
nppsColwert_64s64f(const Npp64s * pSrc, Npp64f * pDst, int nLength);

NppStatus 
nppsColwert_64f32f(const Npp64f * pSrc, Npp32f * pDst, int nLength);

NppStatus 
nppsColwert_16s32f_Sfs(const Npp16s * pSrc, Npp32f * pDst, int nLength, int nScaleFactor);

NppStatus 
nppsColwert_16s64f_Sfs(const Npp16s * pSrc, Npp64f * pDst, int nLength, int nScaleFactor);

NppStatus 
nppsColwert_32s16s_Sfs(const Npp32s * pSrc, Npp16s * pDst, int nLength, int nScaleFactor);

NppStatus 
nppsColwert_32s32f_Sfs(const Npp32s * pSrc, Npp32f * pDst, int nLength, int nScaleFactor);

NppStatus 
nppsColwert_32s64f_Sfs(const Npp32s * pSrc, Npp64f * pDst, int nLength, int nScaleFactor);

NppStatus 
nppsColwert_32f8s_Sfs(const Npp32f * pSrc, Npp8s * pDst, int nLength, NppRoundMode eRoundMode, int nScaleFactor);

NppStatus 
nppsColwert_32f8u_Sfs(const Npp32f * pSrc, Npp8u * pDst, int nLength, NppRoundMode eRoundMode, int nScaleFactor);

NppStatus 
nppsColwert_32f16s_Sfs(const Npp32f * pSrc, Npp16s * pDst, int nLength, NppRoundMode eRoundMode, int nScaleFactor);

NppStatus 
nppsColwert_32f16u_Sfs(const Npp32f * pSrc, Npp16u * pDst, int nLength, NppRoundMode eRoundMode, int nScaleFactor);

NppStatus 
nppsColwert_32f32s_Sfs(const Npp32f * pSrc, Npp32s * pDst, int nLength, NppRoundMode eRoundMode, int nScaleFactor);

NppStatus 
nppsColwert_64s32s_Sfs(const Npp64s * pSrc, Npp32s * pDst, int nLength, NppRoundMode eRoundMode, int nScaleFactor);

NppStatus 
nppsColwert_64f16s_Sfs(const Npp64f * pSrc, Npp16s * pDst, int nLength, NppRoundMode eRoundMode, int nScaleFactor);

NppStatus 
nppsColwert_64f32s_Sfs(const Npp64f * pSrc, Npp32s * pDst, int nLength, NppRoundMode eRoundMode, int nScaleFactor);

NppStatus 
nppsColwert_64f64s_Sfs(const Npp64f * pSrc, Npp64s * pDst, int nLength, NppRoundMode eRoundMode, int nScaleFactor);

/** @} end of Colwert */

/** @} signal_colwert */

/** @defgroup signal_threshold Threshold
 *
 * @{
 *
 */

/** @name Threshold Functions
 * Performs the threshold operation on the samples of a signal by limiting the sample values by a specified constant value.
 *
 * @{
 *
 */

/** 
 * 16-bit signed short signal threshold with constant level.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value to be used to limit each signal sample
 * \param nRelOp NppCmpOp type of thresholding operation (NPP_CMP_LESS or NPP_CMP_GREATER only).
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsThreshold_16s(const Npp16s * pSrc, Npp16s * pDst, int nLength, Npp16s nLevel, NppCmpOp nRelOp);

/** 
 * 16-bit in place signed short signal threshold with constant level.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value to be used to limit each signal sample
 * \param nRelOp NppCmpOp type of thresholding operation (NPP_CMP_LESS or NPP_CMP_GREATER only).
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsThreshold_16s_I(Npp16s * pSrcDst, int nLength, Npp16s nLevel, NppCmpOp nRelOp);

/** 
 * 16-bit signed short complex number signal threshold with constant level.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample
 * \param nRelOp NppCmpOp type of thresholding operation (NPP_CMP_LESS or NPP_CMP_GREATER only).
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsThreshold_16sc(const Npp16sc * pSrc, Npp16sc * pDst, int nLength, Npp16s nLevel, NppCmpOp nRelOp);

/** 
 * 16-bit in place signed short complex number signal threshold with constant level.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample
 * \param nRelOp NppCmpOp type of thresholding operation (NPP_CMP_LESS or NPP_CMP_GREATER only).
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsThreshold_16sc_I(Npp16sc * pSrcDst, int nLength, Npp16s nLevel, NppCmpOp nRelOp);

/** 
 * 32-bit floating point signal threshold with constant level.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value to be used to limit each signal sample
 * \param nRelOp NppCmpOp type of thresholding operation (NPP_CMP_LESS or NPP_CMP_GREATER only).
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsThreshold_32f(const Npp32f * pSrc, Npp32f * pDst, int nLength, Npp32f nLevel, NppCmpOp nRelOp);

/** 
 * 32-bit in place floating point signal threshold with constant level.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value to be used to limit each signal sample
 * \param nRelOp NppCmpOp type of thresholding operation (NPP_CMP_LESS or NPP_CMP_GREATER only).
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsThreshold_32f_I(Npp32f * pSrcDst, int nLength, Npp32f nLevel, NppCmpOp nRelOp);

/** 
 * 32-bit floating point complex number signal threshold with constant level.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample
 * \param nRelOp NppCmpOp type of thresholding operation (NPP_CMP_LESS or NPP_CMP_GREATER only).
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsThreshold_32fc(const Npp32fc * pSrc, Npp32fc * pDst, int nLength, Npp32f nLevel, NppCmpOp nRelOp);

/** 
 * 32-bit in place floating point complex number signal threshold with constant level.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample
 * \param nRelOp NppCmpOp type of thresholding operation (NPP_CMP_LESS or NPP_CMP_GREATER only).
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsThreshold_32fc_I(Npp32fc * pSrcDst, int nLength, Npp32f nLevel, NppCmpOp nRelOp);

/** 
 * 64-bit floating point signal threshold with constant level.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value to be used to limit each signal sample
 * \param nRelOp NppCmpOp type of thresholding operation (NPP_CMP_LESS or NPP_CMP_GREATER only).
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsThreshold_64f(const Npp64f * pSrc, Npp64f * pDst, int nLength, Npp64f nLevel, NppCmpOp nRelOp);

/** 
 * 64-bit in place floating point signal threshold with constant level.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value to be used to limit each signal sample
 * \param nRelOp NppCmpOp type of thresholding operation (NPP_CMP_LESS or NPP_CMP_GREATER only).
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsThreshold_64f_I(Npp64f * pSrcDst, int nLength, Npp64f nLevel, NppCmpOp nRelOp);

/** 
 * 64-bit floating point complex number signal threshold with constant level.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample
 * \param nRelOp NppCmpOp type of thresholding operation (NPP_CMP_LESS or NPP_CMP_GREATER only).
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsThreshold_64fc(const Npp64fc * pSrc, Npp64fc * pDst, int nLength, Npp64f nLevel, NppCmpOp nRelOp);

/** 
 * 64-bit in place floating point complex number signal threshold with constant level.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample
 * \param nRelOp NppCmpOp type of thresholding operation (NPP_CMP_LESS or NPP_CMP_GREATER only).
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsThreshold_64fc_I(Npp64fc * pSrcDst, int nLength, Npp64f nLevel, NppCmpOp nRelOp);

/** 
 * 16-bit signed short signal NPP_CMP_LESS threshold with constant level.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value to be used to limit each signal sample
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsThreshold_LT_16s(const Npp16s * pSrc, Npp16s * pDst, int nLength, Npp16s nLevel);

/** 
 * 16-bit in place signed short signal NPP_CMP_LESS threshold with constant level.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value to be used to limit each signal sample
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsThreshold_LT_16s_I(Npp16s * pSrcDst, int nLength, Npp16s nLevel);

/** 
 * 16-bit signed short complex number signal NPP_CMP_LESS threshold with constant level.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsThreshold_LT_16sc(const Npp16sc * pSrc, Npp16sc * pDst, int nLength, Npp16s nLevel);

/** 
 * 16-bit in place signed short complex number signal NPP_CMP_LESS threshold with constant level.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsThreshold_LT_16sc_I(Npp16sc * pSrcDst, int nLength, Npp16s nLevel);

/** 
 * 32-bit floating point signal NPP_CMP_LESS threshold with constant level.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value to be used to limit each signal sample
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsThreshold_LT_32f(const Npp32f * pSrc, Npp32f * pDst, int nLength, Npp32f nLevel);

/** 
 * 32-bit in place floating point signal NPP_CMP_LESS threshold with constant level.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value to be used to limit each signal sample
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsThreshold_LT_32f_I(Npp32f * pSrcDst, int nLength, Npp32f nLevel);

/** 
 * 32-bit floating point complex number signal NPP_CMP_LESS threshold with constant level.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsThreshold_LT_32fc(const Npp32fc * pSrc, Npp32fc * pDst, int nLength, Npp32f nLevel);

/** 
 * 32-bit in place floating point complex number signal NPP_CMP_LESS threshold with constant level.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsThreshold_LT_32fc_I(Npp32fc * pSrcDst, int nLength, Npp32f nLevel);

/** 
 * 64-bit floating point signal NPP_CMP_LESS threshold with constant level.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value to be used to limit each signal sample
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsThreshold_LT_64f(const Npp64f * pSrc, Npp64f * pDst, int nLength, Npp64f nLevel);

/** 
 * 64-bit in place floating point signal NPP_CMP_LESS threshold with constant level.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value to be used to limit each signal sample
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsThreshold_LT_64f_I(Npp64f * pSrcDst, int nLength, Npp64f nLevel);

/** 
 * 64-bit floating point complex number signal NPP_CMP_LESS threshold with constant level.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsThreshold_LT_64fc(const Npp64fc * pSrc, Npp64fc * pDst, int nLength, Npp64f nLevel);

/** 
 * 64-bit in place floating point complex number signal NPP_CMP_LESS threshold with constant level.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsThreshold_LT_64fc_I(Npp64fc * pSrcDst, int nLength, Npp64f nLevel);

/** 
 * 16-bit signed short signal NPP_CMP_GREATER threshold with constant level.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value to be used to limit each signal sample
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsThreshold_GT_16s(const Npp16s * pSrc, Npp16s * pDst, int nLength, Npp16s nLevel);

/** 
 * 16-bit in place signed short signal NPP_CMP_GREATER threshold with constant level.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value to be used to limit each signal sample
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsThreshold_GT_16s_I(Npp16s * pSrcDst, int nLength, Npp16s nLevel);

/** 
 * 16-bit signed short complex number signal NPP_CMP_GREATER threshold with constant level.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsThreshold_GT_16sc(const Npp16sc * pSrc, Npp16sc * pDst, int nLength, Npp16s nLevel);

/** 
 * 16-bit in place signed short complex number signal NPP_CMP_GREATER threshold with constant level.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsThreshold_GT_16sc_I(Npp16sc * pSrcDst, int nLength, Npp16s nLevel);

/** 
 * 32-bit floating point signal NPP_CMP_GREATER threshold with constant level.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value to be used to limit each signal sample
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsThreshold_GT_32f(const Npp32f * pSrc, Npp32f * pDst, int nLength, Npp32f nLevel);

/** 
 * 32-bit in place floating point signal NPP_CMP_GREATER threshold with constant level.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value to be used to limit each signal sample
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsThreshold_GT_32f_I(Npp32f * pSrcDst, int nLength, Npp32f nLevel);

/** 
 * 32-bit floating point complex number signal NPP_CMP_GREATER threshold with constant level.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsThreshold_GT_32fc(const Npp32fc * pSrc, Npp32fc * pDst, int nLength, Npp32f nLevel);

/** 
 * 32-bit in place floating point complex number signal NPP_CMP_GREATER threshold with constant level.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsThreshold_GT_32fc_I(Npp32fc * pSrcDst, int nLength, Npp32f nLevel);

/** 
 * 64-bit floating point signal NPP_CMP_GREATER threshold with constant level.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value to be used to limit each signal sample
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsThreshold_GT_64f(const Npp64f * pSrc, Npp64f * pDst, int nLength, Npp64f nLevel);

/** 
 * 64-bit in place floating point signal NPP_CMP_GREATER threshold with constant level.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value to be used to limit each signal sample
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsThreshold_GT_64f_I(Npp64f * pSrcDst, int nLength, Npp64f nLevel);

/** 
 * 64-bit floating point complex number signal NPP_CMP_GREATER threshold with constant level.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsThreshold_GT_64fc(const Npp64fc * pSrc, Npp64fc * pDst, int nLength, Npp64f nLevel);

/** 
 * 64-bit in place floating point complex number signal NPP_CMP_GREATER threshold with constant level.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsThreshold_GT_64fc_I(Npp64fc * pSrcDst, int nLength, Npp64f nLevel);

/** 
 * 16-bit signed short signal NPP_CMP_LESS threshold with constant level.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value to be used to limit each signal sample
 * \param lwalue Constant value to replace source value when threshold test is true.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsThreshold_LTVal_16s(const Npp16s * pSrc, Npp16s * pDst, int nLength, Npp16s nLevel, Npp16s lwalue);

/** 
 * 16-bit in place signed short signal NPP_CMP_LESS threshold with constant level.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value to be used to limit each signal sample
 * \param lwalue Constant value to replace source value when threshold test is true.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsThreshold_LTVal_16s_I(Npp16s * pSrcDst, int nLength, Npp16s nLevel, Npp16s lwalue);

/** 
 * 16-bit signed short complex number signal NPP_CMP_LESS threshold with constant level.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample
 * \param lwalue Constant value to replace source value when threshold test is true.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsThreshold_LTVal_16sc(const Npp16sc * pSrc, Npp16sc * pDst, int nLength, Npp16s nLevel, Npp16sc lwalue);

/** 
 * 16-bit in place signed short complex number signal NPP_CMP_LESS threshold with constant level.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample
 * \param lwalue Constant value to replace source value when threshold test is true.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsThreshold_LTVal_16sc_I(Npp16sc * pSrcDst, int nLength, Npp16s nLevel, Npp16sc lwalue);

/** 
 * 32-bit floating point signal NPP_CMP_LESS threshold with constant level.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value to be used to limit each signal sample
 * \param lwalue Constant value to replace source value when threshold test is true.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsThreshold_LTVal_32f(const Npp32f * pSrc, Npp32f * pDst, int nLength, Npp32f nLevel, Npp32f lwalue);

/** 
 * 32-bit in place floating point signal NPP_CMP_LESS threshold with constant level.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value to be used to limit each signal sample
 * \param lwalue Constant value to replace source value when threshold test is true.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsThreshold_LTVal_32f_I(Npp32f * pSrcDst, int nLength, Npp32f nLevel, Npp32f lwalue);

/** 
 * 32-bit floating point complex number signal NPP_CMP_LESS threshold with constant level.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample
 * \param lwalue Constant value to replace source value when threshold test is true.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsThreshold_LTVal_32fc(const Npp32fc * pSrc, Npp32fc * pDst, int nLength, Npp32f nLevel, Npp32fc lwalue);

/** 
 * 32-bit in place floating point complex number signal NPP_CMP_LESS threshold with constant level.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample
 * \param lwalue Constant value to replace source value when threshold test is true.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsThreshold_LTVal_32fc_I(Npp32fc * pSrcDst, int nLength, Npp32f nLevel, Npp32fc lwalue);

/** 
 * 64-bit floating point signal NPP_CMP_LESS threshold with constant level.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value to be used to limit each signal sample
 * \param lwalue Constant value to replace source value when threshold test is true.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsThreshold_LTVal_64f(const Npp64f * pSrc, Npp64f * pDst, int nLength, Npp64f nLevel, Npp64f lwalue);

/** 
 * 64-bit in place floating point signal NPP_CMP_LESS threshold with constant level.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value to be used to limit each signal sample
 * \param lwalue Constant value to replace source value when threshold test is true.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsThreshold_LTVal_64f_I(Npp64f * pSrcDst, int nLength, Npp64f nLevel, Npp64f lwalue);

/** 
 * 64-bit floating point complex number signal NPP_CMP_LESS threshold with constant level.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample
 * \param lwalue Constant value to replace source value when threshold test is true.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsThreshold_LTVal_64fc(const Npp64fc * pSrc, Npp64fc * pDst, int nLength, Npp64f nLevel, Npp64fc lwalue);

/** 
 * 64-bit in place floating point complex number signal NPP_CMP_LESS threshold with constant level.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample
 * \param lwalue Constant value to replace source value when threshold test is true.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsThreshold_LTVal_64fc_I(Npp64fc * pSrcDst, int nLength, Npp64f nLevel, Npp64fc lwalue);

/** 
 * 16-bit signed short signal NPP_CMP_GREATER threshold with constant level.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value to be used to limit each signal sample
 * \param lwalue Constant value to replace source value when threshold test is true.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsThreshold_GTVal_16s(const Npp16s * pSrc, Npp16s * pDst, int nLength, Npp16s nLevel, Npp16s lwalue);

/** 
 * 16-bit in place signed short signal NPP_CMP_GREATER threshold with constant level.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value to be used to limit each signal sample
 * \param lwalue Constant value to replace source value when threshold test is true.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsThreshold_GTVal_16s_I(Npp16s * pSrcDst, int nLength, Npp16s nLevel, Npp16s lwalue);

/** 
 * 16-bit signed short complex number signal NPP_CMP_GREATER threshold with constant level.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample
 * \param lwalue Constant value to replace source value when threshold test is true.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsThreshold_GTVal_16sc(const Npp16sc * pSrc, Npp16sc * pDst, int nLength, Npp16s nLevel, Npp16sc lwalue);

/** 
 * 16-bit in place signed short complex number signal NPP_CMP_GREATER threshold with constant level.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample
 * \param lwalue Constant value to replace source value when threshold test is true.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsThreshold_GTVal_16sc_I(Npp16sc * pSrcDst, int nLength, Npp16s nLevel, Npp16sc lwalue);

/** 
 * 32-bit floating point signal NPP_CMP_GREATER threshold with constant level.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value to be used to limit each signal sample
 * \param lwalue Constant value to replace source value when threshold test is true.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsThreshold_GTVal_32f(const Npp32f * pSrc, Npp32f * pDst, int nLength, Npp32f nLevel, Npp32f lwalue);

/** 
 * 32-bit in place floating point signal NPP_CMP_GREATER threshold with constant level.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value to be used to limit each signal sample
 * \param lwalue Constant value to replace source value when threshold test is true.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsThreshold_GTVal_32f_I(Npp32f * pSrcDst, int nLength, Npp32f nLevel, Npp32f lwalue);

/** 
 * 32-bit floating point complex number signal NPP_CMP_GREATER threshold with constant level.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample
 * \param lwalue Constant value to replace source value when threshold test is true.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsThreshold_GTVal_32fc(const Npp32fc * pSrc, Npp32fc * pDst, int nLength, Npp32f nLevel, Npp32fc lwalue);

/** 
 * 32-bit in place floating point complex number signal NPP_CMP_GREATER threshold with constant level.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample
 * \param lwalue Constant value to replace source value when threshold test is true.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsThreshold_GTVal_32fc_I(Npp32fc * pSrcDst, int nLength, Npp32f nLevel, Npp32fc lwalue);

/** 
 * 64-bit floating point signal NPP_CMP_GREATER threshold with constant level.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value to be used to limit each signal sample
 * \param lwalue Constant value to replace source value when threshold test is true.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsThreshold_GTVal_64f(const Npp64f * pSrc, Npp64f * pDst, int nLength, Npp64f nLevel, Npp64f lwalue);

/** 
 * 64-bit in place floating point signal NPP_CMP_GREATER threshold with constant level.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value to be used to limit each signal sample
 * \param lwalue Constant value to replace source value when threshold test is true.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsThreshold_GTVal_64f_I(Npp64f * pSrcDst, int nLength, Npp64f nLevel, Npp64f lwalue);

/** 
 * 64-bit floating point complex number signal NPP_CMP_GREATER threshold with constant level.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample
 * \param lwalue Constant value to replace source value when threshold test is true.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsThreshold_GTVal_64fc(const Npp64fc * pSrc, Npp64fc * pDst, int nLength, Npp64f nLevel, Npp64fc lwalue);

/** 
 * 64-bit in place floating point complex number signal NPP_CMP_GREATER threshold with constant level.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample
 * \param lwalue Constant value to replace source value when threshold test is true.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsThreshold_GTVal_64fc_I(Npp64fc * pSrcDst, int nLength, Npp64f nLevel, Npp64fc lwalue);

/** @} end of Threshold */

/** @} signal_threshold */

/** @} signal_colwersion_functions */

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* LW_NPPS_COLWERSION_FUNCTIONS_H */
