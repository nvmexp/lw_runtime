 /* Copyright 2009-2016 LWPU Corporation.  All rights reserved. 
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
#ifndef LW_NPPI_THRESHOLD_AND_COMPARE_OPERATIONS_H
#define LW_NPPI_THRESHOLD_AND_COMPARE_OPERATIONS_H
 
/**
 * \file nppi_threshold_and_compare_operations.h
 * NPP Image Processing Functionality.
 */
 
#include "nppdefs.h"


#ifdef __cplusplus
extern "C" {
#endif


/** @defgroup image_threshold_and_compare_operations Threshold and Compare Operations
 *  @ingroup nppi
 *
 * Methods for pixel-wise threshold and compare operations.
 *
 * @{
 *
 * These functions can be found in the nppitc library. Linking to only the sub-libraries that you use can significantly
 * save link time, application load time, and LWCA runtime startup time when using dynamic libraries.
 *
 */

/** 
 * @defgroup image_threshold_operations Threshold Operations
 *
 * Threshold image pixels.
 *
 * @{
 *
 */

/** 
 * 1 channel 8-bit unsigned char threshold.
 * If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nThreshold The threshold value.
 * \param eComparisonOperation The type of comparison operation to be used. The only valid
 *      values are: NPP_CMP_LESS and NPP_CMP_GREATER.
 * \return \ref image_data_error_codes, \ref roi_error_codes, or NPP_NOT_SUPPORTED_MODE_ERROR if an invalid
 * comparison operation type is specified.
 */
NppStatus nppiThreshold_8u_C1R(const Npp8u * pSrc, int nSrcStep,
                                     Npp8u * pDst, int nDstStep, 
                               NppiSize oSizeROI, 
                               const Npp8u nThreshold, NppCmpOp eComparisonOperation); 

/** 
 * 1 channel 8-bit unsigned char in place threshold.
 * If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nThreshold The threshold value.
 * \param eComparisonOperation The type of comparison operation to be used. The only valid
 *      values are: NPP_CMP_LESS and NPP_CMP_GREATER.
 * \return \ref image_data_error_codes, \ref roi_error_codes, or NPP_NOT_SUPPORTED_MODE_ERROR if an invalid
 * comparison operation type is specified.
 */
NppStatus nppiThreshold_8u_C1IR(Npp8u * pSrcDst, int nSrcDstStep, 
                                NppiSize oSizeROI, 
                                const Npp8u nThreshold, NppCmpOp eComparisonOperation); 

/** 
 * 1 channel 16-bit unsigned short threshold.
 * If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nThreshold The threshold value.
 * \param eComparisonOperation The type of comparison operation to be used. The only valid
 *      values are: NPP_CMP_LESS and NPP_CMP_GREATER.
 * \return \ref image_data_error_codes, \ref roi_error_codes, or NPP_NOT_SUPPORTED_MODE_ERROR if an invalid
 * comparison operation type is specified.
 */
NppStatus nppiThreshold_16u_C1R(const Npp16u * pSrc, int nSrcStep,
                                      Npp16u * pDst, int nDstStep, 
                                NppiSize oSizeROI, 
                                const Npp16u nThreshold, NppCmpOp eComparisonOperation); 

/** 
 * 1 channel 16-bit unsigned short in place threshold.
 * If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nThreshold The threshold value.
 * \param eComparisonOperation The type of comparison operation to be used. The only valid
 *      values are: NPP_CMP_LESS and NPP_CMP_GREATER.
 * \return \ref image_data_error_codes, \ref roi_error_codes, or NPP_NOT_SUPPORTED_MODE_ERROR if an invalid
 * comparison operation type is specified.
 */
NppStatus nppiThreshold_16u_C1IR(Npp16u * pSrcDst, int nSrcDstStep, 
                                 NppiSize oSizeROI, 
                                 const Npp16u nThreshold, NppCmpOp eComparisonOperation); 

/** 
 * 1 channel 16-bit signed short threshold.
 * If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nThreshold The threshold value.
 * \param eComparisonOperation The type of comparison operation to be used. The only valid
 *      values are: NPP_CMP_LESS and NPP_CMP_GREATER.
 * \return \ref image_data_error_codes, \ref roi_error_codes, or NPP_NOT_SUPPORTED_MODE_ERROR if an invalid
 * comparison operation type is specified.
 */
NppStatus nppiThreshold_16s_C1R(const Npp16s * pSrc, int nSrcStep,
                                      Npp16s * pDst, int nDstStep, 
                                NppiSize oSizeROI, 
                                const Npp16s nThreshold, NppCmpOp eComparisonOperation); 

/** 
 * 1 channel 16-bit signed short in place threshold.
 * If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nThreshold The threshold value.
 * \param eComparisonOperation The type of comparison operation to be used. The only valid
 *      values are: NPP_CMP_LESS and NPP_CMP_GREATER.
 * \return \ref image_data_error_codes, \ref roi_error_codes, or NPP_NOT_SUPPORTED_MODE_ERROR if an invalid
 * comparison operation type is specified.
 */
NppStatus nppiThreshold_16s_C1IR(Npp16s * pSrcDst, int nSrcDstStep, 
                                 NppiSize oSizeROI, 
                                 const Npp16s nThreshold, NppCmpOp eComparisonOperation); 

/** 
 * 1 channel 32-bit floating point threshold.
 * If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nThreshold The threshold value.
 * \param eComparisonOperation The type of comparison operation to be used. The only valid
 *      values are: NPP_CMP_LESS and NPP_CMP_GREATER.
 * \return \ref image_data_error_codes, \ref roi_error_codes, or NPP_NOT_SUPPORTED_MODE_ERROR if an invalid
 * comparison operation type is specified.
 */
NppStatus nppiThreshold_32f_C1R(const Npp32f * pSrc, int nSrcStep,
                                      Npp32f * pDst, int nDstStep, 
                                NppiSize oSizeROI, 
                                const Npp32f nThreshold, NppCmpOp eComparisonOperation); 

/** 
 * 1 channel 32-bit floating point in place threshold.
 * If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nThreshold The threshold value.
 * \param eComparisonOperation The type of comparison operation to be used. The only valid
 *      values are: NPP_CMP_LESS and NPP_CMP_GREATER.
 * \return \ref image_data_error_codes, \ref roi_error_codes, or NPP_NOT_SUPPORTED_MODE_ERROR if an invalid
 * comparison operation type is specified.
 */
NppStatus nppiThreshold_32f_C1IR(Npp32f * pSrcDst, int nSrcDstStep, 
                                 NppiSize oSizeROI, 
                                 const Npp32f nThreshold, NppCmpOp eComparisonOperation); 

/** 
 * 3 channel 8-bit unsigned char threshold.
 * If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param rThresholds The threshold values, one per color channel.
 * \param eComparisonOperation The type of comparison operation to be used. The only valid
 *      values are: NPP_CMP_LESS and NPP_CMP_GREATER.
 * \return \ref image_data_error_codes, \ref roi_error_codes, or NPP_NOT_SUPPORTED_MODE_ERROR if an invalid
 * comparison operation type is specified.
 */
NppStatus nppiThreshold_8u_C3R(const Npp8u * pSrc, int nSrcStep,
                                     Npp8u * pDst, int nDstStep, 
                                NppiSize oSizeROI, 
                                const Npp8u rThresholds[3], NppCmpOp eComparisonOperation); 

/** 
 * 3 channel 8-bit unsigned char in place threshold.
 * If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param rThresholds The threshold values, one per color channel.
 * \param eComparisonOperation The type of comparison operation to be used. The only valid
 *      values are: NPP_CMP_LESS and NPP_CMP_GREATER.
 * \return \ref image_data_error_codes, \ref roi_error_codes, or NPP_NOT_SUPPORTED_MODE_ERROR if an invalid
 * comparison operation type is specified.
 */
NppStatus nppiThreshold_8u_C3IR(Npp8u * pSrcDst, int nSrcDstStep, 
                                NppiSize oSizeROI, 
                                const Npp8u rThresholds[3], NppCmpOp eComparisonOperation); 

/** 
 * 3 channel 16-bit unsigned short threshold.
 * If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param rThresholds The threshold values, one per color channel.
 * \param eComparisonOperation The type of comparison operation to be used. The only valid
 *      values are: NPP_CMP_LESS and NPP_CMP_GREATER.
 * \return \ref image_data_error_codes, \ref roi_error_codes, or NPP_NOT_SUPPORTED_MODE_ERROR if an invalid
 * comparison operation type is specified.
 */
NppStatus nppiThreshold_16u_C3R(const Npp16u * pSrc, int nSrcStep,
                                      Npp16u * pDst, int nDstStep, 
                                NppiSize oSizeROI, 
                                const Npp16u rThresholds[3], NppCmpOp eComparisonOperation); 

/** 
 * 3 channel 16-bit unsigned short in place threshold.
 * If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param rThresholds The threshold values, one per color channel.
 * \param eComparisonOperation The type of comparison operation to be used. The only valid
 *      values are: NPP_CMP_LESS and NPP_CMP_GREATER.
 * \return \ref image_data_error_codes, \ref roi_error_codes, or NPP_NOT_SUPPORTED_MODE_ERROR if an invalid
 * comparison operation type is specified.
 */
NppStatus nppiThreshold_16u_C3IR(Npp16u * pSrcDst, int nSrcDstStep, 
                                 NppiSize oSizeROI, 
                                 const Npp16u rThresholds[3], NppCmpOp eComparisonOperation); 

/** 
 * 3 channel 16-bit signed short threshold.
 * If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param rThresholds The threshold values, one per color channel.
 * \param eComparisonOperation The type of comparison operation to be used. The only valid
 *      values are: NPP_CMP_LESS and NPP_CMP_GREATER.
 * \return \ref image_data_error_codes, \ref roi_error_codes, or NPP_NOT_SUPPORTED_MODE_ERROR if an invalid
 * comparison operation type is specified.
 */
NppStatus nppiThreshold_16s_C3R(const Npp16s * pSrc, int nSrcStep,
                                      Npp16s * pDst, int nDstStep, 
                                NppiSize oSizeROI, 
                                const Npp16s rThresholds[3], NppCmpOp eComparisonOperation); 

/** 
 * 3 channel 16-bit signed short in place threshold.
 * If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param rThresholds The threshold values, one per color channel.
 * \param eComparisonOperation The type of comparison operation to be used. The only valid
 *      values are: NPP_CMP_LESS and NPP_CMP_GREATER.
 * \return \ref image_data_error_codes, \ref roi_error_codes, or NPP_NOT_SUPPORTED_MODE_ERROR if an invalid
 * comparison operation type is specified.
 */
NppStatus nppiThreshold_16s_C3IR(Npp16s * pSrcDst, int nSrcDstStep, 
                                 NppiSize oSizeROI, 
                                 const Npp16s rThresholds[3], NppCmpOp eComparisonOperation); 

/** 
 * 3 channel 32-bit floating point threshold.
 * If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param rThresholds The threshold values, one per color channel.
 * \param eComparisonOperation The type of comparison operation to be used. The only valid
 *      values are: NPP_CMP_LESS and NPP_CMP_GREATER.
 * \return \ref image_data_error_codes, \ref roi_error_codes, or NPP_NOT_SUPPORTED_MODE_ERROR if an invalid
 * comparison operation type is specified.
 */
NppStatus nppiThreshold_32f_C3R(const Npp32f * pSrc, int nSrcStep,
                                      Npp32f * pDst, int nDstStep, 
                                NppiSize oSizeROI, 
                                const Npp32f rThresholds[3], NppCmpOp eComparisonOperation); 

/** 
 * 3 channel 32-bit floating point in place threshold.
 * If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param rThresholds The threshold values, one per color channel.
 * \param eComparisonOperation The type of comparison operation to be used. The only valid
 *      values are: NPP_CMP_LESS and NPP_CMP_GREATER.
 * \return \ref image_data_error_codes, \ref roi_error_codes, or NPP_NOT_SUPPORTED_MODE_ERROR if an invalid
 * comparison operation type is specified.
 */
NppStatus nppiThreshold_32f_C3IR(Npp32f * pSrcDst, int nSrcDstStep, 
                                 NppiSize oSizeROI, 
                                 const Npp32f rThresholds[3], NppCmpOp eComparisonOperation); 


/** 
 * 4 channel 8-bit unsigned char image threshold, not affecting Alpha.
 * If for a comparison operations OP the predicate (sourcePixel.channel OP nThreshold) is true, the channel
 * value is set to nThreshold, otherwise it is set to sourcePixel.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param rThresholds The threshold values, one per color channel.
 * \param eComparisonOperation The type of comparison operation to be used. The only valid
 *      values are: NPP_CMP_LESS and NPP_CMP_GREATER.
 * \return \ref image_data_error_codes, \ref roi_error_codes, or NPP_NOT_SUPPORTED_MODE_ERROR if an invalid
 * comparison operation type is specified.
 */
NppStatus nppiThreshold_8u_AC4R(const Npp8u * pSrc, int nSrcStep,
                                      Npp8u * pDst, int nDstStep, 
                                NppiSize oSizeROI, 
                                const Npp8u rThresholds[3], NppCmpOp eComparisonOperation);

/** 
 * 4 channel 8-bit unsigned char in place image threshold, not affecting Alpha.
 * If for a comparison operations OP the predicate (sourcePixel.channel OP nThreshold) is true, the channel
 * value is set to nThreshold, otherwise it is set to sourcePixel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param rThresholds The threshold values, one per color channel.
 * \param eComparisonOperation The type of comparison operation to be used. The only valid
 *      values are: NPP_CMP_LESS and NPP_CMP_GREATER.
 * \return \ref image_data_error_codes, \ref roi_error_codes, or NPP_NOT_SUPPORTED_MODE_ERROR if an invalid
 * comparison operation type is specified.
 */
NppStatus nppiThreshold_8u_AC4IR(Npp8u * pSrcDst, int nSrcDstStep, 
                                 NppiSize oSizeROI, 
                                 const Npp8u rThresholds[3], NppCmpOp eComparisonOperation);

/** 
 * 4 channel 16-bit unsigned short image threshold, not affecting Alpha.
 * If for a comparison operations OP the predicate (sourcePixel.channel OP nThreshold) is true, the channel
 * value is set to nThreshold, otherwise it is set to sourcePixel.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param rThresholds The threshold values, one per color channel.
 * \param eComparisonOperation The type of comparison operation to be used. The only valid
 *      values are: NPP_CMP_LESS and NPP_CMP_GREATER.
 * \return \ref image_data_error_codes, \ref roi_error_codes, or NPP_NOT_SUPPORTED_MODE_ERROR if an invalid
 * comparison operation type is specified.
 */
NppStatus nppiThreshold_16u_AC4R(const Npp16u * pSrc, int nSrcStep,
                                       Npp16u * pDst, int nDstStep, 
                                 NppiSize oSizeROI, 
                                 const Npp16u rThresholds[3], NppCmpOp eComparisonOperation);

/** 
 * 4 channel 16-bit unsigned short in place image threshold, not affecting Alpha.
 * If for a comparison operations OP the predicate (sourcePixel.channel OP nThreshold) is true, the channel
 * value is set to nThreshold, otherwise it is set to sourcePixel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param rThresholds The threshold values, one per color channel.
 * \param eComparisonOperation The type of comparison operation to be used. The only valid
 *      values are: NPP_CMP_LESS and NPP_CMP_GREATER.
 * \return \ref image_data_error_codes, \ref roi_error_codes, or NPP_NOT_SUPPORTED_MODE_ERROR if an invalid
 * comparison operation type is specified.
 */
NppStatus nppiThreshold_16u_AC4IR(Npp16u * pSrcDst, int nSrcDstStep, 
                                  NppiSize oSizeROI, 
                                  const Npp16u rThresholds[3], NppCmpOp eComparisonOperation);

/** 
 * 4 channel 16-bit signed short image threshold, not affecting Alpha.
 * If for a comparison operations OP the predicate (sourcePixel.channel OP nThreshold) is true, the channel
 * value is set to nThreshold, otherwise it is set to sourcePixel.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param rThresholds The threshold values, one per color channel.
 * \param eComparisonOperation The type of comparison operation to be used. The only valid
 *      values are: NPP_CMP_LESS and NPP_CMP_GREATER.
 * \return \ref image_data_error_codes, \ref roi_error_codes, or NPP_NOT_SUPPORTED_MODE_ERROR if an invalid
 * comparison operation type is specified.
 */
NppStatus nppiThreshold_16s_AC4R(const Npp16s * pSrc, int nSrcStep,
                                       Npp16s * pDst, int nDstStep, 
                                 NppiSize oSizeROI, 
                                 const Npp16s rThresholds[3], NppCmpOp eComparisonOperation);

/** 
 * 4 channel 16-bit signed short in place image threshold, not affecting Alpha.
 * If for a comparison operations OP the predicate (sourcePixel.channel OP nThreshold) is true, the channel
 * value is set to nThreshold, otherwise it is set to sourcePixel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param rThresholds The threshold values, one per color channel.
 * \param eComparisonOperation The type of comparison operation to be used. The only valid
 *      values are: NPP_CMP_LESS and NPP_CMP_GREATER.
 * \return \ref image_data_error_codes, \ref roi_error_codes, or NPP_NOT_SUPPORTED_MODE_ERROR if an invalid
 * comparison operation type is specified.
 */
NppStatus nppiThreshold_16s_AC4IR(Npp16s * pSrcDst, int nSrcDstStep, 
                                  NppiSize oSizeROI, 
                                  const Npp16s rThresholds[3], NppCmpOp eComparisonOperation);

/** 
 * 4 channel 32-bit floating point image threshold, not affecting Alpha.
 * If for a comparison operations OP the predicate (sourcePixel.channel OP nThreshold) is true, the channel
 * value is set to nThreshold, otherwise it is set to sourcePixel.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param rThresholds The threshold values, one per color channel.
 * \param eComparisonOperation The type of comparison operation to be used. The only valid
 *      values are: NPP_CMP_LESS and NPP_CMP_GREATER.
 * \return \ref image_data_error_codes, \ref roi_error_codes, or NPP_NOT_SUPPORTED_MODE_ERROR if an invalid
 * comparison operation type is specified.
 */
NppStatus nppiThreshold_32f_AC4R(const Npp32f * pSrc, int nSrcStep,
                                       Npp32f * pDst, int nDstStep, 
                                 NppiSize oSizeROI, 
                                 const Npp32f rThresholds[3], NppCmpOp eComparisonOperation);

/** 
 * 4 channel 32-bit floating point in place image threshold, not affecting Alpha.
 * If for a comparison operations OP the predicate (sourcePixel.channel OP nThreshold) is true, the channel
 * value is set to nThreshold, otherwise it is set to sourcePixel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param rThresholds The threshold values, one per color channel.
 * \param eComparisonOperation The type of comparison operation to be used. The only valid
 *      values are: NPP_CMP_LESS and NPP_CMP_GREATER.
 * \return \ref image_data_error_codes, \ref roi_error_codes, or NPP_NOT_SUPPORTED_MODE_ERROR if an invalid
 * comparison operation type is specified.
 */
NppStatus nppiThreshold_32f_AC4IR(Npp32f * pSrcDst, int nSrcDstStep, 
                                  NppiSize oSizeROI, 
                                  const Npp32f rThresholds[3], NppCmpOp eComparisonOperation);

/** 
 * 1 channel 8-bit unsigned char threshold.
 * If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nThreshold The threshold value.
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 */
NppStatus nppiThreshold_GT_8u_C1R(const Npp8u * pSrc, int nSrcStep,
                                        Npp8u * pDst, int nDstStep, 
                                  NppiSize oSizeROI, 
                                  const Npp8u nThreshold); 

/** 
 * 1 channel 8-bit unsigned char in place threshold.
 * If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nThreshold The threshold value.
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 */
NppStatus nppiThreshold_GT_8u_C1IR(Npp8u * pSrcDst, int nSrcDstStep, 
                                   NppiSize oSizeROI, 
                                   const Npp8u nThreshold); 

/** 
 * 1 channel 16-bit unsigned short threshold.
 * If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nThreshold The threshold value.
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 */
NppStatus nppiThreshold_GT_16u_C1R(const Npp16u * pSrc, int nSrcStep,
                                         Npp16u * pDst, int nDstStep, 
                                   NppiSize oSizeROI, 
                                   const Npp16u nThreshold); 

/** 
 * 1 channel 16-bit unsigned short in place threshold.
 * If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nThreshold The threshold value.
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 */
NppStatus nppiThreshold_GT_16u_C1IR(Npp16u * pSrcDst, int nSrcDstStep, 
                                    NppiSize oSizeROI, 
                                    const Npp16u nThreshold); 

/** 
 * 1 channel 16-bit signed short threshold.
 * If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nThreshold The threshold value.
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 */
NppStatus nppiThreshold_GT_16s_C1R(const Npp16s * pSrc, int nSrcStep,
                                         Npp16s * pDst, int nDstStep, 
                                   NppiSize oSizeROI, 
                                   const Npp16s nThreshold); 

/** 
 * 1 channel 16-bit signed short in place threshold.
 * If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nThreshold The threshold value.
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 */
NppStatus nppiThreshold_GT_16s_C1IR(Npp16s * pSrcDst, int nSrcDstStep, 
                                    NppiSize oSizeROI, 
                                    const Npp16s nThreshold); 

/** 
 * 1 channel 32-bit floating point threshold.
 * If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nThreshold The threshold value.
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 */
NppStatus nppiThreshold_GT_32f_C1R(const Npp32f * pSrc, int nSrcStep,
                                         Npp32f * pDst, int nDstStep, 
                                   NppiSize oSizeROI, 
                                   const Npp32f nThreshold); 

/** 
 * 1 channel 32-bit floating point in place threshold.
 * If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nThreshold The threshold value.
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 */
NppStatus nppiThreshold_GT_32f_C1IR(Npp32f * pSrcDst, int nSrcDstStep, 
                                    NppiSize oSizeROI, 
                                    const Npp32f nThreshold); 

/** 
 * 3 channel 8-bit unsigned char threshold.
 * If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param rThresholds The threshold values, one per color channel.
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 */
NppStatus nppiThreshold_GT_8u_C3R(const Npp8u * pSrc, int nSrcStep,
                                        Npp8u * pDst, int nDstStep, 
                                  NppiSize oSizeROI, 
                                  const Npp8u rThresholds[3]); 

/** 
 * 3 channel 8-bit unsigned char in place threshold.
 * If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param rThresholds The threshold values, one per color channel.
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 */
NppStatus nppiThreshold_GT_8u_C3IR(Npp8u * pSrcDst, int nSrcDstStep, 
                                   NppiSize oSizeROI, 
                                   const Npp8u rThresholds[3]); 

/** 
 * 3 channel 16-bit unsigned short threshold.
 * If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param rThresholds The threshold values, one per color channel.
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 */
NppStatus nppiThreshold_GT_16u_C3R(const Npp16u * pSrc, int nSrcStep,
                                         Npp16u * pDst, int nDstStep, 
                                   NppiSize oSizeROI, 
                                   const Npp16u rThresholds[3]); 

/** 
 * 3 channel 16-bit unsigned short in place threshold.
 * If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param rThresholds The threshold values, one per color channel.
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 */
NppStatus nppiThreshold_GT_16u_C3IR(Npp16u * pSrcDst, int nSrcDstStep, 
                                    NppiSize oSizeROI, 
                                    const Npp16u rThresholds[3]); 

/** 
 * 3 channel 16-bit signed short threshold.
 * If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param rThresholds The threshold values, one per color channel.
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 */
NppStatus nppiThreshold_GT_16s_C3R(const Npp16s * pSrc, int nSrcStep,
                                         Npp16s * pDst, int nDstStep, 
                                   NppiSize oSizeROI, 
                                   const Npp16s rThresholds[3]); 

/** 
 * 3 channel 16-bit signed short in place threshold.
 * If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param rThresholds The threshold values, one per color channel.
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 */
NppStatus nppiThreshold_GT_16s_C3IR(Npp16s * pSrcDst, int nSrcDstStep, 
                                    NppiSize oSizeROI, 
                                    const Npp16s rThresholds[3]); 

/** 
 * 3 channel 32-bit floating point threshold.
 * If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param rThresholds The threshold values, one per color channel.
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 */
NppStatus nppiThreshold_GT_32f_C3R(const Npp32f * pSrc, int nSrcStep,
                                         Npp32f * pDst, int nDstStep, 
                                   NppiSize oSizeROI, 
                                   const Npp32f rThresholds[3]); 

/** 
 * 3 channel 32-bit floating point in place threshold.
 * If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param rThresholds The threshold values, one per color channel.
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 */
NppStatus nppiThreshold_GT_32f_C3IR(Npp32f * pSrcDst, int nSrcDstStep, 
                                    NppiSize oSizeROI, 
                                    const Npp32f rThresholds[3]); 


/** 
 * 4 channel 8-bit unsigned char image threshold, not affecting Alpha.
 * If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set
 * value is set to nThreshold, otherwise it is set to sourcePixel.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param rThresholds The threshold values, one per color channel.
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 */
NppStatus nppiThreshold_GT_8u_AC4R(const Npp8u * pSrc, int nSrcStep,
                                         Npp8u * pDst, int nDstStep, 
                                   NppiSize oSizeROI, 
                                   const Npp8u rThresholds[3]);

/** 
 * 4 channel 8-bit unsigned char in place image threshold, not affecting Alpha.
 * If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set
 * value is set to nThreshold, otherwise it is set to sourcePixel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param rThresholds The threshold values, one per color channel.
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 */
NppStatus nppiThreshold_GT_8u_AC4IR(Npp8u * pSrcDst, int nSrcDstStep, 
                                    NppiSize oSizeROI, 
                                    const Npp8u rThresholds[3]);

/** 
 * 4 channel 16-bit unsigned short image threshold, not affecting Alpha.
 * If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set
 * value is set to nThreshold, otherwise it is set to sourcePixel.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param rThresholds The threshold values, one per color channel.
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 */
NppStatus nppiThreshold_GT_16u_AC4R(const Npp16u * pSrc, int nSrcStep,
                                          Npp16u * pDst, int nDstStep, 
                                    NppiSize oSizeROI, 
                                    const Npp16u rThresholds[3]);

/** 
 * 4 channel 16-bit unsigned short in place image threshold, not affecting Alpha.
 * If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set
 * value is set to nThreshold, otherwise it is set to sourcePixel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param rThresholds The threshold values, one per color channel.
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 */
NppStatus nppiThreshold_GT_16u_AC4IR(Npp16u * pSrcDst, int nSrcDstStep, 
                                     NppiSize oSizeROI, 
                                     const Npp16u rThresholds[3]);

/** 
 * 4 channel 16-bit signed short image threshold, not affecting Alpha.
 * If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set
 * value is set to nThreshold, otherwise it is set to sourcePixel.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param rThresholds The threshold values, one per color channel.
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 */
NppStatus nppiThreshold_GT_16s_AC4R(const Npp16s * pSrc, int nSrcStep,
                                          Npp16s * pDst, int nDstStep, 
                                    NppiSize oSizeROI, 
                                    const Npp16s rThresholds[3]);

/** 
 * 4 channel 16-bit signed short in place image threshold, not affecting Alpha.
 * If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set
 * value is set to nThreshold, otherwise it is set to sourcePixel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param rThresholds The threshold values, one per color channel.
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 */
NppStatus nppiThreshold_GT_16s_AC4IR(Npp16s * pSrcDst, int nSrcDstStep, 
                                     NppiSize oSizeROI, 
                                     const Npp16s rThresholds[3]);

/** 
 * 4 channel 32-bit floating point image threshold, not affecting Alpha.
 * If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set
 * value is set to nThreshold, otherwise it is set to sourcePixel.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param rThresholds The threshold values, one per color channel.
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 */
NppStatus nppiThreshold_GT_32f_AC4R(const Npp32f * pSrc, int nSrcStep,
                                          Npp32f * pDst, int nDstStep, 
                                    NppiSize oSizeROI, 
                                    const Npp32f rThresholds[3]);

/** 
 * 4 channel 32-bit floating point in place image threshold, not affecting Alpha.
 * If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set
 * value is set to nThreshold, otherwise it is set to sourcePixel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param rThresholds The threshold values, one per color channel.
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 */
NppStatus nppiThreshold_GT_32f_AC4IR(Npp32f * pSrcDst, int nSrcDstStep, 
                                     NppiSize oSizeROI, 
                                     const Npp32f rThresholds[3]);


/** 
 * 1 channel 8-bit unsigned char threshold.
 * If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nThreshold The threshold value.
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 */
NppStatus nppiThreshold_LT_8u_C1R(const Npp8u * pSrc, int nSrcStep,
                                        Npp8u * pDst, int nDstStep, 
                                  NppiSize oSizeROI, 
                                  const Npp8u nThreshold); 

/** 
 * 1 channel 8-bit unsigned char in place threshold.
 * If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nThreshold The threshold value.
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 */
NppStatus nppiThreshold_LT_8u_C1IR(Npp8u * pSrcDst, int nSrcDstStep, 
                                   NppiSize oSizeROI, 
                                   const Npp8u nThreshold); 

/** 
 * 1 channel 16-bit unsigned short threshold.
 * If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nThreshold The threshold value.
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 */
NppStatus nppiThreshold_LT_16u_C1R(const Npp16u * pSrc, int nSrcStep,
                                         Npp16u * pDst, int nDstStep, 
                                   NppiSize oSizeROI, 
                                   const Npp16u nThreshold); 

/** 
 * 1 channel 16-bit unsigned short in place threshold.
 * If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nThreshold The threshold value.
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 */
NppStatus nppiThreshold_LT_16u_C1IR(Npp16u * pSrcDst, int nSrcDstStep, 
                                    NppiSize oSizeROI, 
                                    const Npp16u nThreshold); 

/** 
 * 1 channel 16-bit signed short threshold.
 * If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nThreshold The threshold value.
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 */
NppStatus nppiThreshold_LT_16s_C1R(const Npp16s * pSrc, int nSrcStep,
                                         Npp16s * pDst, int nDstStep, 
                                   NppiSize oSizeROI, 
                                   const Npp16s nThreshold); 

/** 
 * 1 channel 16-bit signed short in place threshold.
 * If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nThreshold The threshold value.
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 */
NppStatus nppiThreshold_LT_16s_C1IR(Npp16s * pSrcDst, int nSrcDstStep, 
                                    NppiSize oSizeROI, 
                                    const Npp16s nThreshold); 

/** 
 * 1 channel 32-bit floating point threshold.
 * If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nThreshold The threshold value.
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 */
NppStatus nppiThreshold_LT_32f_C1R(const Npp32f * pSrc, int nSrcStep,
                                         Npp32f * pDst, int nDstStep, 
                                   NppiSize oSizeROI, 
                                   const Npp32f nThreshold); 

/** 
 * 1 channel 32-bit floating point in place threshold.
 * If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nThreshold The threshold value.
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 */
NppStatus nppiThreshold_LT_32f_C1IR(Npp32f * pSrcDst, int nSrcDstStep, 
                                    NppiSize oSizeROI, 
                                    const Npp32f nThreshold); 

/** 
 * 3 channel 8-bit unsigned char threshold.
 * If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param rThresholds The threshold values, one per color channel.
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 */
NppStatus nppiThreshold_LT_8u_C3R(const Npp8u * pSrc, int nSrcStep,
                                        Npp8u * pDst, int nDstStep, 
                                  NppiSize oSizeROI, 
                                  const Npp8u rThresholds[3]); 

/** 
 * 3 channel 8-bit unsigned char in place threshold.
 * If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param rThresholds The threshold values, one per color channel.
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 */
NppStatus nppiThreshold_LT_8u_C3IR(Npp8u * pSrcDst, int nSrcDstStep, 
                                   NppiSize oSizeROI, 
                                   const Npp8u rThresholds[3]); 

/** 
 * 3 channel 16-bit unsigned short threshold.
 * If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param rThresholds The threshold values, one per color channel.
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 */
NppStatus nppiThreshold_LT_16u_C3R(const Npp16u * pSrc, int nSrcStep,
                                         Npp16u * pDst, int nDstStep, 
                                   NppiSize oSizeROI, 
                                   const Npp16u rThresholds[3]); 

/** 
 * 3 channel 16-bit unsigned short in place threshold.
 * If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param rThresholds The threshold values, one per color channel.
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 */
NppStatus nppiThreshold_LT_16u_C3IR(Npp16u * pSrcDst, int nSrcDstStep, 
                                    NppiSize oSizeROI, 
                                    const Npp16u rThresholds[3]); 

/** 
 * 3 channel 16-bit signed short threshold.
 * If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param rThresholds The threshold values, one per color channel.
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 */
NppStatus nppiThreshold_LT_16s_C3R(const Npp16s * pSrc, int nSrcStep,
                                         Npp16s * pDst, int nDstStep, 
                                   NppiSize oSizeROI, 
                                   const Npp16s rThresholds[3]); 

/** 
 * 3 channel 16-bit signed short in place threshold.
 * If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param rThresholds The threshold values, one per color channel.
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 */
NppStatus nppiThreshold_LT_16s_C3IR(Npp16s * pSrcDst, int nSrcDstStep, 
                                    NppiSize oSizeROI, 
                                    const Npp16s rThresholds[3]); 

/** 
 * 3 channel 32-bit floating point threshold.
 * If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param rThresholds The threshold values, one per color channel.
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 */
NppStatus nppiThreshold_LT_32f_C3R(const Npp32f * pSrc, int nSrcStep,
                                         Npp32f * pDst, int nDstStep, 
                                   NppiSize oSizeROI, 
                                   const Npp32f rThresholds[3]); 

/** 
 * 3 channel 32-bit floating point in place threshold.
 * If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param rThresholds The threshold values, one per color channel.
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 */
NppStatus nppiThreshold_LT_32f_C3IR(Npp32f * pSrcDst, int nSrcDstStep, 
                                    NppiSize oSizeROI, 
                                    const Npp32f rThresholds[3]); 


/** 
 * 4 channel 8-bit unsigned char image threshold, not affecting Alpha.
 * If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set
 * value is set to nThreshold, otherwise it is set to sourcePixel.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param rThresholds The threshold values, one per color channel.
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 */
NppStatus nppiThreshold_LT_8u_AC4R(const Npp8u * pSrc, int nSrcStep,
                                         Npp8u * pDst, int nDstStep, 
                                   NppiSize oSizeROI, 
                                   const Npp8u rThresholds[3]);

/** 
 * 4 channel 8-bit unsigned char in place image threshold, not affecting Alpha.
 * If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set
 * value is set to nThreshold, otherwise it is set to sourcePixel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param rThresholds The threshold values, one per color channel.
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 */
NppStatus nppiThreshold_LT_8u_AC4IR(Npp8u * pSrcDst, int nSrcDstStep, 
                                    NppiSize oSizeROI, 
                                    const Npp8u rThresholds[3]);

/** 
 * 4 channel 16-bit unsigned short image threshold, not affecting Alpha.
 * If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set
 * value is set to nThreshold, otherwise it is set to sourcePixel.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param rThresholds The threshold values, one per color channel.
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 */
NppStatus nppiThreshold_LT_16u_AC4R(const Npp16u * pSrc, int nSrcStep,
                                          Npp16u * pDst, int nDstStep, 
                                    NppiSize oSizeROI, 
                                    const Npp16u rThresholds[3]);

/** 
 * 4 channel 16-bit unsigned short in place image threshold, not affecting Alpha.
 * If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set
 * value is set to nThreshold, otherwise it is set to sourcePixel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param rThresholds The threshold values, one per color channel.
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 */
NppStatus nppiThreshold_LT_16u_AC4IR(Npp16u * pSrcDst, int nSrcDstStep, 
                                     NppiSize oSizeROI, 
                                     const Npp16u rThresholds[3]);

/** 
 * 4 channel 16-bit signed short image threshold, not affecting Alpha.
 * If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set
 * value is set to nThreshold, otherwise it is set to sourcePixel.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param rThresholds The threshold values, one per color channel.
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 */
NppStatus nppiThreshold_LT_16s_AC4R(const Npp16s * pSrc, int nSrcStep,
                                          Npp16s * pDst, int nDstStep, 
                                    NppiSize oSizeROI, 
                                    const Npp16s rThresholds[3]);

/** 
 * 4 channel 16-bit signed short in place image threshold, not affecting Alpha.
 * If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set
 * value is set to nThreshold, otherwise it is set to sourcePixel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param rThresholds The threshold values, one per color channel.
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 */
NppStatus nppiThreshold_LT_16s_AC4IR(Npp16s * pSrcDst, int nSrcDstStep, 
                                     NppiSize oSizeROI, 
                                     const Npp16s rThresholds[3]);

/** 
 * 4 channel 32-bit floating point image threshold, not affecting Alpha.
 * If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set
 * value is set to nThreshold, otherwise it is set to sourcePixel.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param rThresholds The threshold values, one per color channel.
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 */
NppStatus nppiThreshold_LT_32f_AC4R(const Npp32f * pSrc, int nSrcStep,
                                          Npp32f * pDst, int nDstStep, 
                                    NppiSize oSizeROI, 
                                    const Npp32f rThresholds[3]);

/** 
 * 4 channel 32-bit floating point in place image threshold, not affecting Alpha.
 * If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set
 * value is set to nThreshold, otherwise it is set to sourcePixel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param rThresholds The threshold values, one per color channel.
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 */
NppStatus nppiThreshold_LT_32f_AC4IR(Npp32f * pSrcDst, int nSrcDstStep, 
                                     NppiSize oSizeROI, 
                                     const Npp32f rThresholds[3]);


/** 
 * 1 channel 8-bit unsigned char threshold.
 * If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set
 * to lwalue, otherwise it is set to sourcePixel.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nThreshold The threshold value.
 * \param lwalue The threshold replacement value.
 * \param eComparisonOperation The type of comparison operation to be used. The only valid
 *      values are: NPP_CMP_LESS and NPP_CMP_GREATER.
 * \return \ref image_data_error_codes, \ref roi_error_codes, or NPP_NOT_SUPPORTED_MODE_ERROR if an invalid
 * comparison operation type is specified.
 */
NppStatus nppiThreshold_Val_8u_C1R(const Npp8u * pSrc, int nSrcStep,
                                         Npp8u * pDst, int nDstStep, 
                                   NppiSize oSizeROI, 
                                   const Npp8u nThreshold, const Npp8u lwalue, NppCmpOp eComparisonOperation); 

/** 
 * 1 channel 8-bit unsigned char in place threshold.
 * If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set
 * to lwalue, otherwise it is set to sourcePixel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nThreshold The threshold value.
 * \param lwalue The threshold replacement value.
 * \param eComparisonOperation The type of comparison operation to be used. The only valid
 *      values are: NPP_CMP_LESS and NPP_CMP_GREATER.
 * \return \ref image_data_error_codes, \ref roi_error_codes, or NPP_NOT_SUPPORTED_MODE_ERROR if an invalid
 * comparison operation type is specified.
 */
NppStatus nppiThreshold_Val_8u_C1IR(Npp8u * pSrcDst, int nSrcDstStep, 
                                    NppiSize oSizeROI, 
                                    const Npp8u nThreshold, const Npp8u lwalue, NppCmpOp eComparisonOperation); 

/** 
 * 1 channel 16-bit unsigned short threshold.
 * If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set
 * to lwalue, otherwise it is set to sourcePixel.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nThreshold The threshold value.
 * \param lwalue The threshold replacement value.
 * \param eComparisonOperation The type of comparison operation to be used. The only valid
 *      values are: NPP_CMP_LESS and NPP_CMP_GREATER.
 * \return \ref image_data_error_codes, \ref roi_error_codes, or NPP_NOT_SUPPORTED_MODE_ERROR if an invalid
 * comparison operation type is specified.
 */
NppStatus nppiThreshold_Val_16u_C1R(const Npp16u * pSrc, int nSrcStep,
                                          Npp16u * pDst, int nDstStep, 
                                    NppiSize oSizeROI, 
                                    const Npp16u nThreshold, const Npp16u lwalue, NppCmpOp eComparisonOperation); 

/** 
 * 1 channel 16-bit unsigned short in place threshold.
 * If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set
 * to lwalue, otherwise it is set to sourcePixel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nThreshold The threshold value.
 * \param lwalue The threshold replacement value.
 * \param eComparisonOperation The type of comparison operation to be used. The only valid
 *      values are: NPP_CMP_LESS and NPP_CMP_GREATER.
 * \return \ref image_data_error_codes, \ref roi_error_codes, or NPP_NOT_SUPPORTED_MODE_ERROR if an invalid
 * comparison operation type is specified.
 */
NppStatus nppiThreshold_Val_16u_C1IR(Npp16u * pSrcDst, int nSrcDstStep, 
                                     NppiSize oSizeROI, 
                                     const Npp16u nThreshold, const Npp16u lwalue, NppCmpOp eComparisonOperation); 

/** 
 * 1 channel 16-bit signed short threshold.
 * If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set
 * to lwalue, otherwise it is set to sourcePixel.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nThreshold The threshold value.
 * \param lwalue The threshold replacement value.
 * \param eComparisonOperation The type of comparison operation to be used. The only valid
 *      values are: NPP_CMP_LESS and NPP_CMP_GREATER.
 * \return \ref image_data_error_codes, \ref roi_error_codes, or NPP_NOT_SUPPORTED_MODE_ERROR if an invalid
 * comparison operation type is specified.
 */
NppStatus nppiThreshold_Val_16s_C1R(const Npp16s * pSrc, int nSrcStep,
                                          Npp16s * pDst, int nDstStep, 
                                    NppiSize oSizeROI, 
                                    const Npp16s nThreshold, const Npp16s lwalue, NppCmpOp eComparisonOperation); 

/** 
 * 1 channel 16-bit signed short in place threshold.
 * If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set
 * to lwalue, otherwise it is set to sourcePixel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nThreshold The threshold value.
 * \param lwalue The threshold replacement value.
 * \param eComparisonOperation The type of comparison operation to be used. The only valid
 *      values are: NPP_CMP_LESS and NPP_CMP_GREATER.
 * \return \ref image_data_error_codes, \ref roi_error_codes, or NPP_NOT_SUPPORTED_MODE_ERROR if an invalid
 * comparison operation type is specified.
 */
NppStatus nppiThreshold_Val_16s_C1IR(Npp16s * pSrcDst, int nSrcDstStep, 
                                     NppiSize oSizeROI, 
                                     const Npp16s nThreshold, const Npp16s lwalue, NppCmpOp eComparisonOperation); 
                                     
/** 
 * 1 channel 32-bit floating point threshold.
 * If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set
 * to lwalue, otherwise it is set to sourcePixel.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nThreshold The threshold value.
 * \param lwalue The threshold replacement value.
 * \param eComparisonOperation The type of comparison operation to be used. The only valid
 *      values are: NPP_CMP_LESS and NPP_CMP_GREATER.
 * \return \ref image_data_error_codes, \ref roi_error_codes, or NPP_NOT_SUPPORTED_MODE_ERROR if an invalid
 * comparison operation type is specified.
 */
NppStatus nppiThreshold_Val_32f_C1R(const Npp32f * pSrc, int nSrcStep,
                                          Npp32f * pDst, int nDstStep, 
                                    NppiSize oSizeROI, 
                                    const Npp32f nThreshold, const Npp32f lwalue, NppCmpOp eComparisonOperation); 

/** 
 * 1 channel 32-bit floating point in place threshold.
 * If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set
 * to lwalue, otherwise it is set to sourcePixel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nThreshold The threshold value.
 * \param lwalue The threshold replacement value.
 * \param eComparisonOperation The type of comparison operation to be used. The only valid
 *      values are: NPP_CMP_LESS and NPP_CMP_GREATER.
 * \return \ref image_data_error_codes, \ref roi_error_codes, or NPP_NOT_SUPPORTED_MODE_ERROR if an invalid
 * comparison operation type is specified.
 */
NppStatus nppiThreshold_Val_32f_C1IR(Npp32f * pSrcDst, int nSrcDstStep, 
                                     NppiSize oSizeROI, 
                                     const Npp32f nThreshold, const Npp32f lwalue, NppCmpOp eComparisonOperation); 

/** 
 * 3 channel 8-bit unsigned char threshold.
 * If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set
 * to lwalue, otherwise it is set to sourcePixel.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param rThresholds The threshold values, one per color channel.
 * \param rValues The threshold replacement values, one per color channel.
 * \param eComparisonOperation The type of comparison operation to be used. The only valid
 *      values are: NPP_CMP_LESS and NPP_CMP_GREATER.
 * \return \ref image_data_error_codes, \ref roi_error_codes, or NPP_NOT_SUPPORTED_MODE_ERROR if an invalid
 * comparison operation type is specified.
 */
NppStatus nppiThreshold_Val_8u_C3R(const Npp8u * pSrc, int nSrcStep,
                                         Npp8u * pDst, int nDstStep, 
                                   NppiSize oSizeROI, 
                                   const Npp8u rThresholds[3], const Npp8u rValues[3], NppCmpOp eComparisonOperation); 

/** 
 * 3 channel 8-bit unsigned char in place threshold.
 * If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set
 * to lwalue, otherwise it is set to sourcePixel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param rThresholds The threshold values, one per color channel.
 * \param rValues The threshold replacement values, one per color channel.
 * \param eComparisonOperation The type of comparison operation to be used. The only valid
 *      values are: NPP_CMP_LESS and NPP_CMP_GREATER.
 * \return \ref image_data_error_codes, \ref roi_error_codes, or NPP_NOT_SUPPORTED_MODE_ERROR if an invalid
 * comparison operation type is specified.
 */
NppStatus nppiThreshold_Val_8u_C3IR(Npp8u * pSrcDst, int nSrcDstStep, 
                                    NppiSize oSizeROI, 
                                    const Npp8u rThresholds[3], const Npp8u rValues[3], NppCmpOp eComparisonOperation); 

/** 
 * 3 channel 16-bit unsigned short threshold.
 * If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set
 * to lwalue, otherwise it is set to sourcePixel.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param rThresholds The threshold values, one per color channel.
 * \param rValues The threshold replacement values, one per color channel.
 * \param eComparisonOperation The type of comparison operation to be used. The only valid
 *      values are: NPP_CMP_LESS and NPP_CMP_GREATER.
 * \return \ref image_data_error_codes, \ref roi_error_codes, or NPP_NOT_SUPPORTED_MODE_ERROR if an invalid
 * comparison operation type is specified.
 */
NppStatus nppiThreshold_Val_16u_C3R(const Npp16u * pSrc, int nSrcStep,
                                          Npp16u * pDst, int nDstStep, 
                                    NppiSize oSizeROI, 
                                    const Npp16u rThresholds[3], const Npp16u rValues[3], NppCmpOp eComparisonOperation); 

/** 
 * 3 channel 16-bit unsigned short in place threshold.
 * If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set
 * to lwalue, otherwise it is set to sourcePixel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param rThresholds The threshold values, one per color channel.
 * \param rValues The threshold replacement values, one per color channel.
 * \param eComparisonOperation The type of comparison operation to be used. The only valid
 *      values are: NPP_CMP_LESS and NPP_CMP_GREATER.
 * \return \ref image_data_error_codes, \ref roi_error_codes, or NPP_NOT_SUPPORTED_MODE_ERROR if an invalid
 * comparison operation type is specified.
 */
NppStatus nppiThreshold_Val_16u_C3IR(Npp16u * pSrcDst, int nSrcDstStep, 
                                     NppiSize oSizeROI, 
                                     const Npp16u rThresholds[3], const Npp16u rValues[3], NppCmpOp eComparisonOperation); 

/** 
 * 3 channel 16-bit signed short threshold.
 * If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set
 * to lwalue, otherwise it is set to sourcePixel.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param rThresholds The threshold values, one per color channel.
 * \param rValues The threshold replacement values, one per color channel.
 * \param eComparisonOperation The type of comparison operation to be used. The only valid
 *      values are: NPP_CMP_LESS and NPP_CMP_GREATER.
 * \return \ref image_data_error_codes, \ref roi_error_codes, or NPP_NOT_SUPPORTED_MODE_ERROR if an invalid
 * comparison operation type is specified.
 */
NppStatus nppiThreshold_Val_16s_C3R(const Npp16s * pSrc, int nSrcStep,
                                          Npp16s * pDst, int nDstStep, 
                                    NppiSize oSizeROI, 
                                    const Npp16s rThresholds[3], const Npp16s rValues[3], NppCmpOp eComparisonOperation); 

/** 
 * 3 channel 16-bit signed short in place threshold.
 * If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set
 * to lwalue, otherwise it is set to sourcePixel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param rThresholds The threshold values, one per color channel.
 * \param rValues The threshold replacement values, one per color channel.
 * \param eComparisonOperation The type of comparison operation to be used. The only valid
 *      values are: NPP_CMP_LESS and NPP_CMP_GREATER.
 * \return \ref image_data_error_codes, \ref roi_error_codes, or NPP_NOT_SUPPORTED_MODE_ERROR if an invalid
 * comparison operation type is specified.
 */
NppStatus nppiThreshold_Val_16s_C3IR(Npp16s * pSrcDst, int nSrcDstStep, 
                                     NppiSize oSizeROI, 
                                     const Npp16s rThresholds[3], const Npp16s rValues[3], NppCmpOp eComparisonOperation); 

/** 
 * 3 channel 32-bit floating point threshold.
 * If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set
 * to lwalue, otherwise it is set to sourcePixel.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param rThresholds The threshold values, one per color channel.
 * \param rValues The threshold replacement values, one per color channel.
 * \param eComparisonOperation The type of comparison operation to be used. The only valid
 *      values are: NPP_CMP_LESS and NPP_CMP_GREATER.
 * \return \ref image_data_error_codes, \ref roi_error_codes, or NPP_NOT_SUPPORTED_MODE_ERROR if an invalid
 * comparison operation type is specified.
 */
NppStatus nppiThreshold_Val_32f_C3R(const Npp32f * pSrc, int nSrcStep,
                                          Npp32f * pDst, int nDstStep, 
                                    NppiSize oSizeROI, 
                                    const Npp32f rThresholds[3], const Npp32f rValues[3], NppCmpOp eComparisonOperation); 

/** 
 * 3 channel 32-bit floating point in place threshold.
 * If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set
 * to lwalue, otherwise it is set to sourcePixel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param rThresholds The threshold values, one per color channel.
 * \param rValues The threshold replacement values, one per color channel.
 * \param eComparisonOperation The type of comparison operation to be used. The only valid
 *      values are: NPP_CMP_LESS and NPP_CMP_GREATER.
 * \return \ref image_data_error_codes, \ref roi_error_codes, or NPP_NOT_SUPPORTED_MODE_ERROR if an invalid
 * comparison operation type is specified.
 */
NppStatus nppiThreshold_Val_32f_C3IR(Npp32f * pSrcDst, int nSrcDstStep, 
                                     NppiSize oSizeROI, 
                                     const Npp32f rThresholds[3], const Npp32f rValues[3], NppCmpOp eComparisonOperation); 


/** 
 * 4 channel 8-bit unsigned char image threshold, not affecting Alpha.
 * If for a comparison operations OP the predicate (sourcePixel.channel OP nThreshold) is true, the channel
 * value is set to lwalue, otherwise it is set to sourcePixel.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param rThresholds The threshold values, one per color channel.
 * \param rValues The threshold replacement values, one per color channel.
 * \param eComparisonOperation The type of comparison operation to be used. The only valid
 *      values are: NPP_CMP_LESS and NPP_CMP_GREATER.
 * \return \ref image_data_error_codes, \ref roi_error_codes, or NPP_NOT_SUPPORTED_MODE_ERROR if an invalid
 * comparison operation type is specified.
 */
NppStatus nppiThreshold_Val_8u_AC4R(const Npp8u * pSrc, int nSrcStep,
                                          Npp8u * pDst, int nDstStep, 
                                    NppiSize oSizeROI, 
                                    const Npp8u rThresholds[3], const Npp8u rValues[3], NppCmpOp eComparisonOperation);

/** 
 * 4 channel 8-bit unsigned char in place image threshold, not affecting Alpha.
 * If for a comparison operations OP the predicate (sourcePixel.channel OP nThreshold) is true, the channel
 * value is set to lwalue, otherwise it is set to sourcePixel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param rThresholds The threshold values, one per color channel.
 * \param rValues The threshold replacement values, one per color channel.
 * \param eComparisonOperation The type of comparison operation to be used. The only valid
 *      values are: NPP_CMP_LESS and NPP_CMP_GREATER.
 * \return \ref image_data_error_codes, \ref roi_error_codes, or NPP_NOT_SUPPORTED_MODE_ERROR if an invalid
 * comparison operation type is specified.
 */
NppStatus nppiThreshold_Val_8u_AC4IR(Npp8u * pSrcDst, int nSrcDstStep, 
                                     NppiSize oSizeROI, 
                                     const Npp8u rThresholds[3], const Npp8u rValues[3], NppCmpOp eComparisonOperation);

/** 
 * 4 channel 16-bit unsigned short image threshold, not affecting Alpha.
 * If for a comparison operations OP the predicate (sourcePixel.channel OP nThreshold) is true, the channel
 * value is set to lwalue, otherwise it is set to sourcePixel.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param rThresholds The threshold values, one per color channel.
 * \param rValues The threshold replacement values, one per color channel.
 * \param eComparisonOperation The type of comparison operation to be used. The only valid
 *      values are: NPP_CMP_LESS and NPP_CMP_GREATER.
 * \return \ref image_data_error_codes, \ref roi_error_codes, or NPP_NOT_SUPPORTED_MODE_ERROR if an invalid
 * comparison operation type is specified.
 */
NppStatus nppiThreshold_Val_16u_AC4R(const Npp16u * pSrc, int nSrcStep,
                                           Npp16u * pDst, int nDstStep, 
                                     NppiSize oSizeROI, 
                                     const Npp16u rThresholds[3], const Npp16u rValues[3], NppCmpOp eComparisonOperation);

/** 
 * 4 channel 16-bit unsigned short in place image threshold, not affecting Alpha.
 * If for a comparison operations OP the predicate (sourcePixel.channel OP nThreshold) is true, the channel
 * value is set to lwalue, otherwise it is set to sourcePixel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param rThresholds The threshold values, one per color channel.
 * \param rValues The threshold replacement values, one per color channel.
 * \param eComparisonOperation The type of comparison operation to be used. The only valid
 *      values are: NPP_CMP_LESS and NPP_CMP_GREATER.
 * \return \ref image_data_error_codes, \ref roi_error_codes, or NPP_NOT_SUPPORTED_MODE_ERROR if an invalid
 * comparison operation type is specified.
 */
NppStatus nppiThreshold_Val_16u_AC4IR(Npp16u * pSrcDst, int nSrcDstStep, 
                                      NppiSize oSizeROI, 
                                      const Npp16u rThresholds[3], const Npp16u rValues[3], NppCmpOp eComparisonOperation);

/** 
 * 4 channel 16-bit signed short image threshold, not affecting Alpha.
 * If for a comparison operations OP the predicate (sourcePixel.channel OP nThreshold) is true, the channel
 * value is set to lwalue, otherwise it is set to sourcePixel.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param rThresholds The threshold values, one per color channel.
 * \param rValues The threshold replacement values, one per color channel.
 * \param eComparisonOperation The type of comparison operation to be used. The only valid
 *      values are: NPP_CMP_LESS and NPP_CMP_GREATER.
 * \return \ref image_data_error_codes, \ref roi_error_codes, or NPP_NOT_SUPPORTED_MODE_ERROR if an invalid
 * comparison operation type is specified.
 */
NppStatus nppiThreshold_Val_16s_AC4R(const Npp16s * pSrc, int nSrcStep,
                                           Npp16s * pDst, int nDstStep, 
                                     NppiSize oSizeROI, 
                                     const Npp16s rThresholds[3], const Npp16s rValues[3], NppCmpOp eComparisonOperation);

/** 
 * 4 channel 16-bit signed short in place image threshold, not affecting Alpha.
 * If for a comparison operations OP the predicate (sourcePixel.channel OP nThreshold) is true, the channel
 * value is set to lwalue, otherwise it is set to sourcePixel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param rThresholds The threshold values, one per color channel.
 * \param rValues The threshold replacement values, one per color channel.
 * \param eComparisonOperation The type of comparison operation to be used. The only valid
 *      values are: NPP_CMP_LESS and NPP_CMP_GREATER.
 * \return \ref image_data_error_codes, \ref roi_error_codes, or NPP_NOT_SUPPORTED_MODE_ERROR if an invalid
 * comparison operation type is specified.
 */
NppStatus nppiThreshold_Val_16s_AC4IR(Npp16s * pSrcDst, int nSrcDstStep, 
                                      NppiSize oSizeROI, 
                                      const Npp16s rThresholds[3], const Npp16s rValues[3], NppCmpOp eComparisonOperation);

/** 
 * 4 channel 32-bit floating point image threshold, not affecting Alpha.
 * If for a comparison operations OP the predicate (sourcePixel.channel OP nThreshold) is true, the channel
 * value is set to lwalue, otherwise it is set to sourcePixel.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param rThresholds The threshold values, one per color channel.
 * \param rValues The threshold replacement values, one per color channel.
 * \param eComparisonOperation The type of comparison operation to be used. The only valid
 *      values are: NPP_CMP_LESS and NPP_CMP_GREATER.
 * \return \ref image_data_error_codes, \ref roi_error_codes, or NPP_NOT_SUPPORTED_MODE_ERROR if an invalid
 * comparison operation type is specified.
 */
NppStatus nppiThreshold_Val_32f_AC4R(const Npp32f * pSrc, int nSrcStep,
                                           Npp32f * pDst, int nDstStep, 
                                     NppiSize oSizeROI, 
                                     const Npp32f rThresholds[3], const Npp32f rValues[3], NppCmpOp eComparisonOperation);

/** 
 * 4 channel 32-bit floating point in place image threshold, not affecting Alpha.
 * If for a comparison operations OP the predicate (sourcePixel.channel OP nThreshold) is true, the channel
 * value is set to lwalue, otherwise it is set to sourcePixel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param rThresholds The threshold values, one per color channel.
 * \param rValues The threshold replacement values, one per color channel.
 * \param eComparisonOperation The type of comparison operation to be used. The only valid
 *      values are: NPP_CMP_LESS and NPP_CMP_GREATER.
 * \return \ref image_data_error_codes, \ref roi_error_codes, or NPP_NOT_SUPPORTED_MODE_ERROR if an invalid
 * comparison operation type is specified.
 */
NppStatus nppiThreshold_Val_32f_AC4IR(Npp32f * pSrcDst, int nSrcDstStep, 
                                      NppiSize oSizeROI, 
                                      const Npp32f rThresholds[3], const Npp32f rValues[3], NppCmpOp eComparisonOperation);

/** 
 * 1 channel 8-bit unsigned char threshold.
 * If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set
 * to lwalue, otherwise it is set to sourcePixel.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nThreshold The threshold value.
 * \param lwalue The threshold replacement value.
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 */
NppStatus nppiThreshold_GTVal_8u_C1R(const Npp8u * pSrc, int nSrcStep,
                                           Npp8u * pDst, int nDstStep, 
                                     NppiSize oSizeROI, 
                                     const Npp8u nThreshold, const Npp8u lwalue); 

/** 
 * 1 channel 8-bit unsigned char in place threshold.
 * If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set
 * to lwalue, otherwise it is set to sourcePixel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nThreshold The threshold value.
 * \param lwalue The threshold replacement value.
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 */
NppStatus nppiThreshold_GTVal_8u_C1IR(Npp8u * pSrcDst, int nSrcDstStep, 
                                      NppiSize oSizeROI, 
                                      const Npp8u nThreshold, const Npp8u lwalue); 

/** 
 * 1 channel 16-bit unsigned short threshold.
 * If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set
 * to lwalue, otherwise it is set to sourcePixel.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nThreshold The threshold value.
 * \param lwalue The threshold replacement value.
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 */
NppStatus nppiThreshold_GTVal_16u_C1R(const Npp16u * pSrc, int nSrcStep,
                                            Npp16u * pDst, int nDstStep, 
                                      NppiSize oSizeROI, 
                                      const Npp16u nThreshold, const Npp16u lwalue); 

/** 
 * 1 channel 16-bit unsigned short in place threshold.
 * If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set
 * to lwalue, otherwise it is set to sourcePixel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nThreshold The threshold value.
 * \param lwalue The threshold replacement value.
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 */
NppStatus nppiThreshold_GTVal_16u_C1IR(Npp16u * pSrcDst, int nSrcDstStep, 
                                       NppiSize oSizeROI, 
                                       const Npp16u nThreshold, const Npp16u lwalue); 

/** 
 * 1 channel 16-bit signed short threshold.
 * If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set
 * to lwalue, otherwise it is set to sourcePixel.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nThreshold The threshold value.
 * \param lwalue The threshold replacement value.
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 */
NppStatus nppiThreshold_GTVal_16s_C1R(const Npp16s * pSrc, int nSrcStep,
                                            Npp16s * pDst, int nDstStep, 
                                      NppiSize oSizeROI, 
                                      const Npp16s nThreshold, const Npp16s lwalue); 

/** 
 * 1 channel 16-bit signed short in place threshold.
 * If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set
 * to lwalue, otherwise it is set to sourcePixel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nThreshold The threshold value.
 * \param lwalue The threshold replacement value.
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 */
NppStatus nppiThreshold_GTVal_16s_C1IR(Npp16s * pSrcDst, int nSrcDstStep, 
                                       NppiSize oSizeROI, 
                                       const Npp16s nThreshold, const Npp16s lwalue); 

/** 
 * 1 channel 32-bit floating point threshold.
 * If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set
 * to lwalue, otherwise it is set to sourcePixel.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nThreshold The threshold value.
 * \param lwalue The threshold replacement value.
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 */
NppStatus nppiThreshold_GTVal_32f_C1R(const Npp32f * pSrc, int nSrcStep,
                                            Npp32f * pDst, int nDstStep, 
                                      NppiSize oSizeROI, 
                                      const Npp32f nThreshold, const Npp32f lwalue); 

/** 
 * 1 channel 32-bit floating point in place threshold.
 * If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set
 * to lwalue, otherwise it is set to sourcePixel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nThreshold The threshold value.
 * \param lwalue The threshold replacement values.
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 */
NppStatus nppiThreshold_GTVal_32f_C1IR(Npp32f * pSrcDst, int nSrcDstStep, 
                                       NppiSize oSizeROI, 
                                       const Npp32f nThreshold, const Npp32f lwalue); 

/** 
 * 3 channel 8-bit unsigned char threshold.
 * If for a comparison operations sourcePixel is greater than rThreshold is true, the pixel is set
 * to rValue, otherwise it is set to sourcePixel.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param rThresholds The threshold values, one per color channel.
 * \param rValues The threshold replacement values, one per color channel.
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 */
NppStatus nppiThreshold_GTVal_8u_C3R(const Npp8u * pSrc, int nSrcStep,
                                           Npp8u * pDst, int nDstStep, 
                                     NppiSize oSizeROI, 
                                     const Npp8u rThresholds[3], const Npp8u rValues[3]); 

/** 
 * 3 channel 8-bit unsigned char in place threshold.
 * If for a comparison operations sourcePixel is greater than rThreshold is true, the pixel is set
 * to rValue, otherwise it is set to sourcePixel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param rThresholds The threshold values, one per color channel.
 * \param rValues The threshold replacement values, one per color channel.
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 */
NppStatus nppiThreshold_GTVal_8u_C3IR(Npp8u * pSrcDst, int nSrcDstStep, 
                                      NppiSize oSizeROI, 
                                      const Npp8u rThresholds[3], const Npp8u rValues[3]); 

/** 
 * 3 channel 16-bit unsigned short threshold.
 * If for a comparison operations sourcePixel is greater than rThreshold is true, the pixel is set
 * to rValue, otherwise it is set to sourcePixel.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param rThresholds The threshold values, one per color channel.
 * \param rValues The threshold replacement values, one per color channel.
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 */
NppStatus nppiThreshold_GTVal_16u_C3R(const Npp16u * pSrc, int nSrcStep,
                                            Npp16u * pDst, int nDstStep, 
                                      NppiSize oSizeROI, 
                                      const Npp16u rThresholds[3], const Npp16u rValues[3]); 

/** 
 * 3 channel 16-bit unsigned short in place threshold.
 * If for a comparison operations sourcePixel is greater than rThreshold is true, the pixel is set
 * to rValue, otherwise it is set to sourcePixel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param rThresholds The threshold values, one per color channel.
 * \param rValues The threshold replacement values, one per color channel.
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 */
NppStatus nppiThreshold_GTVal_16u_C3IR(Npp16u * pSrcDst, int nSrcDstStep, 
                                       NppiSize oSizeROI, 
                                       const Npp16u rThresholds[3], const Npp16u rValues[3]); 
                                       
/** 
 * 3 channel 16-bit signed short threshold.
 * If for a comparison operations sourcePixel is greater than rThreshold is true, the pixel is set
 * to rValue, otherwise it is set to sourcePixel.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param rThresholds The threshold values, one per color channel.
 * \param rValues The threshold replacement values, one per color channel.
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 */
NppStatus nppiThreshold_GTVal_16s_C3R(const Npp16s * pSrc, int nSrcStep,
                                            Npp16s * pDst, int nDstStep, 
                                      NppiSize oSizeROI, 
                                      const Npp16s rThresholds[3], const Npp16s rValues[3]); 

/** 
 * 3 channel 16-bit signed short in place threshold.
 * If for a comparison operations sourcePixel is greater than rThreshold is true, the pixel is set
 * to rValue, otherwise it is set to sourcePixel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param rThresholds The threshold values, one per color channel.
 * \param rValues The threshold replacement values, one per color channel.
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 */
NppStatus nppiThreshold_GTVal_16s_C3IR(Npp16s * pSrcDst, int nSrcDstStep, 
                                       NppiSize oSizeROI, 
                                       const Npp16s rThresholds[3], const Npp16s rValues[3]); 

/** 
 * 3 channel 32-bit floating point threshold.
 * If for a comparison operations sourcePixel is greater than rThreshold is true, the pixel is set
 * to rValue, otherwise it is set to sourcePixel.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param rThresholds The threshold values, one per color channel.
 * \param rValues The threshold replacement values, one per color channel.
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 */
NppStatus nppiThreshold_GTVal_32f_C3R(const Npp32f * pSrc, int nSrcStep,
                                            Npp32f * pDst, int nDstStep, 
                                      NppiSize oSizeROI, 
                                      const Npp32f rThresholds[3], const Npp32f rValues[3]); 

/** 
 * 3 channel 32-bit floating point in place threshold.
 * If for a comparison operations sourcePixel is greater than rThreshold is true, the pixel is set
 * to rValue, otherwise it is set to sourcePixel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param rThresholds The threshold values, one per color channel.
 * \param rValues The threshold replacement values, one per color channel.
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 */
NppStatus nppiThreshold_GTVal_32f_C3IR(Npp32f * pSrcDst, int nSrcDstStep, 
                                       NppiSize oSizeROI, 
                                       const Npp32f rThresholds[3], const Npp32f rValues[3]); 

/** 
 * 4 channel 8-bit unsigned char image threshold, not affecting Alpha.
 * If for a comparison operations sourcePixel is greater than rThreshold is true, the pixel is set
 * value is set to rValue, otherwise it is set to sourcePixel.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param rThresholds The threshold values, one per color channel.
 * \param rValues The threshold replacement values, one per color channel.
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 */
NppStatus nppiThreshold_GTVal_8u_AC4R(const Npp8u * pSrc, int nSrcStep,
                                            Npp8u * pDst, int nDstStep, 
                                      NppiSize oSizeROI, 
                                      const Npp8u rThresholds[3], const Npp8u rValues[3]);

/** 
 * 4 channel 8-bit unsigned char in place image threshold, not affecting Alpha.
 * If for a comparison operations sourcePixel is greater than rThreshold is true, the pixel is set
 * value is set to rValue, otherwise it is set to sourcePixel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param rThresholds The threshold values, one per color channel.
 * \param rValues The threshold replacement values, one per color channel.
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 */
NppStatus nppiThreshold_GTVal_8u_AC4IR(Npp8u * pSrcDst, int nSrcDstStep, 
                                       NppiSize oSizeROI, 
                                       const Npp8u rThresholds[3], const Npp8u rValues[3]);

/** 
 * 4 channel 16-bit unsigned short image threshold, not affecting Alpha.
 * If for a comparison operations sourcePixel is greater than rThreshold is true, the pixel is set
 * value is set to rValue, otherwise it is set to sourcePixel.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param rThresholds The threshold values, one per color channel.
 * \param rValues The threshold replacement values, one per color channel.
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 */
NppStatus nppiThreshold_GTVal_16u_AC4R(const Npp16u * pSrc, int nSrcStep,
                                             Npp16u * pDst, int nDstStep, 
                                       NppiSize oSizeROI, 
                                       const Npp16u rThresholds[3], const Npp16u rValues[3]);

/** 
 * 4 channel 16-bit unsigned short in place image threshold, not affecting Alpha.
 * If for a comparison operations sourcePixel is greater than rThreshold is true, the pixel is set
 * value is set to rValue, otherwise it is set to sourcePixel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param rThresholds The threshold values, one per color channel.
 * \param rValues The threshold replacement values, one per color channel.
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 */
NppStatus nppiThreshold_GTVal_16u_AC4IR(Npp16u * pSrcDst, int nSrcDstStep, 
                                        NppiSize oSizeROI, 
                                        const Npp16u rThresholds[3], const Npp16u rValues[3]);

/** 
 * 4 channel 16-bit signed short image threshold, not affecting Alpha.
 * If for a comparison operations sourcePixel is greater than rThreshold is true, the pixel is set
 * value is set to rValue, otherwise it is set to sourcePixel.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param rThresholds The threshold values, one per color channel.
 * \param rValues The threshold replacement values, one per color channel.
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 */
NppStatus nppiThreshold_GTVal_16s_AC4R(const Npp16s * pSrc, int nSrcStep,
                                             Npp16s * pDst, int nDstStep, 
                                       NppiSize oSizeROI, 
                                       const Npp16s rThresholds[3], const Npp16s rValues[3]);

/** 
 * 4 channel 16-bit signed short in place image threshold, not affecting Alpha.
 * If for a comparison operations sourcePixel is greater than rThreshold is true, the pixel is set
 * value is set to rValue, otherwise it is set to sourcePixel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param rThresholds The threshold values, one per color channel.
 * \param rValues The threshold replacement values, one per color channel.
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 */
NppStatus nppiThreshold_GTVal_16s_AC4IR(Npp16s * pSrcDst, int nSrcDstStep, 
                                        NppiSize oSizeROI, 
                                        const Npp16s rThresholds[3], const Npp16s rValues[3]);

/** 
 * 4 channel 32-bit floating point image threshold, not affecting Alpha.
 * If for a comparison operations sourcePixel is greater than rThreshold is true, the pixel is set
 * value is set to rValue, otherwise it is set to sourcePixel.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param rThresholds The threshold values, one per color channel.
 * \param rValues The threshold replacement values, one per color channel.
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 */
NppStatus nppiThreshold_GTVal_32f_AC4R(const Npp32f * pSrc, int nSrcStep,
                                             Npp32f * pDst, int nDstStep, 
                                       NppiSize oSizeROI, 
                                       const Npp32f rThresholds[3], const Npp32f rValues[3]);

/** 
 * 4 channel 32-bit floating point in place image threshold, not affecting Alpha.
 * If for a comparison operations sourcePixel is greater than rThreshold is true, the pixel is set
 * value is set to rValue, otherwise it is set to sourcePixel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param rThresholds The threshold values, one per color channel.
 * \param rValues The threshold replacement values, one per color channel.
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 */
NppStatus nppiThreshold_GTVal_32f_AC4IR(Npp32f * pSrcDst, int nSrcDstStep, 
                                        NppiSize oSizeROI, 
                                        const Npp32f rThresholds[3], const Npp32f rValues[3]);


/** 
 * 1 channel 8-bit unsigned char threshold.
 * If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set
 * to lwalue, otherwise it is set to sourcePixel.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nThreshold The threshold value.
 * \param lwalue The threshold replacement value.
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 */
NppStatus nppiThreshold_LTVal_8u_C1R(const Npp8u * pSrc, int nSrcStep,
                                           Npp8u * pDst, int nDstStep, 
                                     NppiSize oSizeROI, 
                                     const Npp8u nThreshold, const Npp8u lwalue); 

/** 
 * 1 channel 8-bit unsigned char in place threshold.
 * If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set
 * to lwalue, otherwise it is set to sourcePixel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nThreshold The threshold value.
 * \param lwalue The threshold replacement value.
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 */
NppStatus nppiThreshold_LTVal_8u_C1IR(Npp8u * pSrcDst, int nSrcDstStep, 
                                      NppiSize oSizeROI, 
                                      const Npp8u nThreshold, const Npp8u lwalue); 

/** 
 * 1 channel 16-bit unsigned short threshold.
 * If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set
 * to lwalue, otherwise it is set to sourcePixel.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nThreshold The threshold value.
 * \param lwalue The threshold replacement value.
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 */
NppStatus nppiThreshold_LTVal_16u_C1R(const Npp16u * pSrc, int nSrcStep,
                                            Npp16u * pDst, int nDstStep, 
                                      NppiSize oSizeROI, 
                                      const Npp16u nThreshold, const Npp16u lwalue); 

/** 
 * 1 channel 16-bit unsigned short in place threshold.
 * If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set
 * to lwalue, otherwise it is set to sourcePixel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nThreshold The threshold value.
 * \param lwalue The threshold replacement value.
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 */
NppStatus nppiThreshold_LTVal_16u_C1IR(Npp16u * pSrcDst, int nSrcDstStep, 
                                       NppiSize oSizeROI, 
                                       const Npp16u nThreshold, const Npp16u lwalue); 

/** 
 * 1 channel 16-bit signed short threshold.
 * If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set
 * to lwalue, otherwise it is set to sourcePixel.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nThreshold The threshold value.
 * \param lwalue The threshold replacement value.
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 */
NppStatus nppiThreshold_LTVal_16s_C1R(const Npp16s * pSrc, int nSrcStep,
                                            Npp16s * pDst, int nDstStep, 
                                      NppiSize oSizeROI, 
                                      const Npp16s nThreshold, const Npp16s lwalue); 

/** 
 * 1 channel 16-bit signed short in place threshold.
 * If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set
 * to lwalue, otherwise it is set to sourcePixel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nThreshold The threshold value.
 * \param lwalue The threshold replacement value.
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 */
NppStatus nppiThreshold_LTVal_16s_C1IR(Npp16s * pSrcDst, int nSrcDstStep, 
                                       NppiSize oSizeROI, 
                                       const Npp16s nThreshold, const Npp16s lwalue); 

/** 
 * 1 channel 32-bit floating point threshold.
 * If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set
 * to lwalue, otherwise it is set to sourcePixel.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nThreshold The threshold value.
 * \param lwalue The threshold replacement value.
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 */
NppStatus nppiThreshold_LTVal_32f_C1R(const Npp32f * pSrc, int nSrcStep,
                                            Npp32f * pDst, int nDstStep, 
                                      NppiSize oSizeROI, 
                                      const Npp32f nThreshold, const Npp32f lwalue); 

/** 
 * 1 channel 32-bit floating point in place threshold.
 * If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set
 * to lwalue, otherwise it is set to sourcePixel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nThreshold The threshold value.
 * \param lwalue The threshold replacement value.
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 */
NppStatus nppiThreshold_LTVal_32f_C1IR(Npp32f * pSrcDst, int nSrcDstStep, 
                                       NppiSize oSizeROI, 
                                       const Npp32f nThreshold, const Npp32f lwalue); 

/** 
 * 3 channel 8-bit unsigned char threshold.
 * If for a comparison operations sourcePixel is less than rThreshold is true, the pixel is set
 * to rValue, otherwise it is set to sourcePixel.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param rThresholds The threshold values, one per color channel.
 * \param rValues The threshold replacement values, one per color channel.
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 */
NppStatus nppiThreshold_LTVal_8u_C3R(const Npp8u * pSrc, int nSrcStep,
                                           Npp8u * pDst, int nDstStep, 
                                     NppiSize oSizeROI, 
                                     const Npp8u rThresholds[3], const Npp8u rValues[3]); 

/** 
 * 3 channel 8-bit unsigned char in place threshold.
 * If for a comparison operations sourcePixel is less than rThreshold is true, the pixel is set
 * to rValue, otherwise it is set to sourcePixel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param rThresholds The threshold values, one per color channel.
 * \param rValues The threshold replacement values, one per color channel.
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 */
NppStatus nppiThreshold_LTVal_8u_C3IR(Npp8u * pSrcDst, int nSrcDstStep, 
                                      NppiSize oSizeROI, 
                                      const Npp8u rThresholds[3], const Npp8u rValues[3]); 

/** 
 * 3 channel 16-bit unsigned short threshold.
 * If for a comparison operations sourcePixel is less than rThreshold is true, the pixel is set
 * to rValue, otherwise it is set to sourcePixel.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param rThresholds The threshold values, one per color channel.
 * \param rValues The threshold replacement values, one per color channel.
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 */
NppStatus nppiThreshold_LTVal_16u_C3R(const Npp16u * pSrc, int nSrcStep,
                                            Npp16u * pDst, int nDstStep, 
                                      NppiSize oSizeROI, 
                                      const Npp16u rThresholds[3], const Npp16u rValues[3]); 

/** 
 * 3 channel 16-bit unsigned short in place threshold.
 * If for a comparison operations sourcePixel is less than rThreshold is true, the pixel is set
 * to rValue, otherwise it is set to sourcePixel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param rThresholds The threshold values, one per color channel.
 * \param rValues The threshold replacement values, one per color channel.
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 */
NppStatus nppiThreshold_LTVal_16u_C3IR(Npp16u * pSrcDst, int nSrcDstStep, 
                                       NppiSize oSizeROI, 
                                       const Npp16u rThresholds[3], const Npp16u rValues[3]); 

/** 
 * 3 channel 16-bit signed short threshold.
 * If for a comparison operations sourcePixel is less than rThreshold is true, the pixel is set
 * to rValue, otherwise it is set to sourcePixel.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param rThresholds The threshold values, one per color channel.
 * \param rValues The threshold replacement values, one per color channel.
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 */
NppStatus nppiThreshold_LTVal_16s_C3R(const Npp16s * pSrc, int nSrcStep,
                                            Npp16s * pDst, int nDstStep, 
                                      NppiSize oSizeROI, 
                                      const Npp16s rThresholds[3], const Npp16s rValues[3]); 

/** 
 * 3 channel 16-bit signed short in place threshold.
 * If for a comparison operations sourcePixel is less than rThreshold is true, the pixel is set
 * to rValue, otherwise it is set to sourcePixel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param rThresholds The threshold values, one per color channel.
 * \param rValues The threshold replacement values, one per color channel.
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 */
NppStatus nppiThreshold_LTVal_16s_C3IR(Npp16s * pSrcDst, int nSrcDstStep, 
                                       NppiSize oSizeROI, 
                                       const Npp16s rThresholds[3], const Npp16s rValues[3]); 

/** 
 * 3 channel 32-bit floating point threshold.
 * If for a comparison operations sourcePixel is less than rThreshold is true, the pixel is set
 * to rValue, otherwise it is set to sourcePixel.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param rThresholds The threshold values, one per color channel.
 * \param rValues The threshold replacement values, one per color channel.
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 */
NppStatus nppiThreshold_LTVal_32f_C3R(const Npp32f * pSrc, int nSrcStep,
                                            Npp32f * pDst, int nDstStep, 
                                      NppiSize oSizeROI, 
                                      const Npp32f rThresholds[3], const Npp32f rValues[3]); 

/** 
 * 3 channel 32-bit floating point in place threshold.
 * If for a comparison operations sourcePixel is less than rThreshold is true, the pixel is set
 * to rValue, otherwise it is set to sourcePixel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param rThresholds The threshold values, one per color channel.
 * \param rValues The threshold replacement values, one per color channel.
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 */
NppStatus nppiThreshold_LTVal_32f_C3IR(Npp32f * pSrcDst, int nSrcDstStep, 
                                       NppiSize oSizeROI, 
                                       const Npp32f rThresholds[3], const Npp32f rValues[3]); 

/** 
 * 4 channel 8-bit unsigned char image threshold, not affecting Alpha.
 * If for a comparison operations sourcePixel is less than rThreshold is true, the pixel is set
 * value is set to rValue, otherwise it is set to sourcePixel.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param rThresholds The threshold values, one per color channel.
 * \param rValues The threshold replacement values, one per color channel.
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 */
NppStatus nppiThreshold_LTVal_8u_AC4R(const Npp8u * pSrc, int nSrcStep,
                                            Npp8u * pDst, int nDstStep, 
                                      NppiSize oSizeROI, 
                                      const Npp8u rThresholds[3], const Npp8u rValues[3]);

/** 
 * 4 channel 8-bit unsigned char in place image threshold, not affecting Alpha.
 * If for a comparison operations sourcePixel is less than rThreshold is true, the pixel is set
 * value is set to rValue, otherwise it is set to sourcePixel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param rThresholds The threshold values, one per color channel.
 * \param rValues The threshold replacement values, one per color channel.
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 */
NppStatus nppiThreshold_LTVal_8u_AC4IR(Npp8u * pSrcDst, int nSrcDstStep, 
                                       NppiSize oSizeROI, 
                                       const Npp8u rThresholds[3], const Npp8u rValues[3]);

/** 
 * 4 channel 16-bit unsigned short image threshold, not affecting Alpha.
 * If for a comparison operations sourcePixel is less than rThreshold is true, the pixel is set
 * value is set to rValue, otherwise it is set to sourcePixel.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param rThresholds The threshold values, one per color channel.
 * \param rValues The threshold replacement values, one per color channel.
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 */
NppStatus nppiThreshold_LTVal_16u_AC4R(const Npp16u * pSrc, int nSrcStep,
                                             Npp16u * pDst, int nDstStep, 
                                       NppiSize oSizeROI, 
                                       const Npp16u rThresholds[3], const Npp16u rValues[3]);

/** 
 * 4 channel 16-bit unsigned short in place image threshold, not affecting Alpha.
 * If for a comparison operations sourcePixel is less than rThreshold is true, the pixel is set
 * value is set to rValue, otherwise it is set to sourcePixel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param rThresholds The threshold values, one per color channel.
 * \param rValues The threshold replacement values, one per color channel.
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 */
NppStatus nppiThreshold_LTVal_16u_AC4IR(Npp16u * pSrcDst, int nSrcDstStep, 
                                        NppiSize oSizeROI, 
                                        const Npp16u rThresholds[3], const Npp16u rValues[3]);

/** 
 * 4 channel 16-bit signed short image threshold, not affecting Alpha.
 * If for a comparison operations sourcePixel is less than rThreshold is true, the pixel is set
 * value is set to rValue, otherwise it is set to sourcePixel.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param rThresholds The threshold values, one per color channel.
 * \param rValues The threshold replacement values, one per color channel.
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 */
NppStatus nppiThreshold_LTVal_16s_AC4R(const Npp16s * pSrc, int nSrcStep,
                                             Npp16s * pDst, int nDstStep, 
                                       NppiSize oSizeROI, 
                                       const Npp16s rThresholds[3], const Npp16s rValues[3]);

/** 
 * 4 channel 16-bit signed short in place image threshold, not affecting Alpha.
 * If for a comparison operations sourcePixel is less than rThreshold is true, the pixel is set
 * value is set to rValue, otherwise it is set to sourcePixel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param rThresholds The threshold values, one per color channel.
 * \param rValues The threshold replacement values, one per color channel.
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 */
NppStatus nppiThreshold_LTVal_16s_AC4IR(Npp16s * pSrcDst, int nSrcDstStep, 
                                        NppiSize oSizeROI, 
                                        const Npp16s rThresholds[3], const Npp16s rValues[3]);

/** 
 * 4 channel 32-bit floating point image threshold, not affecting Alpha.
 * If for a comparison operations sourcePixel is less than rThreshold is true, the pixel is set
 * value is set to rValue, otherwise it is set to sourcePixel.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param rThresholds The threshold values, one per color channel.
 * \param rValues The threshold replacement values, one per color channel.
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 */
NppStatus nppiThreshold_LTVal_32f_AC4R(const Npp32f * pSrc, int nSrcStep,
                                             Npp32f * pDst, int nDstStep, 
                                       NppiSize oSizeROI, 
                                       const Npp32f rThresholds[3], const Npp32f rValues[3]);

/** 
 * 4 channel 32-bit floating point in place image threshold, not affecting Alpha.
 * If for a comparison operations sourcePixel is less than rThreshold is true, the pixel is set
 * value is set to rValue, otherwise it is set to sourcePixel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param rThresholds The threshold values, one per color channel.
 * \param rValues The threshold replacement values, one per color channel.
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 */
NppStatus nppiThreshold_LTVal_32f_AC4IR(Npp32f * pSrcDst, int nSrcDstStep, 
                                        NppiSize oSizeROI, 
                                        const Npp32f rThresholds[3], const Npp32f rValues[3]);

/** 
 * 1 channel 8-bit unsigned char threshold.
 * If for a comparison operations sourcePixel is less than nThresholdLT is true, the pixel is set
 * to lwalueLT, else if sourcePixel is greater than nThresholdGT the pixel is set to lwalueGT, otherwise it is set to sourcePixel.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nThresholdLT The thresholdLT value.
 * \param lwalueLT The thresholdLT replacement value.
 * \param nThresholdGT The thresholdGT value.
 * \param lwalueGT The thresholdGT replacement value.
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 */
NppStatus nppiThreshold_LTValGTVal_8u_C1R(const Npp8u * pSrc, int nSrcStep,
                                                Npp8u * pDst, int nDstStep, 
                                          NppiSize oSizeROI, 
                                          const Npp8u nThresholdLT, const Npp8u lwalueLT, const Npp8u nThresholdGT, const Npp8u lwalueGT); 

/** 
 * 1 channel 8-bit unsigned char in place threshold.
 * If for a comparison operations sourcePixel is less than nThresholdLT is true, the pixel is set
 * to lwalueLT, else if sourcePixel is greater than nThresholdGT the pixel is set to lwalueGT, otherwise it is set to sourcePixel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nThresholdLT The thresholdLT value.
 * \param lwalueLT The thresholdLT replacement value.
 * \param nThresholdGT The thresholdGT value.
 * \param lwalueGT The thresholdGT replacement value.
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 */
NppStatus nppiThreshold_LTValGTVal_8u_C1IR(Npp8u * pSrcDst, int nSrcDstStep, 
                                           NppiSize oSizeROI, 
                                           const Npp8u nThresholdLT, const Npp8u lwalueLT, const Npp8u nThresholdGT, const Npp8u lwalueGT); 

/** 
 * 1 channel 16-bit unsigned short threshold.
 * If for a comparison operations sourcePixel is less than nThresholdLT is true, the pixel is set
 * to lwalueLT, else if sourcePixel is greater than nThresholdGT the pixel is set to lwalueGT, otherwise it is set to sourcePixel.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nThresholdLT The thresholdLT value.
 * \param lwalueLT The thresholdLT replacement value.
 * \param nThresholdGT The thresholdGT value.
 * \param lwalueGT The thresholdGT replacement value.
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 */
NppStatus nppiThreshold_LTValGTVal_16u_C1R(const Npp16u * pSrc, int nSrcStep,
                                                 Npp16u * pDst, int nDstStep, 
                                           NppiSize oSizeROI, 
                                           const Npp16u nThresholdLT, const Npp16u lwalueLT, const Npp16u nThresholdGT, const Npp16u lwalueGT); 

/** 
 * 1 channel 16-bit unsigned short in place threshold.
 * If for a comparison operations sourcePixel is less than nThresholdLT is true, the pixel is set
 * to lwalueLT, else if sourcePixel is greater than nThresholdGT the pixel is set to lwalueGT, otherwise it is set to sourcePixel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nThresholdLT The thresholdLT value.
 * \param lwalueLT The thresholdLT replacement value.
 * \param nThresholdGT The thresholdGT value.
 * \param lwalueGT The thresholdGT replacement value.
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 */
NppStatus nppiThreshold_LTValGTVal_16u_C1IR(Npp16u * pSrcDst, int nSrcDstStep, 
                                            NppiSize oSizeROI, 
                                            const Npp16u nThresholdLT, const Npp16u lwalueLT, const Npp16u nThresholdGT, const Npp16u lwalueGT); 

/** 
 * 1 channel 16-bit signed short threshold.
 * If for a comparison operations sourcePixel is less than nThresholdLT is true, the pixel is set
 * to lwalueLT, else if sourcePixel is greater than nThresholdGT the pixel is set to lwalueGT, otherwise it is set to sourcePixel.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nThresholdLT The thresholdLT value.
 * \param lwalueLT The thresholdLT replacement value.
 * \param nThresholdGT The thresholdGT value.
 * \param lwalueGT The thresholdGT replacement value.
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 */
NppStatus nppiThreshold_LTValGTVal_16s_C1R(const Npp16s * pSrc, int nSrcStep,
                                                 Npp16s * pDst, int nDstStep, 
                                           NppiSize oSizeROI, 
                                           const Npp16s nThresholdLT, const Npp16s lwalueLT, const Npp16s nThresholdGT, const Npp16s lwalueGT); 

/** 
 * 1 channel 16-bit signed short in place threshold.
 * If for a comparison operations sourcePixel is less than nThresholdLT is true, the pixel is set
 * to lwalueLT, else if sourcePixel is greater than nThresholdGT the pixel is set to lwalueGT, otherwise it is set to sourcePixel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nThresholdLT The thresholdLT value.
 * \param lwalueLT The thresholdLT replacement value.
 * \param nThresholdGT The thresholdGT value.
 * \param lwalueGT The thresholdGT replacement value.
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 */
NppStatus nppiThreshold_LTValGTVal_16s_C1IR(Npp16s * pSrcDst, int nSrcDstStep, 
                                            NppiSize oSizeROI, 
                                            const Npp16s nThresholdLT, const Npp16s lwalueLT, const Npp16s nThresholdGT, const Npp16s lwalueGT); 

/** 
 * 1 channel 32-bit floating point threshold.
 * If for a comparison operations sourcePixel is less than nThresholdLT is true, the pixel is set
 * to lwalueLT, else if sourcePixel is greater than nThresholdGT the pixel is set to lwalueGT, otherwise it is set to sourcePixel.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nThresholdLT The thresholdLT value.
 * \param lwalueLT The thresholdLT replacement value.
 * \param nThresholdGT The thresholdGT value.
 * \param lwalueGT The thresholdGT replacement value.
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 */
NppStatus nppiThreshold_LTValGTVal_32f_C1R(const Npp32f * pSrc, int nSrcStep,
                                                 Npp32f * pDst, int nDstStep, 
                                           NppiSize oSizeROI, 
                                           const Npp32f nThresholdLT, const Npp32f lwalueLT, const Npp32f nThresholdGT, const Npp32f lwalueGT); 

/** 
 * 1 channel 32-bit floating point in place threshold.
 * If for a comparison operations sourcePixel is less than nThresholdLT is true, the pixel is set
 * to lwalueLT, else if sourcePixel is greater than nThresholdGT the pixel is set to lwalueGT, otherwise it is set to sourcePixel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nThresholdLT The thresholdLT value.
 * \param lwalueLT The thresholdLT replacement value.
 * \param nThresholdGT The thresholdGT value.
 * \param lwalueGT The thresholdGT replacement value.
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 */
NppStatus nppiThreshold_LTValGTVal_32f_C1IR(Npp32f * pSrcDst, int nSrcDstStep, 
                                            NppiSize oSizeROI, 
                                            const Npp32f nThresholdLT, const Npp32f lwalueLT, const Npp32f nThresholdGT, const Npp32f lwalueGT); 

/** 
 * 3 channel 8-bit unsigned char threshold.
 * If for a comparison operations sourcePixel is less than rThresholdLT is true, the pixel is set
 * to rValueLT, else if sourcePixel is greater than rThresholdGT the pixel is set to rValueGT, otherwise it is set to sourcePixel.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param rThresholdsLT The thresholdLT values, one per color channel.
 * \param rValuesLT The thresholdLT replacement values, one per color channel.
 * \param rThresholdsGT The thresholdGT values, one per channel.
 * \param rValuesGT The thresholdGT replacement values, one per color channel.
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 */
NppStatus nppiThreshold_LTValGTVal_8u_C3R(const Npp8u * pSrc, int nSrcStep,
                                                Npp8u * pDst, int nDstStep, 
                                          NppiSize oSizeROI, 
                                          const Npp8u rThresholdsLT[3], const Npp8u rValuesLT[3], const Npp8u rThresholdsGT[3], const Npp8u rValuesGT[3]); 

/** 
 * 3 channel 8-bit unsigned char in place threshold.
 * If for a comparison operations sourcePixel is less than rThresholdLT is true, the pixel is set
 * to rValueLT, else if sourcePixel is greater than rThresholdGT the pixel is set to rValueGT, otherwise it is set to sourcePixel.
 * \param pSrcDst \ref destination_image_pointer.
 * \param nSrcDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param rThresholdsLT The thresholdLT values, one per color channel.
 * \param rValuesLT The thresholdLT replacement values, one per color channel.
 * \param rThresholdsGT The thresholdGT values, one per channel.
 * \param rValuesGT The thresholdGT replacement values, one per color channel.
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 */
NppStatus nppiThreshold_LTValGTVal_8u_C3IR(Npp8u * pSrcDst, int nSrcDstStep, 
                                           NppiSize oSizeROI, 
                                           const Npp8u rThresholdsLT[3], const Npp8u rValuesLT[3], const Npp8u rThresholdsGT[3], const Npp8u rValuesGT[3]); 

/** 
 * 3 channel 16-bit unsigned short threshold.
 * If for a comparison operations sourcePixel is less than rThresholdLT is true, the pixel is set
 * to rValueLT, else if sourcePixel is greater than rThresholdGT the pixel is set to rValueGT, otherwise it is set to sourcePixel.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param rThresholdsLT The thresholdLT values, one per color channel.
 * \param rValuesLT The thresholdLT replacement values, one per color channel.
 * \param rThresholdsGT The thresholdGT values, one per channel.
 * \param rValuesGT The thresholdGT replacement values, one per color channel.
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 */
NppStatus nppiThreshold_LTValGTVal_16u_C3R(const Npp16u * pSrc, int nSrcStep,
                                                 Npp16u * pDst, int nDstStep, 
                                           NppiSize oSizeROI, 
                                           const Npp16u rThresholdsLT[3], const Npp16u rValuesLT[3], const Npp16u rThresholdsGT[3], const Npp16u rValuesGT[3]); 

/** 
 * 3 channel 16-bit unsigned short in place threshold.
 * If for a comparison operations sourcePixel is less than rThresholdLT is true, the pixel is set
 * to rValueLT, else if sourcePixel is greater than rThresholdGT the pixel is set to rValueGT, otherwise it is set to sourcePixel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param rThresholdsLT The thresholdLT values, one per color channel.
 * \param rValuesLT The thresholdLT replacement values, one per color channel.
 * \param rThresholdsGT The thresholdGT values, one per channel.
 * \param rValuesGT The thresholdGT replacement values, one per color channel.
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 */
NppStatus nppiThreshold_LTValGTVal_16u_C3IR(Npp16u * pSrcDst, int nSrcDstStep, 
                                            NppiSize oSizeROI, 
                                            const Npp16u rThresholdsLT[3], const Npp16u rValuesLT[3], const Npp16u rThresholdsGT[3], const Npp16u rValuesGT[3]); 

/** 
 * 3 channel 16-bit signed short threshold.
 * If for a comparison operations sourcePixel is less than rThresholdLT is true, the pixel is set
 * to rValueLT, else if sourcePixel is greater than rThresholdGT the pixel is set to rValueGT, otherwise it is set to sourcePixel.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param rThresholdsLT The thresholdLT values, one per color channel.
 * \param rValuesLT The thresholdLT replacement values, one per color channel.
 * \param rThresholdsGT The thresholdGT values, one per channel.
 * \param rValuesGT The thresholdGT replacement values, one per color channel.
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 */
NppStatus nppiThreshold_LTValGTVal_16s_C3R(const Npp16s * pSrc, int nSrcStep,
                                                 Npp16s * pDst, int nDstStep, 
                                           NppiSize oSizeROI, 
                                           const Npp16s rThresholdsLT[3], const Npp16s rValuesLT[3], const Npp16s rThresholdsGT[3], const Npp16s rValuesGT[3]); 

/** 
 * 3 channel 16-bit signed short in place threshold.
 * If for a comparison operations sourcePixel is less than rThresholdLT is true, the pixel is set
 * to rValueLT, else if sourcePixel is greater than rThresholdGT the pixel is set to rValueGT, otherwise it is set to sourcePixel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param rThresholdsLT The thresholdLT values, one per color channel.
 * \param rValuesLT The thresholdLT replacement values, one per color channel.
 * \param rThresholdsGT The thresholdGT values, one per channel.
 * \param rValuesGT The thresholdGT replacement values, one per color channel.
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 */
NppStatus nppiThreshold_LTValGTVal_16s_C3IR(Npp16s * pSrcDst, int nSrcDstStep, 
                                            NppiSize oSizeROI, 
                                            const Npp16s rThresholdsLT[3], const Npp16s rValuesLT[3], const Npp16s rThresholdsGT[3], const Npp16s rValuesGT[3]); 

/** 
 * 3 channel 32-bit floating point threshold.
 * If for a comparison operations sourcePixel is less than rThresholdLT is true, the pixel is set
 * to rValueLT, else if sourcePixel is greater than rThresholdGT the pixel is set to rValueGT, otherwise it is set to sourcePixel.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param rThresholdsLT The thresholdLT values, one per color channel.
 * \param rValuesLT The thresholdLT replacement values, one per color channel.
 * \param rThresholdsGT The thresholdGT values, one per channel.
 * \param rValuesGT The thresholdGT replacement values, one per color channel.
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 */
NppStatus nppiThreshold_LTValGTVal_32f_C3R(const Npp32f * pSrc, int nSrcStep,
                                                 Npp32f * pDst, int nDstStep, 
                                           NppiSize oSizeROI, 
                                           const Npp32f rThresholdsLT[3], const Npp32f rValuesLT[3], const Npp32f rThresholdsGT[3], const Npp32f rValuesGT[3]); 

/** 
 * 3 channel 32-bit floating point in place threshold.
 * If for a comparison operations sourcePixel is less than rThresholdLT is true, the pixel is set
 * to rValueLT, else if sourcePixel is greater than rThresholdGT the pixel is set to rValueGT, otherwise it is set to sourcePixel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param rThresholdsLT The thresholdLT values, one per color channel.
 * \param rValuesLT The thresholdLT replacement values, one per color channel.
 * \param rThresholdsGT The thresholdGT values, one per channel.
 * \param rValuesGT The thresholdGT replacement values, one per color channel.
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 */
NppStatus nppiThreshold_LTValGTVal_32f_C3IR(Npp32f * pSrcDst, int nSrcDstStep, 
                                            NppiSize oSizeROI, 
                                            const Npp32f rThresholdsLT[3], const Npp32f rValuesLT[3], const Npp32f rThresholdsGT[3], const Npp32f rValuesGT[3]); 

/** 
 * 4 channel 8-bit unsigned char image threshold, not affecting Alpha.
 * If for a comparison operations sourcePixel is less than rThresholdLT is true, the pixel is set
 * value is set to rValueLT, else if sourcePixel is greater than rThresholdGT the pixel is set to rValueGT, otherwise it is set to sourcePixel.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param rThresholdsLT The thresholdLT values, one per color channel.
 * \param rValuesLT The thresholdLT replacement values, one per color channel.
 * \param rThresholdsGT The thresholdGT values, one per channel.
 * \param rValuesGT The thresholdGT replacement values, one per color channel.
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 */
NppStatus nppiThreshold_LTValGTVal_8u_AC4R(const Npp8u * pSrc, int nSrcStep,
                                                 Npp8u * pDst, int nDstStep, 
                                           NppiSize oSizeROI, 
                                           const Npp8u rThresholdsLT[3], const Npp8u rValuesLT[3], const Npp8u rThresholdsGT[3], const Npp8u rValuesGT[3]);

/** 
 * 4 channel 8-bit unsigned char in place image threshold, not affecting Alpha.
 * If for a comparison operations sourcePixel is less than rThresholdLT is true, the pixel is set
 * value is set to rValueLT, else if sourcePixel is greater than rThresholdGT the pixel is set to rValueGT, otherwise it is set to sourcePixel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param rThresholdsLT The thresholdLT values, one per color channel.
 * \param rValuesLT The thresholdLT replacement values, one per color channel.
 * \param rThresholdsGT The thresholdGT values, one per channel.
 * \param rValuesGT The thresholdGT replacement values, one per color channel.
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 */
NppStatus nppiThreshold_LTValGTVal_8u_AC4IR(Npp8u * pSrcDst, int nSrcDstStep, 
                                            NppiSize oSizeROI, 
                                            const Npp8u rThresholdsLT[3], const Npp8u rValuesLT[3], const Npp8u rThresholdsGT[3], const Npp8u rValuesGT[3]);

/** 
 * 4 channel 16-bit unsigned short image threshold, not affecting Alpha.
 * If for a comparison operations sourcePixel is less than rThresholdLT is true, the pixel is set
 * value is set to rValueLT, else if sourcePixel is greater than rThresholdGT the pixel is set to rValueGT, otherwise it is set to sourcePixel.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param rThresholdsLT The thresholdLT values, one per color channel.
 * \param rValuesLT The thresholdLT replacement values, one per color channel.
 * \param rThresholdsGT The thresholdGT values, one per channel.
 * \param rValuesGT The thresholdGT replacement values, one per color channel.
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 */
NppStatus nppiThreshold_LTValGTVal_16u_AC4R(const Npp16u * pSrc, int nSrcStep,
                                                  Npp16u * pDst, int nDstStep, 
                                            NppiSize oSizeROI, 
                                            const Npp16u rThresholdsLT[3], const Npp16u rValuesLT[3], const Npp16u rThresholdsGT[3], const Npp16u rValuesGT[3]);

/** 
 * 4 channel 16-bit unsigned short in place image threshold, not affecting Alpha.
 * If for a comparison operations sourcePixel is less than rThresholdLT is true, the pixel is set
 * value is set to rValueLT, else if sourcePixel is greater than rThresholdGT the pixel is set to rValueGT, otherwise it is set to sourcePixel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param rThresholdsLT The thresholdLT values, one per color channel.
 * \param rValuesLT The thresholdLT replacement values, one per color channel.
 * \param rThresholdsGT The thresholdGT values, one per channel.
 * \param rValuesGT The thresholdGT replacement values, one per color channel.
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 */
NppStatus nppiThreshold_LTValGTVal_16u_AC4IR(Npp16u * pSrcDst, int nSrcDstStep, 
                                             NppiSize oSizeROI, 
                                             const Npp16u rThresholdsLT[3], const Npp16u rValuesLT[3], const Npp16u rThresholdsGT[3], const Npp16u rValuesGT[3]);

/** 
 * 4 channel 16-bit signed short image threshold, not affecting Alpha.
 * If for a comparison operations sourcePixel is less than rThresholdLT is true, the pixel is set
 * value is set to rValueLT, else if sourcePixel is greater than rThresholdGT the pixel is set to rValueGT, otherwise it is set to sourcePixel.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param rThresholdsLT The thresholdLT values, one per color channel.
 * \param rValuesLT The thresholdLT replacement values, one per color channel.
 * \param rThresholdsGT The thresholdGT values, one per channel.
 * \param rValuesGT The thresholdGT replacement values, one per color channel.
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 */
NppStatus nppiThreshold_LTValGTVal_16s_AC4R(const Npp16s * pSrc, int nSrcStep,
                                                  Npp16s * pDst, int nDstStep, 
                                            NppiSize oSizeROI, 
                                            const Npp16s rThresholdsLT[3], const Npp16s rValuesLT[3], const Npp16s rThresholdsGT[3], const Npp16s rValuesGT[3]);

/** 
 * 4 channel 16-bit signed short in place image threshold, not affecting Alpha.
 * If for a comparison operations sourcePixel is less than rThresholdLT is true, the pixel is set
 * value is set to rValueLT, else if sourcePixel is greater than rThresholdGT the pixel is set to rValueGT, otherwise it is set to sourcePixel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param rThresholdsLT The thresholdLT values, one per color channel.
 * \param rValuesLT The thresholdLT replacement values, one per color channel.
 * \param rThresholdsGT The thresholdGT values, one per channel.
 * \param rValuesGT The thresholdGT replacement values, one per color channel.
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 */
NppStatus nppiThreshold_LTValGTVal_16s_AC4IR(Npp16s * pSrcDst, int nSrcDstStep, 
                                             NppiSize oSizeROI, 
                                             const Npp16s rThresholdsLT[3], const Npp16s rValuesLT[3], const Npp16s rThresholdsGT[3], const Npp16s rValuesGT[3]);

/** 
 * 4 channel 32-bit floating point image threshold, not affecting Alpha.
 * If for a comparison operations sourcePixel is less than rThresholdLT is true, the pixel is set
 * value is set to rValueLT, else if sourcePixel is greater than rThresholdGT the pixel is set to rValueGT, otherwise it is set to sourcePixel.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param rThresholdsLT The thresholdLT values, one per color channel.
 * \param rValuesLT The thresholdLT replacement values, one per color channel.
 * \param rThresholdsGT The thresholdGT values, one per channel.
 * \param rValuesGT The thresholdGT replacement values, one per color channel.
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 */
NppStatus nppiThreshold_LTValGTVal_32f_AC4R(const Npp32f * pSrc, int nSrcStep,
                                                  Npp32f * pDst, int nDstStep, 
                                            NppiSize oSizeROI, 
                                            const Npp32f rThresholdsLT[3], const Npp32f rValuesLT[3], const Npp32f rThresholdsGT[3], const Npp32f rValuesGT[3]);

/** 
 * 4 channel 32-bit floating point in place image threshold, not affecting Alpha.
 * If for a comparison operations sourcePixel is less than rThresholdLT is true, the pixel is set
 * value is set to rValueLT, else if sourcePixel is greater than rThresholdGT the pixel is set to rValueGT, otherwise it is set to sourcePixel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param rThresholdsLT The thresholdLT values, one per color channel.
 * \param rValuesLT The thresholdLT replacement values, one per color channel.
 * \param rThresholdsGT The thresholdGT values, one per channel.
 * \param rValuesGT The thresholdGT replacement values, one per color channel.
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 */
NppStatus nppiThreshold_LTValGTVal_32f_AC4IR(Npp32f * pSrcDst, int nSrcDstStep, 
                                             NppiSize oSizeROI, 
                                             const Npp32f rThresholdsLT[3], const Npp32f rValuesLT[3], const Npp32f rThresholdsGT[3], const Npp32f rValuesGT[3]);

/** @} image_threshold_operations */

/** @defgroup image_compare_operations Compare Operations
 * Compare the pixels of two images and create a binary result image. In case of multi-channel
 * image types, the condition must be fulfilled for all channels, otherwise the comparison
 * is considered false.
 * The "binary" result image is of type 8u_C1. False is represented by 0, true by NPP_MAX_8U.
 *
 * @{
 *
 */

/** 
 * 1 channel 8-bit unsigned char image compare.
 * Compare pSrc1's pixels with corresponding pixels in pSrc2. 
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eComparisonOperation Specifies the comparison operation to be used in the pixel comparison.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus nppiCompare_8u_C1R(const Npp8u * pSrc1, int nSrc1Step,
                             const Npp8u * pSrc2, int nSrc2Step,
                                   Npp8u * pDst,  int nDstStep,
                             NppiSize oSizeROI, NppCmpOp eComparisonOperation);

/** 
 * 3 channel 8-bit unsigned char image compare.
 * Compare pSrc1's pixels with corresponding pixels in pSrc2. 
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eComparisonOperation Specifies the comparison operation to be used in the pixel comparison.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus nppiCompare_8u_C3R(const Npp8u * pSrc1, int nSrc1Step,
                             const Npp8u * pSrc2, int nSrc2Step,
                                   Npp8u * pDst,  int nDstStep,
                             NppiSize oSizeROI, NppCmpOp eComparisonOperation);

/** 
 * 4 channel 8-bit unsigned char image compare.
 * Compare pSrc1's pixels with corresponding pixels in pSrc2. 
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eComparisonOperation Specifies the comparison operation to be used in the pixel comparison.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus nppiCompare_8u_C4R(const Npp8u * pSrc1, int nSrc1Step,
                             const Npp8u * pSrc2, int nSrc2Step,
                                   Npp8u * pDst,  int nDstStep,
                             NppiSize oSizeROI, NppCmpOp eComparisonOperation);

/** 
 * 4 channel 8-bit unsigned char image compare, not affecting Alpha.
 * Compare pSrc1's pixels with corresponding pixels in pSrc2. 
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eComparisonOperation Specifies the comparison operation to be used in the pixel comparison.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus nppiCompare_8u_AC4R(const Npp8u * pSrc1, int nSrc1Step,
                              const Npp8u * pSrc2, int nSrc2Step,
                                    Npp8u * pDst,  int nDstStep,
                              NppiSize oSizeROI, NppCmpOp eComparisonOperation);

/** 
 * 1 channel 16-bit unsigned short image compare.
 * Compare pSrc1's pixels with corresponding pixels in pSrc2. 
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eComparisonOperation Specifies the comparison operation to be used in the pixel comparison.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus nppiCompare_16u_C1R(const Npp16u * pSrc1, int nSrc1Step,
                              const Npp16u * pSrc2, int nSrc2Step,
                                    Npp8u * pDst,  int nDstStep,
                              NppiSize oSizeROI, NppCmpOp eComparisonOperation);

/** 
 * 3 channel 16-bit unsigned short image compare.
 * Compare pSrc1's pixels with corresponding pixels in pSrc2. 
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eComparisonOperation Specifies the comparison operation to be used in the pixel comparison.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus nppiCompare_16u_C3R(const Npp16u * pSrc1, int nSrc1Step,
                              const Npp16u * pSrc2, int nSrc2Step,
                                    Npp8u * pDst,  int nDstStep,
                              NppiSize oSizeROI, NppCmpOp eComparisonOperation);

/** 
 * 4 channel 16-bit unsigned short image compare.
 * Compare pSrc1's pixels with corresponding pixels in pSrc2. 
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eComparisonOperation Specifies the comparison operation to be used in the pixel comparison.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus nppiCompare_16u_C4R(const Npp16u * pSrc1, int nSrc1Step,
                              const Npp16u * pSrc2, int nSrc2Step,
                                    Npp8u * pDst,  int nDstStep,
                              NppiSize oSizeROI, NppCmpOp eComparisonOperation);

/** 
 * 4 channel 16-bit unsigned short image compare, not affecting Alpha.
 * Compare pSrc1's pixels with corresponding pixels in pSrc2. 
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eComparisonOperation Specifies the comparison operation to be used in the pixel comparison.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus nppiCompare_16u_AC4R(const Npp16u * pSrc1, int nSrc1Step,
                               const Npp16u * pSrc2, int nSrc2Step,
                                     Npp8u * pDst,  int nDstStep,
                               NppiSize oSizeROI, NppCmpOp eComparisonOperation);

/** 
 * 1 channel 16-bit signed short image compare.
 * Compare pSrc1's pixels with corresponding pixels in pSrc2. 
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eComparisonOperation Specifies the comparison operation to be used in the pixel comparison.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus nppiCompare_16s_C1R(const Npp16s * pSrc1, int nSrc1Step,
                              const Npp16s * pSrc2, int nSrc2Step,
                                    Npp8u * pDst,  int nDstStep,
                              NppiSize oSizeROI, NppCmpOp eComparisonOperation);

/** 
 * 3 channel 16-bit signed short image compare.
 * Compare pSrc1's pixels with corresponding pixels in pSrc2. 
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eComparisonOperation Specifies the comparison operation to be used in the pixel comparison.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus nppiCompare_16s_C3R(const Npp16s * pSrc1, int nSrc1Step,
                              const Npp16s * pSrc2, int nSrc2Step,
                                    Npp8u * pDst,  int nDstStep,
                              NppiSize oSizeROI, NppCmpOp eComparisonOperation);

/** 
 * 4 channel 16-bit signed short image compare.
 * Compare pSrc1's pixels with corresponding pixels in pSrc2. 
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eComparisonOperation Specifies the comparison operation to be used in the pixel comparison.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus nppiCompare_16s_C4R(const Npp16s * pSrc1, int nSrc1Step,
                              const Npp16s * pSrc2, int nSrc2Step,
                                    Npp8u * pDst,  int nDstStep,
                              NppiSize oSizeROI, NppCmpOp eComparisonOperation);

/** 
 * 4 channel 16-bit signed short image compare, not affecting Alpha.
 * Compare pSrc1's pixels with corresponding pixels in pSrc2. 
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eComparisonOperation Specifies the comparison operation to be used in the pixel comparison.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus nppiCompare_16s_AC4R(const Npp16s * pSrc1, int nSrc1Step,
                               const Npp16s * pSrc2, int nSrc2Step,
                                     Npp8u * pDst,  int nDstStep,
                               NppiSize oSizeROI, NppCmpOp eComparisonOperation);

/** 
 * 1 channel 32-bit floating point image compare.
 * Compare pSrc1's pixels with corresponding pixels in pSrc2. 
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eComparisonOperation Specifies the comparison operation to be used in the pixel comparison.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus nppiCompare_32f_C1R(const Npp32f * pSrc1, int nSrc1Step,
                              const Npp32f * pSrc2, int nSrc2Step,
                                    Npp8u * pDst,  int nDstStep,
                              NppiSize oSizeROI, NppCmpOp eComparisonOperation);

/** 
 * 3 channel 32-bit floating point image compare.
 * Compare pSrc1's pixels with corresponding pixels in pSrc2. 
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eComparisonOperation Specifies the comparison operation to be used in the pixel comparison.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus nppiCompare_32f_C3R(const Npp32f * pSrc1, int nSrc1Step,
                              const Npp32f * pSrc2, int nSrc2Step,
                                    Npp8u * pDst,  int nDstStep,
                              NppiSize oSizeROI, NppCmpOp eComparisonOperation);

/** 
 * 4 channel 32-bit floating point image compare.
 * Compare pSrc1's pixels with corresponding pixels in pSrc2. 
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eComparisonOperation Specifies the comparison operation to be used in the pixel comparison.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus nppiCompare_32f_C4R(const Npp32f * pSrc1, int nSrc1Step,
                              const Npp32f * pSrc2, int nSrc2Step,
                                    Npp8u * pDst,  int nDstStep,
                              NppiSize oSizeROI, NppCmpOp eComparisonOperation);

/** 
 * 4 channel 32-bit signed floating point compare, not affecting Alpha.
 * Compare pSrc1's pixels with corresponding pixels in pSrc2. 
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eComparisonOperation Specifies the comparison operation to be used in the pixel comparison.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus nppiCompare_32f_AC4R(const Npp32f * pSrc1, int nSrc1Step,
                               const Npp32f * pSrc2, int nSrc2Step,
                                     Npp8u * pDst,  int nDstStep,
                               NppiSize oSizeROI, NppCmpOp eComparisonOperation);

/** 
 * 1 channel 8-bit unsigned char image compare with constant value.
 * Compare pSrc's pixels with constant value. 
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param nConstant constant value.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eComparisonOperation Specifies the comparison operation to be used in the pixel comparison.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus nppiCompareC_8u_C1R(const Npp8u * pSrc, int nSrcStep,
                              const Npp8u nConstant,
                                    Npp8u * pDst,  int nDstStep,
                              NppiSize oSizeROI, NppCmpOp eComparisonOperation);

/** 
 * 3 channel 8-bit unsigned char image compare with constant value.
 * Compare pSrc's pixels with constant value. 
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pConstants pointer to a list of constant values, one per color channel..
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eComparisonOperation Specifies the comparison operation to be used in the pixel comparison.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus nppiCompareC_8u_C3R(const Npp8u * pSrc, int nSrcStep,
                              const Npp8u * pConstants,
                                    Npp8u * pDst,  int nDstStep,
                              NppiSize oSizeROI, NppCmpOp eComparisonOperation);

/** 
 * 4 channel 8-bit unsigned char image compare with constant value.
 * Compare pSrc's pixels with constant value. 
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pConstants pointer to a list of constants, one per color channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eComparisonOperation Specifies the comparison operation to be used in the pixel comparison.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus nppiCompareC_8u_C4R(const Npp8u * pSrc, int nSrcStep,
                              const Npp8u * pConstants,
                                    Npp8u * pDst,  int nDstStep,
                              NppiSize oSizeROI, NppCmpOp eComparisonOperation);

/** 
 * 4 channel 8-bit unsigned char image compare, not affecting Alpha.
 * Compare pSrc's pixels with constant value. 
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pConstants pointer to a list of constants, one per color channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eComparisonOperation Specifies the comparison operation to be used in the pixel comparison.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus nppiCompareC_8u_AC4R(const Npp8u * pSrc, int nSrcStep,
                               const Npp8u * pConstants,
                                     Npp8u * pDst,  int nDstStep,
                               NppiSize oSizeROI, NppCmpOp eComparisonOperation);

/** 
 * 1 channel 16-bit unsigned short image compare with constant value.
 * Compare pSrc's pixels with constant value. 
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param nConstant constant value
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eComparisonOperation Specifies the comparison operation to be used in the pixel comparison.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus nppiCompareC_16u_C1R(const Npp16u * pSrc, int nSrcStep,
                               const Npp16u nConstant,
                                     Npp8u * pDst,  int nDstStep,
                               NppiSize oSizeROI, NppCmpOp eComparisonOperation);

/** 
 * 3 channel 16-bit unsigned short image compare with constant value.
 * Compare pSrc's pixels with constant value. 
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pConstants pointer to a list of constants, one per color channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eComparisonOperation Specifies the comparison operation to be used in the pixel comparison.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus nppiCompareC_16u_C3R(const Npp16u * pSrc, int nSrcStep,
                               const Npp16u * pConstants,
                                     Npp8u * pDst,  int nDstStep,
                               NppiSize oSizeROI, NppCmpOp eComparisonOperation);

/** 
 * 4 channel 16-bit unsigned short image compare with constant value.
 * Compare pSrc's pixels with constant value. 
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pConstants pointer to a list of constants, one per color channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eComparisonOperation Specifies the comparison operation to be used in the pixel comparison.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus nppiCompareC_16u_C4R(const Npp16u * pSrc, int nSrcStep,
                               const Npp16u * pConstants,
                                     Npp8u * pDst,  int nDstStep,
                               NppiSize oSizeROI, NppCmpOp eComparisonOperation);

/** 
 * 4 channel 16-bit unsigned short image compare, not affecting Alpha.
 * Compare pSrc's pixels with constant value. 
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pConstants pointer to a list of constants, one per color channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eComparisonOperation Specifies the comparison operation to be used in the pixel comparison.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus nppiCompareC_16u_AC4R(const Npp16u * pSrc, int nSrcStep,
                                const Npp16u * pConstants,
                                      Npp8u * pDst,  int nDstStep,
                                NppiSize oSizeROI, NppCmpOp eComparisonOperation);

/** 
 * 1 channel 16-bit signed short image compare with constant value.
 * Compare pSrc's pixels with constant value. 
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param nConstant constant value.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eComparisonOperation Specifies the comparison operation to be used in the pixel comparison.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus nppiCompareC_16s_C1R(const Npp16s * pSrc, int nSrcStep,
                               const Npp16s nConstant,
                                     Npp8u * pDst,  int nDstStep,
                               NppiSize oSizeROI, NppCmpOp eComparisonOperation);

/** 
 * 3 channel 16-bit signed short image compare with constant value.
 * Compare pSrc's pixels with constant value. 
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pConstants pointer to a list of constants, one per color channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eComparisonOperation Specifies the comparison operation to be used in the pixel comparison.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus nppiCompareC_16s_C3R(const Npp16s * pSrc, int nSrcStep,
                               const Npp16s * pConstants,
                                     Npp8u * pDst,  int nDstStep,
                               NppiSize oSizeROI, NppCmpOp eComparisonOperation);

/** 
 * 4 channel 16-bit signed short image compare with constant value.
 * Compare pSrc's pixels with constant value. 
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pConstants pointer to a list of constants, one per color channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eComparisonOperation Specifies the comparison operation to be used in the pixel comparison.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus nppiCompareC_16s_C4R(const Npp16s * pSrc, int nSrcStep,
                               const Npp16s * pConstants,
                                     Npp8u * pDst,  int nDstStep,
                               NppiSize oSizeROI, NppCmpOp eComparisonOperation);

/** 
 * 4 channel 16-bit signed short image compare, not affecting Alpha.
 * Compare pSrc's pixels with constant value. 
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pConstants pointer to a list of constants, one per color channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eComparisonOperation Specifies the comparison operation to be used in the pixel comparison.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus nppiCompareC_16s_AC4R(const Npp16s * pSrc, int nSrcStep,
                                const Npp16s * pConstants,
                                      Npp8u * pDst,  int nDstStep,
                                NppiSize oSizeROI, NppCmpOp eComparisonOperation);

/** 
 * 1 channel 32-bit floating point image compare with constant value.
 * Compare pSrc's pixels with constant value. 
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param nConstant constant value
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eComparisonOperation Specifies the comparison operation to be used in the pixel comparison.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus nppiCompareC_32f_C1R(const Npp32f * pSrc, int nSrcStep,
                               const Npp32f nConstant,
                                     Npp8u * pDst,  int nDstStep,
                               NppiSize oSizeROI, NppCmpOp eComparisonOperation);

/** 
 * 3 channel 32-bit floating point image compare with constant value.
 * Compare pSrc's pixels with constant value. 
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pConstants pointer to a list of constants, one per color channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eComparisonOperation Specifies the comparison operation to be used in the pixel comparison.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus nppiCompareC_32f_C3R(const Npp32f * pSrc, int nSrcStep,
                               const Npp32f * pConstants,
                                     Npp8u * pDst,  int nDstStep,
                               NppiSize oSizeROI, NppCmpOp eComparisonOperation);

/** 
 * 4 channel 32-bit floating point image compare with constant value.
 * Compare pSrc's pixels with constant value. 
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pConstants pointer to a list of constants, one per color channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eComparisonOperation Specifies the comparison operation to be used in the pixel comparison.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus nppiCompareC_32f_C4R(const Npp32f * pSrc, int nSrcStep,
                               const Npp32f * pConstants,
                                     Npp8u * pDst,  int nDstStep,
                               NppiSize oSizeROI, NppCmpOp eComparisonOperation);

/** 
 * 4 channel 32-bit signed floating point compare, not affecting Alpha.
 * Compare pSrc's pixels with constant value. 
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pConstants pointer to a list of constants, one per color channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eComparisonOperation Specifies the comparison operation to be used in the pixel comparison.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus nppiCompareC_32f_AC4R(const Npp32f * pSrc, int nSrcStep,
                                const Npp32f * pConstants,
                                      Npp8u * pDst,  int nDstStep,
                                NppiSize oSizeROI, NppCmpOp eComparisonOperation);


/** 
 * 1 channel 32-bit floating point image compare whether two images are equal within epsilon.
 * Compare pSrc1's pixels with corresponding pixels in pSrc2 to determine whether they are equal with a difference of epsilon. 
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nEpsilon epsilon tolerance value to compare to pixel absolute differences
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus nppiCompareEqualEps_32f_C1R(const Npp32f * pSrc1, int nSrc1Step,
                                      const Npp32f * pSrc2, int nSrc2Step,
                                            Npp8u * pDst,   int nDstStep,
                                      NppiSize oSizeROI, Npp32f nEpsilon);

/** 
 * 3 channel 32-bit floating point image compare whether two images are equal within epsilon.
 * Compare pSrc1's pixels with corresponding pixels in pSrc2 to determine whether they are equal with a difference of epsilon. 
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nEpsilon epsilon tolerance value to compare to per color channel pixel absolute differences
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus nppiCompareEqualEps_32f_C3R(const Npp32f * pSrc1, int nSrc1Step,
                                      const Npp32f * pSrc2, int nSrc2Step,
                                            Npp8u * pDst,   int nDstStep,
                                      NppiSize oSizeROI, Npp32f nEpsilon);

/** 
 * 4 channel 32-bit floating point image compare whether two images are equal within epsilon.
 * Compare pSrc1's pixels with corresponding pixels in pSrc2 to determine whether they are equal with a difference of epsilon. 
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nEpsilon epsilon tolerance value to compare to per color channel pixel absolute differences
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus nppiCompareEqualEps_32f_C4R(const Npp32f * pSrc1, int nSrc1Step,
                                      const Npp32f * pSrc2, int nSrc2Step,
                                            Npp8u * pDst,   int nDstStep,
                                      NppiSize oSizeROI, Npp32f nEpsilon);

/** 
 * 4 channel 32-bit signed floating point compare whether two images are equal within epsilon, not affecting Alpha.
 * Compare pSrc1's pixels with corresponding pixels in pSrc2 to determine whether they are equal with a difference of epsilon. 
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nEpsilon epsilon tolerance value to compare to per color channel pixel absolute differences
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus nppiCompareEqualEps_32f_AC4R(const Npp32f * pSrc1, int nSrc1Step,
                                       const Npp32f * pSrc2, int nSrc2Step,
                                             Npp8u * pDst,   int nDstStep,
                                       NppiSize oSizeROI, Npp32f nEpsilon);

/** 
 * 1 channel 32-bit floating point image compare whether image and constant are equal within epsilon.
 * Compare pSrc's pixels with constant value to determine whether they are equal within a difference of epsilon. 
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param nConstant constant value
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nEpsilon epsilon tolerance value to compare to pixel absolute differences
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus nppiCompareEqualEpsC_32f_C1R(const Npp32f * pSrc, int nSrcStep,
                                       const Npp32f nConstant,
                                             Npp8u * pDst,  int nDstStep,
                                       NppiSize oSizeROI, Npp32f nEpsilon);

/** 
 * 3 channel 32-bit floating point image compare whether image and constant are equal within epsilon.
 * Compare pSrc's pixels with constant value to determine whether they are equal within a difference of epsilon. 
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pConstants pointer to a list of constants, one per color channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nEpsilon epsilon tolerance value to compare to per color channel pixel absolute differences
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus nppiCompareEqualEpsC_32f_C3R(const Npp32f * pSrc, int nSrcStep,
                                       const Npp32f * pConstants,
                                             Npp8u * pDst,  int nDstStep,
                                       NppiSize oSizeROI, Npp32f nEpsilon);

/** 
 * 4 channel 32-bit floating point image compare whether image and constant are equal within epsilon.
 * Compare pSrc's pixels with constant value to determine whether they are equal within a difference of epsilon. 
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pConstants pointer to a list of constants, one per color channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nEpsilon epsilon tolerance value to compare to per color channel pixel absolute differences
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus nppiCompareEqualEpsC_32f_C4R(const Npp32f * pSrc, int nSrcStep,
                                       const Npp32f * pConstants,
                                             Npp8u * pDst,  int nDstStep,
                                       NppiSize oSizeROI, Npp32f nEpsilon);

/** 
 * 4 channel 32-bit signed floating point compare whether image and constant are equal within epsilon, not affecting Alpha.
 * Compare pSrc's pixels with constant value to determine whether they are equal within a difference of epsilon. 
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pConstants pointer to a list of constants, one per color channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nEpsilon epsilon tolerance value to compare to per color channel pixel absolute differences
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus nppiCompareEqualEpsC_32f_AC4R(const Npp32f * pSrc, int nSrcStep,
                                        const Npp32f * pConstants,
                                              Npp8u * pDst,  int nDstStep,
                                        NppiSize oSizeROI, Npp32f nEpsilon);


/** @} image_compare_operations */

/** @} image_threshold_and_compare_operations */

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* LW_NPPI_THRESHOLD_AND_COMPARE_OPERATIONS_H */
