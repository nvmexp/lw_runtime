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
#ifndef LW_NPPI_FILTERING_FUNCTIONS_H
#define LW_NPPI_FILTERING_FUNCTIONS_H
 
/**
 * \file nppi_filtering_functions.h
 * NPP Image Processing Functionality.
 */
 
#include "nppdefs.h"


#ifdef __cplusplus
extern "C" {
#endif

/** @defgroup image_filtering_functions Filtering Functions
 *  @ingroup nppi
 *
 * Linear and non-linear image filtering functions.
 *
 * Filtering functions are classified as \ref neighborhood_operations. It is the user's 
 * responsibility to avoid \ref sampling_beyond_image_boundaries. 
 *
 * @{
 *
 * These functions can be found in the nppif library. Linking to only the sub-libraries that you use can significantly
 * save link time, application load time, and LWCA runtime startup time when using dynamic libraries.
 *
 */

/** @defgroup image_1D_linear_filter 1D Linear Filter
 *
 * @{
 *
 */

/** @name FilterColumn
 * Apply colwolution filter with user specified 1D column of weights.  
 * Result pixel is equal to the sum of the products between the kernel
 * coefficients (pKernel array) and corresponding neighboring column pixel
 * values in the source image defined by nKernelDim and nAnchorY, divided by
 * nDivisor. 
 * 
 * @{
 *
 */

/**
 * 8-bit unsigned single-channel 1D column colwolution.
 * 
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *                Coefficients are expected to be stored in reverse order.
 * \param nMaskSize Length of the linear kernel array.
 * \param nAnchor Y offset of the kernel origin frame of reference relative to the
 *                 source pixel.
 * \param nDivisor The factor by which the colwolved summation from the Filter
 *                 operation should be divided.  If equal to the sum of
 *                 coefficients, this will keep the maximum result value within
 *                 full scale.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterColumn_8u_C1R(const Npp8u * pSrc, Npp32s nSrcStep, Npp8u * pDst, Npp32s nDstStep, NppiSize oROI, 
                        const Npp32s * pKernel, Npp32s nMaskSize, Npp32s nAnchor, Npp32s nDivisor);

/**
 * 8-bit unsigned three-channel 1D column colwolution.
 * 
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *                Coefficients are expected to be stored in reverse order.
 * \param nMaskSize Length of the linear kernel array.
 * \param nAnchor Y offset of the kernel origin frame of reference relative to the
 *                 source pixel.
 * \param nDivisor The factor by which the colwolved summation from the Filter
 *                 operation should be divided.  If equal to the sum of
 *                 coefficients, this will keep the maximum result value within
 *                 full scale.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterColumn_8u_C3R(const Npp8u * pSrc, Npp32s nSrcStep, Npp8u * pDst, Npp32s nDstStep, NppiSize oROI, 
                        const Npp32s * pKernel, Npp32s nMaskSize, Npp32s nAnchor, Npp32s nDivisor);

/**
 * 8-bit unsigned four-channel 1D column colwolution.
 * 
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *                Coefficients are expected to be stored in reverse order.
 * \param nMaskSize Length of the linear kernel array.
 * \param nAnchor Y offset of the kernel origin frame of reference relative to the
 *                 source pixel.
 * \param nDivisor The factor by which the colwolved summation from the Filter
 *                 operation should be divided.  If equal to the sum of
 *                 coefficients, this will keep the maximum result value within
 *                 full scale.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterColumn_8u_C4R(const Npp8u * pSrc, Npp32s nSrcStep, Npp8u * pDst, Npp32s nDstStep, NppiSize oROI, 
                        const Npp32s * pKernel, Npp32s nMaskSize, Npp32s nAnchor, Npp32s nDivisor);

/**
 * 8-bit unsigned four-channel 1D column colwolution ignoring alpha-channel.
 * 
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *                Coefficients are expected to be stored in reverse order.
 * \param nMaskSize Length of the linear kernel array.
 * \param nAnchor Y offset of the kernel origin frame of reference relative to the
 *                 source pixel.
 * \param nDivisor The factor by which the colwolved summation from the Filter
 *                 operation should be divided.  If equal to the sum of
 *                 coefficients, this will keep the maximum result value within
 *                 full scale.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterColumn_8u_AC4R(const Npp8u * pSrc, Npp32s nSrcStep, Npp8u * pDst, Npp32s nDstStep, NppiSize oROI, 
                         const Npp32s * pKernel, Npp32s nMaskSize, Npp32s nAnchor, Npp32s nDivisor);

/**
 * 16-bit unsigned single-channel 1D column colwolution.
 * 
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *                Coefficients are expected to be stored in reverse order.
 * \param nMaskSize Length of the linear kernel array.
 * \param nAnchor Y offset of the kernel origin frame of reference relative to the
 *                 source pixel.
 * \param nDivisor The factor by which the colwolved summation from the Filter
 *                 operation should be divided.  If equal to the sum of
 *                 coefficients, this will keep the maximum result value within
 *                 full scale.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterColumn_16u_C1R(const Npp16u * pSrc, Npp32s nSrcStep, Npp16u * pDst, Npp32s nDstStep, NppiSize oROI, 
                         const Npp32s * pKernel, Npp32s nMaskSize, Npp32s nAnchor, Npp32s nDivisor);

/**
 * 16-bit unsigned three-channel 1D column colwolution.
 * 
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *                Coefficients are expected to be stored in reverse order.
 * \param nMaskSize Length of the linear kernel array.
 * \param nAnchor Y offset of the kernel origin frame of reference relative to the
 *                 source pixel.
 * \param nDivisor The factor by which the colwolved summation from the Filter
 *                 operation should be divided.  If equal to the sum of
 *                 coefficients, this will keep the maximum result value within
 *                 full scale.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterColumn_16u_C3R(const Npp16u * pSrc, Npp32s nSrcStep, Npp16u * pDst, Npp32s nDstStep, NppiSize oROI, 
                         const Npp32s * pKernel, Npp32s nMaskSize, Npp32s nAnchor, Npp32s nDivisor);

/**
 * 16-bit unsigned four-channel 1D column colwolution.
 * 
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *                Coefficients are expected to be stored in reverse order.
 * \param nMaskSize Length of the linear kernel array.
 * \param nAnchor Y offset of the kernel origin frame of reference relative to the
 *                 source pixel.
 * \param nDivisor The factor by which the colwolved summation from the Filter
 *                 operation should be divided.  If equal to the sum of
 *                 coefficients, this will keep the maximum result value within
 *                 full scale.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterColumn_16u_C4R(const Npp16u * pSrc, Npp32s nSrcStep, Npp16u * pDst, Npp32s nDstStep, NppiSize oROI, 
                         const Npp32s * pKernel, Npp32s nMaskSize, Npp32s nAnchor, Npp32s nDivisor);

/**
 * 16-bit unsigned four-channel 1D column colwolution ignoring alpha-channel.
 * 
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *                Coefficients are expected to be stored in reverse order.
 * \param nMaskSize Length of the linear kernel array.
 * \param nAnchor Y offset of the kernel origin frame of reference relative to the
 *                 source pixel.
 * \param nDivisor The factor by which the colwolved summation from the Filter
 *                 operation should be divided.  If equal to the sum of
 *                 coefficients, this will keep the maximum result value within
 *                 full scale.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterColumn_16u_AC4R(const Npp16u * pSrc, Npp32s nSrcStep, Npp16u * pDst, Npp32s nDstStep, NppiSize oROI, 
                          const Npp32s * pKernel, Npp32s nMaskSize, Npp32s nAnchor, Npp32s nDivisor);

/**
 * 16-bit single-channel 1D column colwolution.
 * 
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *                Coefficients are expected to be stored in reverse order.
 * \param nMaskSize Length of the linear kernel array.
 * \param nAnchor Y offset of the kernel origin frame of reference relative to the
 *                 source pixel.
 * \param nDivisor The factor by which the colwolved summation from the Filter
 *                 operation should be divided.  If equal to the sum of
 *                 coefficients, this will keep the maximum result value within
 *                 full scale.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterColumn_16s_C1R(const Npp16s * pSrc, Npp32s nSrcStep, Npp16s * pDst, Npp32s nDstStep, NppiSize oROI, 
                         const Npp32s * pKernel, Npp32s nMaskSize, Npp32s nAnchor, Npp32s nDivisor);

/**
 * 16-bit three-channel 1D column colwolution.
 * 
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *                Coefficients are expected to be stored in reverse order.
 * \param nMaskSize Length of the linear kernel array.
 * \param nAnchor Y offset of the kernel origin frame of reference relative to the
 *                 source pixel.
 * \param nDivisor The factor by which the colwolved summation from the Filter
 *                 operation should be divided.  If equal to the sum of
 *                 coefficients, this will keep the maximum result value within
 *                 full scale.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterColumn_16s_C3R(const Npp16s * pSrc, Npp32s nSrcStep, Npp16s * pDst, Npp32s nDstStep, NppiSize oROI, 
                         const Npp32s * pKernel, Npp32s nMaskSize, Npp32s nAnchor, Npp32s nDivisor);

/**
 * 16-bit four-channel 1D column colwolution.
 * 
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *                Coefficients are expected to be stored in reverse order.
 * \param nMaskSize Length of the linear kernel array.
 * \param nAnchor Y offset of the kernel origin frame of reference relative to the
 *                 source pixel.
 * \param nDivisor The factor by which the colwolved summation from the Filter
 *                 operation should be divided.  If equal to the sum of
 *                 coefficients, this will keep the maximum result value within
 *                 full scale.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterColumn_16s_C4R(const Npp16s * pSrc, Npp32s nSrcStep, Npp16s * pDst, Npp32s nDstStep, NppiSize oROI, 
                         const Npp32s * pKernel, Npp32s nMaskSize, Npp32s nAnchor, Npp32s nDivisor);

/**
 * 16-bit four-channel 1D column colwolution ignoring alpha-channel.
 * 
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *                Coefficients are expected to be stored in reverse order.
 * \param nMaskSize Length of the linear kernel array.
 * \param nAnchor Y offset of the kernel origin frame of reference relative to the
 *                 source pixel.
 * \param nDivisor The factor by which the colwolved summation from the Filter
 *                 operation should be divided.  If equal to the sum of
 *                 coefficients, this will keep the maximum result value within
 *                 full scale.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterColumn_16s_AC4R(const Npp16s * pSrc, Npp32s nSrcStep, Npp16s * pDst, Npp32s nDstStep, NppiSize oROI, 
                          const Npp32s * pKernel, Npp32s nMaskSize, Npp32s nAnchor, Npp32s nDivisor);

/**
 * 32-bit float single-channel 1D column colwolution.
 * 
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *                Coefficients are expected to be stored in reverse order.
 * \param nMaskSize Length of the linear kernel array.
 * \param nAnchor Y offset of the kernel origin frame of reference relative to the
 *                 source pixel.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterColumn_32f_C1R(const Npp32f * pSrc, Npp32s nSrcStep, Npp32f * pDst, Npp32s nDstStep, NppiSize oROI, 
                         const Npp32f * pKernel, Npp32s nMaskSize, Npp32s nAnchor);

/**
 * 32-bit float three-channel 1D column colwolution.
 * 
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *                Coefficients are expected to be stored in reverse order.
 * \param nMaskSize Length of the linear kernel array.
 * \param nAnchor Y offset of the kernel origin frame of reference relative to the
 *                 source pixel.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterColumn_32f_C3R(const Npp32f * pSrc, Npp32s nSrcStep, Npp32f * pDst, Npp32s nDstStep, NppiSize oROI, 
                         const Npp32f * pKernel, Npp32s nMaskSize, Npp32s nAnchor);

/**
 * 32-bit float four-channel 1D column colwolution.
 * 
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *                Coefficients are expected to be stored in reverse order.
 * \param nMaskSize Length of the linear kernel array.
 * \param nAnchor Y offset of the kernel origin frame of reference relative to the
 *                 source pixel.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterColumn_32f_C4R(const Npp32f * pSrc, Npp32s nSrcStep, Npp32f * pDst, Npp32s nDstStep, NppiSize oROI, 
                         const Npp32f * pKernel, Npp32s nMaskSize, Npp32s nAnchor);

/**
 * 32-bit float four-channel 1D column colwolution ignoring alpha-channel.
 * 
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *                Coefficients are expected to be stored in reverse order.
 * \param nMaskSize Length of the linear kernel array.
 * \param nAnchor Y offset of the kernel origin frame of reference relative to the
 *                 source pixel.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterColumn_32f_AC4R(const Npp32f * pSrc, Npp32s nSrcStep, Npp32f * pDst, Npp32s nDstStep, NppiSize oROI, 
                          const Npp32f * pKernel, Npp32s nMaskSize, Npp32s nAnchor);

/**
 * 64-bit float single-channel 1D column colwolution.
 * 
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *                Coefficients are expected to be stored in reverse order.
 * \param nMaskSize Length of the linear kernel array.
 * \param nAnchor Y offset of the kernel origin frame of reference relative to the
 *                 source pixel.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterColumn_64f_C1R(const Npp64f * pSrc, Npp32s nSrcStep, Npp64f * pDst, Npp32s nDstStep, NppiSize oROI, 
                         const Npp64f * pKernel, Npp32s nMaskSize, Npp32s nAnchor);


/** @} FilterColumn */

/** @name FilterColumnBorder
 * General purpose 1D colwolution column filter with border control.
 *
 * Pixels under the mask are multiplied by the respective weights in the mask
 * and the results are summed. Before writing the result pixel the sum is scaled
 * back via division by nDivisor. If any portion of the mask overlaps the source
 * image boundary the requested border type operation is applied to all mask pixels
 * which fall outside of the source image.
 *
 * Lwrrently only the NPP_BORDER_REPLICATE border type operation is supported.
 *
 * @{
 *
 */

/**
 * Single channel 8-bit unsigned 1D column colwolution filter with border control.
 * 
 *
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *        Coeffcients are expected to be stored in reverse order.
 * \param nMaskSize Width of the kernel.
 * \param nAnchor X offset of the kernel origin frame of reference
 *        relative to the source pixel.
 * \param nDivisor The factor by which the colwolved summation from the Filter
 *        operation should be divided.  If equal to the sum of coefficients,
 *        this will keep the maximum result value within full scale.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterColumnBorder_8u_C1R(const Npp8u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                              const Npp32s * pKernel, Npp32s nMaskSize, Npp32s nAnchor, Npp32s nDivisor, NppiBorderType eBorderType);
  
/**
 * Three channel 8-bit unsigned 1D column colwolution filter with border control.
 * 
 *
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *        Coeffcients are expected to be stored in reverse order.
 * \param nMaskSize Width of the kernel.
 * \param nAnchor X offset of the kernel origin frame of reference
 *        relative to the source pixel.
 * \param nDivisor The factor by which the colwolved summation from the Filter
 *        operation should be divided.  If equal to the sum of coefficients,
 *        this will keep the maximum result value within full scale.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterColumnBorder_8u_C3R(const Npp8u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                              const Npp32s * pKernel, Npp32s nMaskSize, Npp32s nAnchor, Npp32s nDivisor, NppiBorderType eBorderType);
                 
/**
 * Four channel channel 8-bit unsigned 1D column colwolution filter with border control.
 * 
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *        Coeffcients are expected to be stored in reverse order.
 * \param nMaskSize Width of the kernel.
 * \param nAnchor X offset of the kernel origin frame of reference
 *        relative to the source pixel.
 * \param nDivisor The factor by which the colwolved summation from the Filter
 *        operation should be divided.  If equal to the sum of coefficients,
 *        this will keep the maximum result value within full scale.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterColumnBorder_8u_C4R(const Npp8u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                              const Npp32s * pKernel, Npp32s nMaskSize, Npp32s nAnchor, Npp32s nDivisor, NppiBorderType eBorderType);

/**
 * Four channel 8-bit unsigned colwolution 1D column filter with border control, ignoring alpha channel.
 * 
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *        Coeffcients are expected to be stored in reverse order.
 * \param nMaskSize Width of the kernel.
 * \param nAnchor X offset of the kernel origin frame of reference
 *        relative to the source pixel.
 * \param nDivisor The factor by which the colwolved summation from the Filter
 *        operation should be divided.  If equal to the sum of coefficients,
 *        this will keep the maximum result value within full scale.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterColumnBorder_8u_AC4R(const Npp8u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                               const Npp32s * pKernel, Npp32s nMaskSize, Npp32s nAnchor, Npp32s nDivisor, NppiBorderType eBorderType);

/**
 * Single channel 16-bit unsigned colwolution 1D column filter with border control.
 * 
 *
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *        Coeffcients are expected to be stored in reverse order.
 * \param nMaskSize Width of the kernel.
 * \param nAnchor X offset of the kernel origin frame of reference
 *        relative to the source pixel.
 * \param nDivisor The factor by which the colwolved summation from the Filter
 *        operation should be divided.  If equal to the sum of coefficients,
 *        this will keep the maximum result value within full scale.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterColumnBorder_16u_C1R(const Npp16u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16u * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                               const Npp32s * pKernel, Npp32s nMaskSize, Npp32s nAnchor, Npp32s nDivisor, NppiBorderType eBorderType);
  
/**
 * Three channel 16-bit unsigned 1D column colwolution filter with border control.
 * 
 *
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *        Coeffcients are expected to be stored in reverse order.
 * \param nMaskSize Width of the kernel.
 * \param nAnchor X offset of the kernel origin frame of reference
 *        relative to the source pixel.
 * \param nDivisor The factor by which the colwolved summation from the Filter
 *        operation should be divided.  If equal to the sum of coefficients,
 *        this will keep the maximum result value within full scale.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterColumnBorder_16u_C3R(const Npp16u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16u * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                               const Npp32s * pKernel, Npp32s nMaskSize, Npp32s nAnchor, Npp32s nDivisor, NppiBorderType eBorderType);
                 
/**
 * Four channel channel 16-bit 1D column unsigned colwolution filter with border control.
 * 
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *        Coeffcients are expected to be stored in reverse order.
 * \param nMaskSize Width of the kernel.
 * \param nAnchor X offset of the kernel origin frame of reference
 *        relative to the source pixel.
 * \param nDivisor The factor by which the colwolved summation from the Filter
 *        operation should be divided.  If equal to the sum of coefficients,
 *        this will keep the maximum result value within full scale.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterColumnBorder_16u_C4R(const Npp16u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16u * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                               const Npp32s * pKernel, Npp32s nMaskSize, Npp32s nAnchor, Npp32s nDivisor, NppiBorderType eBorderType);

/**
 * Four channel 16-bit unsigned 1D column colwolution filter with border control, ignoring alpha channel.
 * 
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *        Coeffcients are expected to be stored in reverse order.
 * \param nMaskSize Width of the kernel.
 * \param nAnchor X offset of the kernel origin frame of reference
 *        relative to the source pixel.
 * \param nDivisor The factor by which the colwolved summation from the Filter
 *        operation should be divided.  If equal to the sum of coefficients,
 *        this will keep the maximum result value within full scale.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterColumnBorder_16u_AC4R(const Npp16u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16u * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                                const Npp32s * pKernel, Npp32s nMaskSize, Npp32s nAnchor, Npp32s nDivisor, NppiBorderType eBorderType);

/**
 * Single channel 16-bit 1D column colwolution filter with border control.
 * 
 *
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *        Coeffcients are expected to be stored in reverse order.
 * \param nMaskSize Width of the kernel.
 * \param nAnchor X offset of the kernel origin frame of reference
 *        relative to the source pixel.
 * \param nDivisor The factor by which the colwolved summation from the Filter
 *        operation should be divided.  If equal to the sum of coefficients,
 *        this will keep the maximum result value within full scale.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterColumnBorder_16s_C1R(const Npp16s * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                               const Npp32s * pKernel, Npp32s nMaskSize, Npp32s nAnchor, Npp32s nDivisor, NppiBorderType eBorderType);
  
/**
 * Three channel 16-bit 1D column colwolution filter with border control.
 * 
 *
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *        Coeffcients are expected to be stored in reverse order.
 * \param nMaskSize Width of the kernel.
 * \param nAnchor X offset of the kernel origin frame of reference
 *        relative to the source pixel.
 * \param nDivisor The factor by which the colwolved summation from the Filter
 *        operation should be divided.  If equal to the sum of coefficients,
 *        this will keep the maximum result value within full scale.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterColumnBorder_16s_C3R(const Npp16s * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                               const Npp32s * pKernel, Npp32s nMaskSize, Npp32s nAnchor, Npp32s nDivisor, NppiBorderType eBorderType);
                 
/**
 * Four channel channel 16-bit 1D column colwolution filter with border control.
 * 
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *        Coeffcients are expected to be stored in reverse order.
 * \param nMaskSize Width of the kernel.
 * \param nAnchor X offset of the kernel origin frame of reference
 *        relative to the source pixel.
 * \param nDivisor The factor by which the colwolved summation from the Filter
 *        operation should be divided.  If equal to the sum of coefficients,
 *        this will keep the maximum result value within full scale.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterColumnBorder_16s_C4R(const Npp16s * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                               const Npp32s * pKernel, Npp32s nMaskSize, Npp32s nAnchor, Npp32s nDivisor, NppiBorderType eBorderType);

/**
 * Four channel 16-bit 1D column colwolution filter with border control, ignoring alpha channel.
 * 
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *        Coeffcients are expected to be stored in reverse order.
 * \param nMaskSize Width of the kernel.
 * \param nAnchor X offset of the kernel origin frame of reference
 *        relative to the source pixel.
 * \param nDivisor The factor by which the colwolved summation from the Filter
 *        operation should be divided.  If equal to the sum of coefficients,
 *        this will keep the maximum result value within full scale.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterColumnBorder_16s_AC4R(const Npp16s * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                                const Npp32s * pKernel, Npp32s nMaskSize, Npp32s nAnchor, Npp32s nDivisor, NppiBorderType eBorderType);
 
/**
 * Single channel 32-bit float 1D column colwolution filter with border control.
 * 
 *
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *        Coeffcients are expected to be stored in reverse order.
 * \param nMaskSize Width of the kernel.
 * \param nAnchor X offset of the kernel origin frame of reference
 *        relative to the source pixel.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiFilterColumnBorder_32f_C1R(const Npp32f * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                               const Npp32f * pKernel, Npp32s nMaskSize, Npp32s nAnchor, NppiBorderType eBorderType);

/**
 * Three channel 32-bit float 1D column colwolution filter with border control.
 * 
 *
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *        Coeffcients are expected to be stored in reverse order.
 * \param nMaskSize Width of the kernel.
 * \param nAnchor X offset of the kernel origin frame of reference
 *        relative to the source pixel.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiFilterColumnBorder_32f_C3R(const Npp32f * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                               const Npp32f * pKernel, Npp32s nMaskSize, Npp32s nAnchor, NppiBorderType eBorderType);

/**
 * Four channel 32-bit float 1D column colwolution filter with border control.
 * 
 *
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *        Coeffcients are expected to be stored in reverse order.
 * \param nMaskSize Width of the kernel.
 * \param nAnchor X offset of the kernel origin frame of reference
 *        relative to the source pixel.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiFilterColumnBorder_32f_C4R(const Npp32f * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                               const Npp32f * pKernel, Npp32s nMaskSize, Npp32s nAnchor, NppiBorderType eBorderType);

/**
 * Four channel 32-bit float 1D column colwolution filter with border control, ignoring alpha channel.
 * 
 *
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *        Coeffcients are expected to be stored in reverse order.
 * \param nMaskSize Width of the kernel.
 * \param nAnchor X offset of the kernel origin frame of reference
 *        relative to the source pixel.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiFilterColumnBorder_32f_AC4R(const Npp32f * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                                const Npp32f * pKernel, Npp32s nMaskSize, Npp32s nAnchor, NppiBorderType eBorderType);

/** @} FilterColumnBorder */

/** @name FilterColumn32f
 * 
 * FilterColumn using floating-point weights.
 * 
 * @{
 *
 */

/**
 * 8-bit unsigned single-channel 1D column colwolution.
 * 
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *                Coefficients are expected to be stored in reverse order.
 * \param nMaskSize Length of the linear kernel array.
 * \param nAnchor Y offset of the kernel origin frame of reference relative to the
 *                 source pixel.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterColumn32f_8u_C1R(const Npp8u * pSrc, Npp32s nSrcStep, Npp8u * pDst, Npp32s nDstStep, NppiSize oROI, 
                           const Npp32f * pKernel, Npp32s nMaskSize, Npp32s nAnchor);

/**
 * 8-bit unsigned three-channel 1D column colwolution.
 * 
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *                Coefficients are expected to be stored in reverse order.
 * \param nMaskSize Length of the linear kernel array.
 * \param nAnchor Y offset of the kernel origin frame of reference relative to the
 *                 source pixel.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterColumn32f_8u_C3R(const Npp8u * pSrc, Npp32s nSrcStep, Npp8u * pDst, Npp32s nDstStep, NppiSize oROI, 
                           const Npp32f * pKernel, Npp32s nMaskSize, Npp32s nAnchor);

/**
 * 8-bit unsigned four-channel 1D column colwolution.
 * 
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *                Coefficients are expected to be stored in reverse order.
 * \param nMaskSize Length of the linear kernel array.
 * \param nAnchor Y offset of the kernel origin frame of reference relative to the
 *                 source pixel.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterColumn32f_8u_C4R(const Npp8u * pSrc, Npp32s nSrcStep, Npp8u * pDst, Npp32s nDstStep, NppiSize oROI, 
                           const Npp32f * pKernel, Npp32s nMaskSize, Npp32s nAnchor);

/**
 * 8-bit unsigned four-channel 1D column colwolution ignoring alpha-channel.
 * 
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *                Coefficients are expected to be stored in reverse order.
 * \param nMaskSize Length of the linear kernel array.
 * \param nAnchor Y offset of the kernel origin frame of reference relative to the
 *                 source pixel.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterColumn32f_8u_AC4R(const Npp8u * pSrc, Npp32s nSrcStep, Npp8u * pDst, Npp32s nDstStep, NppiSize oROI, 
                            const Npp32f * pKernel, Npp32s nMaskSize, Npp32s nAnchor);

/**
 * 16-bit unsigned single-channel 1D column colwolution.
 * 
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *                Coefficients are expected to be stored in reverse order.
 * \param nMaskSize Length of the linear kernel array.
 * \param nAnchor Y offset of the kernel origin frame of reference relative to the
 *                 source pixel.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterColumn32f_16u_C1R(const Npp16u * pSrc, Npp32s nSrcStep, Npp16u * pDst, Npp32s nDstStep, NppiSize oROI, 
                            const Npp32f * pKernel, Npp32s nMaskSize, Npp32s nAnchor);

/**
 * 16-bit unsigned three-channel 1D column colwolution.
 * 
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *                Coefficients are expected to be stored in reverse order.
 * \param nMaskSize Length of the linear kernel array.
 * \param nAnchor Y offset of the kernel origin frame of reference relative to the
 *                 source pixel.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterColumn32f_16u_C3R(const Npp16u * pSrc, Npp32s nSrcStep, Npp16u * pDst, Npp32s nDstStep, NppiSize oROI, 
                            const Npp32f * pKernel, Npp32s nMaskSize, Npp32s nAnchor);

/**
 * 16-bit unsigned four-channel 1D column colwolution.
 * 
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *                Coefficients are expected to be stored in reverse order.
 * \param nMaskSize Length of the linear kernel array.
 * \param nAnchor Y offset of the kernel origin frame of reference relative to the
 *                 source pixel.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterColumn32f_16u_C4R(const Npp16u * pSrc, Npp32s nSrcStep, Npp16u * pDst, Npp32s nDstStep, NppiSize oROI, 
                            const Npp32f * pKernel, Npp32s nMaskSize, Npp32s nAnchor);

/**
 * 16-bit unsigned four-channel 1D column colwolution ignoring alpha-channel.
 * 
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *                Coefficients are expected to be stored in reverse order.
 * \param nMaskSize Length of the linear kernel array.
 * \param nAnchor Y offset of the kernel origin frame of reference relative to the
 *                 source pixel.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterColumn32f_16u_AC4R(const Npp16u * pSrc, Npp32s nSrcStep, Npp16u * pDst, Npp32s nDstStep, NppiSize oROI, 
                             const Npp32f * pKernel, Npp32s nMaskSize, Npp32s nAnchor);

/**
 * 16-bit single-channel 1D column colwolution.
 * 
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *                Coefficients are expected to be stored in reverse order.
 * \param nMaskSize Length of the linear kernel array.
 * \param nAnchor Y offset of the kernel origin frame of reference relative to the
 *                 source pixel.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterColumn32f_16s_C1R(const Npp16s * pSrc, Npp32s nSrcStep, Npp16s * pDst, Npp32s nDstStep, NppiSize oROI, 
                            const Npp32f * pKernel, Npp32s nMaskSize, Npp32s nAnchor);

/**
 * 16-bit three-channel 1D column colwolution.
 * 
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *                Coefficients are expected to be stored in reverse order.
 * \param nMaskSize Length of the linear kernel array.
 * \param nAnchor Y offset of the kernel origin frame of reference relative to the
 *                 source pixel.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterColumn32f_16s_C3R(const Npp16s * pSrc, Npp32s nSrcStep, Npp16s * pDst, Npp32s nDstStep, NppiSize oROI, 
                            const Npp32f * pKernel, Npp32s nMaskSize, Npp32s nAnchor);

/**
 * 16-bit four-channel 1D column colwolution.
 * 
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *                Coefficients are expected to be stored in reverse order.
 * \param nMaskSize Length of the linear kernel array.
 * \param nAnchor Y offset of the kernel origin frame of reference relative to the
 *                 source pixel.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterColumn32f_16s_C4R(const Npp16s * pSrc, Npp32s nSrcStep, Npp16s * pDst, Npp32s nDstStep, NppiSize oROI, 
                            const Npp32f * pKernel, Npp32s nMaskSize, Npp32s nAnchor);

/**
 * 16-bit four-channel 1D column colwolution ignoring alpha-channel.
 * 
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *                Coefficients are expected to be stored in reverse order.
 * \param nMaskSize Length of the linear kernel array.
 * \param nAnchor Y offset of the kernel origin frame of reference relative to the
 *                 source pixel.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterColumn32f_16s_AC4R(const Npp16s * pSrc, Npp32s nSrcStep, Npp16s * pDst, Npp32s nDstStep, NppiSize oROI, 
                             const Npp32f * pKernel, Npp32s nMaskSize, Npp32s nAnchor);

/** @} FilterColumn32f */

/** @name FilterColumnBorder32f
 * General purpose 1D column colwolution filter using floating-point weights with border control.
 *
 * Pixels under the mask are multiplied by the respective weights in the mask
 * and the results are summed.  If any portion of the mask overlaps the source
 * image boundary the requested border type operation is applied to all mask pixels
 * which fall outside of the source image.
 *
 * Lwrrently only the NPP_BORDER_REPLICATE border type operation is supported.
 *
 * @{
 *
 */


/**
 * Single channel 8-bit unsigned 1D column colwolution filter with border control.
 * 
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *        Coeffcients are expected to be stored in reverse order.
 * \param nMaskSize Width of the kernel.
 * \param nAnchor X offset of the kernel origin frame of reference
 *        relative to the source pixel.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiFilterColumnBorder32f_8u_C1R(const Npp8u * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u * pDst, int nDstStep, NppiSize oSizeROI, 
                                 const Npp32f * pKernel, Npp32s nMaskSize, Npp32s nAnchor, NppiBorderType eBorderType);
        
/**
 * Three channel 8-bit unsigned 1D column colwolution filter with border control.
 * 
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *        Coeffcients are expected to be stored in reverse order.
 * \param nMaskSize Width of the kernel.
 * \param nAnchor X offset of the kernel origin frame of reference
 *        relative to the source pixel.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiFilterColumnBorder32f_8u_C3R(const Npp8u * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u * pDst, int nDstStep, NppiSize oSizeROI, 
                                 const Npp32f * pKernel, Npp32s nMaskSize, Npp32s nAnchor, NppiBorderType eBorderType);
        
/**
 * Four channel 8-bit unsigned 1D column colwolution filter with border control.
 * 
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *        Coeffcients are expected to be stored in reverse order.
 * \param nMaskSize Width of the kernel.
 * \param nAnchor X offset of the kernel origin frame of reference
 *        relative to the source pixel.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiFilterColumnBorder32f_8u_C4R(const Npp8u * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u * pDst, int nDstStep, NppiSize oSizeROI, 
                                 const Npp32f * pKernel, Npp32s nMaskSize, Npp32s nAnchor, NppiBorderType eBorderType);
        
/**
 * Four channel 8-bit unsigned 1D column colwolution filter with border control, ignorint alpha channel.
 * 
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *        Coeffcients are expected to be stored in reverse order.
 * \param nMaskSize Width of the kernel.
 * \param nAnchor X offset of the kernel origin frame of reference
 *        relative to the source pixel.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiFilterColumnBorder32f_8u_AC4R(const Npp8u * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u * pDst, int nDstStep, NppiSize oSizeROI, 
                                  const Npp32f * pKernel, Npp32s nMaskSize, Npp32s nAnchor, NppiBorderType eBorderType);

/**
 * Single channel 16-bit unsigned 1D column colwolution filter with border control.
 * 
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *        Coeffcients are expected to be stored in reverse order.
 * \param nMaskSize Width of the kernel.
 * \param nAnchor X offset of the kernel origin frame of reference
 *        relative to the source pixel.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiFilterColumnBorder32f_16u_C1R(const Npp16u * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16u * pDst, int nDstStep, NppiSize oSizeROI, 
                                  const Npp32f * pKernel, Npp32s nMaskSize, Npp32s nAnchor, NppiBorderType eBorderType);

/**
 * Three channel 16-bit unsigned 1D column colwolution filter with border control.
 * 
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *        Coeffcients are expected to be stored in reverse order.
 * \param nMaskSize Width of the kernel.
 * \param nAnchor X offset of the kernel origin frame of reference
 *        relative to the source pixel.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiFilterColumnBorder32f_16u_C3R(const Npp16u * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16u * pDst, int nDstStep, NppiSize oSizeROI, 
                                  const Npp32f * pKernel, Npp32s nMaskSize, Npp32s nAnchor, NppiBorderType eBorderType);

/**
 * Four channel 16-bit unsigned 1D column colwolution filter with border control.
 * 
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *        Coeffcients are expected to be stored in reverse order.
 * \param nMaskSize Width of the kernel.
 * \param nAnchor X offset of the kernel origin frame of reference
 *        relative to the source pixel.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiFilterColumnBorder32f_16u_C4R(const Npp16u * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16u * pDst, int nDstStep, NppiSize oSizeROI, 
                                  const Npp32f * pKernel, Npp32s nMaskSize, Npp32s nAnchor, NppiBorderType eBorderType);

/**
 * Four channel 16-bit unsigned 1D column colwolution filter with border control, ignoring alpha channel.
 * 
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *        Coeffcients are expected to be stored in reverse order.
 * \param nMaskSize Width of the kernel.
 * \param nAnchor X offset of the kernel origin frame of reference
 *        relative to the source pixel.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiFilterColumnBorder32f_16u_AC4R(const Npp16u * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16u * pDst, int nDstStep, NppiSize oSizeROI, 
                                   const Npp32f * pKernel, Npp32s nMaskSize, Npp32s nAnchor, NppiBorderType eBorderType);

/**
 * Single channel 16-bit 1D column colwolution filter with border control.
 * 
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *        Coeffcients are expected to be stored in reverse order.
 * \param nMaskSize Width of the kernel.
 * \param nAnchor X offset of the kernel origin frame of reference
 *        relative to the source pixel.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiFilterColumnBorder32f_16s_C1R(const Npp16s * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16s * pDst, int nDstStep, NppiSize oSizeROI, 
                                  const Npp32f * pKernel, Npp32s nMaskSize, Npp32s nAnchor, NppiBorderType eBorderType);

/**
 * Three channel 16-bit 1D column colwolution filter with border control.
 * 
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *        Coeffcients are expected to be stored in reverse order.
 * \param nMaskSize Width of the kernel.
 * \param nAnchor X offset of the kernel origin frame of reference
 *        relative to the source pixel.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiFilterColumnBorder32f_16s_C3R(const Npp16s * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16s * pDst, int nDstStep, NppiSize oSizeROI, 
                                  const Npp32f * pKernel, Npp32s nMaskSize, Npp32s nAnchor, NppiBorderType eBorderType);

/**
 * Four channel 16-bit 1D column colwolution filter with border control.
 * 
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *        Coeffcients are expected to be stored in reverse order.
 * \param nMaskSize Width of the kernel.
 * \param nAnchor X offset of the kernel origin frame of reference
 *        relative to the source pixel.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiFilterColumnBorder32f_16s_C4R(const Npp16s * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16s * pDst, int nDstStep, NppiSize oSizeROI, 
                                  const Npp32f * pKernel, Npp32s nMaskSize, Npp32s nAnchor, NppiBorderType eBorderType);

/**
 * Four channel 16-bit 1D column colwolution filter with border control, ignoring alpha channel.
 * 
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *        Coeffcients are expected to be stored in reverse order.
 * \param nMaskSize Width of the kernel.
 * \param nAnchor X offset of the kernel origin frame of reference
 *        relative to the source pixel.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiFilterColumnBorder32f_16s_AC4R(const Npp16s * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16s * pDst, int nDstStep, NppiSize oSizeROI, 
                                   const Npp32f * pKernel, Npp32s nMaskSize, Npp32s nAnchor, NppiBorderType eBorderType);

/** @} FilterColumnBorder32f */

/** @name FilterRow
 * Apply colwolution filter with user specified 1D row of weights.  
 * Result pixel is equal to the sum of the products between the kernel
 * coefficients (pKernel array) and corresponding neighboring row pixel
 * values in the source image defined by nKernelDim and nAnchorX, divided by
 * nDivisor. 
 * 
 * @{
 *
 */

/**
 * 8-bit unsigned single-channel 1D row colwolution.
 * 
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *                Coefficients are expected to be stored in reverse order.
 * \param nMaskSize Length of the linear kernel array.
 * \param nAnchor X offset of the kernel origin frame of reference relative to the
 *                 source pixel.
 * \param nDivisor The factor by which the colwolved summation from the Filter
 *                 operation should be divided.  If equal to the sum of
 *                 coefficients, this will keep the maximum result value within
 *                 full scale.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterRow_8u_C1R(const Npp8u * pSrc, Npp32s nSrcStep, Npp8u * pDst, Npp32s nDstStep, NppiSize oROI, 
                     const Npp32s * pKernel, Npp32s nMaskSize, Npp32s nAnchor, Npp32s nDivisor);

/**
 * 8-bit unsigned three-channel 1D row colwolution.
 * 
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *                Coefficients are expected to be stored in reverse order.
 * \param nMaskSize Length of the linear kernel array.
 * \param nAnchor X offset of the kernel origin frame of reference relative to the
 *                 source pixel.
 * \param nDivisor The factor by which the colwolved summation from the Filter
 *                 operation should be divided.  If equal to the sum of
 *                 coefficients, this will keep the maximum result value within
 *                 full scale.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterRow_8u_C3R(const Npp8u * pSrc, Npp32s nSrcStep, Npp8u * pDst, Npp32s nDstStep, NppiSize oROI, 
                     const Npp32s * pKernel, Npp32s nMaskSize, Npp32s nAnchor, Npp32s nDivisor);

/**
 * 8-bit unsigned four-channel 1D row colwolution.
 * 
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *                Coefficients are expected to be stored in reverse order.
 * \param nMaskSize Length of the linear kernel array.
 * \param nAnchor X offset of the kernel origin frame of reference relative to the
 *                 source pixel.
 * \param nDivisor The factor by which the colwolved summation from the Filter
 *                 operation should be divided.  If equal to the sum of
 *                 coefficients, this will keep the maximum result value within
 *                 full scale.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterRow_8u_C4R(const Npp8u * pSrc, Npp32s nSrcStep, Npp8u * pDst, Npp32s nDstStep, NppiSize oROI, 
                     const Npp32s * pKernel, Npp32s nMaskSize, Npp32s nAnchor, Npp32s nDivisor);

/**
 * 8-bit unsigned four-channel 1D row colwolution ignoring alpha-channel.
 * 
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *                Coefficients are expected to be stored in reverse order.
 * \param nMaskSize Length of the linear kernel array.
 * \param nAnchor X offset of the kernel origin frame of reference relative to the
 *                 source pixel.
 * \param nDivisor The factor by which the colwolved summation from the Filter
 *                 operation should be divided.  If equal to the sum of
 *                 coefficients, this will keep the maximum result value within
 *                 full scale.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterRow_8u_AC4R(const Npp8u * pSrc, Npp32s nSrcStep, Npp8u * pDst, Npp32s nDstStep, NppiSize oROI, 
                      const Npp32s * pKernel, Npp32s nMaskSize, Npp32s nAnchor, Npp32s nDivisor);

/**
 * 16-bit unsigned single-channel 1D row colwolution.
 * 
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *                Coefficients are expected to be stored in reverse order.
 * \param nMaskSize Length of the linear kernel array.
 * \param nAnchor X offset of the kernel origin frame of reference relative to the
 *                 source pixel.
 * \param nDivisor The factor by which the colwolved summation from the Filter
 *                 operation should be divided.  If equal to the sum of
 *                 coefficients, this will keep the maximum result value within
 *                 full scale.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterRow_16u_C1R(const Npp16u * pSrc, Npp32s nSrcStep, Npp16u * pDst, Npp32s nDstStep, NppiSize oROI, 
                      const Npp32s * pKernel, Npp32s nMaskSize, Npp32s nAnchor, Npp32s nDivisor);

/**
 * 16-bit unsigned three-channel 1D row colwolution.
 * 
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *                Coefficients are expected to be stored in reverse order.
 * \param nMaskSize Length of the linear kernel array.
 * \param nAnchor X offset of the kernel origin frame of reference relative to the
 *                 source pixel.
 * \param nDivisor The factor by which the colwolved summation from the Filter
 *                 operation should be divided.  If equal to the sum of
 *                 coefficients, this will keep the maximum result value within
 *                 full scale.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterRow_16u_C3R(const Npp16u * pSrc, Npp32s nSrcStep, Npp16u * pDst, Npp32s nDstStep, NppiSize oROI, 
                      const Npp32s * pKernel, Npp32s nMaskSize, Npp32s nAnchor, Npp32s nDivisor);

/**
 * 16-bit unsigned four-channel 1D row colwolution.
 * 
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *                Coefficients are expected to be stored in reverse order.
 * \param nMaskSize Length of the linear kernel array.
 * \param nAnchor X offset of the kernel origin frame of reference relative to the
 *                 source pixel.
 * \param nDivisor The factor by which the colwolved summation from the Filter
 *                 operation should be divided.  If equal to the sum of
 *                 coefficients, this will keep the maximum result value within
 *                 full scale.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterRow_16u_C4R(const Npp16u * pSrc, Npp32s nSrcStep, Npp16u * pDst, Npp32s nDstStep, NppiSize oROI, 
                      const Npp32s * pKernel, Npp32s nMaskSize, Npp32s nAnchor, Npp32s nDivisor);

/**
 * 16-bit unsigned four-channel 1D row colwolution ignoring alpha-channel.
 * 
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *                Coefficients are expected to be stored in reverse order.
 * \param nMaskSize Length of the linear kernel array.
 * \param nAnchor X offset of the kernel origin frame of reference relative to the
 *                 source pixel.
 * \param nDivisor The factor by which the colwolved summation from the Filter
 *                 operation should be divided.  If equal to the sum of
 *                 coefficients, this will keep the maximum result value within
 *                 full scale.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterRow_16u_AC4R(const Npp16u * pSrc, Npp32s nSrcStep, Npp16u * pDst, Npp32s nDstStep, NppiSize oROI, 
                       const Npp32s * pKernel, Npp32s nMaskSize, Npp32s nAnchor, Npp32s nDivisor);

/**
 * 16-bit single-channel 1D row colwolution.
 * 
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *                Coefficients are expected to be stored in reverse order.
 * \param nMaskSize Length of the linear kernel array.
 * \param nAnchor X offset of the kernel origin frame of reference relative to the
 *                 source pixel.
 * \param nDivisor The factor by which the colwolved summation from the Filter
 *                 operation should be divided.  If equal to the sum of
 *                 coefficients, this will keep the maximum result value within
 *                 full scale.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterRow_16s_C1R(const Npp16s * pSrc, Npp32s nSrcStep, Npp16s * pDst, Npp32s nDstStep, NppiSize oROI, 
                      const Npp32s * pKernel, Npp32s nMaskSize, Npp32s nAnchor, Npp32s nDivisor);

/**
 * 16-bit three-channel 1D row colwolution.
 * 
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *                Coefficients are expected to be stored in reverse order.
 * \param nMaskSize Length of the linear kernel array.
 * \param nAnchor X offset of the kernel origin frame of reference relative to the
 *                 source pixel.
 * \param nDivisor The factor by which the colwolved summation from the Filter
 *                 operation should be divided.  If equal to the sum of
 *                 coefficients, this will keep the maximum result value within
 *                 full scale.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterRow_16s_C3R(const Npp16s * pSrc, Npp32s nSrcStep, Npp16s * pDst, Npp32s nDstStep, NppiSize oROI, 
                      const Npp32s * pKernel, Npp32s nMaskSize, Npp32s nAnchor, Npp32s nDivisor);

/**
 * 16-bit four-channel 1D row colwolution.
 * 
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *                Coefficients are expected to be stored in reverse order.
 * \param nMaskSize Length of the linear kernel array.
 * \param nAnchor X offset of the kernel origin frame of reference relative to the
 *                 source pixel.
 * \param nDivisor The factor by which the colwolved summation from the Filter
 *                 operation should be divided.  If equal to the sum of
 *                 coefficients, this will keep the maximum result value within
 *                 full scale.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterRow_16s_C4R(const Npp16s * pSrc, Npp32s nSrcStep, Npp16s * pDst, Npp32s nDstStep, NppiSize oROI, 
                      const Npp32s * pKernel, Npp32s nMaskSize, Npp32s nAnchor, Npp32s nDivisor);

/**
 * 16-bit four-channel 1D row colwolution ignoring alpha-channel.
 * 
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *                Coefficients are expected to be stored in reverse order.
 * \param nMaskSize Length of the linear kernel array.
 * \param nAnchor X offset of the kernel origin frame of reference relative to the
 *                 source pixel.
 * \param nDivisor The factor by which the colwolved summation from the Filter
 *                 operation should be divided.  If equal to the sum of
 *                 coefficients, this will keep the maximum result value within
 *                 full scale.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterRow_16s_AC4R(const Npp16s * pSrc, Npp32s nSrcStep, Npp16s * pDst, Npp32s nDstStep, NppiSize oROI, 
                       const Npp32s * pKernel, Npp32s nMaskSize, Npp32s nAnchor, Npp32s nDivisor);


/**
 * 32-bit float single-channel 1D row colwolution.
 * 
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *                Coefficients are expected to be stored in reverse order.
 * \param nMaskSize Length of the linear kernel array.
 * \param nAnchor X offset of the kernel origin frame of reference relative to the
 *                 source pixel.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterRow_32f_C1R(const Npp32f * pSrc, Npp32s nSrcStep, Npp32f * pDst, Npp32s nDstStep, NppiSize oROI, 
                      const Npp32f * pKernel, Npp32s nMaskSize, Npp32s nAnchor);

/**
 * 32-bit float three-channel 1D row colwolution.
 * 
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *                Coefficients are expected to be stored in reverse order.
 * \param nMaskSize Length of the linear kernel array.
 * \param nAnchor X offset of the kernel origin frame of reference relative to the
 *                 source pixel.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterRow_32f_C3R(const Npp32f * pSrc, Npp32s nSrcStep, Npp32f * pDst, Npp32s nDstStep, NppiSize oROI, 
                      const Npp32f * pKernel, Npp32s nMaskSize, Npp32s nAnchor);

/**
 * 32-bit float four-channel 1D row colwolution.
 * 
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *                Coefficients are expected to be stored in reverse order.
 * \param nMaskSize Length of the linear kernel array.
 * \param nAnchor X offset of the kernel origin frame of reference relative to the
 *                 source pixel.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterRow_32f_C4R(const Npp32f * pSrc, Npp32s nSrcStep, Npp32f * pDst, Npp32s nDstStep, NppiSize oROI, 
                      const Npp32f * pKernel, Npp32s nMaskSize, Npp32s nAnchor);

/**
 * 32-bit float four-channel 1D row colwolution ignoring alpha-channel.
 * 
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *                Coefficients are expected to be stored in reverse order.
 * \param nMaskSize Length of the linear kernel array.
 * \param nAnchor X offset of the kernel origin frame of reference relative to the
 *                 source pixel.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterRow_32f_AC4R(const Npp32f * pSrc, Npp32s nSrcStep, Npp32f * pDst, Npp32s nDstStep, NppiSize oROI, 
                       const Npp32f * pKernel, Npp32s nMaskSize, Npp32s nAnchor);

/**
 * 64-bit float single-channel 1D row colwolution.
 * 
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *                Coefficients are expected to be stored in reverse order.
 * \param nMaskSize Length of the linear kernel array.
 * \param nAnchor X offset of the kernel origin frame of reference relative to the
 *                 source pixel.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterRow_64f_C1R(const Npp64f * pSrc, Npp32s nSrcStep, Npp64f * pDst, Npp32s nDstStep, NppiSize oROI, 
                      const Npp64f * pKernel, Npp32s nMaskSize, Npp32s nAnchor);


/** @} FilterRow */

/** @name FilterRowBorder
 * General purpose 1D colwolution row filter with border control.
 *
 * Pixels under the mask are multiplied by the respective weights in the mask
 * and the results are summed. Before writing the result pixel the sum is scaled
 * back via division by nDivisor. If any portion of the mask overlaps the source
 * image boundary the requested border type operation is applied to all mask pixels
 * which fall outside of the source image.
 *
 * Lwrrently only the NPP_BORDER_REPLICATE border type operation is supported.
 *
 * @{
 *
 */

/**
 * Single channel 8-bit unsigned 1D row colwolution filter with border control.
 * 
 *
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *        Coeffcients are expected to be stored in reverse order.
 * \param nMaskSize Width of the kernel.
 * \param nAnchor X offset of the kernel origin frame of reference
 *        relative to the source pixel.
 * \param nDivisor The factor by which the colwolved summation from the Filter
 *        operation should be divided.  If equal to the sum of coefficients,
 *        this will keep the maximum result value within full scale.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterRowBorder_8u_C1R(const Npp8u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                           const Npp32s * pKernel, Npp32s nMaskSize, Npp32s nAnchor, Npp32s nDivisor, NppiBorderType eBorderType);
  
/**
 * Three channel 8-bit unsigned 1D row colwolution filter with border control.
 * 
 *
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *        Coeffcients are expected to be stored in reverse order.
 * \param nMaskSize Width of the kernel.
 * \param nAnchor X offset of the kernel origin frame of reference
 *        relative to the source pixel.
 * \param nDivisor The factor by which the colwolved summation from the Filter
 *        operation should be divided.  If equal to the sum of coefficients,
 *        this will keep the maximum result value within full scale.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterRowBorder_8u_C3R(const Npp8u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                           const Npp32s * pKernel, Npp32s nMaskSize, Npp32s nAnchor, Npp32s nDivisor, NppiBorderType eBorderType);
                 
/**
 * Four channel channel 8-bit unsigned 1D row colwolution filter with border control.
 * 
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *        Coeffcients are expected to be stored in reverse order.
 * \param nMaskSize Width of the kernel.
 * \param nAnchor X offset of the kernel origin frame of reference
 *        relative to the source pixel.
 * \param nDivisor The factor by which the colwolved summation from the Filter
 *        operation should be divided.  If equal to the sum of coefficients,
 *        this will keep the maximum result value within full scale.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterRowBorder_8u_C4R(const Npp8u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                           const Npp32s * pKernel, Npp32s nMaskSize, Npp32s nAnchor, Npp32s nDivisor, NppiBorderType eBorderType);

/**
 * Four channel 8-bit unsigned colwolution 1D row filter with border control, ignoring alpha channel.
 * 
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *        Coeffcients are expected to be stored in reverse order.
 * \param nMaskSize Width of the kernel.
 * \param nAnchor X offset of the kernel origin frame of reference
 *        relative to the source pixel.
 * \param nDivisor The factor by which the colwolved summation from the Filter
 *        operation should be divided.  If equal to the sum of coefficients,
 *        this will keep the maximum result value within full scale.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterRowBorder_8u_AC4R(const Npp8u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                            const Npp32s * pKernel, Npp32s nMaskSize, Npp32s nAnchor, Npp32s nDivisor, NppiBorderType eBorderType);

/**
 * Single channel 16-bit unsigned colwolution 1D row filter with border control.
 * 
 *
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *        Coeffcients are expected to be stored in reverse order.
 * \param nMaskSize Width of the kernel.
 * \param nAnchor X offset of the kernel origin frame of reference
 *        relative to the source pixel.
 * \param nDivisor The factor by which the colwolved summation from the Filter
 *        operation should be divided.  If equal to the sum of coefficients,
 *        this will keep the maximum result value within full scale.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterRowBorder_16u_C1R(const Npp16u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16u * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                            const Npp32s * pKernel, Npp32s nMaskSize, Npp32s nAnchor, Npp32s nDivisor, NppiBorderType eBorderType);
  
/**
 * Three channel 16-bit unsigned 1D row colwolution filter with border control.
 * 
 *
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *        Coeffcients are expected to be stored in reverse order.
 * \param nMaskSize Width of the kernel.
 * \param nAnchor X offset of the kernel origin frame of reference
 *        relative to the source pixel.
 * \param nDivisor The factor by which the colwolved summation from the Filter
 *        operation should be divided.  If equal to the sum of coefficients,
 *        this will keep the maximum result value within full scale.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterRowBorder_16u_C3R(const Npp16u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16u * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                            const Npp32s * pKernel, Npp32s nMaskSize, Npp32s nAnchor, Npp32s nDivisor, NppiBorderType eBorderType);
                 
/**
 * Four channel channel 16-bit 1D row unsigned colwolution filter with border control.
 * 
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *        Coeffcients are expected to be stored in reverse order.
 * \param nMaskSize Width of the kernel.
 * \param nAnchor X offset of the kernel origin frame of reference
 *        relative to the source pixel.
 * \param nDivisor The factor by which the colwolved summation from the Filter
 *        operation should be divided.  If equal to the sum of coefficients,
 *        this will keep the maximum result value within full scale.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterRowBorder_16u_C4R(const Npp16u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16u * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                            const Npp32s * pKernel, Npp32s nMaskSize, Npp32s nAnchor, Npp32s nDivisor, NppiBorderType eBorderType);

/**
 * Four channel 16-bit unsigned 1D row colwolution filter with border control, ignoring alpha channel.
 * 
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *        Coeffcients are expected to be stored in reverse order.
 * \param nMaskSize Width of the kernel.
 * \param nAnchor X offset of the kernel origin frame of reference
 *        relative to the source pixel.
 * \param nDivisor The factor by which the colwolved summation from the Filter
 *        operation should be divided.  If equal to the sum of coefficients,
 *        this will keep the maximum result value within full scale.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterRowBorder_16u_AC4R(const Npp16u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16u * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                             const Npp32s * pKernel, Npp32s nMaskSize, Npp32s nAnchor, Npp32s nDivisor, NppiBorderType eBorderType);

/**
 * Single channel 16-bit 1D row colwolution filter with border control.
 * 
 *
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *        Coeffcients are expected to be stored in reverse order.
 * \param nMaskSize Width of the kernel.
 * \param nAnchor X offset of the kernel origin frame of reference
 *        relative to the source pixel.
 * \param nDivisor The factor by which the colwolved summation from the Filter
 *        operation should be divided.  If equal to the sum of coefficients,
 *        this will keep the maximum result value within full scale.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterRowBorder_16s_C1R(const Npp16s * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                            const Npp32s * pKernel, Npp32s nMaskSize, Npp32s nAnchor, Npp32s nDivisor, NppiBorderType eBorderType);
  
/**
 * Three channel 16-bit 1D row colwolution filter with border control.
 * 
 *
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *        Coeffcients are expected to be stored in reverse order.
 * \param nMaskSize Width of the kernel.
 * \param nAnchor X offset of the kernel origin frame of reference
 *        relative to the source pixel.
 * \param nDivisor The factor by which the colwolved summation from the Filter
 *        operation should be divided.  If equal to the sum of coefficients,
 *        this will keep the maximum result value within full scale.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterRowBorder_16s_C3R(const Npp16s * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                            const Npp32s * pKernel, Npp32s nMaskSize, Npp32s nAnchor, Npp32s nDivisor, NppiBorderType eBorderType);
                 
/**
 * Four channel channel 16-bit 1D row colwolution filter with border control.
 * 
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *        Coeffcients are expected to be stored in reverse order.
 * \param nMaskSize Width of the kernel.
 * \param nAnchor X offset of the kernel origin frame of reference
 *        relative to the source pixel.
 * \param nDivisor The factor by which the colwolved summation from the Filter
 *        operation should be divided.  If equal to the sum of coefficients,
 *        this will keep the maximum result value within full scale.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterRowBorder_16s_C4R(const Npp16s * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                            const Npp32s * pKernel, Npp32s nMaskSize, Npp32s nAnchor, Npp32s nDivisor, NppiBorderType eBorderType);

/**
 * Four channel 16-bit 1D row colwolution filter with border control, ignoring alpha channel.
 * 
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *        Coeffcients are expected to be stored in reverse order.
 * \param nMaskSize Width of the kernel.
 * \param nAnchor X offset of the kernel origin frame of reference
 *        relative to the source pixel.
 * \param nDivisor The factor by which the colwolved summation from the Filter
 *        operation should be divided.  If equal to the sum of coefficients,
 *        this will keep the maximum result value within full scale.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterRowBorder_16s_AC4R(const Npp16s * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                             const Npp32s * pKernel, Npp32s nMaskSize, Npp32s nAnchor, Npp32s nDivisor, NppiBorderType eBorderType);
 
/**
 * Single channel 32-bit float 1D row colwolution filter with border control.
 * 
 *
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *        Coeffcients are expected to be stored in reverse order.
 * \param nMaskSize Width of the kernel.
 * \param nAnchor X offset of the kernel origin frame of reference
 *        relative to the source pixel.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiFilterRowBorder_32f_C1R(const Npp32f * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                            const Npp32f * pKernel, Npp32s nMaskSize, Npp32s nAnchor, NppiBorderType eBorderType);

/**
 * Three channel 32-bit float 1D row colwolution filter with border control.
 * 
 *
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *        Coeffcients are expected to be stored in reverse order.
 * \param nMaskSize Width of the kernel.
 * \param nAnchor X offset of the kernel origin frame of reference
 *        relative to the source pixel.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiFilterRowBorder_32f_C3R(const Npp32f * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                            const Npp32f * pKernel, Npp32s nMaskSize, Npp32s nAnchor, NppiBorderType eBorderType);

/**
 * Four channel 32-bit float 1D row colwolution filter with border control.
 * 
 *
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *        Coeffcients are expected to be stored in reverse order.
 * \param nMaskSize Width of the kernel.
 * \param nAnchor X offset of the kernel origin frame of reference
 *        relative to the source pixel.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiFilterRowBorder_32f_C4R(const Npp32f * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                            const Npp32f * pKernel, Npp32s nMaskSize, Npp32s nAnchor, NppiBorderType eBorderType);

/**
 * Four channel 32-bit float 1D row colwolution filter with border control, ignoring alpha channel.
 * 
 *
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *        Coeffcients are expected to be stored in reverse order.
 * \param nMaskSize Width of the kernel.
 * \param nAnchor X offset of the kernel origin frame of reference
 *        relative to the source pixel.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiFilterRowBorder_32f_AC4R(const Npp32f * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                             const Npp32f * pKernel, Npp32s nMaskSize, Npp32s nAnchor, NppiBorderType eBorderType);

/** @} FilterRowBorder */

/** @name FilterRow32f
 * 
 * FilterRow using floating-point weights.
 * 
 * @{
 *
 */

/**
 * 8-bit unsigned single-channel 1D row colwolution.
 * 
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *                Coefficients are expected to be stored in reverse order.
 * \param nMaskSize Length of the linear kernel array.
 * \param nAnchor X offset of the kernel origin frame of reference relative to the
 *                 source pixel.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterRow32f_8u_C1R(const Npp8u * pSrc, Npp32s nSrcStep, Npp8u * pDst, Npp32s nDstStep, NppiSize oROI, 
                        const Npp32f * pKernel, Npp32s nMaskSize, Npp32s nAnchor);

/**
 * 8-bit unsigned three-channel 1D row colwolution.
 * 
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *                Coefficients are expected to be stored in reverse order.
 * \param nMaskSize Length of the linear kernel array.
 * \param nAnchor X offset of the kernel origin frame of reference relative to the
 *                 source pixel.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterRow32f_8u_C3R(const Npp8u * pSrc, Npp32s nSrcStep, Npp8u * pDst, Npp32s nDstStep, NppiSize oROI, 
                        const Npp32f * pKernel, Npp32s nMaskSize, Npp32s nAnchor);

/**
 * 8-bit unsigned four-channel 1D row colwolution.
 * 
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *                Coefficients are expected to be stored in reverse order.
 * \param nMaskSize Length of the linear kernel array.
 * \param nAnchor X offset of the kernel origin frame of reference relative to the
 *                 source pixel.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterRow32f_8u_C4R(const Npp8u * pSrc, Npp32s nSrcStep, Npp8u * pDst, Npp32s nDstStep, NppiSize oROI, 
                        const Npp32f * pKernel, Npp32s nMaskSize, Npp32s nAnchor);

/**
 * 8-bit unsigned four-channel 1D row colwolution ignoring alpha-channel.
 * 
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *                Coefficients are expected to be stored in reverse order.
 * \param nMaskSize Length of the linear kernel array.
 * \param nAnchor X offset of the kernel origin frame of reference relative to the
 *                 source pixel.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterRow32f_8u_AC4R(const Npp8u * pSrc, Npp32s nSrcStep, Npp8u * pDst, Npp32s nDstStep, NppiSize oROI, 
                         const Npp32f * pKernel, Npp32s nMaskSize, Npp32s nAnchor);

/**
 * 16-bit unsigned single-channel 1D row colwolution.
 * 
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *                Coefficients are expected to be stored in reverse order.
 * \param nMaskSize Length of the linear kernel array.
 * \param nAnchor X offset of the kernel origin frame of reference relative to the
 *                 source pixel.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterRow32f_16u_C1R(const Npp16u * pSrc, Npp32s nSrcStep, Npp16u * pDst, Npp32s nDstStep, NppiSize oROI, 
                         const Npp32f * pKernel, Npp32s nMaskSize, Npp32s nAnchor);

/**
 * 16-bit unsigned three-channel 1D row colwolution.
 * 
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *                Coefficients are expected to be stored in reverse order.
 * \param nMaskSize Length of the linear kernel array.
 * \param nAnchor X offset of the kernel origin frame of reference relative to the
 *                 source pixel.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterRow32f_16u_C3R(const Npp16u * pSrc, Npp32s nSrcStep, Npp16u * pDst, Npp32s nDstStep, NppiSize oROI, 
                         const Npp32f * pKernel, Npp32s nMaskSize, Npp32s nAnchor);

/**
 * 16-bit unsigned four-channel 1D row colwolution.
 * 
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *                Coefficients are expected to be stored in reverse order.
 * \param nMaskSize Length of the linear kernel array.
 * \param nAnchor X offset of the kernel origin frame of reference relative to the
 *                 source pixel.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterRow32f_16u_C4R(const Npp16u * pSrc, Npp32s nSrcStep, Npp16u * pDst, Npp32s nDstStep, NppiSize oROI, 
                         const Npp32f * pKernel, Npp32s nMaskSize, Npp32s nAnchor);

/**
 * 16-bit unsigned four-channel 1D row colwolution ignoring alpha-channel.
 * 
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *                Coefficients are expected to be stored in reverse order.
 * \param nMaskSize Length of the linear kernel array.
 * \param nAnchor X offset of the kernel origin frame of reference relative to the
 *                 source pixel.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterRow32f_16u_AC4R(const Npp16u * pSrc, Npp32s nSrcStep, Npp16u * pDst, Npp32s nDstStep, NppiSize oROI, 
                          const Npp32f * pKernel, Npp32s nMaskSize, Npp32s nAnchor);

/**
 * 16-bit single-channel 1D row colwolution.
 * 
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *                Coefficients are expected to be stored in reverse order.
 * \param nMaskSize Length of the linear kernel array.
 * \param nAnchor X offset of the kernel origin frame of reference relative to the
 *                 source pixel.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterRow32f_16s_C1R(const Npp16s * pSrc, Npp32s nSrcStep, Npp16s * pDst, Npp32s nDstStep, NppiSize oROI, 
                         const Npp32f * pKernel, Npp32s nMaskSize, Npp32s nAnchor);

/**
 * 16-bit three-channel 1D row colwolution.
 * 
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *                Coefficients are expected to be stored in reverse order.
 * \param nMaskSize Length of the linear kernel array.
 * \param nAnchor X offset of the kernel origin frame of reference relative to the
 *                 source pixel.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterRow32f_16s_C3R(const Npp16s * pSrc, Npp32s nSrcStep, Npp16s * pDst, Npp32s nDstStep, NppiSize oROI, 
                         const Npp32f * pKernel, Npp32s nMaskSize, Npp32s nAnchor);

/**
 * 16-bit four-channel 1D row colwolution.
 * 
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *                Coefficients are expected to be stored in reverse order.
 * \param nMaskSize Length of the linear kernel array.
 * \param nAnchor X offset of the kernel origin frame of reference relative to the
 *                 source pixel.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterRow32f_16s_C4R(const Npp16s * pSrc, Npp32s nSrcStep, Npp16s * pDst, Npp32s nDstStep, NppiSize oROI, 
                         const Npp32f * pKernel, Npp32s nMaskSize, Npp32s nAnchor);

/**
 * 16-bit four-channel 1D row colwolution ignoring alpha-channel.
 * 
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *                Coefficients are expected to be stored in reverse order.
 * \param nMaskSize Length of the linear kernel array.
 * \param nAnchor X offset of the kernel origin frame of reference relative to the
 *                 source pixel.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterRow32f_16s_AC4R(const Npp16s * pSrc, Npp32s nSrcStep, Npp16s * pDst, Npp32s nDstStep, NppiSize oROI, 
                          const Npp32f * pKernel, Npp32s nMaskSize, Npp32s nAnchor);

/** @} FilterRow32f */

/** @name FilterRowBorder32f
 * General purpose 1D row colwolution filter using floating-point weights with border control.
 *
 * Pixels under the mask are multiplied by the respective weights in the mask
 * and the results are summed.  If any portion of the mask overlaps the source
 * image boundary the requested border type operation is applied to all mask pixels
 * which fall outside of the source image.
 *
 * Lwrrently only the NPP_BORDER_REPLICATE border type operation is supported.
 *
 * @{
 *
 */


/**
 * Single channel 8-bit unsigned 1D row colwolution filter with border control.
 * 
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *        Coeffcients are expected to be stored in reverse order.
 * \param nMaskSize Width of the kernel.
 * \param nAnchor X offset of the kernel origin frame of reference
 *        relative to the source pixel.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiFilterRowBorder32f_8u_C1R(const Npp8u * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u * pDst, int nDstStep, NppiSize oSizeROI, 
                              const Npp32f * pKernel, Npp32s nMaskSize, Npp32s nAnchor, NppiBorderType eBorderType);
        
/**
 * Three channel 8-bit unsigned 1D row colwolution filter with border control.
 * 
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *        Coeffcients are expected to be stored in reverse order.
 * \param nMaskSize Width of the kernel.
 * \param nAnchor X offset of the kernel origin frame of reference
 *        relative to the source pixel.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiFilterRowBorder32f_8u_C3R(const Npp8u * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u * pDst, int nDstStep, NppiSize oSizeROI, 
                              const Npp32f * pKernel, Npp32s nMaskSize, Npp32s nAnchor, NppiBorderType eBorderType);
        
/**
 * Four channel 8-bit unsigned 1D row colwolution filter with border control.
 * 
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *        Coeffcients are expected to be stored in reverse order.
 * \param nMaskSize Width of the kernel.
 * \param nAnchor X offset of the kernel origin frame of reference
 *        relative to the source pixel.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiFilterRowBorder32f_8u_C4R(const Npp8u * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u * pDst, int nDstStep, NppiSize oSizeROI, 
                              const Npp32f * pKernel, Npp32s nMaskSize, Npp32s nAnchor, NppiBorderType eBorderType);
        
/**
 * Four channel 8-bit unsigned 1D row colwolution filter with border control, ignorint alpha channel.
 * 
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *        Coeffcients are expected to be stored in reverse order.
 * \param nMaskSize Width of the kernel.
 * \param nAnchor X offset of the kernel origin frame of reference
 *        relative to the source pixel.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiFilterRowBorder32f_8u_AC4R(const Npp8u * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u * pDst, int nDstStep, NppiSize oSizeROI, 
                               const Npp32f * pKernel, Npp32s nMaskSize, Npp32s nAnchor, NppiBorderType eBorderType);

/**
 * Single channel 16-bit unsigned 1D row colwolution filter with border control.
 * 
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *        Coeffcients are expected to be stored in reverse order.
 * \param nMaskSize Width of the kernel.
 * \param nAnchor X offset of the kernel origin frame of reference
 *        relative to the source pixel.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiFilterRowBorder32f_16u_C1R(const Npp16u * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16u * pDst, int nDstStep, NppiSize oSizeROI, 
                               const Npp32f * pKernel, Npp32s nMaskSize, Npp32s nAnchor, NppiBorderType eBorderType);

/**
 * Three channel 16-bit unsigned 1D row colwolution filter with border control.
 * 
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *        Coeffcients are expected to be stored in reverse order.
 * \param nMaskSize Width of the kernel.
 * \param nAnchor X offset of the kernel origin frame of reference
 *        relative to the source pixel.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiFilterRowBorder32f_16u_C3R(const Npp16u * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16u * pDst, int nDstStep, NppiSize oSizeROI, 
                               const Npp32f * pKernel, Npp32s nMaskSize, Npp32s nAnchor, NppiBorderType eBorderType);

/**
 * Four channel 16-bit unsigned 1D row colwolution filter with border control.
 * 
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *        Coeffcients are expected to be stored in reverse order.
 * \param nMaskSize Width of the kernel.
 * \param nAnchor X offset of the kernel origin frame of reference
 *        relative to the source pixel.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiFilterRowBorder32f_16u_C4R(const Npp16u * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16u * pDst, int nDstStep, NppiSize oSizeROI, 
                               const Npp32f * pKernel, Npp32s nMaskSize, Npp32s nAnchor, NppiBorderType eBorderType);

/**
 * Four channel 16-bit unsigned 1D row colwolution filter with border control, ignoring alpha channel.
 * 
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *        Coeffcients are expected to be stored in reverse order.
 * \param nMaskSize Width of the kernel.
 * \param nAnchor X offset of the kernel origin frame of reference
 *        relative to the source pixel.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiFilterRowBorder32f_16u_AC4R(const Npp16u * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16u * pDst, int nDstStep, NppiSize oSizeROI, 
                                const Npp32f * pKernel, Npp32s nMaskSize, Npp32s nAnchor, NppiBorderType eBorderType);

/**
 * Single channel 16-bit 1D row colwolution filter with border control.
 * 
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *        Coeffcients are expected to be stored in reverse order.
 * \param nMaskSize Width of the kernel.
 * \param nAnchor X offset of the kernel origin frame of reference
 *        relative to the source pixel.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiFilterRowBorder32f_16s_C1R(const Npp16s * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16s * pDst, int nDstStep, NppiSize oSizeROI, 
                               const Npp32f * pKernel, Npp32s nMaskSize, Npp32s nAnchor, NppiBorderType eBorderType);

/**
 * Three channel 16-bit 1D row colwolution filter with border control.
 * 
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *        Coeffcients are expected to be stored in reverse order.
 * \param nMaskSize Width of the kernel.
 * \param nAnchor X offset of the kernel origin frame of reference
 *        relative to the source pixel.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiFilterRowBorder32f_16s_C3R(const Npp16s * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16s * pDst, int nDstStep, NppiSize oSizeROI, 
                               const Npp32f * pKernel, Npp32s nMaskSize, Npp32s nAnchor, NppiBorderType eBorderType);

/**
 * Four channel 16-bit 1D row colwolution filter with border control.
 * 
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *        Coeffcients are expected to be stored in reverse order.
 * \param nMaskSize Width of the kernel.
 * \param nAnchor X offset of the kernel origin frame of reference
 *        relative to the source pixel.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiFilterRowBorder32f_16s_C4R(const Npp16s * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16s * pDst, int nDstStep, NppiSize oSizeROI, 
                               const Npp32f * pKernel, Npp32s nMaskSize, Npp32s nAnchor, NppiBorderType eBorderType);

/**
 * Four channel 16-bit 1D row colwolution filter with border control, ignoring alpha channel.
 * 
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *        Coeffcients are expected to be stored in reverse order.
 * \param nMaskSize Width of the kernel.
 * \param nAnchor X offset of the kernel origin frame of reference
 *        relative to the source pixel.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiFilterRowBorder32f_16s_AC4R(const Npp16s * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16s * pDst, int nDstStep, NppiSize oSizeROI, 
                                const Npp32f * pKernel, Npp32s nMaskSize, Npp32s nAnchor, NppiBorderType eBorderType);

/** @} FilterRowBorder32f */

/** @} image_1D_linear_filter */

/** @defgroup image_1D_window_sum 1D Window Sum
 *
 * @{
 *
 */

/** @name 1D Window Sum
 *  1D mask Window Sum for 8 and 16 bit images.
 *
 * @{
 *
 */

/**
 * One channel 8-bit unsigned 1D (column) sum to 32f.
 *
 * Apply Column Window Summation filter over a 1D mask region around each
 * source pixel for 1-channel 8 bit/pixel input images with 32-bit floating point
 * output.  
 * Result 32-bit floating point pixel is equal to the sum of the corresponding and
 * neighboring column pixel values in a mask region of the source image defined by
 * nMaskSize and nAnchor. 
 *
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oROI \ref roi_specification.
 * \param nMaskSize Length of the linear kernel array.
 * \param nAnchor Y offset of the kernel origin frame of reference relative to the
 *        source pixel.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus nppiSumWindowColumn_8u32f_C1R(const Npp8u  * pSrc, Npp32s nSrcStep, 
                                              Npp32f * pDst, Npp32s nDstStep, NppiSize oROI, 
                                        Npp32s nMaskSize, Npp32s nAnchor);


/**
 * Three channel 8-bit unsigned 1D (column) sum to 32f.
 *
 * Apply Column Window Summation filter over a 1D mask region around each
 * source pixel for 3-channel 8 bit/pixel input images with 32-bit floating point
 * output.  
 * Result 32-bit floating point pixel is equal to the sum of the corresponding and
 * neighboring column pixel values in a mask region of the source image defined by
 * nMaskSize and nAnchor. 
 *
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oROI \ref roi_specification.
 * \param nMaskSize Length of the linear kernel array.
 * \param nAnchor Y offset of the kernel origin frame of reference relative to the
 *        source pixel.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus nppiSumWindowColumn_8u32f_C3R(const Npp8u  * pSrc, Npp32s nSrcStep, 
                                              Npp32f * pDst, Npp32s nDstStep, NppiSize oROI, 
                                        Npp32s nMaskSize, Npp32s nAnchor);


/**
 * Four channel 8-bit unsigned 1D (column) sum to 32f.
 *
 * Apply Column Window Summation filter over a 1D mask region around each
 * source pixel for 4-channel 8 bit/pixel input images with 32-bit floating point
 * output.  
 * Result 32-bit floating point pixel is equal to the sum of the corresponding and
 * neighboring column pixel values in a mask region of the source image defined by
 * nMaskSize and nAnchor. 
 *
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oROI \ref roi_specification.
 * \param nMaskSize Length of the linear kernel array.
 * \param nAnchor Y offset of the kernel origin frame of reference relative to the
 *        source pixel.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus nppiSumWindowColumn_8u32f_C4R(const Npp8u  * pSrc, Npp32s nSrcStep, 
                                              Npp32f * pDst, Npp32s nDstStep, NppiSize oROI, 
                                        Npp32s nMaskSize, Npp32s nAnchor);

/**
 * One channel 16-bit unsigned 1D (column) sum to 32f.
 *
 * Apply Column Window Summation filter over a 1D mask region around each
 * source pixel for 1-channel 16 bit/pixel input images with 32-bit floating point
 * output.  
 * Result 32-bit floating point pixel is equal to the sum of the corresponding and
 * neighboring column pixel values in a mask region of the source image defined by
 * nMaskSize and nAnchor. 
 *
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oROI \ref roi_specification.
 * \param nMaskSize Length of the linear kernel array.
 * \param nAnchor Y offset of the kernel origin frame of reference relative to the
 *        source pixel.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus nppiSumWindowColumn_16u32f_C1R(const Npp16u * pSrc, Npp32s nSrcStep, 
                                               Npp32f * pDst, Npp32s nDstStep, NppiSize oROI, 
                                         Npp32s nMaskSize, Npp32s nAnchor);

/**
 * Three channel 16-bit unsigned 1D (column) sum to 32f.
 *
 * Apply Column Window Summation filter over a 1D mask region around each
 * source pixel for 3-channel 16 bit/pixel input images with 32-bit floating point
 * output.  
 * Result 32-bit floating point pixel is equal to the sum of the corresponding and
 * neighboring column pixel values in a mask region of the source image defined by
 * nMaskSize and nAnchor. 
 *
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oROI \ref roi_specification.
 * \param nMaskSize Length of the linear kernel array.
 * \param nAnchor Y offset of the kernel origin frame of reference relative to the
 *        source pixel.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus nppiSumWindowColumn_16u32f_C3R(const Npp16u * pSrc, Npp32s nSrcStep, 
                                               Npp32f * pDst, Npp32s nDstStep, NppiSize oROI, 
                                         Npp32s nMaskSize, Npp32s nAnchor);

/**
 * Four channel 16-bit unsigned 1D (column) sum to 32f.
 *
 * Apply Column Window Summation filter over a 1D mask region around each
 * source pixel for 4-channel 16 bit/pixel input images with 32-bit floating point
 * output.  
 * Result 32-bit floating point pixel is equal to the sum of the corresponding and
 * neighboring column pixel values in a mask region of the source image defined by
 * nMaskSize and nAnchor. 
 *
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oROI \ref roi_specification.
 * \param nMaskSize Length of the linear kernel array.
 * \param nAnchor Y offset of the kernel origin frame of reference relative to the
 *        source pixel.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus nppiSumWindowColumn_16u32f_C4R(const Npp16u * pSrc, Npp32s nSrcStep, 
                                               Npp32f * pDst, Npp32s nDstStep, NppiSize oROI, 
                                         Npp32s nMaskSize, Npp32s nAnchor);

/**
 * One channel 16-bit signed 1D (column) sum to 32f.
 *
 * Apply Column Window Summation filter over a 1D mask region around each
 * source pixel for 1-channel 16 bit/pixel input images with 32-bit floating point
 * output.  
 * Result 32-bit floating point pixel is equal to the sum of the corresponding and
 * neighboring column pixel values in a mask region of the source image defined by
 * nMaskSize and nAnchor. 
 *
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oROI \ref roi_specification.
 * \param nMaskSize Length of the linear kernel array.
 * \param nAnchor Y offset of the kernel origin frame of reference relative to the
 *        source pixel.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus nppiSumWindowColumn_16s32f_C1R(const Npp16s * pSrc, Npp32s nSrcStep, 
                                               Npp32f * pDst, Npp32s nDstStep, NppiSize oROI, 
                                         Npp32s nMaskSize, Npp32s nAnchor);

/**
 * Three channel 16-bit signed 1D (column) sum to 32f.
 *
 * Apply Column Window Summation filter over a 1D mask region around each
 * source pixel for 1-channel 16 bit/pixel input images with 32-bit floating point
 * output.  
 * Result 32-bit floating point pixel is equal to the sum of the corresponding and
 * neighboring column pixel values in a mask region of the source image defined by
 * nMaskSize and nAnchor. 
 *
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oROI \ref roi_specification.
 * \param nMaskSize Length of the linear kernel array.
 * \param nAnchor Y offset of the kernel origin frame of reference relative to the
 *        source pixel.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus nppiSumWindowColumn_16s32f_C3R(const Npp16s * pSrc, Npp32s nSrcStep, 
                                               Npp32f * pDst, Npp32s nDstStep, NppiSize oROI, 
                                         Npp32s nMaskSize, Npp32s nAnchor);

/**
 * Four channel 16-bit signed 1D (column) sum to 32f.
 *
 * Apply Column Window Summation filter over a 1D mask region around each
 * source pixel for 4-channel 16 bit/pixel input images with 32-bit floating point
 * output.  
 * Result 32-bit floating point pixel is equal to the sum of the corresponding and
 * neighboring column pixel values in a mask region of the source image defined by
 * nMaskSize and nAnchor. 
 *
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oROI \ref roi_specification.
 * \param nMaskSize Length of the linear kernel array.
 * \param nAnchor Y offset of the kernel origin frame of reference relative to the
 *        source pixel.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus nppiSumWindowColumn_16s32f_C4R(const Npp16s * pSrc, Npp32s nSrcStep, 
                                               Npp32f * pDst, Npp32s nDstStep, NppiSize oROI, 
                                         Npp32s nMaskSize, Npp32s nAnchor);

/**
 * One channel 8-bit unsigned 1D (row) sum to 32f.
 *
 * Apply Row Window Summation filter over a 1D mask region around each source
 * pixel for 1-channel 8-bit pixel input images with 32-bit floating point output.  
 * Result 32-bit floating point pixel is equal to the sum of the corresponding and
 * neighboring row pixel values in a mask region of the source image defined
 * by iKernelDim and iAnchorX. 
 *
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oROI \ref roi_specification.
 * \param nMaskSize Length of the linear kernel array.
 * \param nAnchor X offset of the kernel origin frame of reference relative to the
 *        source pixel.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSumWindowRow_8u32f_C1R(const Npp8u  * pSrc, Npp32s nSrcStep, 
                                 Npp32f * pDst, Npp32s nDstStep, 
                           NppiSize oROI, Npp32s nMaskSize, Npp32s nAnchor);

/**
 * Three channel 8-bit unsigned 1D (row) sum to 32f.
 *
 * Apply Row Window Summation filter over a 1D mask region around each source
 * pixel for 3-channel 8-bit pixel input images with 32-bit floating point output.  
 * Result 32-bit floating point pixel is equal to the sum of the corresponding and
 * neighboring row pixel values in a mask region of the source image defined
 * by iKernelDim and iAnchorX. 
 *
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oROI \ref roi_specification.
 * \param nMaskSize Length of the linear kernel array.
 * \param nAnchor X offset of the kernel origin frame of reference relative to the
 *        source pixel.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSumWindowRow_8u32f_C3R(const Npp8u  * pSrc, Npp32s nSrcStep, 
                                 Npp32f * pDst, Npp32s nDstStep, 
                           NppiSize oROI, Npp32s nMaskSize, Npp32s nAnchor);

/**
 * Four channel 8-bit unsigned 1D (row) sum to 32f.
 *
 * Apply Row Window Summation filter over a 1D mask region around each source
 * pixel for 4-channel 8-bit pixel input images with 32-bit floating point output.  
 * Result 32-bit floating point pixel is equal to the sum of the corresponding and
 * neighboring row pixel values in a mask region of the source image defined
 * by iKernelDim and iAnchorX. 
 *
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oROI \ref roi_specification.
 * \param nMaskSize Length of the linear kernel array.
 * \param nAnchor X offset of the kernel origin frame of reference relative to the
 *        source pixel.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSumWindowRow_8u32f_C4R(const Npp8u  * pSrc, Npp32s nSrcStep, 
                                 Npp32f * pDst, Npp32s nDstStep, 
                           NppiSize oROI, Npp32s nMaskSize, Npp32s nAnchor);

/**
 * One channel 16-bit unsigned 1D (row) sum to 32f.
 *
 * Apply Row Window Summation filter over a 1D mask region around each source
 * pixel for 1-channel 16-bit pixel input images with 32-bit floating point output.  
 * Result 32-bit floating point pixel is equal to the sum of the corresponding and
 * neighboring row pixel values in a mask region of the source image defined
 * by iKernelDim and iAnchorX. 
 *
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oROI \ref roi_specification.
 * \param nMaskSize Length of the linear kernel array.
 * \param nAnchor X offset of the kernel origin frame of reference relative to the
 *        source pixel.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSumWindowRow_16u32f_C1R(const Npp16u * pSrc, Npp32s nSrcStep, 
                                  Npp32f * pDst, Npp32s nDstStep, 
                            NppiSize oROI, Npp32s nMaskSize, Npp32s nAnchor);

/**
 * Three channel 16-bit unsigned 1D (row) sum to 32f.
 *
 * Apply Row Window Summation filter over a 1D mask region around each source
 * pixel for 3-channel 16-bit pixel input images with 32-bit floating point output.  
 * Result 32-bit floating point pixel is equal to the sum of the corresponding and
 * neighboring row pixel values in a mask region of the source image defined
 * by iKernelDim and iAnchorX. 
 *
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oROI \ref roi_specification.
 * \param nMaskSize Length of the linear kernel array.
 * \param nAnchor X offset of the kernel origin frame of reference relative to the
 *        source pixel.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSumWindowRow_16u32f_C3R(const Npp16u * pSrc, Npp32s nSrcStep, 
                                  Npp32f * pDst, Npp32s nDstStep, 
                            NppiSize oROI, Npp32s nMaskSize, Npp32s nAnchor);

/**
 * Four channel 16-bit unsigned 1D (row) sum to 32f.
 *
 * Apply Row Window Summation filter over a 1D mask region around each source
 * pixel for 4-channel 16-bit pixel input images with 32-bit floating point output.  
 * Result 32-bit floating point pixel is equal to the sum of the corresponding and
 * neighboring row pixel values in a mask region of the source image defined
 * by iKernelDim and iAnchorX. 
 *
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oROI \ref roi_specification.
 * \param nMaskSize Length of the linear kernel array.
 * \param nAnchor X offset of the kernel origin frame of reference relative to the
 *        source pixel.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSumWindowRow_16u32f_C4R(const Npp16u * pSrc, Npp32s nSrcStep, 
                                  Npp32f * pDst, Npp32s nDstStep, 
                            NppiSize oROI, Npp32s nMaskSize, Npp32s nAnchor);

/**
 * One channel 16-bit signed 1D (row) sum to 32f.
 *
 * Apply Row Window Summation filter over a 1D mask region around each source
 * pixel for 1-channel 16-bit pixel input images with 32-bit floating point output.  
 * Result 32-bit floating point pixel is equal to the sum of the corresponding and
 * neighboring row pixel values in a mask region of the source image defined
 * by iKernelDim and iAnchorX. 
 *
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oROI \ref roi_specification.
 * \param nMaskSize Length of the linear kernel array.
 * \param nAnchor X offset of the kernel origin frame of reference relative to the
 *        source pixel.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSumWindowRow_16s32f_C1R(const Npp16s * pSrc, Npp32s nSrcStep, 
                                  Npp32f * pDst, Npp32s nDstStep, 
                            NppiSize oROI, Npp32s nMaskSize, Npp32s nAnchor);
/**
 * Three channel 16-bit signed 1D (row) sum to 32f.
 *
 * Apply Row Window Summation filter over a 1D mask region around each source
 * pixel for 3-channel 16-bit pixel input images with 32-bit floating point output.  
 * Result 32-bit floating point pixel is equal to the sum of the corresponding and
 * neighboring row pixel values in a mask region of the source image defined
 * by iKernelDim and iAnchorX. 
 *
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oROI \ref roi_specification.
 * \param nMaskSize Length of the linear kernel array.
 * \param nAnchor X offset of the kernel origin frame of reference relative to the
 *        source pixel.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSumWindowRow_16s32f_C3R(const Npp16s * pSrc, Npp32s nSrcStep, 
                                  Npp32f * pDst, Npp32s nDstStep, 
                            NppiSize oROI, Npp32s nMaskSize, Npp32s nAnchor);

/**
 * Four channel 16-bit signed 1D (row) sum to 32f.
 *
 * Apply Row Window Summation filter over a 1D mask region around each source
 * pixel for 4-channel 16-bit pixel input images with 32-bit floating point output.  
 * Result 32-bit floating point pixel is equal to the sum of the corresponding and
 * neighboring row pixel values in a mask region of the source image defined
 * by iKernelDim and iAnchorX. 
 *
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oROI \ref roi_specification.
 * \param nMaskSize Length of the linear kernel array.
 * \param nAnchor X offset of the kernel origin frame of reference relative to the
 *        source pixel.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSumWindowRow_16s32f_C4R(const Npp16s * pSrc, Npp32s nSrcStep, 
                                  Npp32f * pDst, Npp32s nDstStep, 
                            NppiSize oROI, Npp32s nMaskSize, Npp32s nAnchor);
/** @} */

/** @} image_1D_window_sum */

/** @defgroup image_1D_window_sum_border 1D Window Sum with Border Control
 *
 * @{
 *
 */

/** @name 1D Window Sum Border
 * 1D mask Window Sum for 8 and 16 bit images with border control. 
 * If any portion of the mask overlaps the source image boundary the requested border type operation 
 * is applied to all mask pixels which fall outside of the source image.
 *
 * Lwrrently only the NPP_BORDER_REPLICATE border type operation is supported.
 *
 * @{
 *
 */

/**
 * One channel 8-bit unsigned 1D (column) sum to 32f with border control.
 *
 * Apply Column Window Summation filter over a 1D mask region around each
 * source pixel for 1-channel 8 bit/pixel input images with 32-bit floating point
 * output.  
 * Result 32-bit floating point pixel is equal to the sum of the corresponding and
 * neighboring column pixel values in a mask region of the source image defined by
 * nMaskSize and nAnchor. 
 *
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oROI \ref roi_specification.
 * \param nMaskSize Length of the linear kernel array.
 * \param nAnchor Y offset of the kernel origin frame of reference relative to the
 *        source pixel.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus nppiSumWindowColumnBorder_8u32f_C1R(const Npp8u  * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, 
                                                    Npp32f * pDst, Npp32s nDstStep, NppiSize oROI, 
                                                    Npp32s nMaskSize, Npp32s nAnchor, NppiBorderType eBorderType);


/**
 * Three channel 8-bit unsigned 1D (column) sum to 32f with border control.
 *
 * Apply Column Window Summation filter over a 1D mask region around each
 * source pixel for 3-channel 8 bit/pixel input images with 32-bit floating point
 * output.  
 * Result 32-bit floating point pixel is equal to the sum of the corresponding and
 * neighboring column pixel values in a mask region of the source image defined by
 * nMaskSize and nAnchor. 
 *
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oROI \ref roi_specification.
 * \param nMaskSize Length of the linear kernel array.
 * \param nAnchor Y offset of the kernel origin frame of reference relative to the
 *        source pixel.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus nppiSumWindowColumnBorder_8u32f_C3R(const Npp8u  * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, 
                                                    Npp32f * pDst, Npp32s nDstStep, NppiSize oROI, 
                                                    Npp32s nMaskSize, Npp32s nAnchor, NppiBorderType eBorderType);


/**
 * Four channel 8-bit unsigned 1D (column) sum to 32f with border control.
 *
 * Apply Column Window Summation filter over a 1D mask region around each
 * source pixel for 4-channel 8 bit/pixel input images with 32-bit floating point
 * output.  
 * Result 32-bit floating point pixel is equal to the sum of the corresponding and
 * neighboring column pixel values in a mask region of the source image defined by
 * nMaskSize and nAnchor. 
 *
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oROI \ref roi_specification.
 * \param nMaskSize Length of the linear kernel array.
 * \param nAnchor Y offset of the kernel origin frame of reference relative to the
 *        source pixel.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus nppiSumWindowColumnBorder_8u32f_C4R(const Npp8u  * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, 
                                                    Npp32f * pDst, Npp32s nDstStep, NppiSize oROI, 
                                                    Npp32s nMaskSize, Npp32s nAnchor, NppiBorderType eBorderType);

/**
 * One channel 16-bit unsigned 1D (column) sum to 32f with border control.
 *
 * Apply Column Window Summation filter over a 1D mask region around each
 * source pixel for 1-channel 16 bit/pixel input images with 32-bit floating point
 * output.  
 * Result 32-bit floating point pixel is equal to the sum of the corresponding and
 * neighboring column pixel values in a mask region of the source image defined by
 * nMaskSize and nAnchor. 
 *
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oROI \ref roi_specification.
 * \param nMaskSize Length of the linear kernel array.
 * \param nAnchor Y offset of the kernel origin frame of reference relative to the
 *        source pixel.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus nppiSumWindowColumnBorder_16u32f_C1R(const Npp16u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, 
                                                     Npp32f * pDst, Npp32s nDstStep, NppiSize oROI, 
                                                     Npp32s nMaskSize, Npp32s nAnchor, NppiBorderType eBorderType);

/**
 * Three channel 16-bit unsigned 1D (column) sum to 32f with border control.
 *
 * Apply Column Window Summation filter over a 1D mask region around each
 * source pixel for 3-channel 16 bit/pixel input images with 32-bit floating point
 * output.  
 * Result 32-bit floating point pixel is equal to the sum of the corresponding and
 * neighboring column pixel values in a mask region of the source image defined by
 * nMaskSize and nAnchor. 
 *
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oROI \ref roi_specification.
 * \param nMaskSize Length of the linear kernel array.
 * \param nAnchor Y offset of the kernel origin frame of reference relative to the
 *        source pixel.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus nppiSumWindowColumnBorder_16u32f_C3R(const Npp16u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, 
                                                     Npp32f * pDst, Npp32s nDstStep, NppiSize oROI, 
                                                     Npp32s nMaskSize, Npp32s nAnchor, NppiBorderType eBorderType);

/**
 * Four channel 16-bit unsigned 1D (column) sum to 32f with border control.
 *
 * Apply Column Window Summation filter over a 1D mask region around each
 * source pixel for 4-channel 16 bit/pixel input images with 32-bit floating point
 * output.  
 * Result 32-bit floating point pixel is equal to the sum of the corresponding and
 * neighboring column pixel values in a mask region of the source image defined by
 * nMaskSize and nAnchor. 
 *
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oROI \ref roi_specification.
 * \param nMaskSize Length of the linear kernel array.
 * \param nAnchor Y offset of the kernel origin frame of reference relative to the
 *        source pixel.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus nppiSumWindowColumnBorder_16u32f_C4R(const Npp16u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, 
                                                     Npp32f * pDst, Npp32s nDstStep, NppiSize oROI, 
                                                     Npp32s nMaskSize, Npp32s nAnchor, NppiBorderType eBorderType);

/**
 * One channel 16-bit signed 1D (column) sum to 32f with border control.
 *
 * Apply Column Window Summation filter over a 1D mask region around each
 * source pixel for 1-channel 16 bit/pixel input images with 32-bit floating point
 * output.  
 * Result 32-bit floating point pixel is equal to the sum of the corresponding and
 * neighboring column pixel values in a mask region of the source image defined by
 * nMaskSize and nAnchor. 
 *
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oROI \ref roi_specification.
 * \param nMaskSize Length of the linear kernel array.
 * \param nAnchor Y offset of the kernel origin frame of reference relative to the
 *        source pixel.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus nppiSumWindowColumnBorder_16s32f_C1R(const Npp16s * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, 
                                                     Npp32f * pDst, Npp32s nDstStep, NppiSize oROI, 
                                                     Npp32s nMaskSize, Npp32s nAnchor, NppiBorderType eBorderType);

/**
 * Three channel 16-bit signed 1D (column) sum to 32f with border control.
 *
 * Apply Column Window Summation filter over a 1D mask region around each
 * source pixel for 1-channel 16 bit/pixel input images with 32-bit floating point
 * output.  
 * Result 32-bit floating point pixel is equal to the sum of the corresponding and
 * neighboring column pixel values in a mask region of the source image defined by
 * nMaskSize and nAnchor. 
 *
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oROI \ref roi_specification.
 * \param nMaskSize Length of the linear kernel array.
 * \param nAnchor Y offset of the kernel origin frame of reference relative to the
 *        source pixel.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus nppiSumWindowColumnBorder_16s32f_C3R(const Npp16s * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, 
                                                     Npp32f * pDst, Npp32s nDstStep, NppiSize oROI, 
                                                     Npp32s nMaskSize, Npp32s nAnchor, NppiBorderType eBorderType);

/**
 * Four channel 16-bit signed 1D (column) sum to 32f with border control.
 *
 * Apply Column Window Summation filter over a 1D mask region around each
 * source pixel for 4-channel 16 bit/pixel input images with 32-bit floating point
 * output.  
 * Result 32-bit floating point pixel is equal to the sum of the corresponding and
 * neighboring column pixel values in a mask region of the source image defined by
 * nMaskSize and nAnchor. 
 *
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oROI \ref roi_specification.
 * \param nMaskSize Length of the linear kernel array.
 * \param nAnchor Y offset of the kernel origin frame of reference relative to the
 *        source pixel.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus nppiSumWindowColumnBorder_16s32f_C4R(const Npp16s * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, 
                                                     Npp32f * pDst, Npp32s nDstStep, NppiSize oROI, 
                                                     Npp32s nMaskSize, Npp32s nAnchor, NppiBorderType eBorderType);

/**
 * One channel 8-bit unsigned 1D (row) sum to 32f with border control.
 *
 * Apply Row Window Summation filter over a 1D mask region around each source
 * pixel for 1-channel 8-bit pixel input images with 32-bit floating point output.  
 * Result 32-bit floating point pixel is equal to the sum of the corresponding and
 * neighboring row pixel values in a mask region of the source image defined
 * by iKernelDim and iAnchorX. 
 *
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oROI \ref roi_specification.
 * \param nMaskSize Length of the linear kernel array.
 * \param nAnchor X offset of the kernel origin frame of reference relative to the
 *        source pixel.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSumWindowRowBorder_8u32f_C1R(const Npp8u  * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, 
                                       Npp32f * pDst, Npp32s nDstStep, 
                                       NppiSize oROI, Npp32s nMaskSize, Npp32s nAnchor, NppiBorderType eBorderType);

/**
 * Three channel 8-bit unsigned 1D (row) sum to 32f with border control.
 *
 * Apply Row Window Summation filter over a 1D mask region around each source
 * pixel for 3-channel 8-bit pixel input images with 32-bit floating point output.  
 * Result 32-bit floating point pixel is equal to the sum of the corresponding and
 * neighboring row pixel values in a mask region of the source image defined
 * by iKernelDim and iAnchorX. 
 *
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oROI \ref roi_specification.
 * \param nMaskSize Length of the linear kernel array.
 * \param nAnchor X offset of the kernel origin frame of reference relative to the
 *        source pixel.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSumWindowRowBorder_8u32f_C3R(const Npp8u  * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, 
                                       Npp32f * pDst, Npp32s nDstStep, 
                                       NppiSize oROI, Npp32s nMaskSize, Npp32s nAnchor, NppiBorderType eBorderType);

/**
 * Four channel 8-bit unsigned 1D (row) sum to 32f with border control.
 *
 * Apply Row Window Summation filter over a 1D mask region around each source
 * pixel for 4-channel 8-bit pixel input images with 32-bit floating point output.  
 * Result 32-bit floating point pixel is equal to the sum of the corresponding and
 * neighboring row pixel values in a mask region of the source image defined
 * by iKernelDim and iAnchorX. 
 *
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oROI \ref roi_specification.
 * \param nMaskSize Length of the linear kernel array.
 * \param nAnchor X offset of the kernel origin frame of reference relative to the
 *        source pixel.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSumWindowRowBorder_8u32f_C4R(const Npp8u  * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, 
                                       Npp32f * pDst, Npp32s nDstStep, 
                                       NppiSize oROI, Npp32s nMaskSize, Npp32s nAnchor, NppiBorderType eBorderType);

/**
 * One channel 16-bit unsigned 1D (row) sum to 32f with border control.
 *
 * Apply Row Window Summation filter over a 1D mask region around each source
 * pixel for 1-channel 16-bit pixel input images with 32-bit floating point output.  
 * Result 32-bit floating point pixel is equal to the sum of the corresponding and
 * neighboring row pixel values in a mask region of the source image defined
 * by iKernelDim and iAnchorX. 
 *
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oROI \ref roi_specification.
 * \param nMaskSize Length of the linear kernel array.
 * \param nAnchor X offset of the kernel origin frame of reference relative to the
 *        source pixel.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSumWindowRowBorder_16u32f_C1R(const Npp16u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, 
                                        Npp32f * pDst, Npp32s nDstStep, 
                                        NppiSize oROI, Npp32s nMaskSize, Npp32s nAnchor, NppiBorderType eBorderType);

/**
 * Three channel 16-bit unsigned 1D (row) sum to 32f with border control.
 *
 * Apply Row Window Summation filter over a 1D mask region around each source
 * pixel for 3-channel 16-bit pixel input images with 32-bit floating point output.  
 * Result 32-bit floating point pixel is equal to the sum of the corresponding and
 * neighboring row pixel values in a mask region of the source image defined
 * by iKernelDim and iAnchorX. 
 *
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oROI \ref roi_specification.
 * \param nMaskSize Length of the linear kernel array.
 * \param nAnchor X offset of the kernel origin frame of reference relative to the
 *        source pixel.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSumWindowRowBorder_16u32f_C3R(const Npp16u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, 
                                        Npp32f * pDst, Npp32s nDstStep, 
                                        NppiSize oROI, Npp32s nMaskSize, Npp32s nAnchor, NppiBorderType eBorderType);

/**
 * Four channel 16-bit unsigned 1D (row) sum to 32f with border control.
 *
 * Apply Row Window Summation filter over a 1D mask region around each source
 * pixel for 4-channel 16-bit pixel input images with 32-bit floating point output.  
 * Result 32-bit floating point pixel is equal to the sum of the corresponding and
 * neighboring row pixel values in a mask region of the source image defined
 * by iKernelDim and iAnchorX. 
 *
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oROI \ref roi_specification.
 * \param nMaskSize Length of the linear kernel array.
 * \param nAnchor X offset of the kernel origin frame of reference relative to the
 *        source pixel.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSumWindowRowBorder_16u32f_C4R(const Npp16u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, 
                                        Npp32f * pDst, Npp32s nDstStep, 
                                        NppiSize oROI, Npp32s nMaskSize, Npp32s nAnchor, NppiBorderType eBorderType);

/**
 * One channel 16-bit signed 1D (row) sum to 32f with border control.
 *
 * Apply Row Window Summation filter over a 1D mask region around each source
 * pixel for 1-channel 16-bit pixel input images with 32-bit floating point output.  
 * Result 32-bit floating point pixel is equal to the sum of the corresponding and
 * neighboring row pixel values in a mask region of the source image defined
 * by iKernelDim and iAnchorX. 
 *
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oROI \ref roi_specification.
 * \param nMaskSize Length of the linear kernel array.
 * \param nAnchor X offset of the kernel origin frame of reference relative to the
 *        source pixel.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSumWindowRowBorder_16s32f_C1R(const Npp16s * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, 
                                        Npp32f * pDst, Npp32s nDstStep, 
                                        NppiSize oROI, Npp32s nMaskSize, Npp32s nAnchor, NppiBorderType eBorderType);
/**
 * Three channel 16-bit signed 1D (row) sum to 32f with border control.
 *
 * Apply Row Window Summation filter over a 1D mask region around each source
 * pixel for 3-channel 16-bit pixel input images with 32-bit floating point output.  
 * Result 32-bit floating point pixel is equal to the sum of the corresponding and
 * neighboring row pixel values in a mask region of the source image defined
 * by iKernelDim and iAnchorX. 
 *
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oROI \ref roi_specification.
 * \param nMaskSize Length of the linear kernel array.
 * \param nAnchor X offset of the kernel origin frame of reference relative to the
 *        source pixel.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSumWindowRowBorder_16s32f_C3R(const Npp16s * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, 
                                        Npp32f * pDst, Npp32s nDstStep, 
                                        NppiSize oROI, Npp32s nMaskSize, Npp32s nAnchor, NppiBorderType eBorderType);

/**
 * Four channel 16-bit signed 1D (row) sum to 32f with border control.
 *
 * Apply Row Window Summation filter over a 1D mask region around each source
 * pixel for 4-channel 16-bit pixel input images with 32-bit floating point output.  
 * Result 32-bit floating point pixel is equal to the sum of the corresponding and
 * neighboring row pixel values in a mask region of the source image defined
 * by iKernelDim and iAnchorX. 
 *
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oROI \ref roi_specification.
 * \param nMaskSize Length of the linear kernel array.
 * \param nAnchor X offset of the kernel origin frame of reference relative to the
 *        source pixel.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSumWindowRowBorder_16s32f_C4R(const Npp16s * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, 
                                        Npp32f * pDst, Npp32s nDstStep, 
                                        NppiSize oROI, Npp32s nMaskSize, Npp32s nAnchor, NppiBorderType eBorderType);
/** @} */

/** @} image_1D_window_sum_border */

/** @defgroup image_colwolution Colwolution
 *
 * @{
 *
 */

/** @name Filter
 * General purpose 2D colwolution filter.
 *
 * Pixels under the mask are multiplied by the respective weights in the mask
 * and the results are summed. Before writing the result pixel the sum is scaled
 * back via division by nDivisor.
 *
 * @{
 *
 */

/**
 * Single channel 8-bit unsigned colwolution filter.
 * 
 *
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *        Coeffcients are expected to be stored in reverse order.
 * \param oKernelSize Width and Height of the rectangular kernel.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference
 *        relative to the source pixel.
 * \param nDivisor The factor by which the colwolved summation from the Filter
 *        operation should be divided.  If equal to the sum of coefficients,
 *        this will keep the maximum result value within full scale.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilter_8u_C1R(const Npp8u * pSrc, Npp32s nSrcStep, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                  const Npp32s * pKernel, NppiSize oKernelSize, NppiPoint oAnchor, Npp32s nDivisor);
  
/**
 * Three channel 8-bit unsigned colwolution filter.
 * 
 *
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *        Coeffcients are expected to be stored in reverse order.
 * \param oKernelSize Width and Height of the rectangular kernel.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference
 *        relative to the source pixel.
 * \param nDivisor The factor by which the colwolved summation from the Filter
 *        operation should be divided.  If equal to the sum of coefficients,
 *        this will keep the maximum result value within full scale.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilter_8u_C3R(const Npp8u * pSrc, Npp32s nSrcStep, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                  const Npp32s * pKernel, NppiSize oKernelSize, NppiPoint oAnchor, Npp32s nDivisor);
                 
/**
 * Four channel channel 8-bit unsigned colwolution filter.
 * 
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *        Coeffcients are expected to be stored in reverse order.
 * \param oKernelSize Width and Height of the rectangular kernel.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference
 *        relative to the source pixel.
 * \param nDivisor The factor by which the colwolved summation from the Filter
 *        operation should be divided.  If equal to the sum of coefficients,
 *        this will keep the maximum result value within full scale.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilter_8u_C4R(const Npp8u * pSrc, Npp32s nSrcStep, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                  const Npp32s * pKernel, NppiSize oKernelSize, NppiPoint oAnchor, Npp32s nDivisor);

/**
 * Four channel 8-bit unsigned colwolution filter, ignoring alpha channel.
 * 
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *        Coeffcients are expected to be stored in reverse order.
 * \param oKernelSize Width and Height of the rectangular kernel.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference
 *        relative to the source pixel.
 * \param nDivisor The factor by which the colwolved summation from the Filter
 *        operation should be divided.  If equal to the sum of coefficients,
 *        this will keep the maximum result value within full scale.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilter_8u_AC4R(const Npp8u * pSrc, Npp32s nSrcStep, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                   const Npp32s * pKernel, NppiSize oKernelSize, NppiPoint oAnchor, Npp32s nDivisor);

/**
 * Single channel 16-bit unsigned colwolution filter.
 * 
 *
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *        Coeffcients are expected to be stored in reverse order.
 * \param oKernelSize Width and Height of the rectangular kernel.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference
 *        relative to the source pixel.
 * \param nDivisor The factor by which the colwolved summation from the Filter
 *        operation should be divided.  If equal to the sum of coefficients,
 *        this will keep the maximum result value within full scale.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilter_16u_C1R(const Npp16u * pSrc, Npp32s nSrcStep, Npp16u * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                   const Npp32s * pKernel, NppiSize oKernelSize, NppiPoint oAnchor, Npp32s nDivisor);
  
/**
 * Three channel 16-bit unsigned colwolution filter.
 * 
 *
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *        Coeffcients are expected to be stored in reverse order.
 * \param oKernelSize Width and Height of the rectangular kernel.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference
 *        relative to the source pixel.
 * \param nDivisor The factor by which the colwolved summation from the Filter
 *        operation should be divided.  If equal to the sum of coefficients,
 *        this will keep the maximum result value within full scale.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilter_16u_C3R(const Npp16u * pSrc, Npp32s nSrcStep, Npp16u * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                   const Npp32s * pKernel, NppiSize oKernelSize, NppiPoint oAnchor, Npp32s nDivisor);
                 
/**
 * Four channel channel 16-bit unsigned colwolution filter.
 * 
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *        Coeffcients are expected to be stored in reverse order.
 * \param oKernelSize Width and Height of the rectangular kernel.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference
 *        relative to the source pixel.
 * \param nDivisor The factor by which the colwolved summation from the Filter
 *        operation should be divided.  If equal to the sum of coefficients,
 *        this will keep the maximum result value within full scale.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilter_16u_C4R(const Npp16u * pSrc, Npp32s nSrcStep, Npp16u * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                   const Npp32s * pKernel, NppiSize oKernelSize, NppiPoint oAnchor, Npp32s nDivisor);

/**
 * Four channel 16-bit unsigned colwolution filter, ignoring alpha channel.
 * 
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *        Coeffcients are expected to be stored in reverse order.
 * \param oKernelSize Width and Height of the rectangular kernel.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference
 *        relative to the source pixel.
 * \param nDivisor The factor by which the colwolved summation from the Filter
 *        operation should be divided.  If equal to the sum of coefficients,
 *        this will keep the maximum result value within full scale.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilter_16u_AC4R(const Npp16u * pSrc, Npp32s nSrcStep, Npp16u * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                   const Npp32s * pKernel, NppiSize oKernelSize, NppiPoint oAnchor, Npp32s nDivisor);

/**
 * Single channel 16-bit colwolution filter.
 * 
 *
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *        Coeffcients are expected to be stored in reverse order.
 * \param oKernelSize Width and Height of the rectangular kernel.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference
 *        relative to the source pixel.
 * \param nDivisor The factor by which the colwolved summation from the Filter
 *        operation should be divided.  If equal to the sum of coefficients,
 *        this will keep the maximum result value within full scale.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilter_16s_C1R(const Npp16s * pSrc, Npp32s nSrcStep, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                   const Npp32s * pKernel, NppiSize oKernelSize, NppiPoint oAnchor, Npp32s nDivisor);
  
/**
 * Three channel 16-bit colwolution filter.
 * 
 *
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *        Coeffcients are expected to be stored in reverse order.
 * \param oKernelSize Width and Height of the rectangular kernel.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference
 *        relative to the source pixel.
 * \param nDivisor The factor by which the colwolved summation from the Filter
 *        operation should be divided.  If equal to the sum of coefficients,
 *        this will keep the maximum result value within full scale.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilter_16s_C3R(const Npp16s * pSrc, Npp32s nSrcStep, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                   const Npp32s * pKernel, NppiSize oKernelSize, NppiPoint oAnchor, Npp32s nDivisor);
                 
/**
 * Four channel channel 16-bit colwolution filter.
 * 
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *        Coeffcients are expected to be stored in reverse order.
 * \param oKernelSize Width and Height of the rectangular kernel.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference
 *        relative to the source pixel.
 * \param nDivisor The factor by which the colwolved summation from the Filter
 *        operation should be divided.  If equal to the sum of coefficients,
 *        this will keep the maximum result value within full scale.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilter_16s_C4R(const Npp16s * pSrc, Npp32s nSrcStep, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                   const Npp32s * pKernel, NppiSize oKernelSize, NppiPoint oAnchor, Npp32s nDivisor);

/**
 * Four channel 16-bit colwolution filter, ignoring alpha channel.
 * 
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *        Coeffcients are expected to be stored in reverse order.
 * \param oKernelSize Width and Height of the rectangular kernel.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference
 *        relative to the source pixel.
 * \param nDivisor The factor by which the colwolved summation from the Filter
 *        operation should be divided.  If equal to the sum of coefficients,
 *        this will keep the maximum result value within full scale.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilter_16s_AC4R(const Npp16s * pSrc, Npp32s nSrcStep, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                    const Npp32s * pKernel, NppiSize oKernelSize, NppiPoint oAnchor, Npp32s nDivisor);
 
/**
 * Single channel 32-bit float colwolution filter.
 * 
 *
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *        Coeffcients are expected to be stored in reverse order.
 * \param oKernelSize Width and Height of the rectangular kernel.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference
 *        relative to the source pixel.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiFilter_32f_C1R(const Npp32f * pSrc, Npp32s nSrcStep, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                   const Npp32f * pKernel, NppiSize oKernelSize, NppiPoint oAnchor);

/**
 * Two channel 32-bit float colwolution filter.
 * 
 *
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *        Coeffcients are expected to be stored in reverse order.
 * \param oKernelSize Width and Height of the rectangular kernel.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference
 *        relative to the source pixel.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiFilter_32f_C2R(const Npp32f * pSrc, Npp32s nSrcStep, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                   const Npp32f * pKernel, NppiSize oKernelSize, NppiPoint oAnchor);

/**
 * Three channel 32-bit float colwolution filter.
 * 
 *
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *        Coeffcients are expected to be stored in reverse order.
 * \param oKernelSize Width and Height of the rectangular kernel.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference
 *        relative to the source pixel.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiFilter_32f_C3R(const Npp32f * pSrc, Npp32s nSrcStep, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                   const Npp32f * pKernel, NppiSize oKernelSize, NppiPoint oAnchor);

/**
 * Four channel 32-bit float colwolution filter.
 * 
 *
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *        Coeffcients are expected to be stored in reverse order.
 * \param oKernelSize Width and Height of the rectangular kernel.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference
 *        relative to the source pixel.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiFilter_32f_C4R(const Npp32f * pSrc, Npp32s nSrcStep, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                   const Npp32f * pKernel, NppiSize oKernelSize, NppiPoint oAnchor);

/**
 * Four channel 32-bit float colwolution filter, ignoring alpha channel.
 * 
 *
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *        Coeffcients are expected to be stored in reverse order.
 * \param oKernelSize Width and Height of the rectangular kernel.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference
 *        relative to the source pixel.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiFilter_32f_AC4R(const Npp32f * pSrc, Npp32s nSrcStep, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                    const Npp32f * pKernel, NppiSize oKernelSize, NppiPoint oAnchor);

/**
 * Single channel 64-bit float colwolution filter.
 * 
 *
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *        Coeffcients are expected to be stored in reverse order.
 * \param oKernelSize Width and Height of the rectangular kernel.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference
 *        relative to the source pixel.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiFilter_64f_C1R(const Npp64f * pSrc, Npp32s nSrcStep, Npp64f * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                   const Npp64f * pKernel, NppiSize oKernelSize, NppiPoint oAnchor);


/** @} Filter */

/** @name Filter32f
 * General purpose 2D colwolution filter using floating-point weights.
 *
 * Pixels under the mask are multiplied by the respective weights in the mask
 * and the results are summed. 
 *
 * @{
 *
 */


/**
 * Single channel 8-bit unsigned colwolution filter.
 * 
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *        Coeffcients are expected to be stored in reverse order.
 * \param oKernelSize Width and Height of the rectangular kernel.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference
 *        relative to the source pixel.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiFilter32f_8u_C1R(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst, int nDstStep, NppiSize oSizeROI, 
                     const Npp32f * pKernel, NppiSize oKernelSize, NppiPoint oAnchor);
        
/**
 * Two channel 8-bit unsigned colwolution filter.
 * 
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *        Coeffcients are expected to be stored in reverse order.
 * \param oKernelSize Width and Height of the rectangular kernel.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference
 *        relative to the source pixel.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiFilter32f_8u_C2R(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst, int nDstStep, NppiSize oSizeROI, 
                     const Npp32f * pKernel, NppiSize oKernelSize, NppiPoint oAnchor);
        
/**
 * Three channel 8-bit unsigned colwolution filter.
 * 
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *        Coeffcients are expected to be stored in reverse order.
 * \param oKernelSize Width and Height of the rectangular kernel.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference
 *        relative to the source pixel.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiFilter32f_8u_C3R(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst, int nDstStep, NppiSize oSizeROI, 
                     const Npp32f * pKernel, NppiSize oKernelSize, NppiPoint oAnchor);
        
/**
 * Four channel 8-bit unsigned colwolution filter.
 * 
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *        Coeffcients are expected to be stored in reverse order.
 * \param oKernelSize Width and Height of the rectangular kernel.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference
 *        relative to the source pixel.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiFilter32f_8u_C4R(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst, int nDstStep, NppiSize oSizeROI, 
                     const Npp32f * pKernel, NppiSize oKernelSize, NppiPoint oAnchor);
        
/**
 * Four channel 8-bit unsigned colwolution filter, ignorint alpha channel.
 * 
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *        Coeffcients are expected to be stored in reverse order.
 * \param oKernelSize Width and Height of the rectangular kernel.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference
 *        relative to the source pixel.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiFilter32f_8u_AC4R(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst, int nDstStep, NppiSize oSizeROI, 
                      const Npp32f * pKernel, NppiSize oKernelSize, NppiPoint oAnchor);

/**
 * Single channel 8-bit signed colwolution filter.
 * 
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *        Coeffcients are expected to be stored in reverse order.
 * \param oKernelSize Width and Height of the rectangular kernel.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference
 *        relative to the source pixel.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiFilter32f_8s_C1R(const Npp8s * pSrc, int nSrcStep, Npp8s * pDst, int nDstStep, NppiSize oSizeROI, 
                     const Npp32f * pKernel, NppiSize oKernelSize, NppiPoint oAnchor);

/**
 * Two channel 8-bit signed colwolution filter.
 * 
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *        Coeffcients are expected to be stored in reverse order.
 * \param oKernelSize Width and Height of the rectangular kernel.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference
 *        relative to the source pixel.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiFilter32f_8s_C2R(const Npp8s * pSrc, int nSrcStep, Npp8s * pDst, int nDstStep, NppiSize oSizeROI, 
                     const Npp32f * pKernel, NppiSize oKernelSize, NppiPoint oAnchor);

/**
 * Three channel 8-bit signed colwolution filter.
 * 
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *        Coeffcients are expected to be stored in reverse order.
 * \param oKernelSize Width and Height of the rectangular kernel.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference
 *        relative to the source pixel.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiFilter32f_8s_C3R(const Npp8s * pSrc, int nSrcStep, Npp8s * pDst, int nDstStep, NppiSize oSizeROI, 
                     const Npp32f * pKernel, NppiSize oKernelSize, NppiPoint oAnchor);

/**
 * Four channel 8-bit signed colwolution filter.
 * 
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *        Coeffcients are expected to be stored in reverse order.
 * \param oKernelSize Width and Height of the rectangular kernel.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference
 *        relative to the source pixel.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiFilter32f_8s_C4R(const Npp8s * pSrc, int nSrcStep, Npp8s * pDst, int nDstStep, NppiSize oSizeROI, 
                     const Npp32f * pKernel, NppiSize oKernelSize, NppiPoint oAnchor);

/**
 * Four channel 8-bit signed colwolution filter, ignoring alpha channel.
 * 
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *        Coeffcients are expected to be stored in reverse order.
 * \param oKernelSize Width and Height of the rectangular kernel.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference
 *        relative to the source pixel.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiFilter32f_8s_AC4R(const Npp8s * pSrc, int nSrcStep, Npp8s * pDst, int nDstStep, NppiSize oSizeROI, 
                      const Npp32f * pKernel, NppiSize oKernelSize, NppiPoint oAnchor);

/**
 * Single channel 16-bit unsigned colwolution filter.
 * 
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *        Coeffcients are expected to be stored in reverse order.
 * \param oKernelSize Width and Height of the rectangular kernel.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference
 *        relative to the source pixel.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiFilter32f_16u_C1R(const Npp16u * pSrc, int nSrcStep, Npp16u * pDst, int nDstStep, NppiSize oSizeROI, 
                      const Npp32f * pKernel, NppiSize oKernelSize, NppiPoint oAnchor);

/**
 * Three channel 16-bit unsigned colwolution filter.
 * 
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *        Coeffcients are expected to be stored in reverse order.
 * \param oKernelSize Width and Height of the rectangular kernel.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference
 *        relative to the source pixel.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiFilter32f_16u_C3R(const Npp16u * pSrc, int nSrcStep, Npp16u * pDst, int nDstStep, NppiSize oSizeROI, 
                      const Npp32f * pKernel, NppiSize oKernelSize, NppiPoint oAnchor);

/**
 * Four channel 16-bit unsigned colwolution filter.
 * 
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *        Coeffcients are expected to be stored in reverse order.
 * \param oKernelSize Width and Height of the rectangular kernel.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference
 *        relative to the source pixel.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiFilter32f_16u_C4R(const Npp16u * pSrc, int nSrcStep, Npp16u * pDst, int nDstStep, NppiSize oSizeROI, 
                      const Npp32f * pKernel, NppiSize oKernelSize, NppiPoint oAnchor);

/**
 * Four channel 16-bit unsigned colwolution filter, ignoring alpha channel.
 * 
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *        Coeffcients are expected to be stored in reverse order.
 * \param oKernelSize Width and Height of the rectangular kernel.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference
 *        relative to the source pixel.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiFilter32f_16u_AC4R(const Npp16u * pSrc, int nSrcStep, Npp16u * pDst, int nDstStep, NppiSize oSizeROI, 
                       const Npp32f * pKernel, NppiSize oKernelSize, NppiPoint oAnchor);

/**
 * Single channel 16-bit colwolution filter.
 * 
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *        Coeffcients are expected to be stored in reverse order.
 * \param oKernelSize Width and Height of the rectangular kernel.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference
 *        relative to the source pixel.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiFilter32f_16s_C1R(const Npp16s * pSrc, int nSrcStep, Npp16s * pDst, int nDstStep, NppiSize oSizeROI, 
                      const Npp32f * pKernel, NppiSize oKernelSize, NppiPoint oAnchor);

/**
 * Three channel 16-bit colwolution filter.
 * 
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *        Coeffcients are expected to be stored in reverse order.
 * \param oKernelSize Width and Height of the rectangular kernel.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference
 *        relative to the source pixel.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiFilter32f_16s_C3R(const Npp16s * pSrc, int nSrcStep, Npp16s * pDst, int nDstStep, NppiSize oSizeROI, 
                      const Npp32f * pKernel, NppiSize oKernelSize, NppiPoint oAnchor);

/**
 * Four channel 16-bit colwolution filter.
 * 
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *        Coeffcients are expected to be stored in reverse order.
 * \param oKernelSize Width and Height of the rectangular kernel.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference
 *        relative to the source pixel.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiFilter32f_16s_C4R(const Npp16s * pSrc, int nSrcStep, Npp16s * pDst, int nDstStep, NppiSize oSizeROI, 
                      const Npp32f * pKernel, NppiSize oKernelSize, NppiPoint oAnchor);

/**
 * Four channel 16-bit colwolution filter, ignoring alpha channel.
 * 
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *        Coeffcients are expected to be stored in reverse order.
 * \param oKernelSize Width and Height of the rectangular kernel.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference
 *        relative to the source pixel.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiFilter32f_16s_AC4R(const Npp16s * pSrc, int nSrcStep, Npp16s * pDst, int nDstStep, NppiSize oSizeROI, 
                       const Npp32f * pKernel, NppiSize oKernelSize, NppiPoint oAnchor);


/**
 * Single channel 32-bit colwolution filter.
 * 
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *        Coeffcients are expected to be stored in reverse order.
 * \param oKernelSize Width and Height of the rectangular kernel.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference
 *        relative to the source pixel.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiFilter32f_32s_C1R(const Npp32s * pSrc, int nSrcStep, Npp32s * pDst, int nDstStep, NppiSize oSizeROI, 
                      const Npp32f * pKernel, NppiSize oKernelSize, NppiPoint oAnchor);

/**
 * Three channel 32-bit colwolution filter.
 * 
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *        Coeffcients are expected to be stored in reverse order.
 * \param oKernelSize Width and Height of the rectangular kernel.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference
 *        relative to the source pixel.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiFilter32f_32s_C3R(const Npp32s * pSrc, int nSrcStep, Npp32s * pDst, int nDstStep, NppiSize oSizeROI, 
                      const Npp32f * pKernel, NppiSize oKernelSize, NppiPoint oAnchor);

/**
 * Four channel 32-bit colwolution filter.
 * 
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *        Coeffcients are expected to be stored in reverse order.
 * \param oKernelSize Width and Height of the rectangular kernel.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference
 *        relative to the source pixel.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiFilter32f_32s_C4R(const Npp32s * pSrc, int nSrcStep, Npp32s * pDst, int nDstStep, NppiSize oSizeROI, 
                      const Npp32f * pKernel, NppiSize oKernelSize, NppiPoint oAnchor);

/**
 * Four channel 32-bit colwolution filter, ignoring alpha channel.
 * 
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *        Coeffcients are expected to be stored in reverse order.
 * \param oKernelSize Width and Height of the rectangular kernel.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference
 *        relative to the source pixel.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiFilter32f_32s_AC4R(const Npp32s * pSrc, int nSrcStep, Npp32s * pDst, int nDstStep, NppiSize oSizeROI, 
                       const Npp32f * pKernel, NppiSize oKernelSize, NppiPoint oAnchor);


/**
 * Single channel 8-bit unsigned to 16-bit signed colwolution filter.
 * 
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *        Coeffcients are expected to be stored in reverse order.
 * \param oKernelSize Width and Height of the rectangular kernel.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference
 *        relative to the source pixel.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiFilter32f_8u16s_C1R(const Npp8u * pSrc, int nSrcStep, Npp16s * pDst, int nDstStep, NppiSize oSizeROI, 
                        const Npp32f * pKernel, NppiSize oKernelSize, NppiPoint oAnchor);

/**
 * Three channel 8-bit unsigned to 16-bit signed colwolution filter.
 * 
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *        Coeffcients are expected to be stored in reverse order.
 * \param oKernelSize Width and Height of the rectangular kernel.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference
 *        relative to the source pixel.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiFilter32f_8u16s_C3R(const Npp8u * pSrc, int nSrcStep, Npp16s * pDst, int nDstStep, NppiSize oSizeROI, 
                        const Npp32f * pKernel, NppiSize oKernelSize, NppiPoint oAnchor);

/**
 * Four channel 8-bit unsigned to 16-bit signed colwolution filter.
 * 
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *        Coeffcients are expected to be stored in reverse order.
 * \param oKernelSize Width and Height of the rectangular kernel.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference
 *        relative to the source pixel.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiFilter32f_8u16s_C4R(const Npp8u * pSrc, int nSrcStep, Npp16s * pDst, int nDstStep, NppiSize oSizeROI, 
                        const Npp32f * pKernel, NppiSize oKernelSize, NppiPoint oAnchor);

/**
 * Four channel 8-bit unsigned to 16-bit signed colwolution filter, ignoring alpha channel.
 * 
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *        Coeffcients are expected to be stored in reverse order.
 * \param oKernelSize Width and Height of the rectangular kernel.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference
 *        relative to the source pixel.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiFilter32f_8u16s_AC4R(const Npp8u * pSrc, int nSrcStep, Npp16s * pDst, int nDstStep, NppiSize oSizeROI, 
                         const Npp32f * pKernel, NppiSize oKernelSize, NppiPoint oAnchor);

/**
 * Single channel 8-bit to 16-bit signed colwolution filter.
 * 
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *        Coeffcients are expected to be stored in reverse order.
 * \param oKernelSize Width and Height of the rectangular kernel.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference
 *        relative to the source pixel.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiFilter32f_8s16s_C1R(const Npp8s * pSrc, int nSrcStep, Npp16s * pDst, int nDstStep, NppiSize oSizeROI, 
                        const Npp32f * pKernel, NppiSize oKernelSize, NppiPoint oAnchor);

/**
 * Three channel 8-bit to 16-bit signed colwolution filter.
 * 
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *        Coeffcients are expected to be stored in reverse order.
 * \param oKernelSize Width and Height of the rectangular kernel.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference
 *        relative to the source pixel.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiFilter32f_8s16s_C3R(const Npp8s * pSrc, int nSrcStep, Npp16s * pDst, int nDstStep, NppiSize oSizeROI, 
                        const Npp32f * pKernel, NppiSize oKernelSize, NppiPoint oAnchor);

/**
 * Four channel 8-bit to 16-bit signed colwolution filter.
 * 
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *        Coeffcients are expected to be stored in reverse order.
 * \param oKernelSize Width and Height of the rectangular kernel.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference
 *        relative to the source pixel.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiFilter32f_8s16s_C4R(const Npp8s * pSrc, int nSrcStep, Npp16s * pDst, int nDstStep, NppiSize oSizeROI, 
                        const Npp32f * pKernel, NppiSize oKernelSize, NppiPoint oAnchor);

/**
 * Four channel 8-bit to 16-bit signed colwolution filter, ignoring alpha channel.
 * 
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *        Coeffcients are expected to be stored in reverse order.
 * \param oKernelSize Width and Height of the rectangular kernel.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference
 *        relative to the source pixel.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiFilter32f_8s16s_AC4R(const Npp8s * pSrc, int nSrcStep, Npp16s * pDst, int nDstStep, NppiSize oSizeROI, 
                         const Npp32f * pKernel, NppiSize oKernelSize, NppiPoint oAnchor);


/** @} Filter32f */

/** @name FilterBorder
 * General purpose 2D colwolution filter with border control.
 *
 * Pixels under the mask are multiplied by the respective weights in the mask
 * and the results are summed. Before writing the result pixel the sum is scaled
 * back via division by nDivisor. If any portion of the mask overlaps the source
 * image boundary the requested border type operation is applied to all mask pixels
 * which fall outside of the source image.
 *
 * Lwrrently only the NPP_BORDER_REPLICATE border type operation is supported.
 *
 * @{
 *
 */

/**
 * Single channel 8-bit unsigned colwolution filter with border control.
 * 
 *
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *        Coeffcients are expected to be stored in reverse order.
 * \param oKernelSize Width and Height of the rectangular kernel.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference
 *        relative to the source pixel.
 * \param nDivisor The factor by which the colwolved summation from the Filter
 *        operation should be divided.  If equal to the sum of coefficients,
 *        this will keep the maximum result value within full scale.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterBorder_8u_C1R(const Npp8u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                        const Npp32s * pKernel, NppiSize oKernelSize, NppiPoint oAnchor, Npp32s nDivisor, NppiBorderType eBorderType);
  
/**
 * Three channel 8-bit unsigned colwolution filter with border control.
 * 
 *
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *        Coeffcients are expected to be stored in reverse order.
 * \param oKernelSize Width and Height of the rectangular kernel.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference
 *        relative to the source pixel.
 * \param nDivisor The factor by which the colwolved summation from the Filter
 *        operation should be divided.  If equal to the sum of coefficients,
 *        this will keep the maximum result value within full scale.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterBorder_8u_C3R(const Npp8u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                        const Npp32s * pKernel, NppiSize oKernelSize, NppiPoint oAnchor, Npp32s nDivisor, NppiBorderType eBorderType);
                 
/**
 * Four channel channel 8-bit unsigned colwolution filter with border control.
 * 
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *        Coeffcients are expected to be stored in reverse order.
 * \param oKernelSize Width and Height of the rectangular kernel.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference
 *        relative to the source pixel.
 * \param nDivisor The factor by which the colwolved summation from the Filter
 *        operation should be divided.  If equal to the sum of coefficients,
 *        this will keep the maximum result value within full scale.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterBorder_8u_C4R(const Npp8u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                        const Npp32s * pKernel, NppiSize oKernelSize, NppiPoint oAnchor, Npp32s nDivisor, NppiBorderType eBorderType);

/**
 * Four channel 8-bit unsigned colwolution filter with border control, ignoring alpha channel.
 * 
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *        Coeffcients are expected to be stored in reverse order.
 * \param oKernelSize Width and Height of the rectangular kernel.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference
 *        relative to the source pixel.
 * \param nDivisor The factor by which the colwolved summation from the Filter
 *        operation should be divided.  If equal to the sum of coefficients,
 *        this will keep the maximum result value within full scale.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterBorder_8u_AC4R(const Npp8u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                         const Npp32s * pKernel, NppiSize oKernelSize, NppiPoint oAnchor, Npp32s nDivisor, NppiBorderType eBorderType);

/**
 * Single channel 16-bit unsigned colwolution filter with border control.
 * 
 *
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *        Coeffcients are expected to be stored in reverse order.
 * \param oKernelSize Width and Height of the rectangular kernel.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference
 *        relative to the source pixel.
 * \param nDivisor The factor by which the colwolved summation from the Filter
 *        operation should be divided.  If equal to the sum of coefficients,
 *        this will keep the maximum result value within full scale.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterBorder_16u_C1R(const Npp16u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16u * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                         const Npp32s * pKernel, NppiSize oKernelSize, NppiPoint oAnchor, Npp32s nDivisor, NppiBorderType eBorderType);
  
/**
 * Three channel 16-bit unsigned colwolution filter with border control.
 * 
 *
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *        Coeffcients are expected to be stored in reverse order.
 * \param oKernelSize Width and Height of the rectangular kernel.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference
 *        relative to the source pixel.
 * \param nDivisor The factor by which the colwolved summation from the Filter
 *        operation should be divided.  If equal to the sum of coefficients,
 *        this will keep the maximum result value within full scale.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterBorder_16u_C3R(const Npp16u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16u * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                         const Npp32s * pKernel, NppiSize oKernelSize, NppiPoint oAnchor, Npp32s nDivisor, NppiBorderType eBorderType);
                 
/**
 * Four channel channel 16-bit unsigned colwolution filter with border control.
 * 
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *        Coeffcients are expected to be stored in reverse order.
 * \param oKernelSize Width and Height of the rectangular kernel.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference
 *        relative to the source pixel.
 * \param nDivisor The factor by which the colwolved summation from the Filter
 *        operation should be divided.  If equal to the sum of coefficients,
 *        this will keep the maximum result value within full scale.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterBorder_16u_C4R(const Npp16u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16u * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                         const Npp32s * pKernel, NppiSize oKernelSize, NppiPoint oAnchor, Npp32s nDivisor, NppiBorderType eBorderType);

/**
 * Four channel 16-bit unsigned colwolution filter with border control, ignoring alpha channel.
 * 
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *        Coeffcients are expected to be stored in reverse order.
 * \param oKernelSize Width and Height of the rectangular kernel.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference
 *        relative to the source pixel.
 * \param nDivisor The factor by which the colwolved summation from the Filter
 *        operation should be divided.  If equal to the sum of coefficients,
 *        this will keep the maximum result value within full scale.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterBorder_16u_AC4R(const Npp16u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16u * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                          const Npp32s * pKernel, NppiSize oKernelSize, NppiPoint oAnchor, Npp32s nDivisor, NppiBorderType eBorderType);

/**
 * Single channel 16-bit colwolution filter with border control.
 * 
 *
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *        Coeffcients are expected to be stored in reverse order.
 * \param oKernelSize Width and Height of the rectangular kernel.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference
 *        relative to the source pixel.
 * \param nDivisor The factor by which the colwolved summation from the Filter
 *        operation should be divided.  If equal to the sum of coefficients,
 *        this will keep the maximum result value within full scale.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterBorder_16s_C1R(const Npp16s * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                         const Npp32s * pKernel, NppiSize oKernelSize, NppiPoint oAnchor, Npp32s nDivisor, NppiBorderType eBorderType);
  
/**
 * Three channel 16-bit colwolution filter with border control.
 * 
 *
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *        Coeffcients are expected to be stored in reverse order.
 * \param oKernelSize Width and Height of the rectangular kernel.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference
 *        relative to the source pixel.
 * \param nDivisor The factor by which the colwolved summation from the Filter
 *        operation should be divided.  If equal to the sum of coefficients,
 *        this will keep the maximum result value within full scale.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterBorder_16s_C3R(const Npp16s * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                         const Npp32s * pKernel, NppiSize oKernelSize, NppiPoint oAnchor, Npp32s nDivisor, NppiBorderType eBorderType);
                 
/**
 * Four channel channel 16-bit colwolution filter with border control.
 * 
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *        Coeffcients are expected to be stored in reverse order.
 * \param oKernelSize Width and Height of the rectangular kernel.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference
 *        relative to the source pixel.
 * \param nDivisor The factor by which the colwolved summation from the Filter
 *        operation should be divided.  If equal to the sum of coefficients,
 *        this will keep the maximum result value within full scale.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterBorder_16s_C4R(const Npp16s * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                         const Npp32s * pKernel, NppiSize oKernelSize, NppiPoint oAnchor, Npp32s nDivisor, NppiBorderType eBorderType);

/**
 * Four channel 16-bit colwolution filter with border control, ignoring alpha channel.
 * 
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *        Coeffcients are expected to be stored in reverse order.
 * \param oKernelSize Width and Height of the rectangular kernel.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference
 *        relative to the source pixel.
 * \param nDivisor The factor by which the colwolved summation from the Filter
 *        operation should be divided.  If equal to the sum of coefficients,
 *        this will keep the maximum result value within full scale.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterBorder_16s_AC4R(const Npp16s * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                          const Npp32s * pKernel, NppiSize oKernelSize, NppiPoint oAnchor, Npp32s nDivisor, NppiBorderType eBorderType);
 
/**
 * Single channel 32-bit float colwolution filter with border control.
 * 
 *
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *        Coeffcients are expected to be stored in reverse order.
 * \param oKernelSize Width and Height of the rectangular kernel.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference
 *        relative to the source pixel.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiFilterBorder_32f_C1R(const Npp32f * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                         const Npp32f * pKernel, NppiSize oKernelSize, NppiPoint oAnchor, NppiBorderType eBorderType);

/**
 * Two channel 32-bit float colwolution filter with border control.
 * 
 *
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *        Coeffcients are expected to be stored in reverse order.
 * \param oKernelSize Width and Height of the rectangular kernel.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference
 *        relative to the source pixel.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiFilterBorder_32f_C2R(const Npp32f * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                         const Npp32f * pKernel, NppiSize oKernelSize, NppiPoint oAnchor, NppiBorderType eBorderType);

/**
 * Three channel 32-bit float colwolution filter with border control.
 * 
 *
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *        Coeffcients are expected to be stored in reverse order.
 * \param oKernelSize Width and Height of the rectangular kernel.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference
 *        relative to the source pixel.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiFilterBorder_32f_C3R(const Npp32f * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                         const Npp32f * pKernel, NppiSize oKernelSize, NppiPoint oAnchor, NppiBorderType eBorderType);

/**
 * Four channel 32-bit float colwolution filter with border control.
 * 
 *
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *        Coeffcients are expected to be stored in reverse order.
 * \param oKernelSize Width and Height of the rectangular kernel.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference
 *        relative to the source pixel.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiFilterBorder_32f_C4R(const Npp32f * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                         const Npp32f * pKernel, NppiSize oKernelSize, NppiPoint oAnchor, NppiBorderType eBorderType);

/**
 * Four channel 32-bit float colwolution filter with border control, ignoring alpha channel.
 * 
 *
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *        Coeffcients are expected to be stored in reverse order.
 * \param oKernelSize Width and Height of the rectangular kernel.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference
 *        relative to the source pixel.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiFilterBorder_32f_AC4R(const Npp32f * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                          const Npp32f * pKernel, NppiSize oKernelSize, NppiPoint oAnchor, NppiBorderType eBorderType);

/** @} FilterBorder */

/** @name FilterBorder32f
 * General purpose 2D colwolution filter using floating-point weights with border control.
 *
 * Pixels under the mask are multiplied by the respective weights in the mask
 * and the results are summed. Before writing the result pixel the sum is scaled
 * back via division by nDivisor. If any portion of the mask overlaps the source
 * image boundary the requested border type operation is applied to all mask pixels
 * which fall outside of the source image.
 *
 * Lwrrently only the NPP_BORDER_REPLICATE border type operation is supported.
 *
 * @{
 *
 */


/**
 * Single channel 8-bit unsigned colwolution filter with border control.
 * 
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *        Coeffcients are expected to be stored in reverse order.
 * \param oKernelSize Width and Height of the rectangular kernel.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference
 *        relative to the source pixel.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiFilterBorder32f_8u_C1R(const Npp8u * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u * pDst, int nDstStep, NppiSize oSizeROI, 
                           const Npp32f * pKernel, NppiSize oKernelSize, NppiPoint oAnchor, NppiBorderType eBorderType);
        
/**
 * Two channel 8-bit unsigned colwolution filter with border control.
 * 
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *        Coeffcients are expected to be stored in reverse order.
 * \param oKernelSize Width and Height of the rectangular kernel.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference
 *        relative to the source pixel.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiFilterBorder32f_8u_C2R(const Npp8u * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u * pDst, int nDstStep, NppiSize oSizeROI, 
                           const Npp32f * pKernel, NppiSize oKernelSize, NppiPoint oAnchor, NppiBorderType eBorderType);
        
/**
 * Three channel 8-bit unsigned colwolution filter with border control.
 * 
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *        Coeffcients are expected to be stored in reverse order.
 * \param oKernelSize Width and Height of the rectangular kernel.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference
 *        relative to the source pixel.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiFilterBorder32f_8u_C3R(const Npp8u * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u * pDst, int nDstStep, NppiSize oSizeROI, 
                           const Npp32f * pKernel, NppiSize oKernelSize, NppiPoint oAnchor, NppiBorderType eBorderType);
        
/**
 * Four channel 8-bit unsigned colwolution filter with border control.
 * 
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *        Coeffcients are expected to be stored in reverse order.
 * \param oKernelSize Width and Height of the rectangular kernel.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference
 *        relative to the source pixel.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiFilterBorder32f_8u_C4R(const Npp8u * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u * pDst, int nDstStep, NppiSize oSizeROI, 
                           const Npp32f * pKernel, NppiSize oKernelSize, NppiPoint oAnchor, NppiBorderType eBorderType);
        
/**
 * Four channel 8-bit unsigned colwolution filter with border control, ignorint alpha channel.
 * 
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *        Coeffcients are expected to be stored in reverse order.
 * \param oKernelSize Width and Height of the rectangular kernel.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference
 *        relative to the source pixel.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiFilterBorder32f_8u_AC4R(const Npp8u * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u * pDst, int nDstStep, NppiSize oSizeROI, 
                            const Npp32f * pKernel, NppiSize oKernelSize, NppiPoint oAnchor, NppiBorderType eBorderType);

/**
 * Single channel 8-bit signed colwolution filter with border control.
 * 
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *        Coeffcients are expected to be stored in reverse order.
 * \param oKernelSize Width and Height of the rectangular kernel.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference
 *        relative to the source pixel.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiFilterBorder32f_8s_C1R(const Npp8s * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8s * pDst, int nDstStep, NppiSize oSizeROI, 
                           const Npp32f * pKernel, NppiSize oKernelSize, NppiPoint oAnchor, NppiBorderType eBorderType);

/**
 * Two channel 8-bit signed colwolution filter with border control.
 * 
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *        Coeffcients are expected to be stored in reverse order.
 * \param oKernelSize Width and Height of the rectangular kernel.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference
 *        relative to the source pixel.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiFilterBorder32f_8s_C2R(const Npp8s * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8s * pDst, int nDstStep, NppiSize oSizeROI, 
                           const Npp32f * pKernel, NppiSize oKernelSize, NppiPoint oAnchor, NppiBorderType eBorderType);

/**
 * Three channel 8-bit signed colwolution filter with border control.
 * 
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *        Coeffcients are expected to be stored in reverse order.
 * \param oKernelSize Width and Height of the rectangular kernel.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference
 *        relative to the source pixel.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiFilterBorder32f_8s_C3R(const Npp8s * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8s * pDst, int nDstStep, NppiSize oSizeROI, 
                           const Npp32f * pKernel, NppiSize oKernelSize, NppiPoint oAnchor, NppiBorderType eBorderType);

/**
 * Four channel 8-bit signed colwolution filter with border control.
 * 
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *        Coeffcients are expected to be stored in reverse order.
 * \param oKernelSize Width and Height of the rectangular kernel.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference
 *        relative to the source pixel.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiFilterBorder32f_8s_C4R(const Npp8s * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8s * pDst, int nDstStep, NppiSize oSizeROI, 
                           const Npp32f * pKernel, NppiSize oKernelSize, NppiPoint oAnchor, NppiBorderType eBorderType);

/**
 * Four channel 8-bit signed colwolution filter with border control, ignoring alpha channel.
 * 
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *        Coeffcients are expected to be stored in reverse order.
 * \param oKernelSize Width and Height of the rectangular kernel.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference
 *        relative to the source pixel.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiFilterBorder32f_8s_AC4R(const Npp8s * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8s * pDst, int nDstStep, NppiSize oSizeROI, 
                            const Npp32f * pKernel, NppiSize oKernelSize, NppiPoint oAnchor, NppiBorderType eBorderType);

/**
 * Single channel 16-bit unsigned colwolution filter with border control.
 * 
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *        Coeffcients are expected to be stored in reverse order.
 * \param oKernelSize Width and Height of the rectangular kernel.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference
 *        relative to the source pixel.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiFilterBorder32f_16u_C1R(const Npp16u * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16u * pDst, int nDstStep, NppiSize oSizeROI, 
                            const Npp32f * pKernel, NppiSize oKernelSize, NppiPoint oAnchor, NppiBorderType eBorderType);

/**
 * Three channel 16-bit unsigned colwolution filter with border control.
 * 
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *        Coeffcients are expected to be stored in reverse order.
 * \param oKernelSize Width and Height of the rectangular kernel.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference
 *        relative to the source pixel.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiFilterBorder32f_16u_C3R(const Npp16u * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16u * pDst, int nDstStep, NppiSize oSizeROI, 
                            const Npp32f * pKernel, NppiSize oKernelSize, NppiPoint oAnchor, NppiBorderType eBorderType);

/**
 * Four channel 16-bit unsigned colwolution filter with border control.
 * 
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *        Coeffcients are expected to be stored in reverse order.
 * \param oKernelSize Width and Height of the rectangular kernel.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference
 *        relative to the source pixel.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiFilterBorder32f_16u_C4R(const Npp16u * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16u * pDst, int nDstStep, NppiSize oSizeROI, 
                            const Npp32f * pKernel, NppiSize oKernelSize, NppiPoint oAnchor, NppiBorderType eBorderType);

/**
 * Four channel 16-bit unsigned colwolution filter with border control, ignoring alpha channel.
 * 
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *        Coeffcients are expected to be stored in reverse order.
 * \param oKernelSize Width and Height of the rectangular kernel.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference
 *        relative to the source pixel.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiFilterBorder32f_16u_AC4R(const Npp16u * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16u * pDst, int nDstStep, NppiSize oSizeROI, 
                             const Npp32f * pKernel, NppiSize oKernelSize, NppiPoint oAnchor, NppiBorderType eBorderType);

/**
 * Single channel 16-bit colwolution filter with border control.
 * 
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *        Coeffcients are expected to be stored in reverse order.
 * \param oKernelSize Width and Height of the rectangular kernel.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference
 *        relative to the source pixel.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiFilterBorder32f_16s_C1R(const Npp16s * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16s * pDst, int nDstStep, NppiSize oSizeROI, 
                            const Npp32f * pKernel, NppiSize oKernelSize, NppiPoint oAnchor, NppiBorderType eBorderType);

/**
 * Three channel 16-bit colwolution filter with border control.
 * 
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *        Coeffcients are expected to be stored in reverse order.
 * \param oKernelSize Width and Height of the rectangular kernel.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference
 *        relative to the source pixel.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiFilterBorder32f_16s_C3R(const Npp16s * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16s * pDst, int nDstStep, NppiSize oSizeROI, 
                            const Npp32f * pKernel, NppiSize oKernelSize, NppiPoint oAnchor, NppiBorderType eBorderType);

/**
 * Four channel 16-bit colwolution filter with border control.
 * 
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *        Coeffcients are expected to be stored in reverse order.
 * \param oKernelSize Width and Height of the rectangular kernel.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference
 *        relative to the source pixel.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiFilterBorder32f_16s_C4R(const Npp16s * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16s * pDst, int nDstStep, NppiSize oSizeROI, 
                            const Npp32f * pKernel, NppiSize oKernelSize, NppiPoint oAnchor, NppiBorderType eBorderType);

/**
 * Four channel 16-bit colwolution filter with border control, ignoring alpha channel.
 * 
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *        Coeffcients are expected to be stored in reverse order.
 * \param oKernelSize Width and Height of the rectangular kernel.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference
 *        relative to the source pixel.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiFilterBorder32f_16s_AC4R(const Npp16s * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16s * pDst, int nDstStep, NppiSize oSizeROI, 
                             const Npp32f * pKernel, NppiSize oKernelSize, NppiPoint oAnchor, NppiBorderType eBorderType);


/**
 * Single channel 32-bit colwolution filter with border control.
 * 
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *        Coeffcients are expected to be stored in reverse order.
 * \param oKernelSize Width and Height of the rectangular kernel.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference
 *        relative to the source pixel.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiFilterBorder32f_32s_C1R(const Npp32s * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32s * pDst, int nDstStep, NppiSize oSizeROI, 
                            const Npp32f * pKernel, NppiSize oKernelSize, NppiPoint oAnchor, NppiBorderType eBorderType);

/**
 * Three channel 32-bit colwolution filter with border control.
 * 
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *        Coeffcients are expected to be stored in reverse order.
 * \param oKernelSize Width and Height of the rectangular kernel.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference
 *        relative to the source pixel.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiFilterBorder32f_32s_C3R(const Npp32s * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32s * pDst, int nDstStep, NppiSize oSizeROI, 
                            const Npp32f * pKernel, NppiSize oKernelSize, NppiPoint oAnchor, NppiBorderType eBorderType);

/**
 * Four channel 32-bit colwolution filter with border control.
 * 
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *        Coeffcients are expected to be stored in reverse order.
 * \param oKernelSize Width and Height of the rectangular kernel.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference
 *        relative to the source pixel.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiFilterBorder32f_32s_C4R(const Npp32s * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32s * pDst, int nDstStep, NppiSize oSizeROI, 
                            const Npp32f * pKernel, NppiSize oKernelSize, NppiPoint oAnchor, NppiBorderType eBorderType);

/**
 * Four channel 32-bit colwolution filter with border control, ignoring alpha channel.
 * 
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *        Coeffcients are expected to be stored in reverse order.
 * \param oKernelSize Width and Height of the rectangular kernel.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference
 *        relative to the source pixel.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiFilterBorder32f_32s_AC4R(const Npp32s * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32s * pDst, int nDstStep, NppiSize oSizeROI, 
                             const Npp32f * pKernel, NppiSize oKernelSize, NppiPoint oAnchor, NppiBorderType eBorderType);


/**
 * Single channel 8-bit unsigned to 16-bit signed colwolution filter with border control.
 * 
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *        Coeffcients are expected to be stored in reverse order.
 * \param oKernelSize Width and Height of the rectangular kernel.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference
 *        relative to the source pixel.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiFilterBorder32f_8u16s_C1R(const Npp8u * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16s * pDst, int nDstStep, NppiSize oSizeROI, 
                              const Npp32f * pKernel, NppiSize oKernelSize, NppiPoint oAnchor, NppiBorderType eBorderType);

/**
 * Three channel 8-bit unsigned to 16-bit signed colwolution filter with border control.
 * 
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *        Coeffcients are expected to be stored in reverse order.
 * \param oKernelSize Width and Height of the rectangular kernel.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference
 *        relative to the source pixel.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiFilterBorder32f_8u16s_C3R(const Npp8u * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16s * pDst, int nDstStep, NppiSize oSizeROI, 
                              const Npp32f * pKernel, NppiSize oKernelSize, NppiPoint oAnchor, NppiBorderType eBorderType);

/**
 * Four channel 8-bit unsigned to 16-bit signed colwolution filter with border control.
 * 
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *        Coeffcients are expected to be stored in reverse order.
 * \param oKernelSize Width and Height of the rectangular kernel.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference
 *        relative to the source pixel.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiFilterBorder32f_8u16s_C4R(const Npp8u * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16s * pDst, int nDstStep, NppiSize oSizeROI, 
                             const Npp32f * pKernel, NppiSize oKernelSize, NppiPoint oAnchor, NppiBorderType eBorderType);

/**
 * Four channel 8-bit unsigned to 16-bit signed colwolution filter with border control, ignoring alpha channel.
 * 
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *        Coeffcients are expected to be stored in reverse order.
 * \param oKernelSize Width and Height of the rectangular kernel.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference
 *        relative to the source pixel.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiFilterBorder32f_8u16s_AC4R(const Npp8u * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16s * pDst, int nDstStep, NppiSize oSizeROI, 
                               const Npp32f * pKernel, NppiSize oKernelSize, NppiPoint oAnchor, NppiBorderType eBorderType);

/**
 * Single channel 8-bit to 16-bit signed colwolution filter with border control.
 * 
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *        Coeffcients are expected to be stored in reverse order.
 * \param oKernelSize Width and Height of the rectangular kernel.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference
 *        relative to the source pixel.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiFilterBorder32f_8s16s_C1R(const Npp8s * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16s * pDst, int nDstStep, NppiSize oSizeROI, 
                              const Npp32f * pKernel, NppiSize oKernelSize, NppiPoint oAnchor, NppiBorderType eBorderType);

/**
 * Three channel 8-bit to 16-bit signed colwolution filter with border control.
 * 
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *        Coeffcients are expected to be stored in reverse order.
 * \param oKernelSize Width and Height of the rectangular kernel.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference
 *        relative to the source pixel.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiFilterBorder32f_8s16s_C3R(const Npp8s * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16s * pDst, int nDstStep, NppiSize oSizeROI, 
                              const Npp32f * pKernel, NppiSize oKernelSize, NppiPoint oAnchor, NppiBorderType eBorderType);

/**
 * Four channel 8-bit to 16-bit signed colwolution filter with border control.
 * 
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *        Coeffcients are expected to be stored in reverse order.
 * \param oKernelSize Width and Height of the rectangular kernel.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference
 *        relative to the source pixel.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiFilterBorder32f_8s16s_C4R(const Npp8s * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16s * pDst, int nDstStep, NppiSize oSizeROI, 
                              const Npp32f * pKernel, NppiSize oKernelSize, NppiPoint oAnchor, NppiBorderType eBorderType);

/**
 * Four channel 8-bit to 16-bit signed colwolution filter with border control, ignoring alpha channel.
 * 
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pKernel Pointer to the start address of the kernel coefficient array.
 *        Coeffcients are expected to be stored in reverse order.
 * \param oKernelSize Width and Height of the rectangular kernel.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference
 *        relative to the source pixel.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiFilterBorder32f_8s16s_AC4R(const Npp8s * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16s * pDst, int nDstStep, NppiSize oSizeROI, 
                               const Npp32f * pKernel, NppiSize oKernelSize, NppiPoint oAnchor, NppiBorderType eBorderType);


/** @} FilterBorder32f */

/** @} image_colwolution */

/** @defgroup image_2D_fixed_linear_filters 2D Fixed Linear Filters
 *
 * @{
 *
 */

/** @name FilterBox
 *
 * Computes the average pixel values of the pixels under a rectangular mask.
 *
 * @{
 *
 */

/**
 * Single channel 8-bit unsigned box filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param oMaskSize Width and Height of the neighborhood region for the local
 *        Avg operation.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference relative to
 *        the source pixel.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterBox_8u_C1R(const Npp8u * pSrc, Npp32s nSrcStep, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                     NppiSize oMaskSize, NppiPoint oAnchor);

/**
 * Three channel 8-bit unsigned box filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param oMaskSize Width and Height of the neighborhood region for the local
 *        Avg operation.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference relative to
 *        the source pixel.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterBox_8u_C3R(const Npp8u * pSrc, Npp32s nSrcStep, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                     NppiSize oMaskSize, NppiPoint oAnchor);

/**
 * Four channel 8-bit unsigned box filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param oMaskSize Width and Height of the neighborhood region for the local
 *        Avg operation.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference relative to
 *        the source pixel.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterBox_8u_C4R(const Npp8u * pSrc, Npp32s nSrcStep, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                     NppiSize oMaskSize, NppiPoint oAnchor);

/**
 * Four channel 8-bit unsigned box filter, ignorting alpha channel.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param oMaskSize Width and Height of the neighborhood region for the local
 *        Avg operation.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference relative to
 *        the source pixel.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterBox_8u_AC4R(const Npp8u * pSrc, Npp32s nSrcStep, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                      NppiSize oMaskSize, NppiPoint oAnchor);

/**
 * Single channel 16-bit unsigned box filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param oMaskSize Width and Height of the neighborhood region for the local
 *        Avg operation.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference relative to
 *        the source pixel.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterBox_16u_C1R(const Npp16u * pSrc, Npp32s nSrcStep, Npp16u * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                      NppiSize oMaskSize, NppiPoint oAnchor);

/**
 * Three channel 16-bit unsigned box filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param oMaskSize Width and Height of the neighborhood region for the local
 *        Avg operation.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference relative to
 *        the source pixel.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterBox_16u_C3R(const Npp16u * pSrc, Npp32s nSrcStep, Npp16u * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                      NppiSize oMaskSize, NppiPoint oAnchor);

/**
 * Four channel 16-bit unsigned box filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param oMaskSize Width and Height of the neighborhood region for the local
 *        Avg operation.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference relative to
 *        the source pixel.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterBox_16u_C4R(const Npp16u * pSrc, Npp32s nSrcStep, Npp16u * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                      NppiSize oMaskSize, NppiPoint oAnchor);

/**
 * Four channel 16-bit unsigned box filter, ignorting alpha channel.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param oMaskSize Width and Height of the neighborhood region for the local
 *        Avg operation.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference relative to
 *        the source pixel.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterBox_16u_AC4R(const Npp16u * pSrc, Npp32s nSrcStep, Npp16u * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                       NppiSize oMaskSize, NppiPoint oAnchor);

/**
 * Single channel 16-bit box filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param oMaskSize Width and Height of the neighborhood region for the local
 *        Avg operation.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference relative to
 *        the source pixel.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterBox_16s_C1R(const Npp16s * pSrc, Npp32s nSrcStep, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                      NppiSize oMaskSize, NppiPoint oAnchor);

/**
 * Three channel 16-bit box filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param oMaskSize Width and Height of the neighborhood region for the local
 *        Avg operation.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference relative to
 *        the source pixel.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterBox_16s_C3R(const Npp16s * pSrc, Npp32s nSrcStep, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                      NppiSize oMaskSize, NppiPoint oAnchor);

/**
 * Four channel 16-bit box filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param oMaskSize Width and Height of the neighborhood region for the local
 *        Avg operation.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference relative to
 *        the source pixel.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterBox_16s_C4R(const Npp16s * pSrc, Npp32s nSrcStep, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                      NppiSize oMaskSize, NppiPoint oAnchor);

/**
 * Four channel 16-bit box filter, ignorting alpha channel.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param oMaskSize Width and Height of the neighborhood region for the local
 *        Avg operation.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference relative to
 *        the source pixel.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterBox_16s_AC4R(const Npp16s * pSrc, Npp32s nSrcStep, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                       NppiSize oMaskSize, NppiPoint oAnchor);

/**
 * Single channel 32-bit floating-point box filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param oMaskSize Width and Height of the neighborhood region for the local
 *        Avg operation.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference relative to
 *        the source pixel.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterBox_32f_C1R(const Npp32f * pSrc, Npp32s nSrcStep, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                      NppiSize oMaskSize, NppiPoint oAnchor);

/**
 * Three channel 32-bit floating-point box filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param oMaskSize Width and Height of the neighborhood region for the local
 *        Avg operation.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference relative to
 *        the source pixel.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterBox_32f_C3R(const Npp32f * pSrc, Npp32s nSrcStep, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                      NppiSize oMaskSize, NppiPoint oAnchor);

/**
 * Four channel 32-bit floating-point box filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param oMaskSize Width and Height of the neighborhood region for the local
 *        Avg operation.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference relative to
 *        the source pixel.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterBox_32f_C4R(const Npp32f * pSrc, Npp32s nSrcStep, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                      NppiSize oMaskSize, NppiPoint oAnchor);

/**
 * Four channel 32-bit floating-point box filter, ignorting alpha channel.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param oMaskSize Width and Height of the neighborhood region for the local
 *        Avg operation.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference relative to
 *        the source pixel.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterBox_32f_AC4R(const Npp32f * pSrc, Npp32s nSrcStep, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                       NppiSize oMaskSize, NppiPoint oAnchor);

/**
 * Single channel 64-bit floating-point box filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param oMaskSize Width and Height of the neighborhood region for the local
 *        Avg operation.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference relative to
 *        the source pixel.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterBox_64f_C1R(const Npp64f * pSrc, Npp32s nSrcStep, Npp64f * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                      NppiSize oMaskSize, NppiPoint oAnchor);

/** @} FilterBox */

/** @name FilterBoxBorder
 *
 * Computes the average pixel values of the pixels under a rectangular mask with border control.
 * If any portion of the mask overlaps the source image boundary the requested 
 * border type operation is applied to all mask pixels which fall outside of the source image.
 *
 * Lwrrently only the NPP_BORDER_REPLICATE border type operation is supported. *
 *
 * @{
 *
 */

/**
 * Single channel 8-bit unsigned box filter with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param oMaskSize Width and Height of the neighborhood region for the local
 *        Avg operation.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference relative to
 *        the source pixel.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterBoxBorder_8u_C1R(const Npp8u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                           NppiSize oMaskSize, NppiPoint oAnchor, NppiBorderType eBorderType);

/**
 * Three channel 8-bit unsigned box filter with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param oMaskSize Width and Height of the neighborhood region for the local
 *        Avg operation.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference relative to
 *        the source pixel.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterBoxBorder_8u_C3R(const Npp8u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                           NppiSize oMaskSize, NppiPoint oAnchor, NppiBorderType eBorderType);

/**
 * Four channel 8-bit unsigned box filter with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param oMaskSize Width and Height of the neighborhood region for the local
 *        Avg operation.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference relative to
 *        the source pixel.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterBoxBorder_8u_C4R(const Npp8u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                           NppiSize oMaskSize, NppiPoint oAnchor, NppiBorderType eBorderType);

/**
 * Four channel 8-bit unsigned box filter with border control, ignoring alpha channel.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param oMaskSize Width and Height of the neighborhood region for the local
 *        Avg operation.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference relative to
 *        the source pixel.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterBoxBorder_8u_AC4R(const Npp8u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                            NppiSize oMaskSize, NppiPoint oAnchor, NppiBorderType eBorderType);

/**
 * Single channel 16-bit unsigned box filter with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param oMaskSize Width and Height of the neighborhood region for the local
 *        Avg operation.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference relative to
 *        the source pixel.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterBoxBorder_16u_C1R(const Npp16u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16u * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                            NppiSize oMaskSize, NppiPoint oAnchor, NppiBorderType eBorderType);

/**
 * Three channel 16-bit unsigned box filter with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param oMaskSize Width and Height of the neighborhood region for the local
 *        Avg operation.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference relative to
 *        the source pixel.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterBoxBorder_16u_C3R(const Npp16u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16u * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                            NppiSize oMaskSize, NppiPoint oAnchor, NppiBorderType eBorderType);

/**
 * Four channel 16-bit unsigned box filter with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param oMaskSize Width and Height of the neighborhood region for the local
 *        Avg operation.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference relative to
 *        the source pixel.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterBoxBorder_16u_C4R(const Npp16u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16u * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                            NppiSize oMaskSize, NppiPoint oAnchor, NppiBorderType eBorderType);

/**
 * Four channel 16-bit unsigned box filter with border control, ignoring alpha channel.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param oMaskSize Width and Height of the neighborhood region for the local
 *        Avg operation.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference relative to
 *        the source pixel.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterBoxBorder_16u_AC4R(const Npp16u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16u * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                             NppiSize oMaskSize, NppiPoint oAnchor, NppiBorderType eBorderType);

/**
 * Single channel 16-bit box filter with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param oMaskSize Width and Height of the neighborhood region for the local
 *        Avg operation.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference relative to
 *        the source pixel.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterBoxBorder_16s_C1R(const Npp16s * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                            NppiSize oMaskSize, NppiPoint oAnchor, NppiBorderType eBorderType);

/**
 * Three channel 16-bit box filter with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param oMaskSize Width and Height of the neighborhood region for the local
 *        Avg operation.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference relative to
 *        the source pixel.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterBoxBorder_16s_C3R(const Npp16s * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                            NppiSize oMaskSize, NppiPoint oAnchor, NppiBorderType eBorderType);

/**
 * Four channel 16-bit box filter with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param oMaskSize Width and Height of the neighborhood region for the local
 *        Avg operation.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference relative to
 *        the source pixel.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterBoxBorder_16s_C4R(const Npp16s * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                            NppiSize oMaskSize, NppiPoint oAnchor, NppiBorderType eBorderType);

/**
 * Four channel 16-bit box filter with border control, ignoring alpha channel.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param oMaskSize Width and Height of the neighborhood region for the local
 *        Avg operation.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference relative to
 *        the source pixel.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterBoxBorder_16s_AC4R(const Npp16s * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                             NppiSize oMaskSize, NppiPoint oAnchor, NppiBorderType eBorderType);

/**
 * Single channel 32-bit floating-point box filter with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param oMaskSize Width and Height of the neighborhood region for the local
 *        Avg operation.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference relative to
 *        the source pixel.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterBoxBorder_32f_C1R(const Npp32f * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                            NppiSize oMaskSize, NppiPoint oAnchor, NppiBorderType eBorderType);

/**
 * Three channel 32-bit floating-point box filter with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param oMaskSize Width and Height of the neighborhood region for the local
 *        Avg operation.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference relative to
 *        the source pixel.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterBoxBorder_32f_C3R(const Npp32f * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                            NppiSize oMaskSize, NppiPoint oAnchor, NppiBorderType eBorderType);

/**
 * Four channel 32-bit floating-point box filter with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param oMaskSize Width and Height of the neighborhood region for the local
 *        Avg operation.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference relative to
 *        the source pixel.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterBoxBorder_32f_C4R(const Npp32f * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                            NppiSize oMaskSize, NppiPoint oAnchor, NppiBorderType eBorderType);

/**
 * Four channel 32-bit floating-point box filter with border control, ignoring alpha channel.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param oMaskSize Width and Height of the neighborhood region for the local
 *        Avg operation.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference relative to
 *        the source pixel.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterBoxBorder_32f_AC4R(const Npp32f * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                             NppiSize oMaskSize, NppiPoint oAnchor, NppiBorderType eBorderType);

/** @} FilterBoxBorder */

/** @name FilterThresholdAdaptiveBoxBorder
 *
 * Computes the average pixel values of the pixels under a square mask with border control.
 * If any portion of the mask overlaps the source image boundary the requested 
 * border type operation is applied to all mask pixels which fall outside of the source image.
 * Once the neighborhood average around a source pixel is determined the souce pixel is compared to the average - nDelta
 * and if the source pixel is greater than that average the corresponding destination pixel is set to lwalGT, otherwise lwalLE.
 *
 * Lwrrently only the NPP_BORDER_REPLICATE border type operation is supported. *
 *
 * @{
 *
 */

/**
 * Single channel 8-bit unsigned threshold adaptive box filter with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param oMaskSize Width and Height of the neighborhood region for the local
 *        Avg operation, Width and Height must be equal and odd.
 * \param nDelta Neighborhood average adjustment value
 * \param lwalGT Destination output value if source pixel is greater than average.
 * \param lwalLE Destination output value if source pixel is less than or equal to average.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterThresholdAdaptiveBoxBorder_8u_C1R(const Npp8u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                                            NppiSize oMaskSize, Npp32f nDelta, Npp8u lwalGT, Npp8u lwalLE, NppiBorderType eBorderType);

/** @} FilterThresholdAdaptiveBoxBorder */

/** @} image_2D_fixed_linear_filters */

/** @defgroup image_rank_filters Rank Filters
 *
 * @{
 *
 */

/** @name ImageMax Filter
 *
 * Result pixel value is the maximum of pixel values under the rectangular
 * mask region.
 *
 * @{
 *
 */

/**
 * Single channel 8-bit unsigned maximum filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param oMaskSize Width and Height of the neighborhood region for the local
 *        Max operation.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference
 *        relative to the source pixel.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterMax_8u_C1R(const Npp8u * pSrc, Npp32s nSrcStep, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                     NppiSize oMaskSize, NppiPoint oAnchor);

/**
 * Three channel 8-bit unsigned maximum filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param oMaskSize Width and Height of the neighborhood region for the local
 *        Max operation.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference
 *        relative to the source pixel.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterMax_8u_C3R(const Npp8u * pSrc, Npp32s nSrcStep, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                     NppiSize oMaskSize, NppiPoint oAnchor);

/**
 * Four channel 8-bit unsigned maximum filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param oMaskSize Width and Height of the neighborhood region for the local
 *        Max operation.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference
 *        relative to the source pixel.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiFilterMax_8u_C4R(const Npp8u * pSrc, Npp32s nSrcStep, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                     NppiSize oMaskSize, NppiPoint oAnchor);

/**
 * Four channel 8-bit unsigned maximum filter, ignoring alpha channel.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param oMaskSize Width and Height of the neighborhood region for the local
 *        Max operation.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference
 *        relative to the source pixel.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiFilterMax_8u_AC4R(const Npp8u * pSrc, Npp32s nSrcStep, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                     NppiSize oMaskSize, NppiPoint oAnchor);

/**
 * Single channel 16-bit unsigned maximum filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param oMaskSize Width and Height of the neighborhood region for the local
 *        Max operation.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference
 *        relative to the source pixel.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterMax_16u_C1R(const Npp16u * pSrc, Npp32s nSrcStep, Npp16u * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                      NppiSize oMaskSize, NppiPoint oAnchor);

/**
 * Three channel 16-bit unsigned maximum filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param oMaskSize Width and Height of the neighborhood region for the local
 *        Max operation.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference
 *        relative to the source pixel.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterMax_16u_C3R(const Npp16u * pSrc, Npp32s nSrcStep, Npp16u * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                      NppiSize oMaskSize, NppiPoint oAnchor);

/**
 * Four channel 16-bit unsigned maximum filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param oMaskSize Width and Height of the neighborhood region for the local
 *        Max operation.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference
 *        relative to the source pixel.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiFilterMax_16u_C4R(const Npp16u * pSrc, Npp32s nSrcStep, Npp16u * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                      NppiSize oMaskSize, NppiPoint oAnchor);

/**
 * Four channel 16-bit unsigned maximum filter, ignoring alpha channel.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param oMaskSize Width and Height of the neighborhood region for the local
 *        Max operation.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference
 *        relative to the source pixel.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiFilterMax_16u_AC4R(const Npp16u * pSrc, Npp32s nSrcStep, Npp16u * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                       NppiSize oMaskSize, NppiPoint oAnchor);

/**
 * Single channel 16-bit signed maximum filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param oMaskSize Width and Height of the neighborhood region for the local
 *        Max operation.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference
 *        relative to the source pixel.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterMax_16s_C1R(const Npp16s * pSrc, Npp32s nSrcStep, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                      NppiSize oMaskSize, NppiPoint oAnchor);

/**
 * Three channel 16-bit signed maximum filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param oMaskSize Width and Height of the neighborhood region for the local
 *        Max operation.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference
 *        relative to the source pixel.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterMax_16s_C3R(const Npp16s * pSrc, Npp32s nSrcStep, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                      NppiSize oMaskSize, NppiPoint oAnchor);

/**
 * Four channel 16-bit signed maximum filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param oMaskSize Width and Height of the neighborhood region for the local
 *        Max operation.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference
 *        relative to the source pixel.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiFilterMax_16s_C4R(const Npp16s * pSrc, Npp32s nSrcStep, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                      NppiSize oMaskSize, NppiPoint oAnchor);

/**
 * Four channel 16-bit signed maximum filter, ignoring alpha channel.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param oMaskSize Width and Height of the neighborhood region for the local
 *        Max operation.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference
 *        relative to the source pixel.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiFilterMax_16s_AC4R(const Npp16s * pSrc, Npp32s nSrcStep, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                       NppiSize oMaskSize, NppiPoint oAnchor);

/**
 * Single channel 32-bit floating-point maximum filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param oMaskSize Width and Height of the neighborhood region for the local
 *        Max operation.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference
 *        relative to the source pixel.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterMax_32f_C1R(const Npp32f * pSrc, Npp32s nSrcStep, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                      NppiSize oMaskSize, NppiPoint oAnchor);

/**
 * Three channel 32-bit floating-point maximum filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param oMaskSize Width and Height of the neighborhood region for the local
 *        Max operation.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference
 *        relative to the source pixel.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterMax_32f_C3R(const Npp32f * pSrc, Npp32s nSrcStep, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                      NppiSize oMaskSize, NppiPoint oAnchor);

/**
 * Four channel 32-bit floating-point maximum filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param oMaskSize Width and Height of the neighborhood region for the local
 *        Max operation.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference
 *        relative to the source pixel.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiFilterMax_32f_C4R(const Npp32f * pSrc, Npp32s nSrcStep, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                      NppiSize oMaskSize, NppiPoint oAnchor);

/**
 * Four channel 32-bit floating-point maximum filter, ignoring alpha channel.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param oMaskSize Width and Height of the neighborhood region for the local
 *        Max operation.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference
 *        relative to the source pixel.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiFilterMax_32f_AC4R(const Npp32f * pSrc, Npp32s nSrcStep, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                       NppiSize oMaskSize, NppiPoint oAnchor);

/** @} FilterMax */

/** @name ImageMaxBorder Filter
 *
 * Result pixel value is the maximum of pixel values under the rectangular
 * mask region. If any portion of the mask overlaps the source
 * image boundary the requested border type operation is applied to all mask pixels
 * which fall outside of the source image.
 *
 * Lwrrently only the NPP_BORDER_REPLICATE border type operation is supported.
 *
 * @{
 *
 */

/**
 * Single channel 8-bit unsigned maximum filter with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param oMaskSize Width and Height of the neighborhood region for the local
 *        Max operation.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference
 *        relative to the source pixel.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterMaxBorder_8u_C1R(const Npp8u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                           NppiSize oMaskSize, NppiPoint oAnchor, NppiBorderType eBorderType);

/**
 * Three channel 8-bit unsigned maximum filter with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param oMaskSize Width and Height of the neighborhood region for the local
 *        Max operation.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference
 *        relative to the source pixel.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterMaxBorder_8u_C3R(const Npp8u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                           NppiSize oMaskSize, NppiPoint oAnchor, NppiBorderType eBorderType);

/**
 * Four channel 8-bit unsigned maximum filter with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param oMaskSize Width and Height of the neighborhood region for the local
 *        Max operation.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference
 *        relative to the source pixel.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiFilterMaxBorder_8u_C4R(const Npp8u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                           NppiSize oMaskSize, NppiPoint oAnchor, NppiBorderType eBorderType);

/**
 * Four channel 8-bit unsigned maximum filter with border control, ignoring alpha channel.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param oMaskSize Width and Height of the neighborhood region for the local
 *        Max operation.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference
 *        relative to the source pixel.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiFilterMaxBorder_8u_AC4R(const Npp8u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                            NppiSize oMaskSize, NppiPoint oAnchor, NppiBorderType eBorderType);

/**
 * Single channel 16-bit unsigned maximum filter with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param oMaskSize Width and Height of the neighborhood region for the local
 *        Max operation.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference
 *        relative to the source pixel.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterMaxBorder_16u_C1R(const Npp16u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16u * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                            NppiSize oMaskSize, NppiPoint oAnchor, NppiBorderType eBorderType);

/**
 * Three channel 16-bit unsigned maximum filter with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param oMaskSize Width and Height of the neighborhood region for the local
 *        Max operation.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference
 *        relative to the source pixel.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterMaxBorder_16u_C3R(const Npp16u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16u * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                            NppiSize oMaskSize, NppiPoint oAnchor, NppiBorderType eBorderType);

/**
 * Four channel 16-bit unsigned maximum filter with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param oMaskSize Width and Height of the neighborhood region for the local
 *        Max operation.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference
 *        relative to the source pixel.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiFilterMaxBorder_16u_C4R(const Npp16u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16u * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                            NppiSize oMaskSize, NppiPoint oAnchor, NppiBorderType eBorderType);

/**
 * Four channel 16-bit unsigned maximum filter with border control, ignoring alpha channel.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param oMaskSize Width and Height of the neighborhood region for the local
 *        Max operation.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference
 *        relative to the source pixel.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiFilterMaxBorder_16u_AC4R(const Npp16u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16u * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                             NppiSize oMaskSize, NppiPoint oAnchor, NppiBorderType eBorderType);

/**
 * Single channel 16-bit signed maximum filter with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param oMaskSize Width and Height of the neighborhood region for the local
 *        Max operation.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference
 *        relative to the source pixel.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterMaxBorder_16s_C1R(const Npp16s * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                            NppiSize oMaskSize, NppiPoint oAnchor, NppiBorderType eBorderType);

/**
 * Three channel 16-bit signed maximum filter with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param oMaskSize Width and Height of the neighborhood region for the local
 *        Max operation.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference
 *        relative to the source pixel.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterMaxBorder_16s_C3R(const Npp16s * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                            NppiSize oMaskSize, NppiPoint oAnchor, NppiBorderType eBorderType);

/**
 * Four channel 16-bit signed maximum filter with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param oMaskSize Width and Height of the neighborhood region for the local
 *        Max operation.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference
 *        relative to the source pixel.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiFilterMaxBorder_16s_C4R(const Npp16s * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                            NppiSize oMaskSize, NppiPoint oAnchor, NppiBorderType eBorderType);

/**
 * Four channel 16-bit signed maximum filter with border control, ignoring alpha channel.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param oMaskSize Width and Height of the neighborhood region for the local
 *        Max operation.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference
 *        relative to the source pixel.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiFilterMaxBorder_16s_AC4R(const Npp16s * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                             NppiSize oMaskSize, NppiPoint oAnchor, NppiBorderType eBorderType);

/**
 * Single channel 32-bit floating-point maximum filter with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param oMaskSize Width and Height of the neighborhood region for the local
 *        Max operation.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference
 *        relative to the source pixel.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterMaxBorder_32f_C1R(const Npp32f * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                            NppiSize oMaskSize, NppiPoint oAnchor, NppiBorderType eBorderType);

/**
 * Three channel 32-bit floating-point maximum filter with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param oMaskSize Width and Height of the neighborhood region for the local
 *        Max operation.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference
 *        relative to the source pixel.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterMaxBorder_32f_C3R(const Npp32f * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                            NppiSize oMaskSize, NppiPoint oAnchor, NppiBorderType eBorderType);

/**
 * Four channel 32-bit floating-point maximum filter with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param oMaskSize Width and Height of the neighborhood region for the local
 *        Max operation.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference
 *        relative to the source pixel.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiFilterMaxBorder_32f_C4R(const Npp32f * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                            NppiSize oMaskSize, NppiPoint oAnchor, NppiBorderType eBorderType);

/**
 * Four channel 32-bit floating-point maximum filter with border control, ignoring alpha channel.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param oMaskSize Width and Height of the neighborhood region for the local
 *        Max operation.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference
 *        relative to the source pixel.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiFilterMaxBorder_32f_AC4R(const Npp32f * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                             NppiSize oMaskSize, NppiPoint oAnchor, NppiBorderType eBorderType);

/** @} FilterMaxBorder */

/** @name ImageMin Filter
 *
 * Result pixel value is the minimum of pixel values under the rectangular
 * mask region.
 *
 *
 * @{
 *
 */

/**
 * Single channel 8-bit unsigned minimum filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param oMaskSize Width and Height of the neighborhood region for the local
 *        Max operation.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference
 *        relative to the source pixel.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterMin_8u_C1R(const Npp8u * pSrc, Npp32s nSrcStep, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                     NppiSize oMaskSize, NppiPoint oAnchor);

/**
 * Three channel 8-bit unsigned minimum filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param oMaskSize Width and Height of the neighborhood region for the local
 *        Max operation.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference
 *        relative to the source pixel.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterMin_8u_C3R(const Npp8u * pSrc, Npp32s nSrcStep, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                     NppiSize oMaskSize, NppiPoint oAnchor);

/**
 * Four channel 8-bit unsigned minimum filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param oMaskSize Width and Height of the neighborhood region for the local
 *        Max operation.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference
 *        relative to the source pixel.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterMin_8u_C4R(const Npp8u * pSrc, Npp32s nSrcStep, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                     NppiSize oMaskSize, NppiPoint oAnchor);

/**
 * Four channel 8-bit unsigned minimum filter, ignoring alpha channel.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param oMaskSize Width and Height of the neighborhood region for the local
 *        Max operation.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference
 *        relative to the source pixel.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterMin_8u_AC4R(const Npp8u * pSrc, Npp32s nSrcStep, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                      NppiSize oMaskSize, NppiPoint oAnchor);

/**
 * Single channel 16-bit unsigned minimum filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param oMaskSize Width and Height of the neighborhood region for the local
 *        Max operation.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference
 *        relative to the source pixel.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterMin_16u_C1R(const Npp16u * pSrc, Npp32s nSrcStep, Npp16u * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                      NppiSize oMaskSize, NppiPoint oAnchor);

/**
 * Three channel 16-bit unsigned minimum filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param oMaskSize Width and Height of the neighborhood region for the local
 *        Max operation.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference
 *        relative to the source pixel.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterMin_16u_C3R(const Npp16u * pSrc, Npp32s nSrcStep, Npp16u * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                      NppiSize oMaskSize, NppiPoint oAnchor);

/**
 * Four channel 16-bit unsigned minimum filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param oMaskSize Width and Height of the neighborhood region for the local
 *        Max operation.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference
 *        relative to the source pixel.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterMin_16u_C4R(const Npp16u * pSrc, Npp32s nSrcStep, Npp16u * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                      NppiSize oMaskSize, NppiPoint oAnchor);

/**
 * Four channel 16-bit unsigned minimum filter, ignoring alpha channel.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param oMaskSize Width and Height of the neighborhood region for the local
 *        Max operation.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference
 *        relative to the source pixel.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterMin_16u_AC4R(const Npp16u * pSrc, Npp32s nSrcStep, Npp16u * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                       NppiSize oMaskSize, NppiPoint oAnchor);

/**
 * Single channel 16-bit signed minimum filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param oMaskSize Width and Height of the neighborhood region for the local
 *        Max operation.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference
 *        relative to the source pixel.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterMin_16s_C1R(const Npp16s * pSrc, Npp32s nSrcStep, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                      NppiSize oMaskSize, NppiPoint oAnchor);

/**
 * Three channel 16-bit signed minimum filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param oMaskSize Width and Height of the neighborhood region for the local
 *        Max operation.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference
 *        relative to the source pixel.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterMin_16s_C3R(const Npp16s * pSrc, Npp32s nSrcStep, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                      NppiSize oMaskSize, NppiPoint oAnchor);

/**
 * Four channel 16-bit signed minimum filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param oMaskSize Width and Height of the neighborhood region for the local
 *        Max operation.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference
 *        relative to the source pixel.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterMin_16s_C4R(const Npp16s * pSrc, Npp32s nSrcStep, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                      NppiSize oMaskSize, NppiPoint oAnchor);

/**
 * Four channel 16-bit signed minimum filter, ignoring alpha channel.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param oMaskSize Width and Height of the neighborhood region for the local
 *        Max operation.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference
 *        relative to the source pixel.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterMin_16s_AC4R(const Npp16s * pSrc, Npp32s nSrcStep, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                       NppiSize oMaskSize, NppiPoint oAnchor);


/**
 * Single channel 32-bit floating-point minimum filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param oMaskSize Width and Height of the neighborhood region for the local
 *        Max operation.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference
 *        relative to the source pixel.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterMin_32f_C1R(const Npp32f * pSrc, Npp32s nSrcStep, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                      NppiSize oMaskSize, NppiPoint oAnchor);

/**
 * Three channel 32-bit floating-point minimum filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param oMaskSize Width and Height of the neighborhood region for the local
 *        Max operation.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference
 *        relative to the source pixel.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterMin_32f_C3R(const Npp32f * pSrc, Npp32s nSrcStep, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                      NppiSize oMaskSize, NppiPoint oAnchor);

/**
 * Four channel 32-bit floating-point minimum filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param oMaskSize Width and Height of the neighborhood region for the local
 *        Max operation.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference
 *        relative to the source pixel.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterMin_32f_C4R(const Npp32f * pSrc, Npp32s nSrcStep, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                      NppiSize oMaskSize, NppiPoint oAnchor);

/**
 * Four channel 32-bit floating-point minimum filter, ignoring alpha channel.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param oMaskSize Width and Height of the neighborhood region for the local
 *        Max operation.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference
 *        relative to the source pixel.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterMin_32f_AC4R(const Npp32f * pSrc, Npp32s nSrcStep, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                       NppiSize oMaskSize, NppiPoint oAnchor);

/** @} FilterMin */

/** @name ImageMinBorder Filter
 *
 * Result pixel value is the minimum of pixel values under the rectangular
 * mask region. If any portion of the mask overlaps the source
 * image boundary the requested border type operation is applied to all mask pixels
 * which fall outside of the source image.
 *
 * Lwrrently only the NPP_BORDER_REPLICATE border type operation is supported.
 *
 * @{
 *
 */

/**
 * Single channel 8-bit unsigned minimum filter with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param oMaskSize Width and Height of the neighborhood region for the local
 *        Min operation.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference
 *        relative to the source pixel.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterMinBorder_8u_C1R(const Npp8u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                           NppiSize oMaskSize, NppiPoint oAnchor, NppiBorderType eBorderType);

/**
 * Three channel 8-bit unsigned minimum filter with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param oMaskSize Width and Height of the neighborhood region for the local
 *        Min operation.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference
 *        relative to the source pixel.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterMinBorder_8u_C3R(const Npp8u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                           NppiSize oMaskSize, NppiPoint oAnchor, NppiBorderType eBorderType);

/**
 * Four channel 8-bit unsigned minimum filter with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param oMaskSize Width and Height of the neighborhood region for the local
 *        Min operation.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference
 *        relative to the source pixel.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiFilterMinBorder_8u_C4R(const Npp8u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                           NppiSize oMaskSize, NppiPoint oAnchor, NppiBorderType eBorderType);

/**
 * Four channel 8-bit unsigned minimum filter with border control, ignoring alpha channel.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param oMaskSize Width and Height of the neighborhood region for the local
 *        Min operation.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference
 *        relative to the source pixel.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiFilterMinBorder_8u_AC4R(const Npp8u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                            NppiSize oMaskSize, NppiPoint oAnchor, NppiBorderType eBorderType);

/**
 * Single channel 16-bit unsigned minimum filter with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param oMaskSize Width and Height of the neighborhood region for the local
 *        Min operation.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference
 *        relative to the source pixel.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterMinBorder_16u_C1R(const Npp16u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16u * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                            NppiSize oMaskSize, NppiPoint oAnchor, NppiBorderType eBorderType);

/**
 * Three channel 16-bit unsigned minimum filter with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param oMaskSize Width and Height of the neighborhood region for the local
 *        Min operation.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference
 *        relative to the source pixel.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterMinBorder_16u_C3R(const Npp16u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16u * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                            NppiSize oMaskSize, NppiPoint oAnchor, NppiBorderType eBorderType);

/**
 * Four channel 16-bit unsigned minimum filter with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param oMaskSize Width and Height of the neighborhood region for the local
 *        Min operation.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference
 *        relative to the source pixel.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiFilterMinBorder_16u_C4R(const Npp16u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16u * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                            NppiSize oMaskSize, NppiPoint oAnchor, NppiBorderType eBorderType);

/**
 * Four channel 16-bit unsigned minimum filter with border control, ignoring alpha channel.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param oMaskSize Width and Height of the neighborhood region for the local
 *        Min operation.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference
 *        relative to the source pixel.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiFilterMinBorder_16u_AC4R(const Npp16u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16u * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                             NppiSize oMaskSize, NppiPoint oAnchor, NppiBorderType eBorderType);

/**
 * Single channel 16-bit signed minimum filter with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param oMaskSize Width and Height of the neighborhood region for the local
 *        Min operation.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference
 *        relative to the source pixel.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterMinBorder_16s_C1R(const Npp16s * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                            NppiSize oMaskSize, NppiPoint oAnchor, NppiBorderType eBorderType);

/**
 * Three channel 16-bit signed minimum filter with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param oMaskSize Width and Height of the neighborhood region for the local
 *        Min operation.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference
 *        relative to the source pixel.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterMinBorder_16s_C3R(const Npp16s * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                            NppiSize oMaskSize, NppiPoint oAnchor, NppiBorderType eBorderType);

/**
 * Four channel 16-bit signed minimum filter with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param oMaskSize Width and Height of the neighborhood region for the local
 *        Min operation.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference
 *        relative to the source pixel.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiFilterMinBorder_16s_C4R(const Npp16s * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                            NppiSize oMaskSize, NppiPoint oAnchor, NppiBorderType eBorderType);

/**
 * Four channel 16-bit signed minimum filter with border control, ignoring alpha channel.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param oMaskSize Width and Height of the neighborhood region for the local
 *        Min operation.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference
 *        relative to the source pixel.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiFilterMinBorder_16s_AC4R(const Npp16s * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                             NppiSize oMaskSize, NppiPoint oAnchor, NppiBorderType eBorderType);

/**
 * Single channel 32-bit floating-point minimum filter with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param oMaskSize Width and Height of the neighborhood region for the local
 *        Min operation.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference
 *        relative to the source pixel.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterMinBorder_32f_C1R(const Npp32f * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                            NppiSize oMaskSize, NppiPoint oAnchor, NppiBorderType eBorderType);

/**
 * Three channel 32-bit floating-point minimum filter with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param oMaskSize Width and Height of the neighborhood region for the local
 *        Min operation.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference
 *        relative to the source pixel.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterMinBorder_32f_C3R(const Npp32f * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                            NppiSize oMaskSize, NppiPoint oAnchor, NppiBorderType eBorderType);

/**
 * Four channel 32-bit floating-point minimum filter with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param oMaskSize Width and Height of the neighborhood region for the local
 *        Min operation.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference
 *        relative to the source pixel.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiFilterMinBorder_32f_C4R(const Npp32f * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                            NppiSize oMaskSize, NppiPoint oAnchor, NppiBorderType eBorderType);

/**
 * Four channel 32-bit floating-point minimum filter with border control, ignoring alpha channel.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param oMaskSize Width and Height of the neighborhood region for the local
 *        Min operation.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference
 *        relative to the source pixel.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiFilterMinBorder_32f_AC4R(const Npp32f * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                             NppiSize oMaskSize, NppiPoint oAnchor, NppiBorderType eBorderType);

/** @} FilterMinBorder */

/** @name ImageMedian Filter
 *
 * Result pixel value is the median of pixel values under the rectangular
 * mask region.
 *
 *
 * @{
 *
 */

/**
 * Single channel 8-bit unsigned median filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param oMaskSize Width and Height of the neighborhood region for the local
 *        Median operation.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference
 *        relative to the source pixel.
 * \param pBuffer Pointer to the user-allocated scratch buffer required for the Median operation.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterMedian_8u_C1R(const Npp8u * pSrc, Npp32s nSrcStep, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                     NppiSize oMaskSize, NppiPoint oAnchor, Npp8u * pBuffer);

/**
 * Three channel 8-bit unsigned median filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param oMaskSize Width and Height of the neighborhood region for the local
 *        Median operation.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference
 *        relative to the source pixel.
 * \param pBuffer Pointer to the user-allocated scratch buffer required for the Median operation.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterMedian_8u_C3R(const Npp8u * pSrc, Npp32s nSrcStep, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                     NppiSize oMaskSize, NppiPoint oAnchor, Npp8u * pBuffer);

/**
 * Four channel 8-bit unsigned median filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param oMaskSize Width and Height of the neighborhood region for the local
 *        Median operation.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference
 *        relative to the source pixel.
 * \param pBuffer Pointer to the user-allocated scratch buffer required for the Median operation.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterMedian_8u_C4R(const Npp8u * pSrc, Npp32s nSrcStep, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                     NppiSize oMaskSize, NppiPoint oAnchor, Npp8u * pBuffer);

/**
 * Four channel 8-bit unsigned median filter, ignoring alpha channel.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param oMaskSize Width and Height of the neighborhood region for the local
 *        Median operation.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference
 *        relative to the source pixel.
 * \param pBuffer Pointer to the user-allocated scratch buffer required for the Median operation.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterMedian_8u_AC4R(const Npp8u * pSrc, Npp32s nSrcStep, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                      NppiSize oMaskSize, NppiPoint oAnchor, Npp8u * pBuffer);

/**
 * Single channel 16-bit unsigned median filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param oMaskSize Width and Height of the neighborhood region for the local
 *        Median operation.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference
 *        relative to the source pixel.
 * \param pBuffer Pointer to the user-allocated scratch buffer required for the Median operation.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterMedian_16u_C1R(const Npp16u * pSrc, Npp32s nSrcStep, Npp16u * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                      NppiSize oMaskSize, NppiPoint oAnchor, Npp8u * pBuffer);

/**
 * Three channel 16-bit unsigned median filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param oMaskSize Width and Height of the neighborhood region for the local
 *        Median operation.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference
 *        relative to the source pixel.
 * \param pBuffer Pointer to the user-allocated scratch buffer required for the Median operation.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterMedian_16u_C3R(const Npp16u * pSrc, Npp32s nSrcStep, Npp16u * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                      NppiSize oMaskSize, NppiPoint oAnchor, Npp8u * pBuffer);

/**
 * Four channel 16-bit unsigned median filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param oMaskSize Width and Height of the neighborhood region for the local
 *        Median operation.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference
 *        relative to the source pixel.
 * \param pBuffer Pointer to the user-allocated scratch buffer required for the Median operation.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterMedian_16u_C4R(const Npp16u * pSrc, Npp32s nSrcStep, Npp16u * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                      NppiSize oMaskSize, NppiPoint oAnchor, Npp8u * pBuffer);

/**
 * Four channel 16-bit unsigned median filter, ignoring alpha channel.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param oMaskSize Width and Height of the neighborhood region for the local
 *        Median operation.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference
 *        relative to the source pixel.
 * \param pBuffer Pointer to the user-allocated scratch buffer required for the Median operation.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterMedian_16u_AC4R(const Npp16u * pSrc, Npp32s nSrcStep, Npp16u * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                       NppiSize oMaskSize, NppiPoint oAnchor, Npp8u * pBuffer);

/**
 * Single channel 16-bit signed median filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param oMaskSize Width and Height of the neighborhood region for the local
 *        Median operation.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference
 *        relative to the source pixel.
 * \param pBuffer Pointer to the user-allocated scratch buffer required for the Median operation.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterMedian_16s_C1R(const Npp16s * pSrc, Npp32s nSrcStep, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                      NppiSize oMaskSize, NppiPoint oAnchor, Npp8u * pBuffer);

/**
 * Three channel 16-bit signed median filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param oMaskSize Width and Height of the neighborhood region for the local
 *        Median operation.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference
 *        relative to the source pixel.
 * \param pBuffer Pointer to the user-allocated scratch buffer required for the Median operation.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterMedian_16s_C3R(const Npp16s * pSrc, Npp32s nSrcStep, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                      NppiSize oMaskSize, NppiPoint oAnchor, Npp8u * pBuffer);

/**
 * Four channel 16-bit signed median filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param oMaskSize Width and Height of the neighborhood region for the local
 *        Median operation.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference
 *        relative to the source pixel.
 * \param pBuffer Pointer to the user-allocated scratch buffer required for the Median operation.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterMedian_16s_C4R(const Npp16s * pSrc, Npp32s nSrcStep, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                      NppiSize oMaskSize, NppiPoint oAnchor, Npp8u * pBuffer);

/**
 * Four channel 16-bit signed median filter, ignoring alpha channel.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param oMaskSize Width and Height of the neighborhood region for the local
 *        Median operation.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference
 *        relative to the source pixel.
 * \param pBuffer Pointer to the user-allocated scratch buffer required for the Median operation.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterMedian_16s_AC4R(const Npp16s * pSrc, Npp32s nSrcStep, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                       NppiSize oMaskSize, NppiPoint oAnchor, Npp8u * pBuffer);


/**
 * Single channel 32-bit floating-point median filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param oMaskSize Width and Height of the neighborhood region for the local
 *        Median operation.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference
 *        relative to the source pixel.
 * \param pBuffer Pointer to the user-allocated scratch buffer required for the Median operation.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterMedian_32f_C1R(const Npp32f * pSrc, Npp32s nSrcStep, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                      NppiSize oMaskSize, NppiPoint oAnchor, Npp8u * pBuffer);

/**
 * Three channel 32-bit floating-point median filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param oMaskSize Width and Height of the neighborhood region for the local
 *        Median operation.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference
 *        relative to the source pixel.
 * \param pBuffer Pointer to the user-allocated scratch buffer required for the Median operation.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterMedian_32f_C3R(const Npp32f * pSrc, Npp32s nSrcStep, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                      NppiSize oMaskSize, NppiPoint oAnchor, Npp8u * pBuffer);

/**
 * Four channel 32-bit floating-point median filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param oMaskSize Width and Height of the neighborhood region for the local
 *        Median operation.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference
 *        relative to the source pixel.
 * \param pBuffer Pointer to the user-allocated scratch buffer required for the Median operation.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterMedian_32f_C4R(const Npp32f * pSrc, Npp32s nSrcStep, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                      NppiSize oMaskSize, NppiPoint oAnchor, Npp8u * pBuffer);

/**
 * Four channel 32-bit floating-point median filter, ignoring alpha channel.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param oMaskSize Width and Height of the neighborhood region for the local
 *        Median operation.
 * \param oAnchor X and Y offsets of the kernel origin frame of reference
 *        relative to the source pixel.
 * \param pBuffer Pointer to the user-allocated scratch buffer required for the Median operation.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterMedian_32f_AC4R(const Npp32f * pSrc, Npp32s nSrcStep, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                       NppiSize oMaskSize, NppiPoint oAnchor, Npp8u * pBuffer);
                       






/**
 * Single channel 8-bit unsigned median filter scratch memory size.
 * \param oSizeROI \ref roi_specification.
 * \param oMaskSize Width and Height of the neighborhood region for the local Median operation.
 * \param nBufferSize Pointer to the size of the scratch buffer required for the Median operation.
 * \return \ref image_data_error_codes
 */
NppStatus 
nppiFilterMedianGetBufferSize_8u_C1R(NppiSize oSizeROI, NppiSize oMaskSize, Npp32u * nBufferSize);

/**
 * Three channel 8-bit unsigned median filter scratch memory size.
 * \param oSizeROI \ref roi_specification.
 * \param oMaskSize Width and Height of the neighborhood region for the local Median operation.
 * \param nBufferSize Pointer to the size of the scratch buffer required for the Median operation.
 * \return \ref image_data_error_codes
 */
NppStatus 
nppiFilterMedianGetBufferSize_8u_C3R(NppiSize oSizeROI, NppiSize oMaskSize, Npp32u * nBufferSize);

/**
 * Four channel 8-bit unsigned median filter scratch memory size.
 * \param oSizeROI \ref roi_specification.
 * \param oMaskSize Width and Height of the neighborhood region for the local Median operation.
 * \param nBufferSize Pointer to the size of the scratch buffer required for the Median operation.
 * \return \ref image_data_error_codes
 */
NppStatus 
nppiFilterMedianGetBufferSize_8u_C4R(NppiSize oSizeROI, NppiSize oMaskSize, Npp32u * nBufferSize);

/**
 * Four channel 8-bit unsigned median filter, ignoring alpha channel.
 * \param oSizeROI \ref roi_specification.
 * \param oMaskSize Width and Height of the neighborhood region for the local Median operation.
 * \param nBufferSize Pointer to the size of the scratch buffer required for the Median operation.
 * \return \ref image_data_error_codes
 */
NppStatus 
nppiFilterMedianGetBufferSize_8u_AC4R(NppiSize oSizeROI, NppiSize oMaskSize, Npp32u * nBufferSize);

/**
 * Single channel 16-bit unsigned median filter scratch memory size.
 * \param oSizeROI \ref roi_specification.
 * \param oMaskSize Width and Height of the neighborhood region for the local Median operation.
 * \param nBufferSize Pointer to the size of the scratch buffer required for the Median operation.
 * \return \ref image_data_error_codes
 */
NppStatus 
nppiFilterMedianGetBufferSize_16u_C1R(NppiSize oSizeROI, NppiSize oMaskSize, Npp32u * nBufferSize);

/**
 * Three channel 16-bit unsigned median filter scratch memory size.
 * \param oSizeROI \ref roi_specification.
 * \param oMaskSize Width and Height of the neighborhood region for the local Median operation.
 * \param nBufferSize Pointer to the size of the scratch buffer required for the Median operation.
 * \return \ref image_data_error_codes
 */
NppStatus 
nppiFilterMedianGetBufferSize_16u_C3R(NppiSize oSizeROI, NppiSize oMaskSize, Npp32u * nBufferSize);

/**
 * Four channel 16-bit unsigned median filter scratch memory size.
 * \param oSizeROI \ref roi_specification.
 * \param oMaskSize Width and Height of the neighborhood region for the local Median operation.
 * \param nBufferSize Pointer to the size of the scratch buffer required for the Median operation.
 * \return \ref image_data_error_codes
 */
NppStatus 
nppiFilterMedianGetBufferSize_16u_C4R(NppiSize oSizeROI, NppiSize oMaskSize, Npp32u * nBufferSize);

/**
 * Four channel 16-bit unsigned median filter, ignoring alpha channel.
 * \param oSizeROI \ref roi_specification.
 * \param oMaskSize Width and Height of the neighborhood region for the local Median operation.
 * \param nBufferSize Pointer to the size of the scratch buffer required for the Median operation.
 * \return \ref image_data_error_codes
 */
NppStatus 
nppiFilterMedianGetBufferSize_16u_AC4R(NppiSize oSizeROI, NppiSize oMaskSize, Npp32u * nBufferSize);

/**
 * Single channel 16-bit signed median filter scratch memory size.
 * \param oSizeROI \ref roi_specification.
 * \param oMaskSize Width and Height of the neighborhood region for the local Median operation.
 * \param nBufferSize Pointer to the size of the scratch buffer required for the Median operation.
 * \return \ref image_data_error_codes
 */
NppStatus 
nppiFilterMedianGetBufferSize_16s_C1R(NppiSize oSizeROI, NppiSize oMaskSize, Npp32u * nBufferSize);

/**
 * Three channel 16-bit signed median filter scratch memory size.
 * \param oSizeROI \ref roi_specification.
 * \param oMaskSize Width and Height of the neighborhood region for the local Median operation.
 * \param nBufferSize Pointer to the size of the scratch buffer required for the Median operation.
 * \return \ref image_data_error_codes
 */
NppStatus 
nppiFilterMedianGetBufferSize_16s_C3R(NppiSize oSizeROI, NppiSize oMaskSize, Npp32u * nBufferSize);

/**
 * Four channel 16-bit signed median filter scratch memory size.
 * \param oSizeROI \ref roi_specification.
 * \param oMaskSize Width and Height of the neighborhood region for the local Median operation.
 * \param nBufferSize Pointer to the size of the scratch buffer required for the Median operation.
 * \return \ref image_data_error_codes
 */
NppStatus 
nppiFilterMedianGetBufferSize_16s_C4R(NppiSize oSizeROI, NppiSize oMaskSize, Npp32u * nBufferSize);

/**
 * Four channel 16-bit signed median filter, ignoring alpha channel.
 * \param oSizeROI \ref roi_specification.
 * \param oMaskSize Width and Height of the neighborhood region for the local Median operation.
 * \param nBufferSize Pointer to the size of the scratch buffer required for the Median operation.
 * \return \ref image_data_error_codes
 */
NppStatus 
nppiFilterMedianGetBufferSize_16s_AC4R(NppiSize oSizeROI, NppiSize oMaskSize, Npp32u * nBufferSize);


/**
 * Single channel 32-bit floating-point median filter scratch memory size.
 * \param oSizeROI \ref roi_specification.
 * \param oMaskSize Width and Height of the neighborhood region for the local Median operation.
 * \param nBufferSize Pointer to the size of the scratch buffer required for the Median operation.
 * \return \ref image_data_error_codes
 */
NppStatus 
nppiFilterMedianGetBufferSize_32f_C1R(NppiSize oSizeROI, NppiSize oMaskSize, Npp32u * nBufferSize);

/**
 * Three channel 32-bit floating-point median filter scratch memory size.
 * \param oSizeROI \ref roi_specification.
 * \param oMaskSize Width and Height of the neighborhood region for the local Median operation.
 * \param nBufferSize Pointer to the size of the scratch buffer required for the Median operation.
 * \return \ref image_data_error_codes
 */
NppStatus 
nppiFilterMedianGetBufferSize_32f_C3R(NppiSize oSizeROI, NppiSize oMaskSize, Npp32u * nBufferSize);

/**
 * Four channel 32-bit floating-point median filter scratch memory size.
 * \param oSizeROI \ref roi_specification.
 * \param oMaskSize Width and Height of the neighborhood region for the local Median operation.
 * \param nBufferSize Pointer to the size of the scratch buffer required for the Median operation.
 * \return \ref image_data_error_codes
 */
NppStatus 
nppiFilterMedianGetBufferSize_32f_C4R(NppiSize oSizeROI, NppiSize oMaskSize, Npp32u * nBufferSize);

/**
 * Four channel 32-bit floating-point median filter, ignoring alpha channel.
 * \param oSizeROI \ref roi_specification.
 * \param oMaskSize Width and Height of the neighborhood region for the local Median operation.
 * \param nBufferSize Pointer to the size of the scratch buffer required for the Median operation.
 * \return \ref image_data_error_codes
 */
NppStatus 
nppiFilterMedianGetBufferSize_32f_AC4R(NppiSize oSizeROI, NppiSize oMaskSize, Npp32u * nBufferSize);


/** @} FilterMedian */



/** @} image_rank_filters */

/** @defgroup fixed_filters Fixed Filters
 *
 * Fixed filters perform linear filtering operations (such as colwolutions) with predefined kernels
 * of fixed sizes.  Note that this section also contains a few dynamic kernel filters, namely GaussAdvanced and Bilateral.
 * 
 * Some of the fixed filters have versions with border control.   For these functions, if any portion 
 * of the mask overlaps the source image boundary the requested border type operation is applied to 
 * all mask pixels which fall outside of the source image.
 *
 * Lwrrently only the NPP_BORDER_REPLICATE border type operation is supported for these functions.
 *
 * @{
 *
 */

/** @name FilterPrewittHoriz 
 *
 * Filters the image using a horizontal Prewitt filter kernel:
 *
 * \f[
 *  \left( \begin{array}{rrr}
 *    1 &  1 &  1 \\
 *    0 &  0 &  0 \\
 *   -1 & -1 & -1 \\
 *  \end{array} \right)
 * \f]
 *
 * @{
 *
 */

/**
 * Single channel 8-bit unsigned horizontal Prewitt filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterPrewittHoriz_8u_C1R(const Npp8u * pSrc, Npp32s nSrcStep, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI);

/**
 * Three channel 8-bit unsigned horizontal Prewitt filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterPrewittHoriz_8u_C3R(const Npp8u * pSrc, Npp32s nSrcStep, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI);

/**
 * Four channel 8-bit unsigned horizontal Prewitt filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterPrewittHoriz_8u_C4R(const Npp8u * pSrc, Npp32s nSrcStep, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI);

/**
 * Four channel 8-bit unsigned horizontal Prewitt filter, ignoring alpha channel.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterPrewittHoriz_8u_AC4R(const Npp8u * pSrc, Npp32s nSrcStep, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI);

/**
 * Single channel 16-bit signed horizontal Prewitt filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterPrewittHoriz_16s_C1R(const Npp16s * pSrc, Npp32s nSrcStep, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI);

/**
 * Three channel 16-bit signed horizontal Prewitt filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterPrewittHoriz_16s_C3R(const Npp16s * pSrc, Npp32s nSrcStep, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI);

/**
 * Four channel 16-bit signed horizontal Prewitt filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterPrewittHoriz_16s_C4R(const Npp16s * pSrc, Npp32s nSrcStep, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI);

/**
 * Four channel 16-bit signed horizontal Prewitt filter, ignoring alpha channel.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterPrewittHoriz_16s_AC4R(const Npp16s * pSrc, Npp32s nSrcStep, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI);

/**
 * Single channel 32-bit floating-point horizontal Prewitt filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterPrewittHoriz_32f_C1R(const Npp32f * pSrc, Npp32s nSrcStep, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI);

/**
 * Three channel 32-bit floating-point horizontal Prewitt filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterPrewittHoriz_32f_C3R(const Npp32f * pSrc, Npp32s nSrcStep, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI);

/**
 * Four channel 32-bit floating-point horizontal Prewitt filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterPrewittHoriz_32f_C4R(const Npp32f * pSrc, Npp32s nSrcStep, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI);

/**
 * Four channel 32-bit floating-point horizontal Prewitt filter, ignoring alpha channel.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterPrewittHoriz_32f_AC4R(const Npp32f * pSrc, Npp32s nSrcStep, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI);

/** @} FilterPrewittHoriz */

/** @name FilterPrewittHorizBorder 
 *
 * Filters the image using a horizontal Prewitt filter kernel with border control. If any portion of the mask overlaps the source
 * image boundary the requested border type operation is applied to all mask pixels
 * which fall outside of the source image.
 *
 * Lwrrently only the NPP_BORDER_REPLICATE border type operation is supported.
 *
 * \f[
 *  \left( \begin{array}{rrr}
 *    1 &  1 &  1 \\
 *    0 &  0 &  0 \\
 *   -1 & -1 & -1 \\
 *  \end{array} \right)
 * \f]
 *
 * @{
 *
 */

/**
 * Single channel 8-bit unsigned horizontal Prewitt filter with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterPrewittHorizBorder_8u_C1R(const Npp8u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiBorderType eBorderType);

/**
 * Three channel 8-bit unsigned horizontal Prewitt filter with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterPrewittHorizBorder_8u_C3R(const Npp8u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiBorderType eBorderType);

/**
 * Four channel 8-bit unsigned horizontal Prewitt filter with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterPrewittHorizBorder_8u_C4R(const Npp8u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiBorderType eBorderType);

/**
 * Four channel 8-bit unsigned horizontal Prewitt filter with border control, ignoring alpha channel.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterPrewittHorizBorder_8u_AC4R(const Npp8u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiBorderType eBorderType);

/**
 * Single channel 16-bit signed horizontal Prewitt filter with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterPrewittHorizBorder_16s_C1R(const Npp16s * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiBorderType eBorderType);

/**
 * Three channel 16-bit signed horizontal Prewitt filter with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterPrewittHorizBorder_16s_C3R(const Npp16s * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiBorderType eBorderType);

/**
 * Four channel 16-bit signed horizontal Prewitt filter with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterPrewittHorizBorder_16s_C4R(const Npp16s * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiBorderType eBorderType);

/**
 * Four channel 16-bit signed horizontal Prewitt filter with border control, ignoring alpha channel.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterPrewittHorizBorder_16s_AC4R(const Npp16s * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiBorderType eBorderType);

/**
 * Single channel 32-bit floating-point horizontal Prewitt filter with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterPrewittHorizBorder_32f_C1R(const Npp32f * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiBorderType eBorderType);

/**
 * Three channel 32-bit floating-point horizontal Prewitt filter with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterPrewittHorizBorder_32f_C3R(const Npp32f * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiBorderType eBorderType);

/**
 * Four channel 32-bit floating-point horizontal Prewitt filter with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterPrewittHorizBorder_32f_C4R(const Npp32f * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiBorderType eBorderType);

/**
 * Four channel 32-bit floating-point horizontal Prewitt filter with border control, ignoring alpha channel.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterPrewittHorizBorder_32f_AC4R(const Npp32f * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiBorderType eBorderType);

/** @} FilterPrewittHorizBorder */

/** @name FilterPrewittVert 
 *
 * Filters the image using a vertical Prewitt filter kernel:
 *
 * \f[
 *  \left( \begin{array}{rrr}
 *   -1 & 0 & 1 \\
 *   -1 & 0 & 1 \\
 *   -1 & 0 & 1 \\
 *  \end{array} \right)
 * \f]
 *
 * @{
 *
 */

/**
 * Single channel 8-bit unsigned vertical Prewitt filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterPrewittVert_8u_C1R(const Npp8u * pSrc, Npp32s nSrcStep, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI);

/**
 * Three channel 8-bit unsigned vertical Prewitt filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterPrewittVert_8u_C3R(const Npp8u * pSrc, Npp32s nSrcStep, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI);

/**
 * Four channel 8-bit unsigned vertical Prewitt filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterPrewittVert_8u_C4R(const Npp8u * pSrc, Npp32s nSrcStep, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI);

/**
 * Four channel 8-bit unsigned vertical Prewitt filter, ignoring alpha channel.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterPrewittVert_8u_AC4R(const Npp8u * pSrc, Npp32s nSrcStep, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI);

/**
 * Single channel 16-bit signed vertical Prewitt filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterPrewittVert_16s_C1R(const Npp16s * pSrc, Npp32s nSrcStep, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI);

/**
 * Three channel 16-bit signed vertical Prewitt filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterPrewittVert_16s_C3R(const Npp16s * pSrc, Npp32s nSrcStep, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI);

/**
 * Four channel 16-bit signed vertical Prewitt filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterPrewittVert_16s_C4R(const Npp16s * pSrc, Npp32s nSrcStep, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI);

/**
 * Four channel 16-bit signed vertical Prewitt filter, ignoring alpha channel.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterPrewittVert_16s_AC4R(const Npp16s * pSrc, Npp32s nSrcStep, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI);

/**
 * Single channel 32-bit floating-point vertical Prewitt filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterPrewittVert_32f_C1R(const Npp32f * pSrc, Npp32s nSrcStep, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI);

/**
 * Three channel 32-bit floating-point vertical Prewitt filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterPrewittVert_32f_C3R(const Npp32f * pSrc, Npp32s nSrcStep, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI);

/**
 * Four channel 32-bit floating-point vertical Prewitt filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterPrewittVert_32f_C4R(const Npp32f * pSrc, Npp32s nSrcStep, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI);

/**
 * Four channel 32-bit floating-point vertical Prewitt filter, ignoring alpha channel.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterPrewittVert_32f_AC4R(const Npp32f * pSrc, Npp32s nSrcStep, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI);

/** @} FilterPrewittVert */

/** @name FilterPrewittVertBorder 
 *
 * Filters the image using a vertical Prewitt filter kernel with border control. If any portion of the mask overlaps the source
 * image boundary the requested border type operation is applied to all mask pixels
 * which fall outside of the source image.
 *
 * Lwrrently only the NPP_BORDER_REPLICATE border type operation is supported.
 *
 * \f[
 *  \left( \begin{array}{rrr}
 *   -1 & 0 & 1 \\
 *   -1 & 0 & 1 \\
 *   -1 & 0 & 1 \\
 *  \end{array} \right);
 * \f]
 *
 * @{
 *
 */

/**
 * Single channel 8-bit unsigned vertical Prewitt filter with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterPrewittVertBorder_8u_C1R(const Npp8u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiBorderType eBorderType);

/**
 * Three channel 8-bit unsigned vertical Prewitt filter with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterPrewittVertBorder_8u_C3R(const Npp8u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiBorderType eBorderType);

/**
 * Four channel 8-bit unsigned vertical Prewitt filter with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterPrewittVertBorder_8u_C4R(const Npp8u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiBorderType eBorderType);

/**
 * Four channel 8-bit unsigned vertical Prewitt filter with border control, ignoring alpha channel.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterPrewittVertBorder_8u_AC4R(const Npp8u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiBorderType eBorderType);

/**
 * Single channel 16-bit signed vertical Prewitt filter with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterPrewittVertBorder_16s_C1R(const Npp16s * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiBorderType eBorderType);

/**
 * Three channel 16-bit signed vertical Prewitt filter with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterPrewittVertBorder_16s_C3R(const Npp16s * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiBorderType eBorderType);

/**
 * Four channel 16-bit signed vertical Prewitt filter with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterPrewittVertBorder_16s_C4R(const Npp16s * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiBorderType eBorderType);

/**
 * Four channel 16-bit signed vertical Prewitt filter with border control, ignoring alpha channel.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterPrewittVertBorder_16s_AC4R(const Npp16s * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiBorderType eBorderType);

/**
 * Single channel 32-bit floating-point vertical Prewitt filter with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterPrewittVertBorder_32f_C1R(const Npp32f * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiBorderType eBorderType);

/**
 * Three channel 32-bit floating-point vertical Prewitt filter with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterPrewittVertBorder_32f_C3R(const Npp32f * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiBorderType eBorderType);

/**
 * Four channel 32-bit floating-point vertical Prewitt filter with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterPrewittVertBorder_32f_C4R(const Npp32f * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiBorderType eBorderType);

/**
 * Four channel 32-bit floating-point vertical Prewitt filter with border control, ignoring alpha channel.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterPrewittVertBorder_32f_AC4R(const Npp32f * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiBorderType eBorderType);

/** @} FilterPrewittVertBorder */


/** @name FilterScharrHoriz 
 *
 * Filters the image using a horizontal Scharr filter kernel:
 *
 * \f[
 *  \left( \begin{array}{rrr}
 *    3 &  10 &  3 \\
 *    0 &   0 &  0 \\
 *   -3 & -10 & -3 \\
 *  \end{array} \right)
 * \f]
 *
 * @{
 *
 */

/**
 * Single channel 8-bit unsigned to 16-bit signed horizontal Scharr filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterScharrHoriz_8u16s_C1R(const Npp8u * pSrc, Npp32s nSrcStep, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI);

/**
 * Single channel 8-bit signed to 16-bit signed horizontal Scharr filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterScharrHoriz_8s16s_C1R(const Npp8s * pSrc, Npp32s nSrcStep, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI);

/**
 * Single channel 32-bit floating-point horizontal Scharr filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterScharrHoriz_32f_C1R(const Npp32f * pSrc, Npp32s nSrcStep, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI);

/** @} FilterScharrHoriz */

/** @name FilterScharrVert 
 *
 * Filters the image using a vertical Scharr filter kernel:
 *
 * \f[
 *  \left( \begin{array}{rrr}
 *     3 &   0 &  -3 \\
 *    10 &   0 & -10 \\
 *     3 &   0 &  -3 \\
 *  \end{array} \right)
 * \f]
 *
 * @{
 *
 */

/**
 * Single channel 8-bit unsigned to 16-bit signed vertical Scharr filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterScharrVert_8u16s_C1R(const Npp8u * pSrc, Npp32s nSrcStep, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI);

/**
 * Single channel 8-bit signed to 16-bit signed vertical Scharr filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterScharrVert_8s16s_C1R(const Npp8s * pSrc, Npp32s nSrcStep, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI);

/**
 * Single channel 32-bit floating-point vertical Scharr filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterScharrVert_32f_C1R(const Npp32f * pSrc, Npp32s nSrcStep, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI);

/** @} FilterScharrVert */

/** @name FilterScharrHorizBorder 
 *
 * Filters the image using a horizontal Scharr filter kernel with border control:
 *
 * \f[
 *  \left( \begin{array}{rrr}
 *    3 &  10 &  3 \\
 *    0 &   0 &  0 \\
 *   -3 & -10 & -3 \\
 *  \end{array} \right)
 * \f]
 *
 * @{
 *
 */

/**
 * Single channel 8-bit unsigned to 16-bit signed horizontal Scharr filter kernel with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterScharrHorizBorder_8u16s_C1R(const Npp8u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiBorderType eBorderType);

/**
 * Single channel 8-bit signed to 16-bit signed horizontal Scharr filter kernel with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterScharrHorizBorder_8s16s_C1R(const Npp8s * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiBorderType eBorderType);

/**
 * Single channel 32-bit floating-point horizontal Scharr filter kernel with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterScharrHorizBorder_32f_C1R(const Npp32f * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiBorderType eBorderType);

/** @} FilterScharrHorizBorder */

/** @name FilterScharrVertBorder 
 *
 * Filters the image using a vertical Scharr filter kernel kernel with border control:
 *
 * \f[
 *  \left( \begin{array}{rrr}
 *     3 &   0 &  -3 \\
 *    10 &   0 & -10 \\
 *     3 &   0 &  -3 \\
 *  \end{array} \right)
 * \f]
 *
 * @{
 *
 */

/**
 * Single channel 8-bit unsigned to 16-bit signed vertical Scharr filter kernel with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterScharrVertBorder_8u16s_C1R(const Npp8u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiBorderType eBorderType);

/**
 * Single channel 8-bit signed to 16-bit signed vertical Scharr filter kernel with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterScharrVertBorder_8s16s_C1R(const Npp8s * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiBorderType eBorderType);

/**
 * Single channel 32-bit floating-point vertical Scharr filter kernel with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterScharrVertBorder_32f_C1R(const Npp32f * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiBorderType eBorderType);

/** @} FilterScharrVertBorder */


/** @name FilterSobelHoriz 
 *              
 * Filters the image using a horizontal Sobel filter kernel:
 *
 * \f[
 *  \left( \begin{array}{rrr}
 *    1 &  2 &  1 \\
 *    0 &  0 &  0 \\
 *   -1 & -2 & -1 \\
 *  \end{array} \right)
 *  \left( \begin{array}{rrrrr}
 *    1  &  4 &   6 &  4 &  1 \\
 *    2  &  8 &  12 &  8 &  2 \\
 *    0  &  0 &   0 &  0 &  0 \\
 *    -2 & -8 & -12 & -8 & -2 \\
 *    -1 & -4 &  -6 & -4 & -1 \\
 *  \end{array} \right)
 * \f]
 *
 * @{
 *
 */

/**
 * Single channel 8-bit unsigned horizontal Sobel filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterSobelHoriz_8u_C1R(const Npp8u * pSrc, Npp32s nSrcStep, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI);

/**
 * Three channel 8-bit unsigned horizontal Sobel filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterSobelHoriz_8u_C3R(const Npp8u * pSrc, Npp32s nSrcStep, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI);

/**
 * Four channel 8-bit unsigned horizontal Sobel filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterSobelHoriz_8u_C4R(const Npp8u * pSrc, Npp32s nSrcStep, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI);

/**
 * Four channel 16-bit signed horizontal Sobel filter, ignoring alpha channel.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterSobelHoriz_8u_AC4R(const Npp8u * pSrc, Npp32s nSrcStep, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI);

/**
 * Single channel 16-bit signed horizontal Sobel filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterSobelHoriz_16s_C1R(const Npp16s * pSrc, Npp32s nSrcStep, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI);

/**
 * Three channel 16-bit signed horizontal Sobel filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterSobelHoriz_16s_C3R(const Npp16s * pSrc, Npp32s nSrcStep, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI);

/**
 * Four channel 16-bit signed horizontal Sobel filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterSobelHoriz_16s_C4R(const Npp16s * pSrc, Npp32s nSrcStep, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI);

/**
 * Four channel 8-bit unsigned horizontal Sobel filter, ignoring alpha channel.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterSobelHoriz_16s_AC4R(const Npp16s * pSrc, Npp32s nSrcStep, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI);

/**
 * Single channel 32-bit floating-point horizontal Sobel filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterSobelHoriz_32f_C1R(const Npp32f * pSrc, Npp32s nSrcStep, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI);

/**
 * Three channel 32-bit floating-point horizontal Sobel filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterSobelHoriz_32f_C3R(const Npp32f * pSrc, Npp32s nSrcStep, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI);

/**
 * Four channel 32-bit floating-point horizontal Sobel filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterSobelHoriz_32f_C4R(const Npp32f * pSrc, Npp32s nSrcStep, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI);

/**
 * Four channel 32-bit floating-point horizontal Sobel filter, ignoring alpha channel.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterSobelHoriz_32f_AC4R(const Npp32f * pSrc, Npp32s nSrcStep, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI);

/**
 * Single channel 8-bit unsigned to 16-bit signed horizontal Sobel filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eMaskSize Enumeration value specifying the mask size.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterSobelHoriz_8u16s_C1R(const Npp8u * pSrc, Npp32s nSrcStep, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                               NppiMaskSize eMaskSize);

/**
 * Single channel 8-bit signed to 16-bit signed horizontal Sobel filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eMaskSize Enumeration value specifying the mask size.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterSobelHoriz_8s16s_C1R(const Npp8s * pSrc, Npp32s nSrcStep, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                               NppiMaskSize eMaskSize);

/**
 * Single channel 32-bit floating-point horizontal Sobel filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eMaskSize Enumeration value specifying the mask size.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterSobelHorizMask_32f_C1R(const Npp32f * pSrc, Npp32s nSrcStep, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                                 NppiMaskSize eMaskSize);


/** @} FilterSobelHoriz */

/** @name FilterSobelVert 
 *
 * Filters the image using a vertical Sobel filter kernel:
 *
 * \f[
 *  \left( \begin{array}{rrr}
 *    -1 & 0 & 1 \\
 *    -2 & 0 & 2 \\
 *    -1 & 0 & 1 \\
 *  \end{array} \right)
 *  \left( \begin{array}{rrrrr}
 *    -1 &  -2 & 0 &  2 & 1 \\
 *    -4 &  -8 & 0 &  8 & 4 \\
 *    -6 & -12 & 0 & 12 & 6 \\
 *    -4 &  -8 & 0 &  8 & 4 \\
 *    -1 &  -2 & 0 &  2 & 1 \\
 *  \end{array} \right)
 * \f]
 *
 * @{
 *
 */

/**
 * Single channel 8-bit unsigned vertical Sobel filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterSobelVert_8u_C1R(const Npp8u * pSrc, Npp32s nSrcStep, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI);

/**
 * Three channel 8-bit unsigned vertical Sobel filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterSobelVert_8u_C3R(const Npp8u * pSrc, Npp32s nSrcStep, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI);

/**
 * Four channel 8-bit unsigned vertical Sobel filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterSobelVert_8u_C4R(const Npp8u * pSrc, Npp32s nSrcStep, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI);

/**
 * Four channel 16-bit signed vertical Sobel filter, ignoring alpha channel.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterSobelVert_8u_AC4R(const Npp8u * pSrc, Npp32s nSrcStep, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI);

/**
 * Single channel 16-bit signed vertical Sobel filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterSobelVert_16s_C1R(const Npp16s * pSrc, Npp32s nSrcStep, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI);

/**
 * Three channel 16-bit signed vertical Sobel filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterSobelVert_16s_C3R(const Npp16s * pSrc, Npp32s nSrcStep, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI);

/**
 * Four channel 16-bit signed vertical Sobel filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterSobelVert_16s_C4R(const Npp16s * pSrc, Npp32s nSrcStep, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI);

/**
 * Four channel 8-bit unsigned vertical Sobel filter, ignoring alpha channel.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterSobelVert_16s_AC4R(const Npp16s * pSrc, Npp32s nSrcStep, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI);

/**
 * Single channel 32-bit floating-point vertical Sobel filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterSobelVert_32f_C1R(const Npp32f * pSrc, Npp32s nSrcStep, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI);

/**
 * Three channel 32-bit floating-point vertical Sobel filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterSobelVert_32f_C3R(const Npp32f * pSrc, Npp32s nSrcStep, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI);

/**
 * Four channel 32-bit floating-point vertical Sobel filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterSobelVert_32f_C4R(const Npp32f * pSrc, Npp32s nSrcStep, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI);

/**
 * Four channel 32-bit floating-point vertical Sobel filter, ignoring alpha channel.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterSobelVert_32f_AC4R(const Npp32f * pSrc, Npp32s nSrcStep, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI);

/**
 * Single channel 8-bit unsigned to 16-bit signed vertical Sobel filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eMaskSize Enumeration value specifying the mask size.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterSobelVert_8u16s_C1R(const Npp8u * pSrc, Npp32s nSrcStep, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                               NppiMaskSize eMaskSize);

/**
 * Single channel 8-bit signed to 16-bit signed vertical Sobel filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eMaskSize Enumeration value specifying the mask size.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterSobelVert_8s16s_C1R(const Npp8s * pSrc, Npp32s nSrcStep, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                              NppiMaskSize eMaskSize);

/**
 * Single channel 32-bit floating-point vertical Sobel filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eMaskSize Enumeration value specifying the mask size.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterSobelVertMask_32f_C1R(const Npp32f * pSrc, Npp32s nSrcStep, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                                NppiMaskSize eMaskSize);


/** @} FilterSobelVert */

/** @name FilterSobelHorizSecond
 *
 * Filters the image using a second derivative, horizontal Sobel filter kernel:
 *
 * \f[
 *  \left( \begin{array}{rrr}
 *     1 &   2 &   1 \\
 *    -2 &  -4 &  -2 \\
 *     1 &   2 &   1 \\
 *  \end{array} \right)
 *  \left( \begin{array}{rrrrr}
 *     1  &  4 &   6 &  4 &  1 \\
 *     0  &  0 &   0 &  0 &  0 \\
 *    -2  & -8 & -12 & -8 & -2 \\
 *     0  &  0 &   0 &  0 &  0 \\
 *     1  &  4 &   6 &  4 &  1 \\
 *  \end{array} \right)
 * \f]
 *
 * @{
 *
 */

/**
 * Single channel 8-bit unsigned to 16-bit signed second derivative, horizontal Sobel filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eMaskSize Enumeration value specifying the mask size.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterSobelHorizSecond_8u16s_C1R(const Npp8u * pSrc, Npp32s nSrcStep, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                                     NppiMaskSize eMaskSize);

/**
 * Single channel 8-bit signed to 16-bit signed second derivative, horizontal Sobel filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eMaskSize Enumeration value specifying the mask size.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterSobelHorizSecond_8s16s_C1R(const Npp8s * pSrc, Npp32s nSrcStep, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                                     NppiMaskSize eMaskSize);
/**
 * Single channel 32-bit floating-point second derivative, horizontal Sobel filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eMaskSize Enumeration value specifying the mask size.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterSobelHorizSecond_32f_C1R(const Npp32f * pSrc, Npp32s nSrcStep, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                                   NppiMaskSize eMaskSize);

/** @} FilterSobelHorizSecond */

/** @name FilterSobelVertSecond
 *
 * Filters the image using a second derivative, vertical Sobel filter kernel:
 *
 * \f[
 *  \left( \begin{array}{rrr}
     *    1 & -2 & 1 \\
     *    2 & -4 & 2 \\
     *    1 & -2 & 1 \\
 *  \end{array} \right)
 *  \left( \begin{array}{rrrrr}
 *     1 & 0 &  -2 & 0 & 1 \\
 *     4 & 0 &  -8 & 0 & 4 \\
 *     6 & 0 & -12 & 0 & 6 \\
 *     4 & 0 &  -8 & 0 & 4 \\
 *     1 & 0 &  -2 & 0 & 1 \\
 *  \end{array} \right)
 * \f]
 *
 * @{
 *
 */

/**
 * Single channel 8-bit unsigned to 16-bit signed second derivative, vertical Sobel filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eMaskSize Enumeration value specifying the mask size.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterSobelVertSecond_8u16s_C1R(const Npp8u * pSrc, Npp32s nSrcStep, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                                    NppiMaskSize eMaskSize);

/**
 * Single channel 8-bit signed to 16-bit signed second derivative, vertical Sobel filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eMaskSize Enumeration value specifying the mask size.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterSobelVertSecond_8s16s_C1R(const Npp8s * pSrc, Npp32s nSrcStep, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                                    NppiMaskSize eMaskSize);
/**
 * Single channel 32-bit floating-point second derivative, vertical Sobel filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eMaskSize Enumeration value specifying the mask size.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterSobelVertSecond_32f_C1R(const Npp32f * pSrc, Npp32s nSrcStep, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                                  NppiMaskSize eMaskSize);

/** @} FilterSobelVertSecond */

/** @name FilterSobelCross
 *
 * Filters the image using a second cross derivative Sobel filter kernel:
 *
 * \f[
 *  \left( \begin{array}{rrr}
 *    -1 & 0 &  1 \\
 *     0 & 0 &  0 \\
 *     1 & 0 & -1 \\
 *  \end{array} \right)
 *  \left( \begin{array}{rrrrr}
 *    -1 & -2 & 0 &  2 &  1 \\
 *    -2 & -4 & 0 &  4 &  2 \\
 *     0 &  0 & 0 &  0 &  0 \\
 *     2 &  4 & 0 & -4 & -2 \\
 *     1 &  2 & 0 & -2 & -1 \\
 *  \end{array} \right)
 * \f]
 *
 * @{
 *
 */

/**
 * Single channel 8-bit unsigned to 16-bit signed second cross derivative Sobel filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eMaskSize Enumeration value specifying the mask size.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterSobelCross_8u16s_C1R(const Npp8u * pSrc, Npp32s nSrcStep, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                                    NppiMaskSize eMaskSize);

/**
 * Single channel 8-bit signed to 16-bit signed second cross derivative Sobel filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eMaskSize Enumeration value specifying the mask size.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterSobelCross_8s16s_C1R(const Npp8s * pSrc, Npp32s nSrcStep, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                                    NppiMaskSize eMaskSize);
/**
 * Single channel 32-bit floating-point second cross derivative Sobel filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eMaskSize Enumeration value specifying the mask size.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterSobelCross_32f_C1R(const Npp32f * pSrc, Npp32s nSrcStep, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                                  NppiMaskSize eMaskSize);

/** @} FilterSobelCross */


/** @name FilterSobelHorizBorder 
 *
 * Filters the image using a horizontal Sobel filter kernel with border control:
 *
 * \f[
 *  \left( \begin{array}{rrr}
 *    1 &  2 &  1 \\
 *    0 &  0 &  0 \\
 *   -1 & -2 & -1 \\
 *  \end{array} \right)
 *  \left( \begin{array}{rrrrr}
 *    1  &  4 &   6 &  4 &  1 \\
 *    2  &  8 &  12 &  8 &  2 \\
 *    0  &  0 &   0 &  0 &  0 \\
 *    -2 & -8 & -12 & -8 & -2 \\
 *    -1 & -4 &  -6 & -4 & -1 \\
 *  \end{array} \right)
 * \f]
 *
 * @{
 *
 */

/**
 * Single channel 8-bit unsigned horizontal Sobel filter with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterSobelHorizBorder_8u_C1R(const Npp8u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiBorderType eBorderType);

/**
 * Three channel 8-bit unsigned horizontal Sobel filter with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterSobelHorizBorder_8u_C3R(const Npp8u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiBorderType eBorderType);

/**
 * Four channel 8-bit unsigned horizontal Sobel filter with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterSobelHorizBorder_8u_C4R(const Npp8u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiBorderType eBorderType);

/**
 * Four channel 16-bit signed horizontal Sobel filter with border control, ignoring alpha channel.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterSobelHorizBorder_8u_AC4R(const Npp8u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiBorderType eBorderType);

/**
 * Single channel 16-bit signed horizontal Sobel filter with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterSobelHorizBorder_16s_C1R(const Npp16s * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiBorderType eBorderType);

/**
 * Three channel 16-bit signed horizontal Sobel filter with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterSobelHorizBorder_16s_C3R(const Npp16s * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiBorderType eBorderType);

/**
 * Four channel 16-bit signed horizontal Sobel filter with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterSobelHorizBorder_16s_C4R(const Npp16s * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiBorderType eBorderType);

/**
 * Four channel 8-bit unsigned horizontal Sobel filter with border control, ignoring alpha channel.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterSobelHorizBorder_16s_AC4R(const Npp16s * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiBorderType eBorderType);

/**
 * Single channel 32-bit floating-point horizontal Sobel filter with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterSobelHorizBorder_32f_C1R(const Npp32f * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiBorderType eBorderType);

/**
 * Three channel 32-bit floating-point horizontal Sobel filter with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterSobelHorizBorder_32f_C3R(const Npp32f * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiBorderType eBorderType);

/**
 * Four channel 32-bit floating-point horizontal Sobel filter with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterSobelHorizBorder_32f_C4R(const Npp32f * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiBorderType eBorderType);

/**
 * Four channel 32-bit floating-point horizontal Sobel filter with border control, ignoring alpha channel.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterSobelHorizBorder_32f_AC4R(const Npp32f * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiBorderType eBorderType);

/**
 * Single channel 8-bit unsigned to 16-bit signed horizontal Sobel filter with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eMaskSize Enumeration value specifying the mask size.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterSobelHorizBorder_8u16s_C1R(const Npp8u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                                     NppiMaskSize eMaskSize, NppiBorderType eBorderType);

/**
 * Single channel 8-bit signed to 16-bit signed horizontal Sobel filter with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eMaskSize Enumeration value specifying the mask size.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterSobelHorizBorder_8s16s_C1R(const Npp8s * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                                     NppiMaskSize eMaskSize, NppiBorderType eBorderType);

/**
 * Single channel 32-bit floating-point horizontal Sobel filter with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eMaskSize Enumeration value specifying the mask size.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterSobelHorizMaskBorder_32f_C1R(const Npp32f * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                                       NppiMaskSize eMaskSize, NppiBorderType eBorderType);


/** @} FilterSobelHorizBorder */

/** @name FilterSobelVertBorder 
 *
 * Filters the image using a vertical Sobel filter kernel with border control:
 *
 * \f[
 *  \left( \begin{array}{rrr}
 *    -1 & 0 & 1 \\
 *    -2 & 0 & 2 \\
 *    -1 & 0 & 1 \\
 *  \end{array} \right)
 *  \left( \begin{array}{rrrrr}
 *    -1 &  -2 & 0 &  2 & 1 \\
 *    -4 &  -8 & 0 &  8 & 4 \\
 *    -6 & -12 & 0 & 12 & 6 \\
 *    -4 &  -8 & 0 &  8 & 4 \\
 *    -1 &  -2 & 0 &  2 & 1 \\
 *  \end{array} \right)
 * \f]
 *
 * @{
 *
 */

/**
 * Single channel 8-bit unsigned vertical Sobel filter with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterSobelVertBorder_8u_C1R(const Npp8u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiBorderType eBorderType);

/**
 * Three channel 8-bit unsigned vertical Sobel filter with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterSobelVertBorder_8u_C3R(const Npp8u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiBorderType eBorderType);

/**
 * Four channel 8-bit unsigned vertical Sobel filter with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterSobelVertBorder_8u_C4R(const Npp8u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiBorderType eBorderType);

/**
 * Four channel 16-bit signed vertical Sobel filter with border control, ignoring alpha channel.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterSobelVertBorder_8u_AC4R(const Npp8u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiBorderType eBorderType);

/**
 * Single channel 16-bit signed vertical Sobel filter with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterSobelVertBorder_16s_C1R(const Npp16s * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiBorderType eBorderType);

/**
 * Three channel 16-bit signed vertical Sobel filter with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterSobelVertBorder_16s_C3R(const Npp16s * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiBorderType eBorderType);

/**
 * Four channel 16-bit signed vertical Sobel filter with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterSobelVertBorder_16s_C4R(const Npp16s * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiBorderType eBorderType);

/**
 * Four channel 8-bit unsigned vertical Sobel filter with border control, ignoring alpha channel.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterSobelVertBorder_16s_AC4R(const Npp16s * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiBorderType eBorderType);

/**
 * Single channel 32-bit floating-point vertical Sobel filter with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterSobelVertBorder_32f_C1R(const Npp32f * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiBorderType eBorderType);

/**
 * Three channel 32-bit floating-point vertical Sobel filter with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterSobelVertBorder_32f_C3R(const Npp32f * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiBorderType eBorderType);

/**
 * Four channel 32-bit floating-point vertical Sobel filter with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterSobelVertBorder_32f_C4R(const Npp32f * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiBorderType eBorderType);

/**
 * Four channel 32-bit floating-point vertical Sobel filter with border control, ignoring alpha channel.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterSobelVertBorder_32f_AC4R(const Npp32f * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiBorderType eBorderType);

/**
 * Single channel 8-bit unsigned to 16-bit signed vertical Sobel filter with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eMaskSize Enumeration value specifying the mask size.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterSobelVertBorder_8u16s_C1R(const Npp8u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                                    NppiMaskSize eMaskSize, NppiBorderType eBorderType);

/**
 * Single channel 8-bit signed to 16-bit signed vertical Sobel filter with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eMaskSize Enumeration value specifying the mask size.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterSobelVertBorder_8s16s_C1R(const Npp8s * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                                    NppiMaskSize eMaskSize, NppiBorderType eBorderType);

/**
 * Single channel 32-bit floating-point vertical Sobel filter with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eMaskSize Enumeration value specifying the mask size.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterSobelVertMaskBorder_32f_C1R(const Npp32f * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                                      NppiMaskSize eMaskSize, NppiBorderType eBorderType);


/** @} FilterSobelVertBorder */

/** @name FilterSobelHorizSecondBorder
 *
 * Filters the image using a second derivative, horizontal Sobel filter kernel with border control:
 *
 * \f[
 *  \left( \begin{array}{rrr}
 *     1 &   2 &   1 \\
 *    -2 &  -4 &  -2 \\
 *     1 &   2 &   1 \\
 *  \end{array} \right)
 *  \left( \begin{array}{rrrrr}
 *     1  &  4 &   6 &  4 &  1 \\
 *     0  &  0 &   0 &  0 &  0 \\
 *    -2  & -8 & -12 & -8 & -2 \\
 *     0  &  0 &   0 &  0 &  0 \\
 *     1  &  4 &   6 &  4 &  1 \\
 *  \end{array} \right)
 * \f]
 *
 * @{
 *
 */

/**
 * Single channel 8-bit unsigned to 16-bit signed second derivative, horizontal Sobel filter with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eMaskSize Enumeration value specifying the mask size.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterSobelHorizSecondBorder_8u16s_C1R(const Npp8u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                                           NppiMaskSize eMaskSize, NppiBorderType eBorderType);

/**
 * Single channel 8-bit signed to 16-bit signed second derivative, horizontal Sobel filter with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eMaskSize Enumeration value specifying the mask size.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterSobelHorizSecondBorder_8s16s_C1R(const Npp8s * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                                           NppiMaskSize eMaskSize, NppiBorderType eBorderType);
/**
 * Single channel 32-bit floating-point second derivative, horizontal Sobel filter with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eMaskSize Enumeration value specifying the mask size.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterSobelHorizSecondBorder_32f_C1R(const Npp32f * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                                         NppiMaskSize eMaskSize, NppiBorderType eBorderType);

/** @} FilterSobelHorizSecondBorder */

/** @name FilterSobelVertSecondBorder
 *
 * Filters the image using a second derivative, vertical Sobel filter kernel with border control:
 *
 * \f[
 *  \left( \begin{array}{rrr}
     *    1 & -2 & 1 \\
     *    2 & -4 & 2 \\
     *    1 & -2 & 1 \\
 *  \end{array} \right)
 *  \left( \begin{array}{rrrrr}
 *     1 & 0 &  -2 & 0 & 1 \\
 *     4 & 0 &  -8 & 0 & 4 \\
 *     6 & 0 & -12 & 0 & 6 \\
 *     4 & 0 &  -8 & 0 & 4 \\
 *     1 & 0 &  -2 & 0 & 1 \\
 *  \end{array} \right)
 * \f]
 *
 * @{
 *
 */

/**
 * Single channel 8-bit unsigned to 16-bit signed second derivative, vertical Sobel filter with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eMaskSize Enumeration value specifying the mask size.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterSobelVertSecondBorder_8u16s_C1R(const Npp8u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                                          NppiMaskSize eMaskSize, NppiBorderType eBorderType);

/**
 * Single channel 8-bit signed to 16-bit signed second derivative, vertical Sobel filter with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eMaskSize Enumeration value specifying the mask size.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterSobelVertSecondBorder_8s16s_C1R(const Npp8s * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                                          NppiMaskSize eMaskSize, NppiBorderType eBorderType);
/**
 * Single channel 32-bit floating-point second derivative, vertical Sobel filter with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eMaskSize Enumeration value specifying the mask size.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterSobelVertSecondBorder_32f_C1R(const Npp32f * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                                        NppiMaskSize eMaskSize, NppiBorderType eBorderType);

/** @} FilterSobelVertSecondBorder */

/** @name FilterSobelCrossBorder
 *
 * Filters the image using a second cross derivative Sobel filter kernel with border control:
 *
 * \f[
 *  \left( \begin{array}{rrr}
 *    -1 & 0 &  1 \\
 *     0 & 0 &  0 \\
 *     1 & 0 & -1 \\
 *  \end{array} \right)
 *  \left( \begin{array}{rrrrr}
 *    -1 & -2 & 0 &  2 &  1 \\
 *    -2 & -4 & 0 &  4 &  2 \\
 *     0 &  0 & 0 &  0 &  0 \\
 *     2 &  4 & 0 & -4 & -2 \\
 *     1 &  2 & 0 & -2 & -1 \\
 *  \end{array} \right)
 * \f]
 *
 * @{
 *
 */

/**
 * Single channel 8-bit unsigned to 16-bit signed second cross derivative Sobel filter with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eMaskSize Enumeration value specifying the mask size.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterSobelCrossBorder_8u16s_C1R(const Npp8u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                                     NppiMaskSize eMaskSize, NppiBorderType eBorderType);

/**
 * Single channel 8-bit signed to 16-bit signed second cross derivative Sobel filter with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eMaskSize Enumeration value specifying the mask size.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterSobelCrossBorder_8s16s_C1R(const Npp8s * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                                     NppiMaskSize eMaskSize, NppiBorderType eBorderType);
/**
 * Single channel 32-bit floating-point second cross derivative Sobel filter with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eMaskSize Enumeration value specifying the mask size.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterSobelCrossBorder_32f_C1R(const Npp32f * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                                   NppiMaskSize eMaskSize, NppiBorderType eBorderType);

/** @} FilterSobelCrossBorder */

/** @name FilterRobertsDown
 *
 * Filters the image using a horizontal Roberts filter kernel:
 *
 * \f[
 *  \left( \begin{array}{rrr}
 *   0 & 0 &  0 \\
 *   0 & 1 &  0 \\
 *   0 & 0 & -1 \\
 *  \end{array} \right)
 * \f]
 *
 * @{
 *
 */

/**
 * Single channel 8-bit unsigned horizontal Roberts filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterRobertsDown_8u_C1R(const Npp8u * pSrc, Npp32s nSrcStep, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI);

/**
 * Three channel 8-bit unsigned horizontal Roberts filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterRobertsDown_8u_C3R(const Npp8u * pSrc, Npp32s nSrcStep, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI);

/**
 * Four channel 8-bit unsigned horizontal Roberts filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterRobertsDown_8u_C4R(const Npp8u * pSrc, Npp32s nSrcStep, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI);

/**
 * Four channel 8-bit unsigned horizontal Roberts filter, ignoring alpha-channel.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterRobertsDown_8u_AC4R(const Npp8u * pSrc, Npp32s nSrcStep, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI);

/**
 * Single channel 16-bit signed horizontal Roberts filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterRobertsDown_16s_C1R(const Npp16s * pSrc, Npp32s nSrcStep, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI);

/**
 * Three channel 16-bit signed horizontal Roberts filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterRobertsDown_16s_C3R(const Npp16s * pSrc, Npp32s nSrcStep, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI);

/**
 * Four channel 16-bit signed horizontal Roberts filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterRobertsDown_16s_C4R(const Npp16s * pSrc, Npp32s nSrcStep, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI);

/**
 * Four channel 16-bit signed horizontal Roberts filter, ignoring alpha-channel.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterRobertsDown_16s_AC4R(const Npp16s * pSrc, Npp32s nSrcStep, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI);

/**
 * Single channel 32-bit floating-point horizontal Roberts filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterRobertsDown_32f_C1R(const Npp32f * pSrc, Npp32s nSrcStep, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI);

/**
 * Three channel 32-bit floating-point horizontal Roberts filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterRobertsDown_32f_C3R(const Npp32f * pSrc, Npp32s nSrcStep, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI);

/**
 * Four channel 32-bit floating-point horizontal Roberts filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterRobertsDown_32f_C4R(const Npp32f * pSrc, Npp32s nSrcStep, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI);

/**
 * Four channel 32-bit floating-point horizontal Roberts filter, ignoring alpha-channel.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterRobertsDown_32f_AC4R(const Npp32f * pSrc, Npp32s nSrcStep, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI);

/** @} FilterRobertsDown */

/** @name FilterRobertsDownBorder
 *
 * Filters the image using a horizontal Roberts filter kernel with border control.  If any portion of the mask overlaps the source
 * image boundary the requested border type operation is applied to all mask pixels
 * which fall outside of the source image.
 *
 * Lwrrently only the NPP_BORDER_REPLICATE border type operation is supported.
 *
 * \f[
 *  \left( \begin{array}{rrr}
 *   0 & 0 &  0 \\
 *   0 & 1 &  0 \\
 *   0 & 0 & -1 \\
 *  \end{array} \right)
 * \f]
 *
 * @{
 *
 */

/**
 * Single channel 8-bit unsigned horizontal Roberts filter with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterRobertsDownBorder_8u_C1R(const Npp8u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiBorderType eBorderType);

/**
 * Three channel 8-bit unsigned horizontal Roberts filter with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterRobertsDownBorder_8u_C3R(const Npp8u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiBorderType eBorderType);

/**
 * Four channel 8-bit unsigned horizontal Roberts filter with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterRobertsDownBorder_8u_C4R(const Npp8u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiBorderType eBorderType);

/**
 * Four channel 8-bit unsigned horizontal Roberts filter with border control, ignoring alpha-channel.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterRobertsDownBorder_8u_AC4R(const Npp8u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiBorderType eBorderType);

/**
 * Single channel 16-bit signed horizontal Roberts filter with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterRobertsDownBorder_16s_C1R(const Npp16s * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiBorderType eBorderType);

/**
 * Three channel 16-bit signed horizontal Roberts filter with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterRobertsDownBorder_16s_C3R(const Npp16s * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiBorderType eBorderType);

/**
 * Four channel 16-bit signed horizontal Roberts filter with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterRobertsDownBorder_16s_C4R(const Npp16s * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiBorderType eBorderType);

/**
 * Four channel 16-bit signed horizontal Roberts filter with border control, ignoring alpha-channel.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterRobertsDownBorder_16s_AC4R(const Npp16s * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiBorderType eBorderType);

/**
 * Single channel 32-bit floating-point horizontal Roberts filter with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterRobertsDownBorder_32f_C1R(const Npp32f * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiBorderType eBorderType);

/**
 * Three channel 32-bit floating-point horizontal Roberts filter with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterRobertsDownBorder_32f_C3R(const Npp32f * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiBorderType eBorderType);

/**
 * Four channel 32-bit floating-point horizontal Roberts filter with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterRobertsDownBorder_32f_C4R(const Npp32f * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiBorderType eBorderType);

/**
 * Four channel 32-bit floating-point horizontal Roberts filter with border control, ignoring alpha-channel.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterRobertsDownBorder_32f_AC4R(const Npp32f * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiBorderType eBorderType);

/** @} FilterRobertsDownBorder */

/** @name FilterRobertsUp
 *
 * Filters the image using a vertical Roberts filter kernel:
 *
 * \f[
 *  \left( \begin{array}{rrr}
 *   0 & 0 &  0 \\
 *   0 & 1 &  0 \\
 *  -1 & 0 &  0 \\
 *  \end{array} \right)
 * \f]
 *
 * @{
 *
 */

/**
 * Single channel 8-bit unsigned vertical Roberts filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterRobertsUp_8u_C1R(const Npp8u * pSrc, Npp32s nSrcStep, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI);

/**
 * Three channel 8-bit unsigned vertical Roberts filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterRobertsUp_8u_C3R(const Npp8u * pSrc, Npp32s nSrcStep, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI);

/**
 * Four channel 8-bit unsigned vertical Roberts filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterRobertsUp_8u_C4R(const Npp8u * pSrc, Npp32s nSrcStep, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI);

/**
 * Four channel 8-bit unsigned vertical Roberts filter, ignoring alpha-channel.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterRobertsUp_8u_AC4R(const Npp8u * pSrc, Npp32s nSrcStep, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI);

/**
 * Single channel 16-bit signed vertical Roberts filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterRobertsUp_16s_C1R(const Npp16s * pSrc, Npp32s nSrcStep, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI);

/**
 * Three channel 16-bit signed vertical Roberts filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterRobertsUp_16s_C3R(const Npp16s * pSrc, Npp32s nSrcStep, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI);

/**
 * Four channel 16-bit signed vertical Roberts filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterRobertsUp_16s_C4R(const Npp16s * pSrc, Npp32s nSrcStep, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI);

/**
 * Four channel 16-bit signed vertical Roberts filter, ignoring alpha-channel.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterRobertsUp_16s_AC4R(const Npp16s * pSrc, Npp32s nSrcStep, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI);

/**
 * Single channel 32-bit floating-point vertical Roberts filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterRobertsUp_32f_C1R(const Npp32f * pSrc, Npp32s nSrcStep, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI);

/**
 * Three channel 32-bit floating-point vertical Roberts filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterRobertsUp_32f_C3R(const Npp32f * pSrc, Npp32s nSrcStep, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI);

/**
 * Four channel 32-bit floating-point vertical Roberts filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterRobertsUp_32f_C4R(const Npp32f * pSrc, Npp32s nSrcStep, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI);

/**
 * Four channel 32-bit floating-point vertical Roberts filter, ignoring alpha-channel.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterRobertsUp_32f_AC4R(const Npp32f * pSrc, Npp32s nSrcStep, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI);

/** @} FilterRobertsUp */

/** @name FilterRobertsUpBorder
 *
 * Filters the image using a vertical Roberts filter kernel with border control.  If any portion of the mask overlaps the source
 * image boundary the requested border type operation is applied to all mask pixels
 * which fall outside of the source image.
 *
 * Lwrrently only the NPP_BORDER_REPLICATE border type operation is supported.
 *
 * \f[
 *  \left( \begin{array}{rrr}
 *   0 & 0 &  0 \\
 *   0 & 1 &  0 \\
 *  -1 & 0 &  0 \\
 *  \end{array} \right)
 * \f]
 *
 * @{
 *
 */

/**
 * Single channel 8-bit unsigned vertical Roberts filter with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterRobertsUpBorder_8u_C1R(const Npp8u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiBorderType eBorderType);

/**
 * Three channel 8-bit unsigned vertical Roberts filter with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterRobertsUpBorder_8u_C3R(const Npp8u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiBorderType eBorderType);

/**
 * Four channel 8-bit unsigned vertical Roberts filter with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterRobertsUpBorder_8u_C4R(const Npp8u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiBorderType eBorderType);

/**
 * Four channel 8-bit unsigned vertical Roberts filter with border control, ignoring alpha-channel.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterRobertsUpBorder_8u_AC4R(const Npp8u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiBorderType eBorderType);

/**
 * Single channel 16-bit signed vertical Roberts filter with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterRobertsUpBorder_16s_C1R(const Npp16s * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiBorderType eBorderType);

/**
 * Three channel 16-bit signed vertical Roberts filter with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterRobertsUpBorder_16s_C3R(const Npp16s * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiBorderType eBorderType);

/**
 * Four channel 16-bit signed vertical Roberts filter with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterRobertsUpBorder_16s_C4R(const Npp16s * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiBorderType eBorderType);

/**
 * Four channel 16-bit signed vertical Roberts filter with border control, ignoring alpha-channel.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterRobertsUpBorder_16s_AC4R(const Npp16s * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiBorderType eBorderType);

/**
 * Single channel 32-bit floating-point vertical Roberts filter with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterRobertsUpBorder_32f_C1R(const Npp32f * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiBorderType eBorderType);

/**
 * Three channel 32-bit floating-point vertical Roberts filter with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterRobertsUpBorder_32f_C3R(const Npp32f * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiBorderType eBorderType);

/**
 * Four channel 32-bit floating-point vertical Roberts filter with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterRobertsUpBorder_32f_C4R(const Npp32f * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiBorderType eBorderType);

/**
 * Four channel 32-bit floating-point vertical Roberts filter with border control, ignoring alpha-channel.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterRobertsUpBorder_32f_AC4R(const Npp32f * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiBorderType eBorderType);

/** @} FilterRobertsUpBorder */


/** @name FilterLaplace
 *
 * Filters the image using a Laplacian filter kernel:
 *
 * \f[
 *  \left( \begin{array}{rrr}
 *   -1 & -1 & -1 \\
 *   -1 &  8 & -1 \\
 *   -1 & -1 & -1 \\
 *  \end{array} \right)
  *  \left( \begin{array}{rrrrr}
 *   -1 & -3 & -4 & -3 & -1 \\
 *   -3 &  0 &  6 &  0 & -3 \\
 *   -4 &  6 & 20 &  6 & -4 \\
 *   -3 &  0 &  6 &  0 & -3 \\
 *   -1 & -3 & -4 & -3 & -1 \\
 *  \end{array} \right)

 * \f]
 *
 * @{
 *
 */

/**
 * Single channel 8-bit unsigned Laplace filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eMaskSize Enumeration value specifying the mask size.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterLaplace_8u_C1R(const Npp8u * pSrc, Npp32s nSrcStep, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                         NppiMaskSize eMaskSize);

/**
 * Three channel 8-bit unsigned Laplace filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eMaskSize Enumeration value specifying the mask size.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterLaplace_8u_C3R(const Npp8u * pSrc, Npp32s nSrcStep, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                         NppiMaskSize eMaskSize);

/**
 * Four channel 8-bit unsigned Laplace filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eMaskSize Enumeration value specifying the mask size.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterLaplace_8u_C4R(const Npp8u * pSrc, Npp32s nSrcStep, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                         NppiMaskSize eMaskSize);

/**
 * Four channel 8-bit unsigned Laplace filter, ignoring alpha channel.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eMaskSize Enumeration value specifying the mask size.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterLaplace_8u_AC4R(const Npp8u * pSrc, Npp32s nSrcStep, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                          NppiMaskSize eMaskSize);

/**
 * Single channel 16-bit signed Laplace filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eMaskSize Enumeration value specifying the mask size.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterLaplace_16s_C1R(const Npp16s * pSrc, Npp32s nSrcStep, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                          NppiMaskSize eMaskSize);

/**
 * Three channel 16-bit signed Laplace filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eMaskSize Enumeration value specifying the mask size.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterLaplace_16s_C3R(const Npp16s * pSrc, Npp32s nSrcStep, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                          NppiMaskSize eMaskSize);

/**
 * Four channel 16-bit signed Laplace filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eMaskSize Enumeration value specifying the mask size.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterLaplace_16s_C4R(const Npp16s * pSrc, Npp32s nSrcStep, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                          NppiMaskSize eMaskSize);

/**
 * Four channel 16-bit signed Laplace filter, ignoring alpha channel.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eMaskSize Enumeration value specifying the mask size.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterLaplace_16s_AC4R(const Npp16s * pSrc, Npp32s nSrcStep, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                           NppiMaskSize eMaskSize);

/**
 * Single channel 32-bit floating-point Laplace filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eMaskSize Enumeration value specifying the mask size.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterLaplace_32f_C1R(const Npp32f * pSrc, Npp32s nSrcStep, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                          NppiMaskSize eMaskSize);

/**
 * Three channel 32-bit floating-point Laplace filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eMaskSize Enumeration value specifying the mask size.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterLaplace_32f_C3R(const Npp32f * pSrc, Npp32s nSrcStep, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                          NppiMaskSize eMaskSize);

/**
 * Four channel 32-bit floating-point Laplace filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eMaskSize Enumeration value specifying the mask size.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterLaplace_32f_C4R(const Npp32f * pSrc, Npp32s nSrcStep, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                          NppiMaskSize eMaskSize);

/**
 * Four channel 32-bit floating-point Laplace filter, ignoring alpha channel.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eMaskSize Enumeration value specifying the mask size.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterLaplace_32f_AC4R(const Npp32f * pSrc, Npp32s nSrcStep, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                           NppiMaskSize eMaskSize);

/**
 * Single channel 8-bit unsigned to 16-bit signed Laplace filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eMaskSize Enumeration value specifying the mask size.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterLaplace_8u16s_C1R(const Npp8u * pSrc, Npp32s nSrcStep, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                            NppiMaskSize eMaskSize);

/**
 * Single channel 8-bit signed to 16-bit signed Laplace filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eMaskSize Enumeration value specifying the mask size.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterLaplace_8s16s_C1R(const Npp8s * pSrc, Npp32s nSrcStep, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                            NppiMaskSize eMaskSize);

/** @} FilterLaplace */

/** @name FilterLaplaceBorder
 *
 * Filters the image using a Laplacian filter kernel with border control. If any portion of the mask overlaps the source
 * image boundary the requested border type operation is applied to all mask pixels
 * which fall outside of the source image.
 *
 * Lwrrently only the NPP_BORDER_REPLICATE border type operation is supported.
 *
 * \f[
 *  \left( \begin{array}{rrr}
 *   -1 & -1 & -1 \\
 *   -1 &  8 & -1 \\
 *   -1 & -1 & -1 \\
 *  \end{array} \right)
  *  \left( \begin{array}{rrrrr}
 *   -1 & -3 & -4 & -3 & -1 \\
 *   -3 &  0 &  6 &  0 & -3 \\
 *   -4 &  6 & 20 &  6 & -4 \\
 *   -3 &  0 &  6 &  0 & -3 \\
 *   -1 & -3 & -4 & -3 & -1 \\
 *  \end{array} \right)

 * \f]
 *
 * @{
 *
 */

/**
 * Single channel 8-bit unsigned Laplace filter with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eMaskSize Enumeration value specifying the mask size.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterLaplaceBorder_8u_C1R(const Npp8u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                               NppiMaskSize eMaskSize, NppiBorderType eBorderType);

/**
 * Three channel 8-bit unsigned Laplace filter with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eMaskSize Enumeration value specifying the mask size.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterLaplaceBorder_8u_C3R(const Npp8u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                               NppiMaskSize eMaskSize, NppiBorderType eBorderType);

/**
 * Four channel 8-bit unsigned Laplace filter with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eMaskSize Enumeration value specifying the mask size.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterLaplaceBorder_8u_C4R(const Npp8u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                               NppiMaskSize eMaskSize, NppiBorderType eBorderType);

/**
 * Four channel 8-bit unsigned Laplace filter with border control, ignoring alpha channel.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eMaskSize Enumeration value specifying the mask size.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterLaplaceBorder_8u_AC4R(const Npp8u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                                NppiMaskSize eMaskSize, NppiBorderType eBorderType);

/**
 * Single channel 16-bit signed Laplace filter with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eMaskSize Enumeration value specifying the mask size.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterLaplaceBorder_16s_C1R(const Npp16s * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                                NppiMaskSize eMaskSize, NppiBorderType eBorderType);

/**
 * Three channel 16-bit signed Laplace filter with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eMaskSize Enumeration value specifying the mask size.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterLaplaceBorder_16s_C3R(const Npp16s * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                                NppiMaskSize eMaskSize, NppiBorderType eBorderType);

/**
 * Four channel 16-bit signed Laplace filter with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eMaskSize Enumeration value specifying the mask size.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterLaplaceBorder_16s_C4R(const Npp16s * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                                NppiMaskSize eMaskSize, NppiBorderType eBorderType);

/**
 * Four channel 16-bit signed Laplace filter with border control, ignoring alpha channel.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eMaskSize Enumeration value specifying the mask size.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterLaplaceBorder_16s_AC4R(const Npp16s * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                                 NppiMaskSize eMaskSize, NppiBorderType eBorderType);

/**
 * Single channel 32-bit floating-point Laplace filter with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eMaskSize Enumeration value specifying the mask size.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterLaplaceBorder_32f_C1R(const Npp32f * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                                NppiMaskSize eMaskSize, NppiBorderType eBorderType);

/**
 * Three channel 32-bit floating-point Laplace filter with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eMaskSize Enumeration value specifying the mask size.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterLaplaceBorder_32f_C3R(const Npp32f * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                                NppiMaskSize eMaskSize, NppiBorderType eBorderType);

/**
 * Four channel 32-bit floating-point Laplace filter with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eMaskSize Enumeration value specifying the mask size.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterLaplaceBorder_32f_C4R(const Npp32f * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                                NppiMaskSize eMaskSize, NppiBorderType eBorderType);

/**
 * Four channel 32-bit floating-point Laplace filter with border control, ignoring alpha channel.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eMaskSize Enumeration value specifying the mask size.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterLaplaceBorder_32f_AC4R(const Npp32f * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                                 NppiMaskSize eMaskSize, NppiBorderType eBorderType);

/**
 * Single channel 8-bit unsigned to 16-bit signed Laplace filter with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eMaskSize Enumeration value specifying the mask size.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterLaplaceBorder_8u16s_C1R(const Npp8u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                                  NppiMaskSize eMaskSize, NppiBorderType eBorderType);

/**
 * Single channel 8-bit signed to 16-bit signed Laplace filter with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eMaskSize Enumeration value specifying the mask size.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterLaplaceBorder_8s16s_C1R(const Npp8s * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                                  NppiMaskSize eMaskSize, NppiBorderType eBorderType);

/** @} FilterLaplaceBorder */

/** @name FilterGauss
 *
 * Filters the image using a Gaussian filter kernel:
 *
 * Note that all FilterGauss functions lwrrently support mask sizes up to 15x15. Filter kernels for these functions are callwlated
 * using a sigma value of 0.4F + (mask width / 2) * 0.6F.
 *
 * @{
 *
 */

/**
 * Single channel 8-bit unsigned Gauss filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eMaskSize Enumeration value specifying the mask size.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterGauss_8u_C1R(const Npp8u * pSrc, Npp32s nSrcStep, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                       NppiMaskSize eMaskSize);

/**
 * Three channel 8-bit unsigned Gauss filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eMaskSize Enumeration value specifying the mask size.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterGauss_8u_C3R(const Npp8u * pSrc, Npp32s nSrcStep, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                       NppiMaskSize eMaskSize);

/**
 * Four channel 8-bit unsigned Gauss filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eMaskSize Enumeration value specifying the mask size.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterGauss_8u_C4R(const Npp8u * pSrc, Npp32s nSrcStep, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                       NppiMaskSize eMaskSize);

/**
 * Four channel 8-bit unsigned Gauss filter, ignoring alpha channel.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eMaskSize Enumeration value specifying the mask size.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterGauss_8u_AC4R(const Npp8u * pSrc, Npp32s nSrcStep, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                        NppiMaskSize eMaskSize);

/**
 * Single channel 16-bit unsigned Gauss filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eMaskSize Enumeration value specifying the mask size.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterGauss_16u_C1R(const Npp16u * pSrc, Npp32s nSrcStep, Npp16u * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                        NppiMaskSize eMaskSize);

/**
 * Three channel 16-bit unsigned Gauss filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eMaskSize Enumeration value specifying the mask size.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterGauss_16u_C3R(const Npp16u * pSrc, Npp32s nSrcStep, Npp16u * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                        NppiMaskSize eMaskSize);

/**
 * Four channel 16-bit unsigned Gauss filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eMaskSize Enumeration value specifying the mask size.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterGauss_16u_C4R(const Npp16u * pSrc, Npp32s nSrcStep, Npp16u * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                        NppiMaskSize eMaskSize);

/**
 * Four channel 16-bit unsigned Gauss filter, ignoring alpha channel.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eMaskSize Enumeration value specifying the mask size.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterGauss_16u_AC4R(const Npp16u * pSrc, Npp32s nSrcStep, Npp16u * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                         NppiMaskSize eMaskSize);

/**
 * Single channel 16-bit signed Gauss filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eMaskSize Enumeration value specifying the mask size.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterGauss_16s_C1R(const Npp16s * pSrc, Npp32s nSrcStep, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                        NppiMaskSize eMaskSize);

/**
 * Three channel 16-bit signed Gauss filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eMaskSize Enumeration value specifying the mask size.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterGauss_16s_C3R(const Npp16s * pSrc, Npp32s nSrcStep, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                        NppiMaskSize eMaskSize);

/**
 * Four channel 16-bit signed Gauss filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eMaskSize Enumeration value specifying the mask size.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterGauss_16s_C4R(const Npp16s * pSrc, Npp32s nSrcStep, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                        NppiMaskSize eMaskSize);

/**
 * Four channel 16-bit signed Gauss filter, ignoring alpha channel.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eMaskSize Enumeration value specifying the mask size.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterGauss_16s_AC4R(const Npp16s * pSrc, Npp32s nSrcStep, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                         NppiMaskSize eMaskSize);

/**
 * Single channel 32-bit floating-point Gauss filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eMaskSize Enumeration value specifying the mask size.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterGauss_32f_C1R(const Npp32f * pSrc, Npp32s nSrcStep, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                        NppiMaskSize eMaskSize);

/**
 * Three channel 32-bit floating-point Gauss filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eMaskSize Enumeration value specifying the mask size.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterGauss_32f_C3R(const Npp32f * pSrc, Npp32s nSrcStep, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                        NppiMaskSize eMaskSize);

/**
 * Four channel 32-bit floating-point Gauss filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eMaskSize Enumeration value specifying the mask size.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterGauss_32f_C4R(const Npp32f * pSrc, Npp32s nSrcStep, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                        NppiMaskSize eMaskSize);

/**
 * Four channel 32-bit floating-point Gauss filter, ignoring alpha channel.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eMaskSize Enumeration value specifying the mask size.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterGauss_32f_AC4R(const Npp32f * pSrc, Npp32s nSrcStep, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                         NppiMaskSize eMaskSize);

/** @} FilterGauss */

/** @name FilterGaussAdvanced
 *
 * Filters the image using a separable Gaussian filter kernel with user supplied floating point coefficients:
 *
 * @{
 *
 */                                                  

/**
 * Single channel 8-bit unsigned Gauss filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nFilterTaps The number of filter taps where nFilterTaps =  2 * ((int)((float)ceil(radius) + 0.5F) ) + 1.
 * \param pKernel Pointer to an array of nFilterTaps kernel coefficients which sum to 1.0F. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterGaussAdvanced_8u_C1R(const Npp8u * pSrc, Npp32s nSrcStep, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                               const int nFilterTaps, const Npp32f * pKernel);

/**
 * Three channel 8-bit unsigned Gauss filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nFilterTaps The number of filter taps where nFilterTaps =  2 * ((int)((float)ceil(radius) + 0.5F) ) + 1.
 * \param pKernel Pointer to an array of nFilterTaps kernel coefficients which sum to 1.0F. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterGaussAdvanced_8u_C3R(const Npp8u * pSrc, Npp32s nSrcStep, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                               const int nFilterTaps, const Npp32f * pKernel);

/**
 * Four channel 8-bit unsigned Gauss filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nFilterTaps The number of filter taps where nFilterTaps =  2 * ((int)((float)ceil(radius) + 0.5F) ) + 1.
 * \param pKernel Pointer to an array of nFilterTaps kernel coefficients which sum to 1.0F. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterGaussAdvanced_8u_C4R(const Npp8u * pSrc, Npp32s nSrcStep, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                               const int nFilterTaps, const Npp32f * pKernel);

/**
 * Four channel 8-bit unsigned Gauss filter, ignoring alpha channel.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nFilterTaps The number of filter taps where nFilterTaps =  2 * ((int)((float)ceil(radius) + 0.5F) ) + 1.
 * \param pKernel Pointer to an array of nFilterTaps kernel coefficients which sum to 1.0F. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterGaussAdvanced_8u_AC4R(const Npp8u * pSrc, Npp32s nSrcStep, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                                const int nFilterTaps, const Npp32f * pKernel);

/**
 * Single channel 16-bit unsigned Gauss filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nFilterTaps The number of filter taps where nFilterTaps =  2 * ((int)((float)ceil(radius) + 0.5F) ) + 1.
 * \param pKernel Pointer to an array of nFilterTaps kernel coefficients which sum to 1.0F. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterGaussAdvanced_16u_C1R(const Npp16u * pSrc, Npp32s nSrcStep, Npp16u * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                                const int nFilterTaps, const Npp32f * pKernel);

/**
 * Three channel 16-bit unsigned Gauss filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nFilterTaps The number of filter taps where nFilterTaps =  2 * ((int)((float)ceil(radius) + 0.5F) ) + 1.
 * \param pKernel Pointer to an array of nFilterTaps kernel coefficients which sum to 1.0F. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterGaussAdvanced_16u_C3R(const Npp16u * pSrc, Npp32s nSrcStep, Npp16u * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                                const int nFilterTaps, const Npp32f * pKernel);

/**
 * Four channel 16-bit unsigned Gauss filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nFilterTaps The number of filter taps where nFilterTaps =  2 * ((int)((float)ceil(radius) + 0.5F) ) + 1.
 * \param pKernel Pointer to an array of nFilterTaps kernel coefficients which sum to 1.0F. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterGaussAdvanced_16u_C4R(const Npp16u * pSrc, Npp32s nSrcStep, Npp16u * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                                const int nFilterTaps, const Npp32f * pKernel);

/**
 * Four channel 16-bit unsigned Gauss filter, ignoring alpha channel.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nFilterTaps The number of filter taps where nFilterTaps =  2 * ((int)((float)ceil(radius) + 0.5F) ) + 1.
 * \param pKernel Pointer to an array of nFilterTaps kernel coefficients which sum to 1.0F. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterGaussAdvanced_16u_AC4R(const Npp16u * pSrc, Npp32s nSrcStep, Npp16u * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                                 const int nFilterTaps, const Npp32f * pKernel);

/**
 * Single channel 16-bit signed Gauss filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nFilterTaps The number of filter taps where nFilterTaps =  2 * ((int)((float)ceil(radius) + 0.5F) ) + 1.
 * \param pKernel Pointer to an array of nFilterTaps kernel coefficients which sum to 1.0F. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterGaussAdvanced_16s_C1R(const Npp16s * pSrc, Npp32s nSrcStep, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                                const int nFilterTaps, const Npp32f * pKernel);

/**
 * Three channel 16-bit signed Gauss filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nFilterTaps The number of filter taps where nFilterTaps =  2 * ((int)((float)ceil(radius) + 0.5F) ) + 1.
 * \param pKernel Pointer to an array of nFilterTaps kernel coefficients which sum to 1.0F. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterGaussAdvanced_16s_C3R(const Npp16s * pSrc, Npp32s nSrcStep, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                                const int nFilterTaps, const Npp32f * pKernel);

/**
 * Four channel 16-bit signed Gauss filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nFilterTaps The number of filter taps where nFilterTaps =  2 * ((int)((float)ceil(radius) + 0.5F) ) + 1.
 * \param pKernel Pointer to an array of nFilterTaps kernel coefficients which sum to 1.0F. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterGaussAdvanced_16s_C4R(const Npp16s * pSrc, Npp32s nSrcStep, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                                const int nFilterTaps, const Npp32f * pKernel);

/**
 * Four channel 16-bit signed Gauss filter, ignoring alpha channel.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nFilterTaps The number of filter taps where nFilterTaps =  2 * ((int)((float)ceil(radius) + 0.5F) ) + 1.
 * \param pKernel Pointer to an array of nFilterTaps kernel coefficients which sum to 1.0F. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterGaussAdvanced_16s_AC4R(const Npp16s * pSrc, Npp32s nSrcStep, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                                 const int nFilterTaps, const Npp32f * pKernel);

/**
 * Single channel 32-bit floating-point Gauss filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nFilterTaps The number of filter taps where nFilterTaps =  2 * ((int)((float)ceil(radius) + 0.5F) ) + 1.
 * \param pKernel Pointer to an array of nFilterTaps kernel coefficients which sum to 1.0F. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterGaussAdvanced_32f_C1R(const Npp32f * pSrc, Npp32s nSrcStep, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                                const int nFilterTaps, const Npp32f * pKernel);

/**
 * Three channel 32-bit floating-point Gauss filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nFilterTaps The number of filter taps where nFilterTaps =  2 * ((int)((float)ceil(radius) + 0.5F) ) + 1.
 * \param pKernel Pointer to an array of nFilterTaps kernel coefficients which sum to 1.0F. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterGaussAdvanced_32f_C3R(const Npp32f * pSrc, Npp32s nSrcStep, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                                const int nFilterTaps, const Npp32f * pKernel);

/**
 * Four channel 32-bit floating-point Gauss filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nFilterTaps The number of filter taps where nFilterTaps =  2 * ((int)((float)ceil(radius) + 0.5F) ) + 1.
 * \param pKernel Pointer to an array of nFilterTaps kernel coefficients which sum to 1.0F. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterGaussAdvanced_32f_C4R(const Npp32f * pSrc, Npp32s nSrcStep, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                                const int nFilterTaps, const Npp32f * pKernel);

/**
 * Four channel 32-bit floating-point Gauss filter, ignoring alpha channel.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nFilterTaps The number of filter taps where nFilterTaps =  2 * ((int)((float)ceil(radius) + 0.5F) ) + 1.
 * \param pKernel Pointer to an array of nFilterTaps kernel coefficients which sum to 1.0F. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterGaussAdvanced_32f_AC4R(const Npp32f * pSrc, Npp32s nSrcStep, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                                 const int nFilterTaps, const Npp32f * pKernel);

/** @} FilterGaussAdvanced */

/** @name FilterGaussBorder
 *
 * If any portion of the mask overlaps the source
 * image boundary the requested border type operation is applied to all mask pixels
 * which fall outside of the source image.
 *
 * Lwrrently only the NPP_BORDER_REPLICATE border type operation is supported.
 *
 * Note that all FilterGaussBorder functions lwrrently support mask sizes up to 15x15. Filter kernels for these functions are callwlated
 * using a sigma value of 0.4F + (mask width / 2) * 0.6F.
 *
 * @{
 *
 */

/**
 * Single channel 8-bit unsigned Gauss filter with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eMaskSize Enumeration value specifying the mask size.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterGaussBorder_8u_C1R(const Npp8u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                             NppiMaskSize eMaskSize, NppiBorderType eBorderType);

/**
 * Three channel 8-bit unsigned Gauss filter with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eMaskSize Enumeration value specifying the mask size.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterGaussBorder_8u_C3R(const Npp8u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                             NppiMaskSize eMaskSize, NppiBorderType eBorderType);

/**
 * Four channel 8-bit unsigned Gauss filter with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eMaskSize Enumeration value specifying the mask size.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterGaussBorder_8u_C4R(const Npp8u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                             NppiMaskSize eMaskSize, NppiBorderType eBorderType);

/**
 * Four channel 8-bit unsigned Gauss filter with border control, ignoring alpha channel.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eMaskSize Enumeration value specifying the mask size.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterGaussBorder_8u_AC4R(const Npp8u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                              NppiMaskSize eMaskSize, NppiBorderType eBorderType);

/**
 * Single channel 16-bit unsigned Gauss filter with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eMaskSize Enumeration value specifying the mask size.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterGaussBorder_16u_C1R(const Npp16u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16u * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                              NppiMaskSize eMaskSize, NppiBorderType eBorderType);

/**
 * Three channel 16-bit unsigned Gauss filter with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eMaskSize Enumeration value specifying the mask size.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterGaussBorder_16u_C3R(const Npp16u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16u * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                              NppiMaskSize eMaskSize, NppiBorderType eBorderType);

/**
 * Four channel 16-bit unsigned Gauss filter with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eMaskSize Enumeration value specifying the mask size.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterGaussBorder_16u_C4R(const Npp16u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16u * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                              NppiMaskSize eMaskSize, NppiBorderType eBorderType);

/**
 * Four channel 16-bit unsigned Gauss filter with border control, ignoring alpha channel.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eMaskSize Enumeration value specifying the mask size.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterGaussBorder_16u_AC4R(const Npp16u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16u * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                               NppiMaskSize eMaskSize, NppiBorderType eBorderType);

/**
 * Single channel 16-bit signed Gauss filter with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eMaskSize Enumeration value specifying the mask size.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterGaussBorder_16s_C1R(const Npp16s * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                              NppiMaskSize eMaskSize, NppiBorderType eBorderType);

/**
 * Three channel 16-bit signed Gauss filter with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eMaskSize Enumeration value specifying the mask size.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterGaussBorder_16s_C3R(const Npp16s * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                              NppiMaskSize eMaskSize, NppiBorderType eBorderType);

/**
 * Four channel 16-bit signed Gauss filter with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eMaskSize Enumeration value specifying the mask size.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterGaussBorder_16s_C4R(const Npp16s * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                              NppiMaskSize eMaskSize, NppiBorderType eBorderType);

/**
 * Four channel 16-bit signed Gauss filter with border control, ignoring alpha channel.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eMaskSize Enumeration value specifying the mask size.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterGaussBorder_16s_AC4R(const Npp16s * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                               NppiMaskSize eMaskSize, NppiBorderType eBorderType);

/**
 * Single channel 32-bit floating-point Gauss filter with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eMaskSize Enumeration value specifying the mask size.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterGaussBorder_32f_C1R(const Npp32f * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                              NppiMaskSize eMaskSize, NppiBorderType eBorderType);

/**
 * Three channel 32-bit floating-point Gauss filter with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eMaskSize Enumeration value specifying the mask size.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterGaussBorder_32f_C3R(const Npp32f * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                              NppiMaskSize eMaskSize, NppiBorderType eBorderType);

/**
 * Four channel 32-bit floating-point Gauss filter with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eMaskSize Enumeration value specifying the mask size.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterGaussBorder_32f_C4R(const Npp32f * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                              NppiMaskSize eMaskSize, NppiBorderType eBorderType);

/**
 * Four channel 32-bit floating-point Gauss filter with border control, ignoring alpha channel.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eMaskSize Enumeration value specifying the mask size.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterGaussBorder_32f_AC4R(const Npp32f * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                               NppiMaskSize eMaskSize, NppiBorderType eBorderType);

/** @} FilterGaussBorder */

/** @name FilterGaussAdvancedBorder
 *
 * Filters the image using a separable Gaussian filter kernel with user supplied floating point coefficients with border control:
 * If any portion of the mask overlaps the source image boundary the requested border type operation is applied to all mask pixels
 * which fall outside of the source image.
 *
 * Lwrrently only the NPP_BORDER_REPLICATE and NPP_BORDER_MIRROR border type operations are supported.
 *
 * @{
 *
 */

/**
 * Single channel 8-bit unsigned Gauss filter with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nFilterTaps The number of filter taps where nFilterTaps =  2 * ((int)((float)ceil(radius) + 0.5F) ) + 1.
 * \param pKernel Pointer to an array of nFilterTaps kernel coefficients which sum to 1.0F. 
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterGaussAdvancedBorder_8u_C1R(const Npp8u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                                     const int nFilterTaps, const Npp32f * pKernel, NppiBorderType eBorderType);

/**
 * Three channel 8-bit unsigned Gauss filter with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nFilterTaps The number of filter taps where nFilterTaps =  2 * ((int)((float)ceil(radius) + 0.5F) ) + 1.
 * \param pKernel Pointer to an array of nFilterTaps kernel coefficients which sum to 1.0F. 
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterGaussAdvancedBorder_8u_C3R(const Npp8u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                                     const int nFilterTaps, const Npp32f * pKernel, NppiBorderType eBorderType);

/**
 * Four channel 8-bit unsigned Gauss filter with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nFilterTaps The number of filter taps where nFilterTaps =  2 * ((int)((float)ceil(radius) + 0.5F) ) + 1.
 * \param pKernel Pointer to an array of nFilterTaps kernel coefficients which sum to 1.0F. 
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterGaussAdvancedBorder_8u_C4R(const Npp8u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                                     const int nFilterTaps, const Npp32f * pKernel, NppiBorderType eBorderType);

/**
 * Four channel 8-bit unsigned Gauss filter with border control, ignoring alpha channel.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nFilterTaps The number of filter taps where nFilterTaps =  2 * ((int)((float)ceil(radius) + 0.5F) ) + 1.
 * \param pKernel Pointer to an array of nFilterTaps kernel coefficients which sum to 1.0F. 
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterGaussAdvancedBorder_8u_AC4R(const Npp8u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                                      const int nFilterTaps, const Npp32f * pKernel, NppiBorderType eBorderType);

/**
 * Single channel 16-bit unsigned Gauss filter with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nFilterTaps The number of filter taps where nFilterTaps =  2 * ((int)((float)ceil(radius) + 0.5F) ) + 1.
 * \param pKernel Pointer to an array of nFilterTaps kernel coefficients which sum to 1.0F. 
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterGaussAdvancedBorder_16u_C1R(const Npp16u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16u * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                                      const int nFilterTaps, const Npp32f * pKernel, NppiBorderType eBorderType);

/**
 * Three channel 16-bit unsigned Gauss filter with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nFilterTaps The number of filter taps where nFilterTaps =  2 * ((int)((float)ceil(radius) + 0.5F) ) + 1.
 * \param pKernel Pointer to an array of nFilterTaps kernel coefficients which sum to 1.0F. 
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterGaussAdvancedBorder_16u_C3R(const Npp16u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16u * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                                      const int nFilterTaps, const Npp32f * pKernel, NppiBorderType eBorderType);

/**
 * Four channel 16-bit unsigned Gauss filter with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nFilterTaps The number of filter taps where nFilterTaps =  2 * ((int)((float)ceil(radius) + 0.5F) ) + 1.
 * \param pKernel Pointer to an array of nFilterTaps kernel coefficients which sum to 1.0F. 
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterGaussAdvancedBorder_16u_C4R(const Npp16u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16u * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                                      const int nFilterTaps, const Npp32f * pKernel, NppiBorderType eBorderType);

/**
 * Four channel 16-bit unsigned Gauss filter with border control, ignoring alpha channel.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nFilterTaps The number of filter taps where nFilterTaps =  2 * ((int)((float)ceil(radius) + 0.5F) ) + 1.
 * \param pKernel Pointer to an array of nFilterTaps kernel coefficients which sum to 1.0F. 
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterGaussAdvancedBorder_16u_AC4R(const Npp16u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16u * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                                       const int nFilterTaps, const Npp32f * pKernel, NppiBorderType eBorderType);

/**
 * Single channel 16-bit signed Gauss filter with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nFilterTaps The number of filter taps where nFilterTaps =  2 * ((int)((float)ceil(radius) + 0.5F) ) + 1.
 * \param pKernel Pointer to an array of nFilterTaps kernel coefficients which sum to 1.0F. 
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterGaussAdvancedBorder_16s_C1R(const Npp16s * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                                      const int nFilterTaps, const Npp32f * pKernel, NppiBorderType eBorderType);

/**
 * Three channel 16-bit signed Gauss filter with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nFilterTaps The number of filter taps where nFilterTaps =  2 * ((int)((float)ceil(radius) + 0.5F) ) + 1.
 * \param pKernel Pointer to an array of nFilterTaps kernel coefficients which sum to 1.0F. 
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterGaussAdvancedBorder_16s_C3R(const Npp16s * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                                      const int nFilterTaps, const Npp32f * pKernel, NppiBorderType eBorderType);

/**
 * Four channel 16-bit signed Gauss filter with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nFilterTaps The number of filter taps where nFilterTaps =  2 * ((int)((float)ceil(radius) + 0.5F) ) + 1.
 * \param pKernel Pointer to an array of nFilterTaps kernel coefficients which sum to 1.0F. 
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterGaussAdvancedBorder_16s_C4R(const Npp16s * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                                      const int nFilterTaps, const Npp32f * pKernel, NppiBorderType eBorderType);

/**
 * Four channel 16-bit signed Gauss filter with border control, ignoring alpha channel.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nFilterTaps The number of filter taps where nFilterTaps =  2 * ((int)((float)ceil(radius) + 0.5F) ) + 1.
 * \param pKernel Pointer to an array of nFilterTaps kernel coefficients which sum to 1.0F. 
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterGaussAdvancedBorder_16s_AC4R(const Npp16s * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                                       const int nFilterTaps, const Npp32f * pKernel, NppiBorderType eBorderType);

/**
 * Single channel 32-bit floating-point Gauss filter with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nFilterTaps The number of filter taps where nFilterTaps =  2 * ((int)((float)ceil(radius) + 0.5F) ) + 1.
 * \param pKernel Pointer to an array of nFilterTaps kernel coefficients which sum to 1.0F. 
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterGaussAdvancedBorder_32f_C1R(const Npp32f * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                                      const int nFilterTaps, const Npp32f * pKernel, NppiBorderType eBorderType);

/**
 * Three channel 32-bit floating-point Gauss filter with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nFilterTaps The number of filter taps where nFilterTaps =  2 * ((int)((float)ceil(radius) + 0.5F) ) + 1.
 * \param pKernel Pointer to an array of nFilterTaps kernel coefficients which sum to 1.0F. 
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterGaussAdvancedBorder_32f_C3R(const Npp32f * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                                      const int nFilterTaps, const Npp32f * pKernel, NppiBorderType eBorderType);

/**
 * Four channel 32-bit floating-point Gauss filter with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nFilterTaps The number of filter taps where nFilterTaps =  2 * ((int)((float)ceil(radius) + 0.5F) ) + 1.
 * \param pKernel Pointer to an array of nFilterTaps kernel coefficients which sum to 1.0F. 
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterGaussAdvancedBorder_32f_C4R(const Npp32f * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                                      const int nFilterTaps, const Npp32f * pKernel, NppiBorderType eBorderType);

/**
 * Four channel 32-bit floating-point Gauss filter with border control, ignoring alpha channel.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nFilterTaps The number of filter taps where nFilterTaps =  2 * ((int)((float)ceil(radius) + 0.5F) ) + 1.
 * \param pKernel Pointer to an array of nFilterTaps kernel coefficients which sum to 1.0F. 
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterGaussAdvancedBorder_32f_AC4R(const Npp32f * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                                       const int nFilterTaps, const Npp32f * pKernel, NppiBorderType eBorderType);

/** @} FilterGaussAdvancedBorder */

/** @name FilterGaussPyramidLayerDownBorder
 *
 * Filters the image using a separable Gaussian filter kernel with user supplied floating point coefficients with downsampling and border control.
 * If the downsampling rate is equivalent to an integer value then unnecessary source pixels are just skipped.
 * If any portion of the mask overlaps the source image boundary the requested border type operation is applied to all mask pixels
 * which fall outside of the source image.
 *
 * Lwrrently only the NPP_BORDER_MIRROR and NPP_BORDER_REPLICATE border type operations are supported.
 *
 * @{
 *
 */

/**
 * Callwlate destination image SizeROI width and height from source image ROI width and height and downsampling rate.
 * It is highly recommended that this function be use to determine the destination image ROI for consistent results. 
 *
 * \param nSrcROIWidth The desired source image ROI width, must be <= oSrcSize.width.
 * \param nSrcROIHeight The desired source image ROI height, must be <= oSrcSize.height.
 * \param pDstSizeROI Host memory pointer to the destination image roi_specification.
 * \param nRate The downsampling rate to be used.  For integer equivalent rates unnecessary source pixels are just skipped.
 *              For non-integer rates the source image is bilinear interpolated. nRate must be > 1.0F and <= 10.0F. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */

NppStatus 
nppiGetFilterGaussPyramidLayerDownBorderDstROI(int nSrcROIWidth, int nSrcROIHeight, NppiSize * pDstSizeROI, Npp32f nRate);

/**
 * Single channel 8-bit unsigned Gauss filter with downsampling and border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nRate The downsampling rate to be used.  For integer equivalent rates unnecessary source pixels are just skipped.
 *              For non-integer rates the source image is bilinear interpolated. nRate must be > 1.0F and <= 10.0F. 
 * \param nFilterTaps The number of filter taps where nFilterTaps =  2 * ((int)((float)ceil(radius) + 0.5F) ) + 1.
 * \param pKernel Pointer to an array of nFilterTaps kernel coefficients which sum to 1.0F. 
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterGaussPyramidLayerDownBorder_8u_C1R(const Npp8u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                                             Npp32f nRate, const int nFilterTaps, const Npp32f * pKernel, NppiBorderType eBorderType);

/**
 * Three channel 8-bit unsigned Gauss filter with downsampling and border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nRate The downsampling rate to be used.  For integer equivalent rates unnecessary source pixels are just skipped.
 *              For non-integer rates the source image is bilinear interpolated. nRate must be > 1.0F and <= 10.0F. 
 * \param nFilterTaps The number of filter taps where nFilterTaps =  2 * ((int)((float)ceil(radius) + 0.5F) ) + 1.
 * \param pKernel Pointer to an array of nFilterTaps kernel coefficients which sum to 1.0F. 
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterGaussPyramidLayerDownBorder_8u_C3R(const Npp8u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                                             Npp32f nRate, const int nFilterTaps, const Npp32f * pKernel, NppiBorderType eBorderType);

/**
 * Single channel 16-bit unsigned Gauss filter with downsampling and border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nRate The downsampling rate to be used.  For integer equivalent rates unnecessary source pixels are just skipped.
 *              For non-integer rates the source image is bilinear interpolated. nRate must be > 1.0F and <= 10.0F. 
 * \param nFilterTaps The number of filter taps where nFilterTaps =  2 * ((int)((float)ceil(radius) + 0.5F) ) + 1.
 * \param pKernel Pointer to an array of nFilterTaps kernel coefficients which sum to 1.0F. 
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterGaussPyramidLayerDownBorder_16u_C1R(const Npp16u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16u * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                                              Npp32f nRate, const int nFilterTaps, const Npp32f * pKernel, NppiBorderType eBorderType);

/**
 * Three channel 16-bit unsigned Gauss filter with downsampling and border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nRate The downsampling rate to be used.  For integer equivalent rates unnecessary source pixels are just skipped.
 *              For non-integer rates the source image is bilinear interpolated. nRate must be > 1.0F and <= 10.0F. 
 * \param nFilterTaps The number of filter taps where nFilterTaps =  2 * ((int)((float)ceil(radius) + 0.5F) ) + 1.
 * \param pKernel Pointer to an array of nFilterTaps kernel coefficients which sum to 1.0F. 
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterGaussPyramidLayerDownBorder_16u_C3R(const Npp16u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16u * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                                              Npp32f nRate, const int nFilterTaps, const Npp32f * pKernel, NppiBorderType eBorderType);

/**
 * Single channel 32-bit floating-point Gauss filter downsampling and with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nRate The downsampling rate to be used.  For integer equivalent rates unnecessary source pixels are just skipped.
 *              For non-integer rates the source image is bilinear interpolated. nRate must be > 1.0F and <= 10.0F. 
 * \param nFilterTaps The number of filter taps where nFilterTaps =  2 * ((int)((float)ceil(radius) + 0.5F) ) + 1.
 * \param pKernel Pointer to an array of nFilterTaps kernel coefficients which sum to 1.0F. 
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterGaussPyramidLayerDownBorder_32f_C1R(const Npp32f * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                                              Npp32f nRate, const int nFilterTaps, const Npp32f * pKernel, NppiBorderType eBorderType);

/**
 * Three channel 32-bit floating-point Gauss filter with downsampling and border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nRate The downsampling rate to be used.  For integer equivalent rates unnecessary source pixels are just skipped.
 *              For non-integer rates the source image is bilinear interpolated. nRate must be > 1.0F and <= 10.0F. 
 * \param nFilterTaps The number of filter taps where nFilterTaps =  2 * ((int)((float)ceil(radius) + 0.5F) ) + 1.
 * \param pKernel Pointer to an array of nFilterTaps kernel coefficients which sum to 1.0F. 
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterGaussPyramidLayerDownBorder_32f_C3R(const Npp32f * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                                              Npp32f nRate, const int nFilterTaps, const Npp32f * pKernel, NppiBorderType eBorderType);

/** @} FilterGaussPyramidLayerDownBorder */

/** @name FilterGaussPyramidLayerUpBorder
 *
 * Filters the image using a separable Gaussian filter kernel with user supplied floating point coefficients with upsampling and border control.
 * If the upsampling rate is equivalent to an integer value then unnecessary source pixels are just skipped.
 * If any portion of the mask overlaps the source image boundary the requested border type operation is applied to all mask pixels
 * which fall outside of the source image.
 *
 * Lwrrently only the NPP_BORDER_MIRROR and NPP_BORDER_REPLICATE border type operations are supported.
 *
 * @{
 *
 */

/**
 * Callwlate destination image minimum and maximum SizeROI width and height from source image ROI width and height and upsampling rate.
 * It is highly recommended that this function be use to determine the best destination image ROI for consistent results. 
 *
 * \param nSrcROIWidth The desired source image ROI width, must be <= oSrcSize.width.
 * \param nSrcROIHeight The desired source image ROI height, must be <= oSrcSize.height.
 * \param pDstSizeROIMin Host memory pointer to the minimum recommended destination image roi_specification.
 * \param pDstSizeROIMax Host memory pointer to the maximum recommended destination image roi_specification.
 * \param nRate The upsampling rate to be used.  For integer equivalent rates unnecessary source pixels are just skipped.
 *              For non-integer rates the source image is bilinear interpolated. nRate must be > 1.0F and <= 10.0F. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */

NppStatus 
nppiGetFilterGaussPyramidLayerUpBorderDstROI(int nSrcROIWidth, int nSrcROIHeight, NppiSize * pDstSizeROIMin, NppiSize * pDstSizeROIMax, Npp32f nRate);

/**
 * Single channel 8-bit unsigned Gauss filter with upsampling and border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nRate The upsampling rate to be used.  For integer equivalent rates unnecessary source pixels are just skipped.
 *              For non-integer rates the source image is bilinear interpolated. nRate must be > 1.0F and <= 10.0F. 
 * \param nFilterTaps The number of filter taps where nFilterTaps =  2 * ((int)((float)ceil(radius) + 0.5F) ) + 1.
 * \param pKernel Pointer to an array of nFilterTaps kernel coefficients which sum to 1.0F. 
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterGaussPyramidLayerUpBorder_8u_C1R(const Npp8u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                                           Npp32f nRate, const int nFilterTaps, const Npp32f * pKernel, NppiBorderType eBorderType);

/**
 * Three channel 8-bit unsigned Gauss filter with upsampling and border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nRate The upsampling rate to be used.  For integer equivalent rates unnecessary source pixels are just skipped.
 *              For non-integer rates the source image is bilinear interpolated. nRate must be > 1.0F and <= 10.0F. 
 * \param nFilterTaps The number of filter taps where nFilterTaps =  2 * ((int)((float)ceil(radius) + 0.5F) ) + 1.
 * \param pKernel Pointer to an array of nFilterTaps kernel coefficients which sum to 1.0F. 
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterGaussPyramidLayerUpBorder_8u_C3R(const Npp8u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                                           Npp32f nRate, const int nFilterTaps, const Npp32f * pKernel, NppiBorderType eBorderType);

/**
 * Single channel 16-bit unsigned Gauss filter with upsampling and border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nRate The upsampling rate to be used.  For integer equivalent rates unnecessary source pixels are just skipped.
 *              For non-integer rates the source image is bilinear interpolated. nRate must be > 1.0F and <= 10.0F. 
 * \param nFilterTaps The number of filter taps where nFilterTaps =  2 * ((int)((float)ceil(radius) + 0.5F) ) + 1.
 * \param pKernel Pointer to an array of nFilterTaps kernel coefficients which sum to 1.0F. 
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterGaussPyramidLayerUpBorder_16u_C1R(const Npp16u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16u * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                                            Npp32f nRate, const int nFilterTaps, const Npp32f * pKernel, NppiBorderType eBorderType);

/**
 * Three channel 16-bit unsigned Gauss filter with upsampling and border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nRate The upsampling rate to be used.  For integer equivalent rates unnecessary source pixels are just skipped.
 *              For non-integer rates the source image is bilinear interpolated. nRate must be > 1.0F and <= 10.0F. 
 * \param nFilterTaps The number of filter taps where nFilterTaps =  2 * ((int)((float)ceil(radius) + 0.5F) ) + 1.
 * \param pKernel Pointer to an array of nFilterTaps kernel coefficients which sum to 1.0F. 
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterGaussPyramidLayerUpBorder_16u_C3R(const Npp16u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16u * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                                            Npp32f nRate, const int nFilterTaps, const Npp32f * pKernel, NppiBorderType eBorderType);

/**
 * Single channel 32-bit floating-point Gauss filter upsampling and with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nRate The upsampling rate to be used.  For integer equivalent rates unnecessary source pixels are just skipped.
 *              For non-integer rates the source image is bilinear interpolated. nRate must be > 1.0F and <= 10.0F. 
 * \param nFilterTaps The number of filter taps where nFilterTaps =  2 * ((int)((float)ceil(radius) + 0.5F) ) + 1.
 * \param pKernel Pointer to an array of nFilterTaps kernel coefficients which sum to 1.0F. 
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterGaussPyramidLayerUpBorder_32f_C1R(const Npp32f * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                                            Npp32f nRate, const int nFilterTaps, const Npp32f * pKernel, NppiBorderType eBorderType);

/**
 * Three channel 32-bit floating-point Gauss filter with upsampling and border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nRate The upsampling rate to be used.  For integer equivalent rates unnecessary source pixels are just skipped.
 *              For non-integer rates the source image is bilinear interpolated. nRate must be > 1.0F and <= 10.0F. 
 * \param nFilterTaps The number of filter taps where nFilterTaps =  2 * ((int)((float)ceil(radius) + 0.5F) ) + 1.
 * \param pKernel Pointer to an array of nFilterTaps kernel coefficients which sum to 1.0F. 
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterGaussPyramidLayerUpBorder_32f_C3R(const Npp32f * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                                            Npp32f nRate, const int nFilterTaps, const Npp32f * pKernel, NppiBorderType eBorderType);

/** @} FilterGaussPyramidLayerUpBorder */

/** @name FilterBilateralGaussBorder
 *
 * Filters the image using a bilateral Gaussian filter kernel with border control:
 * If any portion of the mask overlaps the source image boundary the requested border type operation is applied to all mask pixels
 * which fall outside of the source image.
 *
 * For this filter the anchor point is always the central element of the kernel. 
 * Coefficients of the bilateral filter kernel depend on their position in the kernel and 
 * on the value of some source image pixels overlayed by the filter kernel. 
 * Only source image pixels with both coordinates divisible by nDistanceBetweenSrcPixels are used in callwlations.
 *
 * The value of an output pixel \f$d\f$ is 
 * \f[d = \frac{\sum_{h=-nRadius}^{nRadius}\sum_{w=-nRadius}^{nRadius}W1(h,w)\cdot W2(h,w)\cdot S(h,w)}{\sum_{h=-nRadius}^{nRadius}\sum_{w=-nRadius}^{nRadius}W1(h,w)\cdot W2(h,w)}\f]
 * where h and w are the corresponding kernel width and height indexes, 
 * S(h,w) is the value of the source image pixel overlayed by filter kernel position (h,w),
 * W1(h,w) is func(lwalSquareSigma, (S(h,w) - S(0,0))) where S(0,0) is the value of the source image pixel at the center of the kernel,
 * W2(h,w) is func(nPosSquareSigma, sqrt(h*h+w*w)), and func is the following formula
 * \f[func(S,I) = exp(-\frac{I^2}{2.0F\cdot S^2})\f]
 *
 * Lwrrently only the NPP_BORDER_REPLICATE border type operations are supported.
 *
 * @{
 *
 */

/**
 * Single channel 8-bit unsigned bilateral Gauss filter with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nRadius The radius of the round filter kernel to be used.  A radius of 1 indicates a filter kernel size of 3 by 3, 2 indicates 5 by 5, etc.
 *        Radius values from 1 to 32 are supported.
 * \param nStepBetweenSrcPixels The step size between adjacent source image pixels processed by the filter kernel, most commonly 1. 
 * \param lwalSquareSigma The square of the sigma for the relative intensity distance between a source image pixel in the filter kernel 
 *        and the source image pixel at the center of the filter kernel.
 * \param nPosSquareSigma The square of the sigma for the relative geometric distance between a source image pixel in the filter kernel 
 *        and the source image pixel at the center of the filter kernel.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterBilateralGaussBorder_8u_C1R(const Npp8u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                                      const int nRadius, const int nStepBetweenSrcPixels, const Npp32f lwalSquareSigma, const Npp32f nPosSquareSigma, NppiBorderType eBorderType);

/**
 * Three channel 8-bit unsigned bilateral Gauss filter with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nRadius The radius of the round filter kernel to be used.  A radius of 1 indicates a filter kernel size of 3 by 3, 2 indicates 5 by 5, etc.
 *        Radius values from 1 to 32 are supported.
 * \param nStepBetweenSrcPixels The step size between adjacent source image pixels processed by the filter kernel, most commonly 1. 
 * \param lwalSquareSigma The square of the sigma for the relative intensity distance between a source image pixel in the filter kernel 
 *        and the source image pixel at the center of the filter kernel.
 * \param nPosSquareSigma The square of the sigma for the relative geometric distance between a source image pixel in the filter kernel 
 *        and the source image pixel at the center of the filter kernel.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterBilateralGaussBorder_8u_C3R(const Npp8u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                                      const int nRadius, const int nStepBetweenSrcPixels, const Npp32f lwalSquareSigma, const Npp32f nPosSquareSigma, NppiBorderType eBorderType);

/**
 * Single channel 16-bit unsigned bilateral Gauss filter with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nRadius The radius of the round filter kernel to be used.  A radius of 1 indicates a filter kernel size of 3 by 3, 2 indicates 5 by 5, etc.
 *        Radius values from 1 to 32 are supported.
 * \param nStepBetweenSrcPixels The step size between adjacent source image pixels processed by the filter kernel, most commonly 1. 
 * \param lwalSquareSigma The square of the sigma for the relative intensity distance between a source image pixel in the filter kernel 
 *        and the source image pixel at the center of the filter kernel.
 * \param nPosSquareSigma The square of the sigma for the relative geometric distance between a source image pixel in the filter kernel 
 *        and the source image pixel at the center of the filter kernel.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterBilateralGaussBorder_16u_C1R(const Npp16u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16u * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                                       const int nRadius, const int nStepBetweenSrcPixels, const Npp32f lwalSquareSigma, const Npp32f nPosSquareSigma, NppiBorderType eBorderType);

/**
 * Three channel 16-bit unsigned bilateral Gauss filter with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nRadius The radius of the round filter kernel to be used.  A radius of 1 indicates a filter kernel size of 3 by 3, 2 indicates 5 by 5, etc.
 *        Radius values from 1 to 32 are supported.
 * \param nStepBetweenSrcPixels The step size between adjacent source image pixels processed by the filter kernel, most commonly 1. 
 * \param lwalSquareSigma The square of the sigma for the relative intensity distance between a source image pixel in the filter kernel 
 *        and the source image pixel at the center of the filter kernel.
 * \param nPosSquareSigma The square of the sigma for the relative geometric distance between a source image pixel in the filter kernel 
 *        and the source image pixel at the center of the filter kernel.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterBilateralGaussBorder_16u_C3R(const Npp16u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16u * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                                       const int nRadius, const int nStepBetweenSrcPixels, const Npp32f lwalSquareSigma, const Npp32f nPosSquareSigma, NppiBorderType eBorderType);

/**
 * One channel 32-bit floating-point bilateral Gauss filter with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nRadius The radius of the round filter kernel to be used.  A radius of 1 indicates a filter kernel size of 3 by 3, 2 indicates 5 by 5, etc.
 *        Radius values from 1 to 32 are supported.
 * \param nStepBetweenSrcPixels The step size between adjacent source image pixels processed by the filter kernel, most commonly 1. 
 * \param lwalSquareSigma The square of the sigma for the relative intensity distance between a source image pixel in the filter kernel 
 *        and the source image pixel at the center of the filter kernel.
 * \param nPosSquareSigma The square of the sigma for the relative geometric distance between a source image pixel in the filter kernel 
 *        and the source image pixel at the center of the filter kernel.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterBilateralGaussBorder_32f_C1R(const Npp32f * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                                       const int nRadius, const int nStepBetweenSrcPixels, const Npp32f lwalSquareSigma, const Npp32f nPosSquareSigma, NppiBorderType eBorderType);

/**
 * Three channel 32-bit floating-point bilateral Gauss filter with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nRadius The radius of the round filter kernel to be used.  A radius of 1 indicates a filter kernel size of 3 by 3, 2 indicates 5 by 5, etc.
 *        Radius values from 1 to 32 are supported.
 * \param nStepBetweenSrcPixels The step size between adjacent source image pixels processed by the filter kernel, most commonly 1. 
 * \param lwalSquareSigma The square of the sigma for the relative intensity distance between a source image pixel in the filter kernel 
 *        and the source image pixel at the center of the filter kernel.
 * \param nPosSquareSigma The square of the sigma for the relative geometric distance between a source image pixel in the filter kernel 
 *        and the source image pixel at the center of the filter kernel.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterBilateralGaussBorder_32f_C3R(const Npp32f * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                                       const int nRadius, const int nStepBetweenSrcPixels, const Npp32f lwalSquareSigma, const Npp32f nPosSquareSigma, NppiBorderType eBorderType);

/** @} FilterBilateralGaussBorder */

/** @name FilterHighPass
 *
 * Filters the image using a high-pass filter kernel:
 *
 * \f[
 *  \left( \begin{array}{rrr}
 *      -1 & -1 & -1 \\
 *      -1 &  8 & -1 \\
 *      -1 & -1 & -1 \\
 *  \end{array} \right)
 *  \left( \begin{array}{rrrrr}
 *      -1 & -1 & -1 & -1 & -1 \\
 *      -1 & -1 & -1 & -1 & -1 \\
 *      -1 & -1 & 24 & -1 & -1 \\
 *      -1 & -1 & -1 & -1 & -1 \\
 *      -1 & -1 & -1 & -1 & -1 \\
 *  \end{array} \right)
 * \f]
 *
 * @{
 *
 */

/**
 * Single channel 8-bit unsigned high-pass filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eMaskSize Enumeration value specifying the mask size.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterHighPass_8u_C1R(const Npp8u * pSrc, Npp32s nSrcStep, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                          NppiMaskSize eMaskSize);

/**
 * Three channel 8-bit unsigned high-pass filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eMaskSize Enumeration value specifying the mask size.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterHighPass_8u_C3R(const Npp8u * pSrc, Npp32s nSrcStep, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                          NppiMaskSize eMaskSize);

/**
 * Four channel 8-bit unsigned high-pass filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eMaskSize Enumeration value specifying the mask size.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterHighPass_8u_C4R(const Npp8u * pSrc, Npp32s nSrcStep, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                          NppiMaskSize eMaskSize);

/**
 * Four channel 8-bit unsigned high-pass filter, ignoring alpha channel.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eMaskSize Enumeration value specifying the mask size.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterHighPass_8u_AC4R(const Npp8u * pSrc, Npp32s nSrcStep, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                           NppiMaskSize eMaskSize);

/**
 * Single channel 16-bit unsigned high-pass filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eMaskSize Enumeration value specifying the mask size.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterHighPass_16u_C1R(const Npp16u * pSrc, Npp32s nSrcStep, Npp16u * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                           NppiMaskSize eMaskSize);

/**
 * Three channel 16-bit unsigned high-pass filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eMaskSize Enumeration value specifying the mask size.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterHighPass_16u_C3R(const Npp16u * pSrc, Npp32s nSrcStep, Npp16u * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                           NppiMaskSize eMaskSize);

/**
 * Four channel 16-bit unsigned high-pass filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eMaskSize Enumeration value specifying the mask size.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterHighPass_16u_C4R(const Npp16u * pSrc, Npp32s nSrcStep, Npp16u * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                           NppiMaskSize eMaskSize);

/**
 * Four channel 16-bit unsigned high-pass filter, ignoring alpha channel.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eMaskSize Enumeration value specifying the mask size.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterHighPass_16u_AC4R(const Npp16u * pSrc, Npp32s nSrcStep, Npp16u * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                            NppiMaskSize eMaskSize);

/**
 * Single channel 16-bit signed high-pass filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eMaskSize Enumeration value specifying the mask size.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterHighPass_16s_C1R(const Npp16s * pSrc, Npp32s nSrcStep, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                           NppiMaskSize eMaskSize);

/**
 * Three channel 16-bit signed high-pass filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eMaskSize Enumeration value specifying the mask size.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterHighPass_16s_C3R(const Npp16s * pSrc, Npp32s nSrcStep, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                           NppiMaskSize eMaskSize);

/**
 * Four channel 16-bit signed high-pass filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eMaskSize Enumeration value specifying the mask size.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterHighPass_16s_C4R(const Npp16s * pSrc, Npp32s nSrcStep, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                           NppiMaskSize eMaskSize);

/**
 * Four channel 16-bit signed high-pass filter, ignoring alpha channel.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eMaskSize Enumeration value specifying the mask size.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterHighPass_16s_AC4R(const Npp16s * pSrc, Npp32s nSrcStep, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                            NppiMaskSize eMaskSize);

/**
 * Single channel 32-bit floating-point high-pass filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eMaskSize Enumeration value specifying the mask size.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterHighPass_32f_C1R(const Npp32f * pSrc, Npp32s nSrcStep, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                           NppiMaskSize eMaskSize);

/**
 * Three channel 32-bit floating-point high-pass filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eMaskSize Enumeration value specifying the mask size.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterHighPass_32f_C3R(const Npp32f * pSrc, Npp32s nSrcStep, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                           NppiMaskSize eMaskSize);

/**
 * Four channel 32-bit floating-point high-pass filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eMaskSize Enumeration value specifying the mask size.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterHighPass_32f_C4R(const Npp32f * pSrc, Npp32s nSrcStep, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                           NppiMaskSize eMaskSize);

/**
 * Four channel 32-bit floating-point high-pass filter, ignoring alpha channel.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eMaskSize Enumeration value specifying the mask size.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterHighPass_32f_AC4R(const Npp32f * pSrc, Npp32s nSrcStep, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                            NppiMaskSize eMaskSize);

/** @} FilterHighPass */

/** @name FilterHighPassBorder
 *
 * Filters the image using a high-pass filter kernel with border control.
 * If any portion of the mask overlaps the source image boundary the requested 
 * border type operation is applied to all mask pixels which fall outside of the source image.
 *
 * Lwrrently only the NPP_BORDER_REPLICATE border type operation is supported. 
 *
 *
 * \f[
 *  \left( \begin{array}{rrr}
 *      -1 & -1 & -1 \\
 *      -1 &  8 & -1 \\
 *      -1 & -1 & -1 \\
 *  \end{array} \right)
 *  \left( \begin{array}{rrrrr}
 *      -1 & -1 & -1 & -1 & -1 \\
 *      -1 & -1 & -1 & -1 & -1 \\
 *      -1 & -1 & 24 & -1 & -1 \\
 *      -1 & -1 & -1 & -1 & -1 \\
 *      -1 & -1 & -1 & -1 & -1 \\
 *  \end{array} \right)
 * \f]
 *
 * @{
 *
 */

/**
 * Single channel 8-bit unsigned high-pass filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eMaskSize Enumeration value specifying the mask size.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterHighPassBorder_8u_C1R(const Npp8u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                                NppiMaskSize eMaskSize, NppiBorderType eBorderType);

/**
 * Three channel 8-bit unsigned high-pass filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eMaskSize Enumeration value specifying the mask size.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterHighPassBorder_8u_C3R(const Npp8u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                                NppiMaskSize eMaskSize, NppiBorderType eBorderType);

/**
 * Four channel 8-bit unsigned high-pass filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eMaskSize Enumeration value specifying the mask size.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterHighPassBorder_8u_C4R(const Npp8u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                                NppiMaskSize eMaskSize, NppiBorderType eBorderType);

/**
 * Four channel 8-bit unsigned high-pass filter, ignoring alpha channel.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eMaskSize Enumeration value specifying the mask size.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterHighPassBorder_8u_AC4R(const Npp8u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                                 NppiMaskSize eMaskSize, NppiBorderType eBorderType);

/**
 * Single channel 16-bit unsigned high-pass filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eMaskSize Enumeration value specifying the mask size.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterHighPassBorder_16u_C1R(const Npp16u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16u * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                                 NppiMaskSize eMaskSize, NppiBorderType eBorderType);

/**
 * Three channel 16-bit unsigned high-pass filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eMaskSize Enumeration value specifying the mask size.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterHighPassBorder_16u_C3R(const Npp16u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16u * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                                 NppiMaskSize eMaskSize, NppiBorderType eBorderType);

/**
 * Four channel 16-bit unsigned high-pass filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eMaskSize Enumeration value specifying the mask size.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterHighPassBorder_16u_C4R(const Npp16u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16u * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                                 NppiMaskSize eMaskSize, NppiBorderType eBorderType);

/**
 * Four channel 16-bit unsigned high-pass filter, ignoring alpha channel.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eMaskSize Enumeration value specifying the mask size.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterHighPassBorder_16u_AC4R(const Npp16u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16u * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                                  NppiMaskSize eMaskSize, NppiBorderType eBorderType);

/**
 * Single channel 16-bit signed high-pass filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eMaskSize Enumeration value specifying the mask size.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterHighPassBorder_16s_C1R(const Npp16s * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                                 NppiMaskSize eMaskSize, NppiBorderType eBorderType);

/**
 * Three channel 16-bit signed high-pass filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eMaskSize Enumeration value specifying the mask size.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterHighPassBorder_16s_C3R(const Npp16s * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                                 NppiMaskSize eMaskSize, NppiBorderType eBorderType);

/**
 * Four channel 16-bit signed high-pass filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eMaskSize Enumeration value specifying the mask size.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterHighPassBorder_16s_C4R(const Npp16s * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                                 NppiMaskSize eMaskSize, NppiBorderType eBorderType);

/**
 * Four channel 16-bit signed high-pass filter, ignoring alpha channel.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eMaskSize Enumeration value specifying the mask size.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterHighPassBorder_16s_AC4R(const Npp16s * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                                  NppiMaskSize eMaskSize, NppiBorderType eBorderType);

/**
 * Single channel 32-bit floating-point high-pass filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eMaskSize Enumeration value specifying the mask size.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterHighPassBorder_32f_C1R(const Npp32f * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                                 NppiMaskSize eMaskSize, NppiBorderType eBorderType);

/**
 * Three channel 32-bit floating-point high-pass filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eMaskSize Enumeration value specifying the mask size.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterHighPassBorder_32f_C3R(const Npp32f * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                                 NppiMaskSize eMaskSize, NppiBorderType eBorderType);

/**
 * Four channel 32-bit floating-point high-pass filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eMaskSize Enumeration value specifying the mask size.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterHighPassBorder_32f_C4R(const Npp32f * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                                 NppiMaskSize eMaskSize, NppiBorderType eBorderType);

/**
 * Four channel 32-bit floating-point high-pass filter, ignoring alpha channel.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eMaskSize Enumeration value specifying the mask size.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterHighPassBorder_32f_AC4R(const Npp32f * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                                  NppiMaskSize eMaskSize, NppiBorderType eBorderType);

/** @} FilterHighPassBorder */

/** @name FilterLowPass
 *
 * Filters the image using a low-pass filter kernel:
 *
 * \f[
 *  \left( \begin{array}{rrr}
 *      1/9 & 1/9 & 1/9 \\
 *      1/9 & 1/9 & 1/9 \\
 *      1/9 & 1/9 & 1/9 \\
 *  \end{array} \right)
 *  \left( \begin{array}{rrrrr}
 *      1/25 & 1/25 & 1/25 & 1/25 & 1/25 \\
 *      1/25 & 1/25 & 1/25 & 1/25 & 1/25 \\
 *      1/25 & 1/25 & 1/25 & 1/25 & 1/25 \\
 *      1/25 & 1/25 & 1/25 & 1/25 & 1/25 \\
 *      1/25 & 1/25 & 1/25 & 1/25 & 1/25 \\
 *  \end{array} \right)
 * \f]
 *
 * @{
 *
 */

/**
 * Single channel 8-bit unsigned low-pass filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eMaskSize Enumeration value specifying the mask size.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterLowPass_8u_C1R(const Npp8u * pSrc, Npp32s nSrcStep, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                         NppiMaskSize eMaskSize);

/**
 * Three channel 8-bit unsigned low-pass filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eMaskSize Enumeration value specifying the mask size.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterLowPass_8u_C3R(const Npp8u * pSrc, Npp32s nSrcStep, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                         NppiMaskSize eMaskSize);

/**
 * Four channel 8-bit unsigned low-pass filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eMaskSize Enumeration value specifying the mask size.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterLowPass_8u_C4R(const Npp8u * pSrc, Npp32s nSrcStep, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                         NppiMaskSize eMaskSize);

/**
 * Four channel 8-bit unsigned low-pass filter, ignoring alpha channel.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eMaskSize Enumeration value specifying the mask size.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterLowPass_8u_AC4R(const Npp8u * pSrc, Npp32s nSrcStep, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                          NppiMaskSize eMaskSize);

/**
 * Single channel 16-bit unsigned low-pass filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eMaskSize Enumeration value specifying the mask size.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterLowPass_16u_C1R(const Npp16u * pSrc, Npp32s nSrcStep, Npp16u * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                          NppiMaskSize eMaskSize);

/**
 * Three channel 16-bit unsigned low-pass filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eMaskSize Enumeration value specifying the mask size.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterLowPass_16u_C3R(const Npp16u * pSrc, Npp32s nSrcStep, Npp16u * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                          NppiMaskSize eMaskSize);

/**
 * Four channel 16-bit unsigned low-pass filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eMaskSize Enumeration value specifying the mask size.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterLowPass_16u_C4R(const Npp16u * pSrc, Npp32s nSrcStep, Npp16u * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                          NppiMaskSize eMaskSize);

/**
 * Four channel 16-bit unsigned low-pass filter, ignoring alpha channel.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eMaskSize Enumeration value specifying the mask size.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterLowPass_16u_AC4R(const Npp16u * pSrc, Npp32s nSrcStep, Npp16u * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                           NppiMaskSize eMaskSize);

/**
 * Single channel 16-bit signed low-pass filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eMaskSize Enumeration value specifying the mask size.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterLowPass_16s_C1R(const Npp16s * pSrc, Npp32s nSrcStep, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                          NppiMaskSize eMaskSize);

/**
 * Three channel 16-bit signed low-pass filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eMaskSize Enumeration value specifying the mask size.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterLowPass_16s_C3R(const Npp16s * pSrc, Npp32s nSrcStep, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                          NppiMaskSize eMaskSize);

/**
 * Four channel 16-bit signed low-pass filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eMaskSize Enumeration value specifying the mask size.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterLowPass_16s_C4R(const Npp16s * pSrc, Npp32s nSrcStep, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                          NppiMaskSize eMaskSize);

/**
 * Four channel 16-bit signed low-pass filter, ignoring alpha channel.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eMaskSize Enumeration value specifying the mask size.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterLowPass_16s_AC4R(const Npp16s * pSrc, Npp32s nSrcStep, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                           NppiMaskSize eMaskSize);

/**
 * Single channel 32-bit floating-point low-pass filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eMaskSize Enumeration value specifying the mask size.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterLowPass_32f_C1R(const Npp32f * pSrc, Npp32s nSrcStep, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                          NppiMaskSize eMaskSize);

/**
 * Three channel 32-bit floating-point low-pass filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eMaskSize Enumeration value specifying the mask size.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterLowPass_32f_C3R(const Npp32f * pSrc, Npp32s nSrcStep, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                          NppiMaskSize eMaskSize);

/**
 * Four channel 32-bit floating-point low-pass filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eMaskSize Enumeration value specifying the mask size.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterLowPass_32f_C4R(const Npp32f * pSrc, Npp32s nSrcStep, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                          NppiMaskSize eMaskSize);

/**
 * Four channel 32-bit floating-point high-pass filter, ignoring alpha channel.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eMaskSize Enumeration value specifying the mask size.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterLowPass_32f_AC4R(const Npp32f * pSrc, Npp32s nSrcStep, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                           NppiMaskSize eMaskSize);

/** @} FilterLowPass */

/** @name FilterLowPassBorder
 *
 * Filters the image using a low-pass filter kernel with border control.
 * If any portion of the mask overlaps the source image boundary the requested 
 * border type operation is applied to all mask pixels which fall outside of the source image.
 *
 * Lwrrently only the NPP_BORDER_REPLICATE border type operation is supported. 
 *
 *
 * \f[
 *  \left( \begin{array}{rrr}
 *      1/9 & 1/9 & 1/9 \\
 *      1/9 & 1/9 & 1/9 \\
 *      1/9 & 1/9 & 1/9 \\
 *  \end{array} \right)
 *  \left( \begin{array}{rrrrr}
 *      1/25 & 1/25 & 1/25 & 1/25 & 1/25 \\
 *      1/25 & 1/25 & 1/25 & 1/25 & 1/25 \\
 *      1/25 & 1/25 & 1/25 & 1/25 & 1/25 \\
 *      1/25 & 1/25 & 1/25 & 1/25 & 1/25 \\
 *      1/25 & 1/25 & 1/25 & 1/25 & 1/25 \\
 *  \end{array} \right)
 * \f]
 *
 * @{
 *
 */

/**
 * Single channel 8-bit unsigned high-pass filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eMaskSize Enumeration value specifying the mask size.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterLowPassBorder_8u_C1R(const Npp8u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                                NppiMaskSize eMaskSize, NppiBorderType eBorderType);

/**
 * Three channel 8-bit unsigned high-pass filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eMaskSize Enumeration value specifying the mask size.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterLowPassBorder_8u_C3R(const Npp8u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                                NppiMaskSize eMaskSize, NppiBorderType eBorderType);

/**
 * Four channel 8-bit unsigned high-pass filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eMaskSize Enumeration value specifying the mask size.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterLowPassBorder_8u_C4R(const Npp8u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                                NppiMaskSize eMaskSize, NppiBorderType eBorderType);

/**
 * Four channel 8-bit unsigned high-pass filter, ignoring alpha channel.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eMaskSize Enumeration value specifying the mask size.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterLowPassBorder_8u_AC4R(const Npp8u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                                 NppiMaskSize eMaskSize, NppiBorderType eBorderType);

/**
 * Single channel 16-bit unsigned high-pass filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eMaskSize Enumeration value specifying the mask size.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterLowPassBorder_16u_C1R(const Npp16u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16u * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                                 NppiMaskSize eMaskSize, NppiBorderType eBorderType);

/**
 * Three channel 16-bit unsigned high-pass filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eMaskSize Enumeration value specifying the mask size.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterLowPassBorder_16u_C3R(const Npp16u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16u * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                                 NppiMaskSize eMaskSize, NppiBorderType eBorderType);

/**
 * Four channel 16-bit unsigned high-pass filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eMaskSize Enumeration value specifying the mask size.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterLowPassBorder_16u_C4R(const Npp16u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16u * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                                 NppiMaskSize eMaskSize, NppiBorderType eBorderType);

/**
 * Four channel 16-bit unsigned high-pass filter, ignoring alpha channel.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eMaskSize Enumeration value specifying the mask size.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterLowPassBorder_16u_AC4R(const Npp16u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16u * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                                  NppiMaskSize eMaskSize, NppiBorderType eBorderType);

/**
 * Single channel 16-bit signed high-pass filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eMaskSize Enumeration value specifying the mask size.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterLowPassBorder_16s_C1R(const Npp16s * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                                 NppiMaskSize eMaskSize, NppiBorderType eBorderType);

/**
 * Three channel 16-bit signed high-pass filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eMaskSize Enumeration value specifying the mask size.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterLowPassBorder_16s_C3R(const Npp16s * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                                 NppiMaskSize eMaskSize, NppiBorderType eBorderType);

/**
 * Four channel 16-bit signed high-pass filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eMaskSize Enumeration value specifying the mask size.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterLowPassBorder_16s_C4R(const Npp16s * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                                 NppiMaskSize eMaskSize, NppiBorderType eBorderType);

/**
 * Four channel 16-bit signed high-pass filter, ignoring alpha channel.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eMaskSize Enumeration value specifying the mask size.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterLowPassBorder_16s_AC4R(const Npp16s * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                                  NppiMaskSize eMaskSize, NppiBorderType eBorderType);

/**
 * Single channel 32-bit floating-point high-pass filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eMaskSize Enumeration value specifying the mask size.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterLowPassBorder_32f_C1R(const Npp32f * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                                 NppiMaskSize eMaskSize, NppiBorderType eBorderType);

/**
 * Three channel 32-bit floating-point high-pass filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eMaskSize Enumeration value specifying the mask size.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterLowPassBorder_32f_C3R(const Npp32f * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                                 NppiMaskSize eMaskSize, NppiBorderType eBorderType);

/**
 * Four channel 32-bit floating-point high-pass filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eMaskSize Enumeration value specifying the mask size.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterLowPassBorder_32f_C4R(const Npp32f * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                                 NppiMaskSize eMaskSize, NppiBorderType eBorderType);

/**
 * Four channel 32-bit floating-point high-pass filter, ignoring alpha channel.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eMaskSize Enumeration value specifying the mask size.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterLowPassBorder_32f_AC4R(const Npp32f * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI,
                                  NppiMaskSize eMaskSize, NppiBorderType eBorderType);

/** @} FilterLowPassBorder */

/** @name FilterSharpen
 *
 * Filters the image using a sharpening filter kernel:
 *
 * \f[
 *  \left( \begin{array}{rrr}
 *      -1/8 & -1/8 & -1/8 \\
 *      -1/8 & 16/8 & -1/8 \\
 *      -1/8 & -1/8 & -1/8 \\
 *  \end{array} \right)
 * \f]
 *
 * @{
 *
 */

/**
 * Single channel 8-bit unsigned sharpening filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterSharpen_8u_C1R(const Npp8u * pSrc, Npp32s nSrcStep, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI);

/**
 * Three channel 8-bit unsigned sharpening filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterSharpen_8u_C3R(const Npp8u * pSrc, Npp32s nSrcStep, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI);

/**
 * Four channel 8-bit unsigned sharpening filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterSharpen_8u_C4R(const Npp8u * pSrc, Npp32s nSrcStep, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI);

/**
 * Four channel 8-bit unsigned sharpening filter, ignoring alpha channel.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterSharpen_8u_AC4R(const Npp8u * pSrc, Npp32s nSrcStep, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI);

/**
 * Single channel 16-bit unsigned sharpening filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterSharpen_16u_C1R(const Npp16u * pSrc, Npp32s nSrcStep, Npp16u * pDst, Npp32s nDstStep, NppiSize oSizeROI);

/**
 * Three channel 16-bit unsigned sharpening filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterSharpen_16u_C3R(const Npp16u * pSrc, Npp32s nSrcStep, Npp16u * pDst, Npp32s nDstStep, NppiSize oSizeROI);

/**
 * Four channel 16-bit unsigned sharpening filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterSharpen_16u_C4R(const Npp16u * pSrc, Npp32s nSrcStep, Npp16u * pDst, Npp32s nDstStep, NppiSize oSizeROI);

/**
 * Four channel 16-bit unsigned sharpening filter, ignoring alpha channel.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterSharpen_16u_AC4R(const Npp16u * pSrc, Npp32s nSrcStep, Npp16u * pDst, Npp32s nDstStep, NppiSize oSizeROI);

/**
 * Single channel 16-bit signed sharpening filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterSharpen_16s_C1R(const Npp16s * pSrc, Npp32s nSrcStep, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI);

/**
 * Three channel 16-bit signed sharpening filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterSharpen_16s_C3R(const Npp16s * pSrc, Npp32s nSrcStep, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI);

/**
 * Four channel 16-bit signed sharpening filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterSharpen_16s_C4R(const Npp16s * pSrc, Npp32s nSrcStep, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI);

/**
 * Four channel 16-bit signed sharpening filter, ignoring alpha channel.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterSharpen_16s_AC4R(const Npp16s * pSrc, Npp32s nSrcStep, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI);

/**
 * Single channel 32-bit floating-point sharpening filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterSharpen_32f_C1R(const Npp32f * pSrc, Npp32s nSrcStep, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI);

/**
 * Three channel 32-bit floating-point sharpening filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterSharpen_32f_C3R(const Npp32f * pSrc, Npp32s nSrcStep, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI);

/**
 * Four channel 32-bit floating-point sharpening filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterSharpen_32f_C4R(const Npp32f * pSrc, Npp32s nSrcStep, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI);

/**
 * Four channel 32-bit floating-point sharpening filter, ignoring alpha channel.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterSharpen_32f_AC4R(const Npp32f * pSrc, Npp32s nSrcStep, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI);

/** @} FilterSharpen */

/** @name FilterSharpenBorder
 *
 * Filters the image using a sharpening filter kernel with border control. If any portion of the 3x3 mask overlaps the source
 * image boundary the requested border type operation is applied to all mask pixels
 * which fall outside of the source image.
 *
 * Lwrrently only the NPP_BORDER_REPLICATE border type operation is supported.
 *
 * \f[
 *  \left( \begin{array}{rrr}
 *      -1/8 & -1/8 & -1/8 \\
 *      -1/8 & 16/8 & -1/8 \\
 *      -1/8 & -1/8 & -1/8 \\
 *  \end{array} \right)
 * \f]
 *
 * @{
 *
 */

/**
 * Single channel 8-bit unsigned sharpening filter with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterSharpenBorder_8u_C1R(const Npp8u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiBorderType eBorderType);

/**
 * Three channel 8-bit unsigned sharpening filter with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterSharpenBorder_8u_C3R(const Npp8u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiBorderType eBorderType);

/**
 * Four channel 8-bit unsigned sharpening filter with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterSharpenBorder_8u_C4R(const Npp8u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiBorderType eBorderType);

/**
 * Four channel 8-bit unsigned sharpening filter with border control, ignoring alpha channel.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterSharpenBorder_8u_AC4R(const Npp8u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiBorderType eBorderType);

/**
 * Single channel 16-bit unsigned sharpening filter with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterSharpenBorder_16u_C1R(const Npp16u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16u * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiBorderType eBorderType);

/**
 * Three channel 16-bit unsigned sharpening filter with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterSharpenBorder_16u_C3R(const Npp16u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16u * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiBorderType eBorderType);

/**
 * Four channel 16-bit unsigned sharpening filter with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterSharpenBorder_16u_C4R(const Npp16u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16u * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiBorderType eBorderType);

/**
 * Four channel 16-bit unsigned sharpening filter with border control, ignoring alpha channel.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterSharpenBorder_16u_AC4R(const Npp16u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16u * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiBorderType eBorderType);

/**
 * Single channel 16-bit signed sharpening filter with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterSharpenBorder_16s_C1R(const Npp16s * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiBorderType eBorderType);

/**
 * Three channel 16-bit signed sharpening filter with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterSharpenBorder_16s_C3R(const Npp16s * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiBorderType eBorderType);

/**
 * Four channel 16-bit signed sharpening filter with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterSharpenBorder_16s_C4R(const Npp16s * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiBorderType eBorderType);

/**
 * Four channel 16-bit signed sharpening filter with border control, ignoring alpha channel.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterSharpenBorder_16s_AC4R(const Npp16s * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiBorderType eBorderType);

/**
 * Single channel 32-bit floating-point sharpening filter with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterSharpenBorder_32f_C1R(const Npp32f * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiBorderType eBorderType);

/**
 * Three channel 32-bit floating-point sharpening filter with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterSharpenBorder_32f_C3R(const Npp32f * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiBorderType eBorderType);

/**
 * Four channel 32-bit floating-point sharpening filter with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterSharpenBorder_32f_C4R(const Npp32f * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiBorderType eBorderType);

/**
 * Four channel 32-bit floating-point sharpening filter with border control, ignoring alpha channel.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterSharpenBorder_32f_AC4R(const Npp32f * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiBorderType eBorderType);

/** @} FilterSharpenBorder */

/** @name FilterUnsharpBorder
 *
 * Filters the image using a unsharp-mask sharpening filter kernel with border control.
 *
 * The algorithm ilwolves the following steps:
 * Smooth the original image with a Gaussian filter, with the width controlled by the nRadius.
 * Subtract the smoothed image from the original to create a high-pass filtered image.
 * Apply any clipping needed on the high-pass image, as controlled by the nThreshold.
 * Add a certain percentage of the high-pass filtered image to the original image, 
 * with the percentage controlled by the nWeight.
 * In pseudocode this algorithm can be written as:
 * HighPass = Image - Gaussian(Image)
 * Result = Image + nWeight * HighPass * ( |HighPass| >= nThreshold ) 
 * where nWeight is the amount, nThreshold is the threshold, and >= indicates a Boolean operation, 1 if true, or 0 otherwise.
 *
 * If any portion of the mask overlaps the source image boundary, the requested border type 
 * operation is applied to all mask pixels which fall outside of the source image.
 *
 * Lwrrently only the NPP_BORDER_REPLICATE border type operation is supported.
 *
 * @{
 *
 */

/**
 * Single channel 8-bit unsigned unsharp filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nRadius The radius of the Gaussian filter, in pixles, not counting the center pixel.
 * \param nSigma The standard deviation of the Gaussian filter, in pixel.
 * \param nWeight The percentage of the difference between the original and the high pass image that is added back into the original.
 * \param nThreshold The threshold neede to apply the difference amount.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \param pDeviceBuffer Pointer to the user-allocated device scratch buffer required for the unsharp operation.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterUnsharpBorder_8u_C1R(const Npp8u * pSrc, Npp32s nSrcStep, NppiPoint oSrcOffset, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI, Npp32f nRadius, Npp32f nSigma, Npp32f nWeight, Npp32f nThreshold, NppiBorderType eBorderType, Npp8u * pDeviceBuffer);

/**
 * Three channel 8-bit unsigned unsharp filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nRadius The radius of the Gaussian filter, in pixles, not counting the center pixel.
 * \param nSigma The standard deviation of the Gaussian filter, in pixel.
 * \param nWeight The percentage of the difference between the original and the high pass image that is added back into the original.
 * \param nThreshold The threshold neede to apply the difference amount.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \param pDeviceBuffer Pointer to the user-allocated device scratch buffer required for the unsharp operation.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterUnsharpBorder_8u_C3R(const Npp8u * pSrc, Npp32s nSrcStep, NppiPoint oSrcOffset, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI, Npp32f nRadius, Npp32f nSigma, Npp32f nWeight, Npp32f nThreshold, NppiBorderType eBorderType, Npp8u * pDeviceBuffer);

/**
 * Four channel 8-bit unsigned unsharp filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nRadius The radius of the Gaussian filter, in pixles, not counting the center pixel.
 * \param nSigma The standard deviation of the Gaussian filter, in pixel.
 * \param nWeight The percentage of the difference between the original and the high pass image that is added back into the original.
 * \param nThreshold The threshold neede to apply the difference amount.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \param pDeviceBuffer Pointer to the user-allocated device scratch buffer required for the unsharp operation.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterUnsharpBorder_8u_C4R(const Npp8u * pSrc, Npp32s nSrcStep, NppiPoint oSrcOffset, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI, Npp32f nRadius, Npp32f nSigma, Npp32f nWeight, Npp32f nThreshold, NppiBorderType eBorderType, Npp8u * pDeviceBuffer);

/**
 * Four channel 8-bit unsigned unsharp filter (alpha channel is not processed).
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nRadius The radius of the Gaussian filter, in pixles, not counting the center pixel.
 * \param nSigma The standard deviation of the Gaussian filter, in pixel.
 * \param nWeight The percentage of the difference between the original and the high pass image that is added back into the original.
 * \param nThreshold The threshold neede to apply the difference amount.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \param pDeviceBuffer Pointer to the user-allocated device scratch buffer required for the unsharp operation.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterUnsharpBorder_8u_AC4R(const Npp8u * pSrc, Npp32s nSrcStep, NppiPoint oSrcOffset, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI, Npp32f nRadius, Npp32f nSigma, Npp32f nWeight, Npp32f nThreshold, NppiBorderType eBorderType, Npp8u * pDeviceBuffer);

/**
 * Single channel 16-bit unsigned unsharp filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nRadius The radius of the Gaussian filter, in pixles, not counting the center pixel.
 * \param nSigma The standard deviation of the Gaussian filter, in pixel.
 * \param nWeight The percentage of the difference between the original and the high pass image that is added back into the original.
 * \param nThreshold The threshold neede to apply the difference amount.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \param pDeviceBuffer Pointer to the user-allocated device scratch buffer required for the unsharp operation.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterUnsharpBorder_16u_C1R(const Npp16u * pSrc, Npp32s nSrcStep, NppiPoint oSrcOffset, Npp16u * pDst, Npp32s nDstStep, NppiSize oSizeROI, Npp32f nRadius, Npp32f nSigma, Npp32f nWeight, Npp32f nThreshold, NppiBorderType eBorderType, Npp8u * pDeviceBuffer);

/**
 * Three channel 16-bit unsigned unsharp filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nRadius The radius of the Gaussian filter, in pixles, not counting the center pixel.
 * \param nSigma The standard deviation of the Gaussian filter, in pixel.
 * \param nWeight The percentage of the difference between the original and the high pass image that is added back into the original.
 * \param nThreshold The threshold neede to apply the difference amount.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \param pDeviceBuffer Pointer to the user-allocated device scratch buffer required for the unsharp operation.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterUnsharpBorder_16u_C3R(const Npp16u * pSrc, Npp32s nSrcStep, NppiPoint oSrcOffset, Npp16u * pDst, Npp32s nDstStep, NppiSize oSizeROI, Npp32f nRadius, Npp32f nSigma, Npp32f nWeight, Npp32f nThreshold, NppiBorderType eBorderType, Npp8u * pDeviceBuffer);

/**
 * Four channel 16-bit unsigned unsharp filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nRadius The radius of the Gaussian filter, in pixles, not counting the center pixel.
 * \param nSigma The standard deviation of the Gaussian filter, in pixel.
 * \param nWeight The percentage of the difference between the original and the high pass image that is added back into the original.
 * \param nThreshold The threshold neede to apply the difference amount.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \param pDeviceBuffer Pointer to the user-allocated device scratch buffer required for the unsharp operation.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterUnsharpBorder_16u_C4R(const Npp16u * pSrc, Npp32s nSrcStep, NppiPoint oSrcOffset, Npp16u * pDst, Npp32s nDstStep, NppiSize oSizeROI, Npp32f nRadius, Npp32f nSigma, Npp32f nWeight, Npp32f nThreshold, NppiBorderType eBorderType, Npp8u * pDeviceBuffer);

/**
 * Four channel 16-bit unsigned unsharp filter (alpha channel is not processed).
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nRadius The radius of the Gaussian filter, in pixles, not counting the center pixel.
 * \param nSigma The standard deviation of the Gaussian filter, in pixel.
 * \param nWeight The percentage of the difference between the original and the high pass image that is added back into the original.
 * \param nThreshold The threshold neede to apply the difference amount.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \param pDeviceBuffer Pointer to the user-allocated device scratch buffer required for the unsharp operation.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterUnsharpBorder_16u_AC4R(const Npp16u * pSrc, Npp32s nSrcStep, NppiPoint oSrcOffset, Npp16u * pDst, Npp32s nDstStep, NppiSize oSizeROI, Npp32f nRadius, Npp32f nSigma, Npp32f nWeight, Npp32f nThreshold, NppiBorderType eBorderType, Npp8u * pDeviceBuffer);

/**
 * Single channel 16-bit signed unsharp filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nRadius The radius of the Gaussian filter, in pixles, not counting the center pixel.
 * \param nSigma The standard deviation of the Gaussian filter, in pixel.
 * \param nWeight The percentage of the difference between the original and the high pass image that is added back into the original.
 * \param nThreshold The threshold neede to apply the difference amount.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \param pDeviceBuffer Pointer to the user-allocated device scratch buffer required for the unsharp operation.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterUnsharpBorder_16s_C1R(const Npp16s * pSrc, Npp32s nSrcStep, NppiPoint oSrcOffset, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI, Npp32f nRadius, Npp32f nSigma, Npp32f nWeight, Npp32f nThreshold, NppiBorderType eBorderType, Npp8u * pDeviceBuffer);

/**
 * Three channel 16-bit signed unsharp filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nRadius The radius of the Gaussian filter, in pixles, not counting the center pixel.
 * \param nSigma The standard deviation of the Gaussian filter, in pixel.
 * \param nWeight The percentage of the difference between the original and the high pass image that is added back into the original.
 * \param nThreshold The threshold neede to apply the difference amount.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \param pDeviceBuffer Pointer to the user-allocated device scratch buffer required for the unsharp operation.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterUnsharpBorder_16s_C3R(const Npp16s * pSrc, Npp32s nSrcStep, NppiPoint oSrcOffset, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI, Npp32f nRadius, Npp32f nSigma, Npp32f nWeight, Npp32f nThreshold, NppiBorderType eBorderType, Npp8u * pDeviceBuffer);

/**
 * Four channel 16-bit signed unsharp filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nRadius The radius of the Gaussian filter, in pixles, not counting the center pixel.
 * \param nSigma The standard deviation of the Gaussian filter, in pixel.
 * \param nWeight The percentage of the difference between the original and the high pass image that is added back into the original.
 * \param nThreshold The threshold neede to apply the difference amount.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \param pDeviceBuffer Pointer to the user-allocated device scratch buffer required for the unsharp operation.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterUnsharpBorder_16s_C4R(const Npp16s * pSrc, Npp32s nSrcStep, NppiPoint oSrcOffset, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI, Npp32f nRadius, Npp32f nSigma, Npp32f nWeight, Npp32f nThreshold, NppiBorderType eBorderType, Npp8u * pDeviceBuffer);

/**
 * Four channel 16-bit signed unsharp filter (alpha channel is not processed).
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nRadius The radius of the Gaussian filter, in pixles, not counting the center pixel.
 * \param nSigma The standard deviation of the Gaussian filter, in pixel.
 * \param nWeight The percentage of the difference between the original and the high pass image that is added back into the original.
 * \param nThreshold The threshold neede to apply the difference amount.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \param pDeviceBuffer Pointer to the user-allocated device scratch buffer required for the unsharp operation.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterUnsharpBorder_16s_AC4R(const Npp16s * pSrc, Npp32s nSrcStep, NppiPoint oSrcOffset, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI, Npp32f nRadius, Npp32f nSigma, Npp32f nWeight, Npp32f nThreshold, NppiBorderType eBorderType, Npp8u * pDeviceBuffer);

/**
 * Single channel 32-bit floating point unsharp filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nRadius The radius of the Gaussian filter, in pixles, not counting the center pixel.
 * \param nSigma The standard deviation of the Gaussian filter, in pixel.
 * \param nWeight The percentage of the difference between the original and the high pass image that is added back into the original.
 * \param nThreshold The threshold neede to apply the difference amount.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \param pDeviceBuffer Pointer to the user-allocated device scratch buffer required for the unsharp operation.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterUnsharpBorder_32f_C1R(const Npp32f * pSrc, Npp32s nSrcStep, NppiPoint oSrcOffset, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI, Npp32f nRadius, Npp32f nSigma, Npp32f nWeight, Npp32f nThreshold, NppiBorderType eBorderType, Npp8u * pDeviceBuffer);

/**
 * Three channel 32-bit floating point unsharp filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nRadius The radius of the Gaussian filter, in pixles, not counting the center pixel.
 * \param nSigma The standard deviation of the Gaussian filter, in pixel.
 * \param nWeight The percentage of the difference between the original and the high pass image that is added back into the original.
 * \param nThreshold The threshold neede to apply the difference amount.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \param pDeviceBuffer Pointer to the user-allocated device scratch buffer required for the unsharp operation.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterUnsharpBorder_32f_C3R(const Npp32f * pSrc, Npp32s nSrcStep, NppiPoint oSrcOffset, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI, Npp32f nRadius, Npp32f nSigma, Npp32f nWeight, Npp32f nThreshold, NppiBorderType eBorderType, Npp8u * pDeviceBuffer);

/**
 * Four channel 32-bit floating point unsharp filter.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nRadius The radius of the Gaussian filter, in pixles, not counting the center pixel.
 * \param nSigma The standard deviation of the Gaussian filter, in pixel.
 * \param nWeight The percentage of the difference between the original and the high pass image that is added back into the original.
 * \param nThreshold The threshold neede to apply the difference amount.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \param pDeviceBuffer Pointer to the user-allocated device scratch buffer required for the unsharp operation.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterUnsharpBorder_32f_C4R(const Npp32f * pSrc, Npp32s nSrcStep, NppiPoint oSrcOffset, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI, Npp32f nRadius, Npp32f nSigma, Npp32f nWeight, Npp32f nThreshold, NppiBorderType eBorderType, Npp8u * pDeviceBuffer);

/**
 * Four channel 32-bit floating point unsharp filter (alpha channel is not processed).
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nRadius The radius of the Gaussian filter, in pixles, not counting the center pixel.
 * \param nSigma The standard deviation of the Gaussian filter, in pixel.
 * \param nWeight The percentage of the difference between the original and the high pass image that is added back into the original.
 * \param nThreshold The threshold neede to apply the difference amount.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \param pDeviceBuffer Pointer to the user-allocated device scratch buffer required for the unsharp operation.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterUnsharpBorder_32f_AC4R(const Npp32f * pSrc, Npp32s nSrcStep, NppiPoint oSrcOffset, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI, Npp32f nRadius, Npp32f nSigma, Npp32f nWeight, Npp32f nThreshold, NppiBorderType eBorderType, Npp8u * pDeviceBuffer);

/**
 * Single channel 8-bit unsigned unsharp filter scratch memory size.
 * \param nRadius The radius of the Gaussian filter, in pixles, not counting the center pixel.
 * \param nSigma The standard deviation of the Gaussian filter, in pixel.
 * \param hpBufferSize Pointer to the size of the scratch buffer required for the unsharp operation.
 * \return \ref image_data_error_codes
 */
NppStatus 
nppiFilterUnsharpGetBufferSize_8u_C1R(const Npp32f nRadius, const Npp32f nSigma, int * hpBufferSize);

/**
 * Three channel 8-bit unsigned unsharp filter scratch memory size.
 * \param nRadius The radius of the Gaussian filter, in pixles, not counting the center pixel.
 * \param nSigma The standard deviation of the Gaussian filter, in pixel.
 * \param hpBufferSize Pointer to the size of the scratch buffer required for the unsharp operation.
 * \return \ref image_data_error_codes
 */
NppStatus 
nppiFilterUnsharpGetBufferSize_8u_C3R(const Npp32f nRadius, const Npp32f nSigma, int * hpBufferSize);

/**
 * Four channel 8-bit unsigned unsharp filter scratch memory size.
 * \param nRadius The radius of the Gaussian filter, in pixles, not counting the center pixel.
 * \param nSigma The standard deviation of the Gaussian filter, in pixel.
 * \param hpBufferSize Pointer to the size of the scratch buffer required for the unsharp operation.
 * \return \ref image_data_error_codes
 */
NppStatus 
nppiFilterUnsharpGetBufferSize_8u_C4R(const Npp32f nRadius, const Npp32f nSigma, int * hpBufferSize);

/**
 * Four channel 8-bit unsigned unsharp filter scratch memory size (alpha channel is not processed).
 * \param nRadius The radius of the Gaussian filter, in pixles, not counting the center pixel.
 * \param nSigma The standard deviation of the Gaussian filter, in pixel.
 * \param hpBufferSize Pointer to the size of the scratch buffer required for the unsharp operation.
 * \return \ref image_data_error_codes
 */
NppStatus 
nppiFilterUnsharpGetBufferSize_8u_AC4R(const Npp32f nRadius, const Npp32f nSigma, int * hpBufferSize);

/**
 * Single channel 16-bit unsigned unsharp filter scratch memory size.
 * \param nRadius The radius of the Gaussian filter, in pixles, not counting the center pixel.
 * \param nSigma The standard deviation of the Gaussian filter, in pixel.
 * \param hpBufferSize Pointer to the size of the scratch buffer required for the unsharp operation.
 * \return \ref image_data_error_codes
 */
NppStatus 
nppiFilterUnsharpGetBufferSize_16u_C1R(const Npp32f nRadius, const Npp32f nSigma, int * hpBufferSize);

/**
 * Three channel 16-bit unsigned unsharp filter scratch memory size.
 * \param nRadius The radius of the Gaussian filter, in pixles, not counting the center pixel.
 * \param nSigma The standard deviation of the Gaussian filter, in pixel.
 * \param hpBufferSize Pointer to the size of the scratch buffer required for the unsharp operation.
 * \return \ref image_data_error_codes
 */
NppStatus 
nppiFilterUnsharpGetBufferSize_16u_C3R(const Npp32f nRadius, const Npp32f nSigma, int * hpBufferSize);

/**
 * Four channel 16-bit unsigned unsharp filter scratch memory size.
 * \param nRadius The radius of the Gaussian filter, in pixles, not counting the center pixel.
 * \param nSigma The standard deviation of the Gaussian filter, in pixel.
 * \param hpBufferSize Pointer to the size of the scratch buffer required for the unsharp operation.
 * \return \ref image_data_error_codes
 */
NppStatus 
nppiFilterUnsharpGetBufferSize_16u_C4R(const Npp32f nRadius, const Npp32f nSigma, int * hpBufferSize);

/**
 * Four channel 16-bit unsigned unsharp filter scratch memory size (alpha channel is not processed).
 * \param nRadius The radius of the Gaussian filter, in pixles, not counting the center pixel.
 * \param nSigma The standard deviation of the Gaussian filter, in pixel.
 * \param hpBufferSize Pointer to the size of the scratch buffer required for the unsharp operation.
 * \return \ref image_data_error_codes
 */
NppStatus 
nppiFilterUnsharpGetBufferSize_16u_AC4R(const Npp32f nRadius, const Npp32f nSigma, int * hpBufferSize);

/**
 * Single channel 16-bit signed unsharp filter scratch memory size.
 * \param nRadius The radius of the Gaussian filter, in pixles, not counting the center pixel.
 * \param nSigma The standard deviation of the Gaussian filter, in pixel.
 * \param hpBufferSize Pointer to the size of the scratch buffer required for the unsharp operation.
 * \return \ref image_data_error_codes
 */
NppStatus 
nppiFilterUnsharpGetBufferSize_16s_C1R(const Npp32f nRadius, const Npp32f nSigma, int * hpBufferSize);

/**
 * Three channel 16-bit signed unsharp filter scratch memory size.
 * \param nRadius The radius of the Gaussian filter, in pixles, not counting the center pixel.
 * \param nSigma The standard deviation of the Gaussian filter, in pixel.
 * \param hpBufferSize Pointer to the size of the scratch buffer required for the unsharp operation.
 * \return \ref image_data_error_codes
 */
NppStatus 
nppiFilterUnsharpGetBufferSize_16s_C3R(const Npp32f nRadius, const Npp32f nSigma, int * hpBufferSize);

/**
 * Four channel 16-bit signed unsharp filter scratch memory size.
 * \param nRadius The radius of the Gaussian filter, in pixles, not counting the center pixel.
 * \param nSigma The standard deviation of the Gaussian filter, in pixel.
 * \param hpBufferSize Pointer to the size of the scratch buffer required for the unsharp operation.
 * \return \ref image_data_error_codes
 */
NppStatus 
nppiFilterUnsharpGetBufferSize_16s_C4R(const Npp32f nRadius, const Npp32f nSigma, int * hpBufferSize);

/**
 * Four channel 16-bit signed unsharp filter scratch memory size (alpha channel is not processed).
 * \param nRadius The radius of the Gaussian filter, in pixles, not counting the center pixel.
 * \param nSigma The standard deviation of the Gaussian filter, in pixel.
 * \param hpBufferSize Pointer to the size of the scratch buffer required for the unsharp operation.
 * \return \ref image_data_error_codes
 */
NppStatus 
nppiFilterUnsharpGetBufferSize_16s_AC4R(const Npp32f nRadius, const Npp32f nSigma, int * hpBufferSize);

/**
 * Single channel 32-bit floating point unsharp filter scratch memory size.
 * \param nRadius The radius of the Gaussian filter, in pixles, not counting the center pixel.
 * \param nSigma The standard deviation of the Gaussian filter, in pixel.
 * \param hpBufferSize Pointer to the size of the scratch buffer required for the unsharp operation.
 * \return \ref image_data_error_codes
 */
NppStatus 
nppiFilterUnsharpGetBufferSize_32f_C1R(const Npp32f nRadius, const Npp32f nSigma, int * hpBufferSize);

/**
 * Three channel 32-bit floating point unsharp filter scratch memory size.
 * \param nRadius The radius of the Gaussian filter, in pixles, not counting the center pixel.
 * \param nSigma The standard deviation of the Gaussian filter, in pixel.
 * \param hpBufferSize Pointer to the size of the scratch buffer required for the unsharp operation.
 * \return \ref image_data_error_codes
 */
NppStatus 
nppiFilterUnsharpGetBufferSize_32f_C3R(const Npp32f nRadius, const Npp32f nSigma, int * hpBufferSize);

/**
 * Four channel 32-bit floating point unsharp filter scratch memory size.
 * \param nRadius The radius of the Gaussian filter, in pixles, not counting the center pixel.
 * \param nSigma The standard deviation of the Gaussian filter, in pixel.
 * \param hpBufferSize Pointer to the size of the scratch buffer required for the unsharp operation.
 * \return \ref image_data_error_codes
 */
NppStatus 
nppiFilterUnsharpGetBufferSize_32f_C4R(const Npp32f nRadius, const Npp32f nSigma, int * hpBufferSize);

/**
 * Four channel 32-bit floating point unsharp filter scratch memory size (alpha channel is not processed).
 * \param nRadius The radius of the Gaussian filter, in pixles, not counting the center pixel.
 * \param nSigma The standard deviation of the Gaussian filter, in pixel.
 * \param hpBufferSize Pointer to the size of the scratch buffer required for the unsharp operation.
 * \return \ref image_data_error_codes
 */
NppStatus 
nppiFilterUnsharpGetBufferSize_32f_AC4R(const Npp32f nRadius, const Npp32f nSigma, int * hpBufferSize);

/** @} FilterUnsharp */

/** @name FilterWienerBorder
 * Noise removal filtering of an image using an adaptive Wiener filter with border control.
 *
 * Pixels under the source mask are used to generate statistics about the local neighborhood 
 * which are then used to control the amount of adaptive noise filtering locally applied.
 *
 * Lwrrently only the NPP_BORDER_REPLICATE border type operation is supported.
 *
 * @{
 *
 * For each pixel in the source image the function estimates the local mean and variance in
 * the neighborhood defined by oMaskSize relative to the primary source pixel located at oAnchor.x and oAnchor.y. 
 * Given an oMaskSize with width \f$W\f$ and height \f$H\f$, the mean, variance, and destination pixel value
 * will be computed per channel as
 * \f[Mean = \frac{1}{W\cdot H}\sum_{j=0}^{H-1}\sum_{i=0}^{W-1}pSrc(j,i)\f]
 * \f[Variance^2 = \frac{1}{W\cdot H}\sum_{j=0}^{H-1}\sum_{i=0}^{W-1}(pSrc(j,i)^2-Mean^2)\f]
 * \f[pDst(j,i) = Mean+\frac{(Variance^2-NoiseVariance)}{Variance^2}\cdot {(pSrc(j,i)-Mean)}\f]
 *
 */

/**
 * Single channel 8-bit unsigned Wiener filter with border control.
 * 
 *
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param oMaskSize Pixel Width and Height of the rectangular region of interest surrounding the source pixel.
 * \param oAnchor Positive X and Y relative offsets of primary pixel in region of interest surrounding the source pixel relative to bottom right of oMaskSize.
 * \param aNoise Fixed size array of per-channel noise variance level value in range of 0.0F to 1.0F.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiFilterWienerBorder_8u_C1R(const Npp8u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                                    NppiSize oMaskSize, NppiPoint oAnchor, Npp32f aNoise[1], NppiBorderType eBorderType);

/**
 * Three channel 8-bit unsigned Wiener filter with border control.
 * 
 *
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param oMaskSize Pixel Width and Height of the rectangular region of interest surrounding the source pixel.
 * \param oAnchor Positive X and Y relative offsets of primary pixel in region of interest surrounding the source pixel relative to bottom right of oMaskSize.
 * \param aNoise Fixed size array of per-channel noise variance level value in range of 0.0F to 1.0F.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiFilterWienerBorder_8u_C3R(const Npp8u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                                    NppiSize oMaskSize, NppiPoint oAnchor, Npp32f aNoise[3], NppiBorderType eBorderType);

/**
 * Four channel 8-bit unsigned Wiener filter with border control.
 * 
 *
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param oMaskSize Pixel Width and Height of the rectangular region of interest surrounding the source pixel.
 * \param oAnchor Positive X and Y relative offsets of primary pixel in region of interest surrounding the source pixel relative to bottom right of oMaskSize.
 * \param aNoise Fixed size array of per-channel noise variance level value in range of 0.0F to 1.0F.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiFilterWienerBorder_8u_C4R(const Npp8u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                                    NppiSize oMaskSize, NppiPoint oAnchor, Npp32f aNoise[4], NppiBorderType eBorderType);

/**
 * Four channel 8-bit unsigned Wiener filter with border control, ignoring alpha channel.
 * 
 *
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param oMaskSize Pixel Width and Height of the rectangular region of interest surrounding the source pixel.
 * \param oAnchor Positive X and Y relative offsets of primary pixel in region of interest surrounding the source pixel relative to bottom right of oMaskSize.
 * \param aNoise Fixed size array of per-channel noise variance level value in range of 0.0F to 1.0F.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiFilterWienerBorder_8u_AC4R(const Npp8u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                                     NppiSize oMaskSize, NppiPoint oAnchor, Npp32f aNoise[3], NppiBorderType eBorderType);

/**
 * Single channel 16-bit signed Wiener filter with border control.
 * 
 *
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param oMaskSize Pixel Width and Height of the rectangular region of interest surrounding the source pixel.
 * \param oAnchor Positive X and Y relative offsets of primary pixel in region of interest surrounding the source pixel relative to bottom right of oMaskSize.
 * \param aNoise Fixed size array of per-channel noise variance level value in range of 0.0F to 1.0F.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiFilterWienerBorder_16s_C1R(const Npp16s * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                                     NppiSize oMaskSize, NppiPoint oAnchor, Npp32f aNoise[1], NppiBorderType eBorderType);

/**
 * Three channel 16-bit signed Wiener filter with border control.
 * 
 *
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param oMaskSize Pixel Width and Height of the rectangular region of interest surrounding the source pixel.
 * \param oAnchor Positive X and Y relative offsets of primary pixel in region of interest surrounding the source pixel relative to bottom right of oMaskSize.
 * \param aNoise Fixed size array of per-channel noise variance level value in range of 0.0F to 1.0F.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiFilterWienerBorder_16s_C3R(const Npp16s * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                                     NppiSize oMaskSize, NppiPoint oAnchor, Npp32f aNoise[3], NppiBorderType eBorderType);

/**
 * Four channel 16-bit signed Wiener filter with border control.
 * 
 *
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param oMaskSize Pixel Width and Height of the rectangular region of interest surrounding the source pixel.
 * \param oAnchor Positive X and Y relative offsets of primary pixel in region of interest surrounding the source pixel relative to bottom right of oMaskSize.
 * \param aNoise Fixed size array of per-channel noise variance level value in range of 0.0F to 1.0F.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiFilterWienerBorder_16s_C4R(const Npp16s * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                                     NppiSize oMaskSize, NppiPoint oAnchor, Npp32f aNoise[4], NppiBorderType eBorderType);

/**
 * Four channel 16-bit signed Wiener filter with border control, ignoring alpha channel.
 * 
 *
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param oMaskSize Pixel Width and Height of the rectangular region of interest surrounding the source pixel.
 * \param oAnchor Positive X and Y relative offsets of primary pixel in region of interest surrounding the source pixel relative to bottom right of oMaskSize.
 * \param aNoise Fixed size array of per-channel noise variance level value in range of 0.0F to 1.0F.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiFilterWienerBorder_16s_AC4R(const Npp16s * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16s * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                                      NppiSize oMaskSize, NppiPoint oAnchor, Npp32f aNoise[3], NppiBorderType eBorderType);

/**
 * Single channel 32-bit float Wiener filter with border control.
 * 
 *
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param oMaskSize Pixel Width and Height of the rectangular region of interest surrounding the source pixel.
 * \param oAnchor Positive X and Y relative offsets of primary pixel in region of interest surrounding the source pixel relative to bottom right of oMaskSize.
 * \param aNoise Fixed size array of per-channel noise variance level value in range of 0.0F to 1.0F.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiFilterWienerBorder_32f_C1R(const Npp32f * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                                     NppiSize oMaskSize, NppiPoint oAnchor, Npp32f aNoise[1], NppiBorderType eBorderType);

/**
 * Three channel 32-bit float Wiener filter with border control.
 * 
 *
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param oMaskSize Pixel Width and Height of the rectangular region of interest surrounding the source pixel.
 * \param oAnchor Positive X and Y relative offsets of primary pixel in region of interest surrounding the source pixel relative to bottom right of oMaskSize.
 * \param aNoise Fixed size array of per-channel noise variance level value in range of 0.0F to 1.0F.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiFilterWienerBorder_32f_C3R(const Npp32f * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                                     NppiSize oMaskSize, NppiPoint oAnchor, Npp32f aNoise[3], NppiBorderType eBorderType);

/**
 * Four channel 32-bit float Wiener filter with border control.
 * 
 *
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param oMaskSize Pixel Width and Height of the rectangular region of interest surrounding the source pixel.
 * \param oAnchor Positive X and Y relative offsets of primary pixel in region of interest surrounding the source pixel relative to bottom right of oMaskSize.
 * \param aNoise Fixed size array of per-channel noise variance level value in range of 0.0F to 1.0F.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiFilterWienerBorder_32f_C4R(const Npp32f * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                                     NppiSize oMaskSize, NppiPoint oAnchor, Npp32f aNoise[4], NppiBorderType eBorderType);

/**
 * Four channel 32-bit float Wiener filter with border control, ignoring alpha channel.
 * 
 *
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param oMaskSize Pixel Width and Height of the rectangular region of interest surrounding the source pixel.
 * \param oAnchor Positive X and Y relative offsets of primary pixel in region of interest surrounding the source pixel relative to bottom right of oMaskSize.
 * \param aNoise Fixed size array of per-channel noise variance level value in range of 0.0F to 1.0F.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiFilterWienerBorder_32f_AC4R(const Npp32f * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                                      NppiSize oMaskSize, NppiPoint oAnchor, Npp32f aNoise[3], NppiBorderType eBorderType);

/** @} FilterWienerBorder */

/** @name GradientVectorPrewittBorder
 * 
 *  RGB Color to Prewitt Gradient Vector colwersion using user selected fixed mask size and gradient distance method.
 *  Functions support up to 4 optional single channel output gradient vectors, X (vertical), Y (horizontal), magnitude, and angle
 *  with user selectable distance methods.  Output for a particular vector is disabled by supplying a NULL pointer for that
 *  vector. X and Y gradient vectors are in cartesian form in the destination data type.  
 *  Magnitude vectors are polar gradient form in the destination data type, angle is always in floating point polar gradient format.
 *  Only fixed mask sizes of 3x3 are supported.
 *  Only nppiNormL1 (sum) and nppiNormL2 (sqrt of sum of squares) distance methods are lwrrently supported.
 *
 * Lwrrently only the NPP_BORDER_REPLICATE border type operation is supported.  Borderless output can be accomplished by using a
 * larger source image than the destination and adjusting oSrcSize and oSrcOffset parameters accordingly.
 *
 * The following fixed kernel mask is used for producing the pDstX (vertical) output image.
 *
 * \f[
 *  \left( \begin{array}{rrr}
 *   -1 & 0 & 1 \\
 *   -1 & 0 & 1 \\
 *   -1 & 0 & 1 \\
 *  \end{array} \right)
 * \f]
 *  
 * The following fixed kernel mask is used for producing the pDstY (horizontal) output image.
 *
 * \f[
 *  \left( \begin{array}{rrr}
 *    1 &  1 &  1 \\
 *    0 &  0 &  0 \\
 *   -1 & -1 & -1 \\
 *  \end{array} \right)
 * \f]
 *
 * For the C1R versions of the function the pDstMag output image value for L1 normalization consists of 
 * the absolute value of the pDstX value plus the absolute value of the pDstY value at that particular image pixel location.
 * For the C1R versions of the function the pDstMag output image value for L2 normalization consists of 
 * the square root of the pDstX value squared plus the pDstY value squared at that particular image pixel location.
 * For the C1R versions of the function the pDstAngle output image value consists of the arctangent (atan2) of 
 * the pDstY value and the pDstX value at that particular image pixel location.
 *
 * For the C3C1R versions of the function, regardless of the selected normalization method, 
 * the L2 normalization value is first determined for each or the pDstX and pDstY values for each source channel then the largest L2
 * normalization value (largest gradient) is used to select which of the 3 pDstX channel values are output to the pDstX image or 
 * pDstY channel values are output to the pDstY image.
 * For the C3C1R versions of the function the pDstMag output image value for L1 normalizaton consists of the same technique
 * used for the C1R version for each source image channel.  Then the largest L2 normalization value is again used to select which
 * of the 3 pDstMag channel values to output to the pDstMag image.
 * For the C3C1R versions of the function the pDstMag output image value for L2 normalizaton consists of just outputting
 * the largest per source channel L2 normalization value to the pDstMag image.
 * For the C3C1R versions of the function the pDstAngle output image value consists of the same technique used for the C1R version
 * callwlated for each source image channel.  Then the largest L2 normalization value is again used to select which of the 3 angle
 * values to output to the pDstAngle image. 
 *
 * @{
 *
 */

/**
 * 1 channel 8-bit unsigned packed RGB to optional 1 channel 16-bit signed X (vertical), Y (horizontal), magnitude, 
 * and/or 32-bit floating point angle gradient vectors with user selectable fixed mask size and distance method with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDstX X vector destination_image_pointer.
 * \param nDstXStep X vector destination_image_line_step.
 * \param pDstY Y vector destination_image_pointer.
 * \param nDstYStep Y vector destination_image_line_step.
 * \param pDstMag magnitude destination_image_pointer.
 * \param nDstMagStep magnitude destination_image_line_step.
 * \param pDstAngle angle destination_image_pointer.
 * \param nDstAngleStep angle destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eMaskSize fixed filter mask size to use.
 * \param eNorm gradient distance method to use.
 * \param eBorderType source image border type to use use.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus nppiGradientVectorPrewittBorder_8u16s_C1R(const Npp8u * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, 
                                                          Npp16s * pDstX, int nDstXStep, Npp16s * pDstY, int nDstYStep, Npp16s * pDstMag, int nDstMagStep, Npp32f * pDstAngle, int nDstAngleStep,
                                                    NppiSize oSizeROI, NppiMaskSize eMaskSize, NppiNorm eNorm, NppiBorderType eBorderType);

/**
 * 3 channel 8-bit unsigned packed RGB to optional 1 channel 16-bit signed X (vertical), Y (horizontal), magnitude, 
 * and/or 32-bit floating point angle gradient vectors with user selectable fixed mask size and distance method with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDstX X vector destination_image_pointer.
 * \param nDstXStep X vector destination_image_line_step.
 * \param pDstY Y vector destination_image_pointer.
 * \param nDstYStep Y vector destination_image_line_step.
 * \param pDstMag magnitude destination_image_pointer.
 * \param nDstMagStep magnitude destination_image_line_step.
 * \param pDstAngle angle destination_image_pointer.
 * \param nDstAngleStep angle destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eMaskSize fixed filter mask size to use.
 * \param eNorm gradient distance method to use.
 * \param eBorderType source image border type to use use.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus nppiGradientVectorPrewittBorder_8u16s_C3C1R(const Npp8u * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, 
                                                            Npp16s * pDstX, int nDstXStep, Npp16s * pDstY, int nDstYStep, Npp16s * pDstMag, int nDstMagStep, Npp32f * pDstAngle, int nDstAngleStep,
                                                      NppiSize oSizeROI, NppiMaskSize eMaskSize, NppiNorm eNorm, NppiBorderType eBorderType);

/**
 * 1 channel 16-bit signed packed RGB to optional 1 channel 32-bit floating point X (vertical), Y (horizontal), magnitude, 
 * and/or 32-bit floating point angle gradient vectors with user selectable fixed mask size and distance method with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDstX X vector destination_image_pointer.
 * \param nDstXStep X vector destination_image_line_step.
 * \param pDstY Y vector destination_image_pointer.
 * \param nDstYStep Y vector destination_image_line_step.
 * \param pDstMag magnitude destination_image_pointer.
 * \param nDstMagStep magnitude destination_image_line_step.
 * \param pDstAngle angle destination_image_pointer.
 * \param nDstAngleStep angle destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eMaskSize fixed filter mask size to use.
 * \param eNorm gradient distance method to use.
 * \param eBorderType source image border type to use use.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus nppiGradientVectorPrewittBorder_16s32f_C1R(const Npp16s * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, 
                                                           Npp32f * pDstX, int nDstXStep, Npp32f * pDstY, int nDstYStep, Npp32f * pDstMag, int nDstMagStep, Npp32f * pDstAngle, int nDstAngleStep,
                                                     NppiSize oSizeROI, NppiMaskSize eMaskSize, NppiNorm eNorm, NppiBorderType eBorderType);

/**
 * 3 channel 16-bit signed packed RGB to optional 1 channel 32-bit floating point X (vertical), Y (horizontal), magnitude, 
 * and/or 32-bit floating point angle gradient vectors with user selectable fixed mask size and distance method with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDstX X vector destination_image_pointer.
 * \param nDstXStep X vector destination_image_line_step.
 * \param pDstY Y vector destination_image_pointer.
 * \param nDstYStep Y vector destination_image_line_step.
 * \param pDstMag magnitude destination_image_pointer.
 * \param nDstMagStep magnitude destination_image_line_step.
 * \param pDstAngle angle destination_image_pointer.
 * \param nDstAngleStep angle destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eMaskSize fixed filter mask size to use.
 * \param eNorm gradient distance method to use.
 * \param eBorderType source image border type to use use.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus nppiGradientVectorPrewittBorder_16s32f_C3C1R(const Npp16s * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, 
                                                             Npp32f * pDstX, int nDstXStep, Npp32f * pDstY, int nDstYStep, Npp32f * pDstMag, int nDstMagStep, Npp32f * pDstAngle, int nDstAngleStep,
                                                       NppiSize oSizeROI, NppiMaskSize eMaskSize, NppiNorm eNorm, NppiBorderType eBorderType);

/**
 * 1 channel 16-bit unsigned packed RGB to optional 1 channel 32-bit floating point X (vertical), Y (horizontal), magnitude, 
 * and/or 32-bit floating point angle gradient vectors with user selectable fixed mask size and distance method with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDstX X vector destination_image_pointer.
 * \param nDstXStep X vector destination_image_line_step.
 * \param pDstY Y vector destination_image_pointer.
 * \param nDstYStep Y vector destination_image_line_step.
 * \param pDstMag magnitude destination_image_pointer.
 * \param nDstMagStep magnitude destination_image_line_step.
 * \param pDstAngle angle destination_image_pointer.
 * \param nDstAngleStep angle destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eMaskSize fixed filter mask size to use.
 * \param eNorm gradient distance method to use.
 * \param eBorderType source image border type to use use.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus nppiGradientVectorPrewittBorder_16u32f_C1R(const Npp16u * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, 
                                                           Npp32f * pDstX, int nDstXStep, Npp32f * pDstY, int nDstYStep, Npp32f * pDstMag, int nDstMagStep, Npp32f * pDstAngle, int nDstAngleStep,
                                                     NppiSize oSizeROI, NppiMaskSize eMaskSize, NppiNorm eNorm, NppiBorderType eBorderType);

/**
 * 3 channel 16-bit unsigned packed RGB to optional 1 channel 32-bit floating point X (vertical), Y (horizontal), magnitude, 
 * and/or 32-bit floating point angle gradient vectors with user selectable fixed mask size and distance method with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDstX X vector destination_image_pointer.
 * \param nDstXStep X vector destination_image_line_step.
 * \param pDstY Y vector destination_image_pointer.
 * \param nDstYStep Y vector destination_image_line_step.
 * \param pDstMag magnitude destination_image_pointer.
 * \param nDstMagStep magnitude destination_image_line_step.
 * \param pDstAngle angle destination_image_pointer.
 * \param nDstAngleStep angle destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eMaskSize fixed filter mask size to use.
 * \param eNorm gradient distance method to use.
 * \param eBorderType source image border type to use use.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus nppiGradientVectorPrewittBorder_16u32f_C3C1R(const Npp16u * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, 
                                                             Npp32f * pDstX, int nDstXStep, Npp32f * pDstY, int nDstYStep, Npp32f * pDstMag, int nDstMagStep, Npp32f * pDstAngle, int nDstAngleStep,
                                                       NppiSize oSizeROI, NppiMaskSize eMaskSize, NppiNorm eNorm, NppiBorderType eBorderType);

/**
 * 1 channel 32-bit floating point packed RGB to optional 1 channel 32-bit floating point X (vertical), Y (horizontal), magnitude, 
 * and/or 32-bit floating point angle gradient vectors with user selectable fixed mask size and distance method with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDstX X vector destination_image_pointer.
 * \param nDstXStep X vector destination_image_line_step.
 * \param pDstY Y vector destination_image_pointer.
 * \param nDstYStep Y vector destination_image_line_step.
 * \param pDstMag magnitude destination_image_pointer.
 * \param nDstMagStep magnitude destination_image_line_step.
 * \param pDstAngle angle destination_image_pointer.
 * \param nDstAngleStep angle destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eMaskSize fixed filter mask size to use.
 * \param eNorm gradient distance method to use.
 * \param eBorderType source image border type to use use.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus nppiGradientVectorPrewittBorder_32f_C1R(const Npp32f * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, 
                                                        Npp32f * pDstX, int nDstXStep, Npp32f * pDstY, int nDstYStep, Npp32f * pDstMag, int nDstMagStep, Npp32f * pDstAngle, int nDstAngleStep,
                                                  NppiSize oSizeROI, NppiMaskSize eMaskSize, NppiNorm eNorm, NppiBorderType eBorderType);

/**
 * 3 channel 32-bit floating point packed RGB to optional 1 channel 32-bit floating point X (vertical), Y (horizontal), magnitude, 
 * and/or 32-bit floating point angle gradient vectors with user selectable fixed mask size and distance method with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDstX X vector destination_image_pointer.
 * \param nDstXStep X vector destination_image_line_step.
 * \param pDstY Y vector destination_image_pointer.
 * \param nDstYStep Y vector destination_image_line_step.
 * \param pDstMag magnitude destination_image_pointer.
 * \param nDstMagStep magnitude destination_image_line_step.
 * \param pDstAngle angle destination_image_pointer.
 * \param nDstAngleStep angle destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eMaskSize fixed filter mask size to use.
 * \param eNorm gradient distance method to use.
 * \param eBorderType source image border type to use use.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus nppiGradientVectorPrewittBorder_32f_C3C1R(const Npp32f * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, 
                                                          Npp32f * pDstX, int nDstXStep, Npp32f * pDstY, int nDstYStep, Npp32f * pDstMag, int nDstMagStep, Npp32f * pDstAngle, int nDstAngleStep,
                                                    NppiSize oSizeROI, NppiMaskSize eMaskSize, NppiNorm eNorm, NppiBorderType eBorderType);


/** @} GradientVectorPrewittBorder */

/** @name GradientVectorScharrBorder
 * 
 *  RGB Color to Scharr Gradient Vector colwersion using user selected fixed mask size and gradient distance method.
 *  Functions support up to 4 optional single channel output gradient vectors, X (vertical), Y (horizontal), magnitude, and angle
 *  with user selectable distance methods.  Output for a particular vector is disabled by supplying a NULL pointer for that
 *  vector. X and Y gradient vectors are in cartesian form in the destination data type.  
 *  Magnitude vectors are polar gradient form in the destination data type, angle is always in floating point polar gradient format.
 *  Only fixed mask sizes of 3x3 are supported.
 *  Only nppiNormL1 (sum) and nppiNormL2 (sqrt of sum of squares) distance methods are lwrrently supported.
 *
 * Lwrrently only the NPP_BORDER_REPLICATE border type operation is supported.  Borderless output can be accomplished by using a
 * larger source image than the destination and adjusting oSrcSize and oSrcOffset parameters accordingly.
 *
 * The following fixed kernel mask is used for producing the pDstX (vertical) output image.
 *
 * \f[
 *  \left( \begin{array}{rrr}
 *    3 & 0 &  -3 \\
 *   10 & 0 & -10 \\
 *    3 & 0 &  -3 \\
 *  \end{array} \right)
 * \f]
 *  
 * The following fixed kernel mask is used for producing the pDstY (horizontal) output image.
 *
 * \f[
 *  \left( \begin{array}{rrr}
 *    3 &  10 &  3 \\
 *    0 &   0 &  0 \\
 *   -3 & -10 & -3 \\
 *  \end{array} \right)
 * \f]
 *
 * For the C1R versions of the function the pDstMag output image value for L1 normalization consists of 
 * the absolute value of the pDstX value plus the absolute value of the pDstY value at that particular image pixel location.
 * For the C1R versions of the function the pDstMag output image value for L2 normalization consists of 
 * the square root of the pDstX value squared plus the pDstY value squared at that particular image pixel location.
 * For the C1R versions of the function the pDstAngle output image value consists of the arctangent (atan2) of 
 * the pDstY value and the pDstX value at that particular image pixel location.
 *
 * For the C3C1R versions of the function, regardless of the selected normalization method, 
 * the L2 normalization value is first determined for each or the pDstX and pDstY values for each source channel then the largest L2
 * normalization value (largest gradient) is used to select which of the 3 pDstX channel values are output to the pDstX image or 
 * pDstY channel values are output to the pDstY image.
 * For the C3C1R versions of the function the pDstMag output image value for L1 normalizaton consists of the same technique
 * used for the C1R version for each source image channel.  Then the largest L2 normalization value is again used to select which
 * of the 3 pDstMag channel values to output to the pDstMag image.
 * For the C3C1R versions of the function the pDstMag output image value for L2 normalizaton consists of just outputting
 * the largest per source channel L2 normalization value to the pDstMag image.
 * For the C3C1R versions of the function the pDstAngle output image value consists of the same technique used for the C1R version
 * callwlated for each source image channel.  Then the largest L2 normalization value is again used to select which of the 3 angle
 * values to output to the pDstAngle image. 
 *
 * @{
 *
 */

/**
 * 1 channel 8-bit unsigned packed RGB to optional 1 channel 16-bit signed X (vertical), Y (horizontal), magnitude, 
 * and/or 32-bit floating point angle gradient vectors with user selectable fixed mask size and distance method with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDstX X vector destination_image_pointer.
 * \param nDstXStep X vector destination_image_line_step.
 * \param pDstY Y vector destination_image_pointer.
 * \param nDstYStep Y vector destination_image_line_step.
 * \param pDstMag magnitude destination_image_pointer.
 * \param nDstMagStep magnitude destination_image_line_step.
 * \param pDstAngle angle destination_image_pointer.
 * \param nDstAngleStep angle destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eMaskSize fixed filter mask size to use.
 * \param eNorm gradient distance method to use.
 * \param eBorderType source image border type to use use.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus nppiGradientVectorScharrBorder_8u16s_C1R(const Npp8u * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, 
                                                         Npp16s * pDstX, int nDstXStep, Npp16s * pDstY, int nDstYStep, Npp16s * pDstMag, int nDstMagStep, Npp32f * pDstAngle, int nDstAngleStep,
                                                   NppiSize oSizeROI, NppiMaskSize eMaskSize, NppiNorm eNorm, NppiBorderType eBorderType);

/**
 * 3 channel 8-bit unsigned packed RGB to optional 1 channel 16-bit signed X (vertical), Y (horizontal), magnitude, 
 * and/or 32-bit floating point angle gradient vectors with user selectable fixed mask size and distance method with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDstX X vector destination_image_pointer.
 * \param nDstXStep X vector destination_image_line_step.
 * \param pDstY Y vector destination_image_pointer.
 * \param nDstYStep Y vector destination_image_line_step.
 * \param pDstMag magnitude destination_image_pointer.
 * \param nDstMagStep magnitude destination_image_line_step.
 * \param pDstAngle angle destination_image_pointer.
 * \param nDstAngleStep angle destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eMaskSize fixed filter mask size to use.
 * \param eNorm gradient distance method to use.
 * \param eBorderType source image border type to use use.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus nppiGradientVectorScharrBorder_8u16s_C3C1R(const Npp8u * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, 
                                                           Npp16s * pDstX, int nDstXStep, Npp16s * pDstY, int nDstYStep, Npp16s * pDstMag, int nDstMagStep, Npp32f * pDstAngle, int nDstAngleStep,
                                                     NppiSize oSizeROI, NppiMaskSize eMaskSize, NppiNorm eNorm, NppiBorderType eBorderType);

/**
 * 1 channel 16-bit signed packed RGB to optional 1 channel 32-bit floating point X (vertical), Y (horizontal), magnitude, 
 * and/or 32-bit floating point angle gradient vectors with user selectable fixed mask size and distance method with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDstX X vector destination_image_pointer.
 * \param nDstXStep X vector destination_image_line_step.
 * \param pDstY Y vector destination_image_pointer.
 * \param nDstYStep Y vector destination_image_line_step.
 * \param pDstMag magnitude destination_image_pointer.
 * \param nDstMagStep magnitude destination_image_line_step.
 * \param pDstAngle angle destination_image_pointer.
 * \param nDstAngleStep angle destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eMaskSize fixed filter mask size to use.
 * \param eNorm gradient distance method to use.
 * \param eBorderType source image border type to use use.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus nppiGradientVectorScharrBorder_16s32f_C1R(const Npp16s * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, 
                                                          Npp32f * pDstX, int nDstXStep, Npp32f * pDstY, int nDstYStep, Npp32f * pDstMag, int nDstMagStep, Npp32f * pDstAngle, int nDstAngleStep,
                                                    NppiSize oSizeROI, NppiMaskSize eMaskSize, NppiNorm eNorm, NppiBorderType eBorderType);

/**
 * 3 channel 16-bit signed packed RGB to optional 1 channel 32-bit floating point X (vertical), Y (horizontal), magnitude, 
 * and/or 32-bit floating point angle gradient vectors with user selectable fixed mask size and distance method with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDstX X vector destination_image_pointer.
 * \param nDstXStep X vector destination_image_line_step.
 * \param pDstY Y vector destination_image_pointer.
 * \param nDstYStep Y vector destination_image_line_step.
 * \param pDstMag magnitude destination_image_pointer.
 * \param nDstMagStep magnitude destination_image_line_step.
 * \param pDstAngle angle destination_image_pointer.
 * \param nDstAngleStep angle destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eMaskSize fixed filter mask size to use.
 * \param eNorm gradient distance method to use.
 * \param eBorderType source image border type to use use.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus nppiGradientVectorScharrBorder_16s32f_C3C1R(const Npp16s * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, 
                                                            Npp32f * pDstX, int nDstXStep, Npp32f * pDstY, int nDstYStep, Npp32f * pDstMag, int nDstMagStep, Npp32f * pDstAngle, int nDstAngleStep,
                                                      NppiSize oSizeROI, NppiMaskSize eMaskSize, NppiNorm eNorm, NppiBorderType eBorderType);

/**
 * 1 channel 16-bit unsigned packed RGB to optional 1 channel 32-bit floating point X (vertical), Y (horizontal), magnitude, 
 * and/or 32-bit floating point angle gradient vectors with user selectable fixed mask size and distance method with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDstX X vector destination_image_pointer.
 * \param nDstXStep X vector destination_image_line_step.
 * \param pDstY Y vector destination_image_pointer.
 * \param nDstYStep Y vector destination_image_line_step.
 * \param pDstMag magnitude destination_image_pointer.
 * \param nDstMagStep magnitude destination_image_line_step.
 * \param pDstAngle angle destination_image_pointer.
 * \param nDstAngleStep angle destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eMaskSize fixed filter mask size to use.
 * \param eNorm gradient distance method to use.
 * \param eBorderType source image border type to use use.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus nppiGradientVectorScharrBorder_16u32f_C1R(const Npp16u * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, 
                                                          Npp32f * pDstX, int nDstXStep, Npp32f * pDstY, int nDstYStep, Npp32f * pDstMag, int nDstMagStep, Npp32f * pDstAngle, int nDstAngleStep,
                                                    NppiSize oSizeROI, NppiMaskSize eMaskSize, NppiNorm eNorm, NppiBorderType eBorderType);

/**
 * 3 channel 16-bit unsigned packed RGB to optional 1 channel 32-bit floating point X (vertical), Y (horizontal), magnitude, 
 * and/or 32-bit floating point angle gradient vectors with user selectable fixed mask size and distance method with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDstX X vector destination_image_pointer.
 * \param nDstXStep X vector destination_image_line_step.
 * \param pDstY Y vector destination_image_pointer.
 * \param nDstYStep Y vector destination_image_line_step.
 * \param pDstMag magnitude destination_image_pointer.
 * \param nDstMagStep magnitude destination_image_line_step.
 * \param pDstAngle angle destination_image_pointer.
 * \param nDstAngleStep angle destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eMaskSize fixed filter mask size to use.
 * \param eNorm gradient distance method to use.
 * \param eBorderType source image border type to use use.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus nppiGradientVectorScharrBorder_16u32f_C3C1R(const Npp16u * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, 
                                                            Npp32f * pDstX, int nDstXStep, Npp32f * pDstY, int nDstYStep, Npp32f * pDstMag, int nDstMagStep, Npp32f * pDstAngle, int nDstAngleStep,
                                                      NppiSize oSizeROI, NppiMaskSize eMaskSize, NppiNorm eNorm, NppiBorderType eBorderType);

/**
 * 1 channel 32-bit floating point packed RGB to optional 1 channel 32-bit floating point X (vertical), Y (horizontal), magnitude, 
 * and/or 32-bit floating point angle gradient vectors with user selectable fixed mask size and distance method with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDstX X vector destination_image_pointer.
 * \param nDstXStep X vector destination_image_line_step.
 * \param pDstY Y vector destination_image_pointer.
 * \param nDstYStep Y vector destination_image_line_step.
 * \param pDstMag magnitude destination_image_pointer.
 * \param nDstMagStep magnitude destination_image_line_step.
 * \param pDstAngle angle destination_image_pointer.
 * \param nDstAngleStep angle destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eMaskSize fixed filter mask size to use.
 * \param eNorm gradient distance method to use.
 * \param eBorderType source image border type to use use.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus nppiGradientVectorScharrBorder_32f_C1R(const Npp32f * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, 
                                                       Npp32f * pDstX, int nDstXStep, Npp32f * pDstY, int nDstYStep, Npp32f * pDstMag, int nDstMagStep, Npp32f * pDstAngle, int nDstAngleStep,
                                                 NppiSize oSizeROI, NppiMaskSize eMaskSize, NppiNorm eNorm, NppiBorderType eBorderType);

/**
 * 3 channel 32-bit floating point packed RGB to optional 1 channel 32-bit floating point X (vertical), Y (horizontal), magnitude, 
 * and/or 32-bit floating point angle gradient vectors with user selectable fixed mask size and distance method with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDstX X vector destination_image_pointer.
 * \param nDstXStep X vector destination_image_line_step.
 * \param pDstY Y vector destination_image_pointer.
 * \param nDstYStep Y vector destination_image_line_step.
 * \param pDstMag magnitude destination_image_pointer.
 * \param nDstMagStep magnitude destination_image_line_step.
 * \param pDstAngle angle destination_image_pointer.
 * \param nDstAngleStep angle destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eMaskSize fixed filter mask size to use.
 * \param eNorm gradient distance method to use.
 * \param eBorderType source image border type to use use.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus nppiGradientVectorScharrBorder_32f_C3C1R(const Npp32f * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, 
                                                         Npp32f * pDstX, int nDstXStep, Npp32f * pDstY, int nDstYStep, Npp32f * pDstMag, int nDstMagStep, Npp32f * pDstAngle, int nDstAngleStep,
                                                   NppiSize oSizeROI, NppiMaskSize eMaskSize, NppiNorm eNorm, NppiBorderType eBorderType);


/** @} GradientVectorScharrBorder */

/** @name GradientVectorSobelBorder
 * 
 *  RGB Color to Sobel Gradient Vector colwersion using user selected fixed mask size and gradient distance method.
 *  Functions support up to 4 optional single channel output gradient vectors, X (vertical), Y (horizontal), magnitude, and angle
 *  with user selectable distance methods.  Output for a particular vector is disabled by supplying a NULL pointer for that
 *  vector. X and Y gradient vectors are in cartesian form in the destination data type.  
 *  Magnitude vectors are polar gradient form in the destination data type, angle is always in floating point polar gradient format.
 *  Only fixed mask sizes of 3x3 and 5x5 are supported.
 *  Only nppiNormL1 (sum) and nppiNormL2 (sqrt of sum of squares) distance methods are lwrrently supported.
 *
 * Lwrrently only the NPP_BORDER_REPLICATE border type operation is supported.  Borderless output can be accomplished by using a
 * larger source image than the destination and adjusting oSrcSize and oSrcOffset parameters accordingly.
 *
 * One of the following fixed kernel masks are used for producing the 3x3 or 5x5 pDstX (vertical) output image depending on selected mask size.
 *
 * \f[
 *  \left( \begin{array}{rrr}
 *   -1 & 0 & 1 \\
 *   -2 & 0 & 2 \\
 *   -1 & 0 & 1 \\
 *  \end{array} \right)
 * \f]
 *  
 *
 * \f[
 *  \left( \begin{array}{rrrrr}
 *   -1 &  -2 & 0 &  2 & 1 \\
 *   -4 &  -8 & 0 &  8 & 4 \\
 *   -6 & -12 & 0 & 12 & 6 \\
 *   -4 &  -8 & 0 &  8 & 4 \\
 *   -1 &  -2 & 0 &  2 & 1 \\
 *  \end{array} \right)
 * \f]
 *  
 * One of the following fixed kernel masks are used for producing the 3x3 or 5x5 pDstY (horizontal) output image depending on selected mask size.
 *
 * \f[
 *  \left( \begin{array}{rrr}
 *    1 &  2 &  1 \\
 *    0 &  0 &  0 \\
 *   -1 & -2 & -1 \\
 *  \end{array} \right)
 * \f]
 *
 *
 * \f[
 *  \left( \begin{array}{rrrrr}
 *    1 &  4 &   6 &  4 &  1 \\
 *    2 &  8 &  12 &  8 &  2 \\
 *    0 &  0 &   0 &  0 &  0 \\
 *   -2 & -8 & -12 & -8 & -2 \\
 *   -1 & -4 &  -6 & -4 & -1 \\
 *  \end{array} \right)
 * \f]
 *
 * For the C1R versions of the function the pDstMag output image value for L1 normalization consists of 
 * the absolute value of the pDstX value plus the absolute value of the pDstY value at that particular image pixel location.
 * For the C1R versions of the function the pDstMag output image value for L2 normalization consists of 
 * the square root of the pDstX value squared plus the pDstY value squared at that particular image pixel location.
 * For the C1R versions of the function the pDstAngle output image value consists of the arctangent (atan2) of 
 * the pDstY value and the pDstX value at that particular image pixel location.
 *
 * For the C3C1R versions of the function, regardless of the selected normalization method, 
 * the L2 normalization value is first determined for each or the pDstX and pDstY values for each source channel then the largest L2
 * normalization value (largest gradient) is used to select which of the 3 pDstX channel values are output to the pDstX image or 
 * pDstY channel values are output to the pDstY image.
 * For the C3C1R versions of the function the pDstMag output image value for L1 normalizaton consists of the same technique
 * used for the C1R version for each source image channel.  Then the largest L2 normalization value is again used to select which
 * of the 3 pDstMag channel values to output to the pDstMag image.
 * For the C3C1R versions of the function the pDstMag output image value for L2 normalizaton consists of just outputting
 * the largest per source channel L2 normalization value to the pDstMag image.
 * For the C3C1R versions of the function the pDstAngle output image value consists of the same technique used for the C1R version
 * callwlated for each source image channel.  Then the largest L2 normalization value is again used to select which of the 3 angle
 * values to output to the pDstAngle image. 
 *
 * @{
 *
 */

/**
 * 1 channel 8-bit unsigned packed RGB to optional 1 channel 16-bit signed X (vertical), Y (horizontal), magnitude, 
 * and/or 32-bit floating point angle gradient vectors with user selectable fixed mask size and distance method with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDstX X vector destination_image_pointer.
 * \param nDstXStep X vector destination_image_line_step.
 * \param pDstY Y vector destination_image_pointer.
 * \param nDstYStep Y vector destination_image_line_step.
 * \param pDstMag magnitude destination_image_pointer.
 * \param nDstMagStep magnitude destination_image_line_step.
 * \param pDstAngle angle destination_image_pointer.
 * \param nDstAngleStep angle destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eMaskSize fixed filter mask size to use.
 * \param eNorm gradient distance method to use.
 * \param eBorderType source image border type to use use.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus nppiGradientVectorSobelBorder_8u16s_C1R(const Npp8u * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, 
                                                        Npp16s * pDstX, int nDstXStep, Npp16s * pDstY, int nDstYStep, Npp16s * pDstMag, int nDstMagStep, Npp32f * pDstAngle, int nDstAngleStep,
                                                  NppiSize oSizeROI, NppiMaskSize eMaskSize, NppiNorm eNorm, NppiBorderType eBorderType);

/**
 * 3 channel 8-bit unsigned packed RGB to optional 1 channel 16-bit signed X (vertical), Y (horizontal), magnitude, 
 * and/or 32-bit floating point angle gradient vectors with user selectable fixed mask size and distance method with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDstX X vector destination_image_pointer.
 * \param nDstXStep X vector destination_image_line_step.
 * \param pDstY Y vector destination_image_pointer.
 * \param nDstYStep Y vector destination_image_line_step.
 * \param pDstMag magnitude destination_image_pointer.
 * \param nDstMagStep magnitude destination_image_line_step.
 * \param pDstAngle angle destination_image_pointer.
 * \param nDstAngleStep angle destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eMaskSize fixed filter mask size to use.
 * \param eNorm gradient distance method to use.
 * \param eBorderType source image border type to use use.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus nppiGradientVectorSobelBorder_8u16s_C3C1R(const Npp8u * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, 
                                                          Npp16s * pDstX, int nDstXStep, Npp16s * pDstY, int nDstYStep, Npp16s * pDstMag, int nDstMagStep, Npp32f * pDstAngle, int nDstAngleStep,
                                                    NppiSize oSizeROI, NppiMaskSize eMaskSize, NppiNorm eNorm, NppiBorderType eBorderType);

/**
 * 1 channel 16-bit signed packed RGB to optional 1 channel 32-bit floating point X (vertical), Y (horizontal), magnitude, 
 * and/or 32-bit floating point angle gradient vectors with user selectable fixed mask size and distance method with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDstX X vector destination_image_pointer.
 * \param nDstXStep X vector destination_image_line_step.
 * \param pDstY Y vector destination_image_pointer.
 * \param nDstYStep Y vector destination_image_line_step.
 * \param pDstMag magnitude destination_image_pointer.
 * \param nDstMagStep magnitude destination_image_line_step.
 * \param pDstAngle angle destination_image_pointer.
 * \param nDstAngleStep angle destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eMaskSize fixed filter mask size to use.
 * \param eNorm gradient distance method to use.
 * \param eBorderType source image border type to use use.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus nppiGradientVectorSobelBorder_16s32f_C1R(const Npp16s * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, 
                                                         Npp32f * pDstX, int nDstXStep, Npp32f * pDstY, int nDstYStep, Npp32f * pDstMag, int nDstMagStep, Npp32f * pDstAngle, int nDstAngleStep,
                                                   NppiSize oSizeROI, NppiMaskSize eMaskSize, NppiNorm eNorm, NppiBorderType eBorderType);

/**
 * 3 channel 16-bit signed packed RGB to optional 1 channel 32-bit floating point X (vertical), Y (horizontal), magnitude, 
 * and/or 32-bit floating point angle gradient vectors with user selectable fixed mask size and distance method with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDstX X vector destination_image_pointer.
 * \param nDstXStep X vector destination_image_line_step.
 * \param pDstY Y vector destination_image_pointer.
 * \param nDstYStep Y vector destination_image_line_step.
 * \param pDstMag magnitude destination_image_pointer.
 * \param nDstMagStep magnitude destination_image_line_step.
 * \param pDstAngle angle destination_image_pointer.
 * \param nDstAngleStep angle destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eMaskSize fixed filter mask size to use.
 * \param eNorm gradient distance method to use.
 * \param eBorderType source image border type to use use.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus nppiGradientVectorSobelBorder_16s32f_C3C1R(const Npp16s * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, 
                                                           Npp32f * pDstX, int nDstXStep, Npp32f * pDstY, int nDstYStep, Npp32f * pDstMag, int nDstMagStep, Npp32f * pDstAngle, int nDstAngleStep,
                                                     NppiSize oSizeROI, NppiMaskSize eMaskSize, NppiNorm eNorm, NppiBorderType eBorderType);

/**
 * 1 channel 16-bit unsigned packed RGB to optional 1 channel 32-bit floating point X (vertical), Y (horizontal), magnitude, 
 * and/or 32-bit floating point angle gradient vectors with user selectable fixed mask size and distance method with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDstX X vector destination_image_pointer.
 * \param nDstXStep X vector destination_image_line_step.
 * \param pDstY Y vector destination_image_pointer.
 * \param nDstYStep Y vector destination_image_line_step.
 * \param pDstMag magnitude destination_image_pointer.
 * \param nDstMagStep magnitude destination_image_line_step.
 * \param pDstAngle angle destination_image_pointer.
 * \param nDstAngleStep angle destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eMaskSize fixed filter mask size to use.
 * \param eNorm gradient distance method to use.
 * \param eBorderType source image border type to use use.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus nppiGradientVectorSobelBorder_16u32f_C1R(const Npp16u * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, 
                                                         Npp32f * pDstX, int nDstXStep, Npp32f * pDstY, int nDstYStep, Npp32f * pDstMag, int nDstMagStep, Npp32f * pDstAngle, int nDstAngleStep,
                                                   NppiSize oSizeROI, NppiMaskSize eMaskSize, NppiNorm eNorm, NppiBorderType eBorderType);

/**
 * 3 channel 16-bit unsigned packed RGB to optional 1 channel 32-bit floating point X (vertical), Y (horizontal), magnitude, 
 * and/or 32-bit floating point angle gradient vectors with user selectable fixed mask size and distance method with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDstX X vector destination_image_pointer.
 * \param nDstXStep X vector destination_image_line_step.
 * \param pDstY Y vector destination_image_pointer.
 * \param nDstYStep Y vector destination_image_line_step.
 * \param pDstMag magnitude destination_image_pointer.
 * \param nDstMagStep magnitude destination_image_line_step.
 * \param pDstAngle angle destination_image_pointer.
 * \param nDstAngleStep angle destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eMaskSize fixed filter mask size to use.
 * \param eNorm gradient distance method to use.
 * \param eBorderType source image border type to use use.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus nppiGradientVectorSobelBorder_16u32f_C3C1R(const Npp16u * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, 
                                                           Npp32f * pDstX, int nDstXStep, Npp32f * pDstY, int nDstYStep, Npp32f * pDstMag, int nDstMagStep, Npp32f * pDstAngle, int nDstAngleStep,
                                                     NppiSize oSizeROI, NppiMaskSize eMaskSize, NppiNorm eNorm, NppiBorderType eBorderType);

/**
 * 1 channel 32-bit floating point packed RGB to optional 1 channel 32-bit floating point X (vertical), Y (horizontal), magnitude, 
 * and/or 32-bit floating point angle gradient vectors with user selectable fixed mask size and distance method with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDstX X vector destination_image_pointer.
 * \param nDstXStep X vector destination_image_line_step.
 * \param pDstY Y vector destination_image_pointer.
 * \param nDstYStep Y vector destination_image_line_step.
 * \param pDstMag magnitude destination_image_pointer.
 * \param nDstMagStep magnitude destination_image_line_step.
 * \param pDstAngle angle destination_image_pointer.
 * \param nDstAngleStep angle destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eMaskSize fixed filter mask size to use.
 * \param eNorm gradient distance method to use.
 * \param eBorderType source image border type to use use.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus nppiGradientVectorSobelBorder_32f_C1R(const Npp32f * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, 
                                                      Npp32f * pDstX, int nDstXStep, Npp32f * pDstY, int nDstYStep, Npp32f * pDstMag, int nDstMagStep, Npp32f * pDstAngle, int nDstAngleStep,
                                                NppiSize oSizeROI, NppiMaskSize eMaskSize, NppiNorm eNorm, NppiBorderType eBorderType);

/**
 * 3 channel 32-bit floating point packed RGB to optional 1 channel 32-bit floating point X (vertical), Y (horizontal), magnitude, 
 * and/or 32-bit floating point angle gradient vectors with user selectable fixed mask size and distance method with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDstX X vector destination_image_pointer.
 * \param nDstXStep X vector destination_image_line_step.
 * \param pDstY Y vector destination_image_pointer.
 * \param nDstYStep Y vector destination_image_line_step.
 * \param pDstMag magnitude destination_image_pointer.
 * \param nDstMagStep magnitude destination_image_line_step.
 * \param pDstAngle angle destination_image_pointer.
 * \param nDstAngleStep angle destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eMaskSize fixed filter mask size to use.
 * \param eNorm gradient distance method to use.
 * \param eBorderType source image border type to use use.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus nppiGradientVectorSobelBorder_32f_C3C1R(const Npp32f * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, 
                                                        Npp32f * pDstX, int nDstXStep, Npp32f * pDstY, int nDstYStep, Npp32f * pDstMag, int nDstMagStep, Npp32f * pDstAngle, int nDstAngleStep,
                                                  NppiSize oSizeROI, NppiMaskSize eMaskSize, NppiNorm eNorm, NppiBorderType eBorderType);


/** @} GradientVectorSobelBorder */

/** @name FilterCannyBorder
 * 
 *  Performs Canny edge detection on a single channel 8-bit grayscale image and outputs a single channel 8-bit image consisting of 0x00 and 0xFF
 *  values with 0xFF representing edge pixels.  The algorithm consists of three phases.  The first phase generates two output images consisting
 *  of a single channel 16-bit signed image containing magnitude values and a single channel 32-bit floating point image containing the angular
 *  direction of those magnitude values.   This phase is accomplished by calling the appropriate GradientVectorBorder filter function based on
 *  the filter type, filter mask size, and norm type requested.  The next phase uses those magnitude and direction images to suppress non-maximum
 *  magnitude values which are lower than the values of either of its two nearest neighbors in the same direction as the test magnitude pixel in 
 *  the 3x3 surrounding magnitude pixel neighborhood.  This phase outputs a new magnitude image with non-maximum pixel values suppressed.  Finally, in the
 *  third phase, the new magnitude image is passed through a hysteresis threshold filter that filters out any magnitude values that are not connected
 *  to another edge magnitude value.   In this phase, any magnitude value above the high threshold value is automatically accepted, any magnitude
 *  value below the low threshold value is automatically rejected.  For magnitude values that lie between the low and high threshold, values are
 *  only accepted if one of their two neighbors in the same direction in the 3x3 neighborhood around them lies above the low threshold value.  In other words,
 *  if they are connected to an active edge.   J. Canny recommends that the ratio of high to low threshold limit be in the range two or three to one, 
 *  based on predicted signal-to-noise ratios. The final output of the third phase consists of a single channel 8-bit unsigned image of 0x00 and 0xFF 
 *  values based on whether they are accepted or rejected during threshold testing.
 *    
 * Lwrrently only the NPP_BORDER_REPLICATE border type operation is supported.  Borderless output can be accomplished by using a
 * larger source image than the destination and adjusting oSrcSize and oSrcOffset parameters accordingly.
 *
 * @{
 *
 */

/**
 * Callwlate scratch buffer size needed for the FilterCannyBorder function based on destination image SizeROI width and height.
 *
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */

NppStatus 
nppiFilterCannyBorderGetBufferSize(NppiSize oSizeROI, int * hpBufferSize);

/**
 * 1 channel 8-bit unsigned grayscale to 1 channel 8-bit unsigned black (0x00) and white (0xFF) image with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst output edge destination_image_pointer.
 * \param nDstStep output edge destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eFilterType selects between Sobel or Scharr filter type.
 * \param eMaskSize fixed filter mask size to use.
 * \param nLowThreshold low hysteresis threshold value.
 * \param nHighThreshold high hysteresis threshold value.
 * \param eNorm gradient distance method to use.
 * \param eBorderType source image border type to use use.
 * \param pDeviceBuffer pointer to scratch DEVICE memory buffer of size hpBufferSize (see nppiFilterCannyBorderGetBufferSize() above)
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterCannyBorder_8u_C1R(const Npp8u * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, 
                                   Npp8u * pDst, int nDstStep, NppiSize oSizeROI, NppiDifferentialKernel eFilterType,
                             NppiMaskSize eMaskSize, Npp16s nLowThreshold, Npp16s nHighThreshold, NppiNorm eNorm, 
                             NppiBorderType eBorderType, Npp8u * pDeviceBuffer);

/** @} FilterCannyBorder */

/** @name FilterHarrisCornersBorder
 * 
 *  Performs Harris Corner detection on a single channel 8-bit grayscale image and outputs a single channel 32-bit floating point image 
 *  consisting the corner response at each pixel of the image.  The algorithm consists of two phases.  The first phase generates the floating
 *  point product of XX, YY, and XY gradients at each pixel in the image.  The type of gradient used is controlled by the eFilterType and eMaskSize parameters.
 *  The second phase averages those products over a window of either 3x3 or 5x5 pixels around the center pixel then generates the Harris corner
 *  response at that pixel which is output in the destination image. The Harris response value is determined as H = ((XX * YY - XY * XY) - 
 *  (nK * ((XX + YY) * (XX + YY)))) * nScale.
 *    
 * Lwrrently only the NPP_BORDER_REPLICATE border type operation is supported.  Borderless output can be accomplished by using a
 * larger source image than the destination and adjusting oSrcSize and oSrcOffset parameters accordingly.
 *
 * @{
 *
 */

/**
 * Callwlate scratch buffer size needed for the FilterHarrisCornersBorder function based on destination image SizeROI width and height.
 *
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */

NppStatus 
nppiFilterHarrisCornersBorderGetBufferSize(NppiSize oSizeROI, int * hpBufferSize);

/**
 * 1 channel 8-bit unsigned grayscale to 1 channel 32-bit floating point Harris corners response image with border control.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param pDst output edge destination_image_pointer.
 * \param nDstStep output edge destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eFilterType selects between Sobel or Scharr filter type.
 * \param eMaskSize fixed filter mask size to use (3x3 or 5x5 for Sobel).
 * \param eAvgWindowSize fixed window mask size to use (3x3 or 5x5).
 * \param nK Harris Corners constant (commonly used value is 0.04F).
 * \param nScale output is scaled by this scale factor.
 * \param eBorderType source image border type to use use.
 * \param pDeviceBuffer pointer to scratch DEVICE memory buffer of size hpBufferSize (see nppiFilterHarrisCornersBorderGetBufferSize() above)
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterHarrisCornersBorder_8u32f_C1R(const Npp8u * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, 
                                              Npp32f * pDst, int nDstStep, NppiSize oSizeROI, NppiDifferentialKernel eFilterType,
                                        NppiMaskSize eMaskSize, NppiMaskSize eAvgWindowSize, Npp32f nK, Npp32f nScale, 
                                        NppiBorderType eBorderType, Npp8u * pDeviceBuffer);

/** @} FilterHarrisCornersBorder */

/** @name FilterHoughLine
 * 
 *  Extracts Hough lines from a single channel 8-bit binarized (0, 255) source feature (canny edges, etc.) image and outputs a list of lines in point polar format
 *  representing the length (rho) and angle (theta) of each line from the origin of the normal to the line using the formula rho = x cos(theta) + y sin(theta).
 *  The level of discretization, nDelta, is specified as an input parameter. The performance and effectiveness of this function highly depends on
 *  this parameter with higher performance for larger numbers and more detailed results for lower numbers.  Also, lines are not guaranteed to
 *  be added to the pDeviceLines list in the same order from one call to the next.  However, all of the same lines will still be generated as long as
 *  nMaxLineCount is set large enough so that they all can fit in the list. To colwert lines in point polar format back to cartesian lines
 *  use the following formula:
 *  \code
 *
 *  Npp32f nHough = ((sqrt(2.0F) * static_cast<Npp32f>(oSizeROI.height > oSizeROI.width ? oSizeROI.height 
 *                                                                                      : oSizeROI.width)) / 2.0F); 
 *  int nAclwmulatorsHeight = nDelta.rho > 1.0F ? static_cast<int>(ceil(nHough * 2.0F)) 
 *                                              : static_cast<int>(ceil((nHough * 2.0F) / nDelta.rho));
 *  int nCenterX = oSizeROI.width >> 1;
 *  int nCenterY = oSizeROI.height >> 1;
 *  Npp32f nThetaRad = static_cast<Npp32f>(deviceline.theta) * 0.0174532925199433F;
 *  Npp32f nSinTheta = sin(nThetaRad);
 *  Npp32f nCosTheta = cos(nThetaRad);
 *  int nX1, nY1, nX2, nY2;
 *
 *  if (deviceline.theta >= 45 && deviceline.theta <= 135) // degrees
 *  {
 *      // y = (rho - x cos(theta)) / sin(theta)
 *      nX1 = minimum cartesian X boundary value;
 *      nY1 = static_cast<int>((static_cast<Npp32f>(deviceline.rho - (nAclwmulatorsHeight >> 1)) - 
 *                             ((nX1 - nCenterX) * nCosTheta)) / nSinTheta + nCenterY);
 *      nX2 = maximum cartesian X boundary value;
 *      nY2 = static_cast<int>((static_cast<Npp32f>(deviceline.rho - (nAclwmulatorsHeight >> 1)) - 
 *                             ((nX2 - nCenterX) * nCosTheta)) / nSinTheta + nCenterY);
 *  }
 *  else
 *  {
 *      // x = (rho - y sin(theta)) / cos(theta)
 *      nY1 = minimum cartesian Y boundary value;
 *      nX1 = static_cast<int>((static_cast<Npp32f>(deviceline.rho - (nAclwmulatorsHeight >> 1)) - 
 *                             ((nY1 - nCenterY) * nSinTheta)) / nCosTheta + nCenterX);
 *      nY2 = maximum cartesian Y boundary value;
 *      nX2 = static_cast<int>((static_cast<Npp32f>(deviceline.rho - (nAclwmulatorsHeight >> 1)) - 
 *                             ((nY2 - nCenterY) * nSinTheta)) / nCosTheta + nCenterX);
 *  }
 *  \endcode
 *    
 * @{
 *
 */

/**
 * Callwlate scratch buffer size needed for the FilterHoughLine or FilterHoughLineRegion functions based on destination image SizeROI width and height and nDelta parameters.
 *
 * \param oSizeROI \ref roi_specification.
 * \param nDelta rho radial increment and theta angular increment that will be used in the FilterHoughLine or FilterHoughLineRegion function call.
 * \param nMaxLineCount The maximum number of lines expected from the FilterHoughLine or FilterHoughLineRegion function call.
 * \param hpBufferSize Required buffer size in bytes. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */

NppStatus 
nppiFilterHoughLineGetBufferSize(NppiSize oSizeROI, NppPointPolar nDelta, int nMaxLineCount, int * hpBufferSize);

/**
 * 1 channel 8-bit unsigned binarized (0, 255) source feature (canny edges, etc.) source image to list of lines in point polar format
 * representing the length (rho) and angle (theta) of each line from the origin of the normal to the line using the formula rho = x cos(theta) + y sin(theta).
 * The level of discretization, nDelta, is specified as an input parameter. The performance and effectiveness of this function highly depends on
 * this parameter with higher performance for larger numbers and more detailed results for lower numbers. nDelta must have the same values as
 * those used in the nppiFilterHoughLineGetBufferSize() function call.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nDelta Discretization steps, range 0.0F < radial increment nDelta.rho < 3.0F, 1.0F recommended, range 0.25F < angular increment nDelta.theta < 3.0F, 1.0F recommended.
 * \param nThreshold Minimum number of points to accept a line.
 * \param pDeviceLines Device pointer to (nMaxLineCount * sizeof(NppPointPolar) line objects.
 * \param nMaxLineCount The maximum number of lines to output.
 * \param pDeviceLineCount The number of lines detected by this function up to nMaxLineCount.
 * \param pDeviceBuffer pointer to scratch DEVICE memory buffer of size hpBufferSize (see nppiFilterHoughLineGetBufferSize() above)
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterHoughLine_8u32f_C1R(const Npp8u * pSrc, int nSrcStep, NppiSize oSizeROI, NppPointPolar nDelta, int nThreshold, 
                                    NppPointPolar * pDeviceLines, int nMaxLineCount, int * pDeviceLineCount, Npp8u * pDeviceBuffer);


/**
 * 1 channel 8-bit unsigned binarized (0, 255) source feature (canny edges, etc.) source image to list of lines in point polar format
 * representing the length (rho) and angle (theta) of each line from the origin of the normal to the line using the formula rho = x cos(theta) + y sin(theta).
 * The level of discretization, nDelta, is specified as an input parameter. The performance and effectiveness of this function highly depends on
 * this parameter with higher performance for larger numbers and more detailed results for lower numbers. nDelta must have the same values as
 * those used in the nppiFilterHoughLineGetBufferSize() function call. The oDstROI region limits are used to limit accepted lines to those that fall within
 * those limits.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nDelta Discretization steps, range 0.0F < radial increment nDelta.rho < 3.0F, 1.0F recommended, range 0.25F < angular increment nDelta.theta < 3.0F, 1.0F recommended.
 * \param nThreshold Minimum number of points to accept a line.
 * \param pDeviceLines Device pointer to (nMaxLineCount * sizeof(NppPointPolar) line objects.
 * \param oDstROI Region limits with oDstROI[0].rho <= accepted rho <= oDstROI[1].rho and oDstROI[0].theta <= accepted theta <= oDstROI[1].theta.
 * \param nMaxLineCount The maximum number of lines to output.
 * \param pDeviceLineCount The number of lines detected by this function up to nMaxLineCount.
 * \param pDeviceBuffer pointer to scratch DEVICE memory buffer of size hpBufferSize (see nppiFilterHoughLineGetBufferSize() above)
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiFilterHoughLineRegion_8u32f_C1R(const Npp8u * pSrc, int nSrcStep, NppiSize oSizeROI, NppPointPolar nDelta, int nThreshold, 
                                          NppPointPolar * pDeviceLines, NppPointPolar oDstROI[2], int nMaxLineCount, int * pDeviceLineCount, Npp8u * pDeviceBuffer);

/** @} FilterHoughLine */

/** @name HistogramOfOrientedGradientsBorder
 * 
 * Performs Histogram Of Oriented Gradients operation on source image generating separate windows of Histogram Descriptors for each requested location.
 *
 * This function implements the simplest form of functionality described by N. Dalal and B. Triggs. Histograms of Oriented Gradients for Human Detection. INRIA, 2005.
 * It supports overlapped contrast normalized block histogram output with L2 normalization only, no threshold clipping, and no pre or post gaussian smoothing of input images or 
 * histogram output values. It supports both single channel grayscale source images and three channel color images.  For color images, the color channel with the
 * highest magnitude value is used as that pixel's magnitude. Output is row order only. 
 * Descriptors are output conselwtively with no separation padding if multiple descriptor output is requested (one desriptor per source image location).
 * For example, common HOG parameters are 9 histogram bins per 8 by 8 pixel cell, 2 by 2 cells per block, 
 * with a descriptor window size of 64 horizontal by 128 vertical pixels yielding 7 by 15 overlapping blocks 
 * (1 cell overlap in both horizontal and vertical directions).  This results in 9 bins * 4 cells * 7 horizontal overlapping blocks * 15 vertical overlapping blocks or 3780 
 * 32-bit floating point output values (bins) per descriptor window. 
 * 
 * The number of horizontal overlapping block histogram bins per descriptor window width is determined by
 * (((oHOGConfig.detectionWindowSize.width / oHOGConfig.histogramBlockSize) * 2) - 1) * oHOGConfig.nHistogramBins. 
 * The number of vertical overlapping block histograms per descriptor window height is determined by 
 * (((oHOGConfig.detectionWindowSize.height / oHOGConfig.histogramBlockSize) * 2) - 1)
 * The offset of each descriptor window in the descriptors output buffer is therefore 
 * horizontal histogram bins per descriptor window width * vertical histograms per descriptor window height 32-bit floating point values 
 * relative to the previous descriptor window output.
 *
 * The algorithm uses a 1D centered derivative mask of [-1, 0, +1] when generating input magnitude and angle gradients. 
 * Magnitudes are added to the two nearest histogram bins of oriented gradients between 0 and 180 degrees using a weighted linear interpolation of each
 * magnitude value across the 2 nearest angular bin orientations. 2D overlapping blocks of histogram bins consisting of the bins from 2D arrangements of cells are
 * then contrast normalized using L2 normalization and output to the corresponding histogram descriptor window for that particular window
 * location in the window locations list.
 *
 * Some restrictions include:
 *
 * \code
 * #define NPP_HOG_MAX_CELL_SIZE                          (16)
 * #define NPP_HOG_MAX_BLOCK_SIZE                         (64)
 * #define NPP_HOG_MAX_BINS_PER_CELL                      (16)
 * #define NPP_HOG_MAX_CELLS_PER_DESCRIPTOR              (256)
 * #define NPP_HOG_MAX_OVERLAPPING_BLOCKS_PER_DESCRIPTOR (256)
 * #define NPP_HOG_MAX_DESCRIPTOR_LOCATIONS_PER_CALL     (128)
 * \endcode 
 * 
 * Lwrrently only the NPP_BORDER_REPLICATE border type operation is supported.
 *    
 * @{
 *
 */

/**
 * Validates requested HOG configuration and callwlates scratch buffer size needed for the HistogramOfGradientsBorder function 
 * based on requested HOG configuration, source image ROI, and number and locations of descriptor window locations.
 *
 * \param oHOGConfig Requested HOG configuration parameters structure.
 * \param hpLocations Host pointer to array of NppiPoint source pixel starting locations of requested descriptor windows. Important: hpLocations is a 
 *        <em>host pointer.</em>
 * \param nLocations Number of NppiPoint in pLocations array. 
 * \param oSizeROI \ref roi_specification of source image.
 * \param hpBufferSize Required buffer size in bytes. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */

NppStatus 
nppiHistogramOfGradientsBorderGetBufferSize(const NppiHOGConfig oHOGConfig, const NppiPoint * hpLocations, int nLocations, NppiSize oSizeROI, int * hpBufferSize);

/**
 * Validates requested HOG configuration and callwlates output window descriptors buffer size needed for the HistogramOfGradientsBorder function 
 * based on requested HOG configuration, and number of descriptor window locations, one descriptor window is output for each location.
 * Descriptor windows are located sequentially and contiguously in the descriptors buffer.
 *
 * The number of horizontal overlapping block histogram bins per descriptor window width is determined by
 * (((oHOGConfig.detectionWindowSize.width / oHOGConfig.histogramBlockSize) * 2) - 1) * oHOGConfig.nHistogramBins. 
 * The number of vertical overlapping block histograms per descriptor window height is determined by 
 * (((oHOGConfig.detectionWindowSize.height / oHOGConfig.histogramBlockSize) * 2) - 1)
 * The offset of each descriptor window in the descriptors output buffer is therefore 
 * horizontal histogram bins per descriptor window width * vertical histograms per descriptor window height floating point values 
 * relative to the previous descriptor window output.  
 *
 * \param oHOGConfig Requested HOG configuration parameters structure.
 * \param nLocations Number of NppiPoint in pLocations array. 
 * \param hpDescriptorsSize Required buffer size in bytes of output windows descriptors for nLocations descriptor windows. Important: hpDescriptorsSize is a 
 *        <em>host pointer.</em>
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */

NppStatus 
nppiHistogramOfGradientsBorderGetDescriptorsSize(const NppiHOGConfig oHOGConfig, int nLocations, int * hpDescriptorsSize);

/**
 * 1 channel 8-bit unsigned grayscale per source image descriptor window location with source image border control 
 * to per descriptor window destination floating point histogram of gradients. Requires first calling nppiHistogramOfGradientsBorderGetBufferSize function
 * call to get required scratch (host) working buffer size and nppiHistogramOfGradientsBorderGetDescriptorsSize() function call to get
 * total size for nLocations of output histogram block descriptor windows.  
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param hpLocations Host pointer to array of NppiPoint source pixel starting locations of requested descriptor windows. Important: hpLocations is a 
 *        <em>host pointer.</em>
 * \param nLocations Number of NppiPoint in pLocations array.
 * \param pDstWindowDescriptorBuffer Output device memory buffer pointer of size hpDescriptorsSize bytes to first of nLoc descriptor windows (see nppiHistogramOfGradientsBorderGetDescriptorsSize() above).
 * \param oSizeROI \ref roi_specification of source image.
 * \param oHOGConfig Requested HOG configuration parameters structure.
 * \param pScratchBuffer Device memory buffer pointer of size hpBufferSize bytes to scratch memory buffer (see nppiHistogramOfGradientsBorderGetBufferSize() above).
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiHistogramOfGradientsBorder_8u32f_C1R(const Npp8u * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, 
                                         const NppiPoint * hpLocations, int nLocations, Npp32f * pDstWindowDescriptorBuffer, 
                                         NppiSize oSizeROI, const NppiHOGConfig oHOGConfig, Npp8u * pScratchBuffer, NppiBorderType eBorderType);


/**
 * 3 channel 8-bit unsigned color per source image descriptor window location with source image border control 
 * to per descriptor window destination floating point histogram of gradients. Requires first calling nppiHistogramOfGradientsBorderGetBufferSize function
 * call to get required scratch (host) working buffer size and nppiHistogramOfGradientsBorderGetDescriptorsSize() function call to get
 * total size for nLocations of output histogram block descriptor windows.  
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param hpLocations Host pointer to array of NppiPoint source pixel starting locations of requested descriptor windows. Important: hpLocations is a 
 *        <em>host pointer.</em>
 * \param nLocations Number of NppiPoint in pLocations array. 
 * \param pDstWindowDescriptorBuffer Output device memory buffer pointer of size hpDescriptorsSize bytes to first of nLoc descriptor windows (see nppiHistogramOfGradientsBorderGetDescriptorsSize() above).
 * \param oSizeROI \ref roi_specification of source image.
 * \param oHOGConfig Requested HOG configuration parameters structure.
 * \param pScratchBuffer Device memory buffer pointer of size hpBufferSize bytes to scratch memory buffer (see nppiHistogramOfGradientsBorderGetBufferSize() above).
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiHistogramOfGradientsBorder_8u32f_C3R(const Npp8u * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, 
                                         const NppiPoint * hpLocations, int nLocations, Npp32f * pDstWindowDescriptorBuffer, 
                                         NppiSize oSizeROI, const NppiHOGConfig oHOGConfig, Npp8u * pScratchBuffer, NppiBorderType eBorderType);

/**
 * 1 channel 16-bit unsigned grayscale per source image descriptor window location with source image border control 
 * to per descriptor window destination floating point histogram of gradients. Requires first calling nppiHistogramOfGradientsBorderGetBufferSize function
 * call to get required scratch (host) working buffer size and nppiHistogramOfGradientsBorderGetDescriptorsSize() function call to get
 * total size for nLocations of output histogram block descriptor windows.  
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param hpLocations Host pointer to array of NppiPoint source pixel starting locations of requested descriptor windows. Important: hpLocations is a 
 *        <em>host pointer.</em>
 * \param nLocations Number of NppiPoint in pLocations array. 
 * \param pDstWindowDescriptorBuffer Output device memory buffer pointer of size hpDescriptorsSize bytes to first of nLoc descriptor windows (see nppiHistogramOfGradientsBorderGetDescriptorsSize() above).
 * \param oSizeROI \ref roi_specification of source image.
 * \param oHOGConfig Requested HOG configuration parameters structure.
 * \param pScratchBuffer Device memory buffer pointer of size hpBufferSize bytes to scratch memory buffer (see nppiHistogramOfGradientsBorderGetBufferSize() above).
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiHistogramOfGradientsBorder_16u32f_C1R(const Npp16u * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, 
                                          const NppiPoint * hpLocations, int nLocations, Npp32f * pDstWindowDescriptorBuffer, 
                                          NppiSize oSizeROI, const NppiHOGConfig oHOGConfig, Npp8u * pScratchBuffer, NppiBorderType eBorderType);


/**
 * 3 channel 16-bit unsigned color per source image descriptor window location with source image border control 
 * to per descriptor window destination floating point histogram of gradients. Requires first calling nppiHistogramOfGradientsBorderGetBufferSize function
 * call to get required scratch (host) working buffer size and nppiHistogramOfGradientsBorderGetDescriptorsSize() function call to get
 * total size for nLocations of output histogram block descriptor windows.  
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param hpLocations Host pointer to array of NppiPoint source pixel starting locations of requested descriptor windows. Important: hpLocations is a 
 *        <em>host pointer.</em>
 * \param nLocations Number of NppiPoint in pLocations array. 
 * \param pDstWindowDescriptorBuffer Output device memory buffer pointer of size hpDescriptorsSize bytes to first of nLoc descriptor windows (see nppiHistogramOfGradientsBorderGetDescriptorsSize() above).
 * \param oSizeROI \ref roi_specification of source image.
 * \param oHOGConfig Requested HOG configuration parameters structure.
 * \param pScratchBuffer Device memory buffer pointer of size hpBufferSize bytes to scratch memory buffer (see nppiHistogramOfGradientsBorderGetBufferSize() above).
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiHistogramOfGradientsBorder_16u32f_C3R(const Npp16u * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, 
                                          const NppiPoint * hpLocations, int nLocations, Npp32f * pDstWindowDescriptorBuffer, 
                                          NppiSize oSizeROI, const NppiHOGConfig oHOGConfig, Npp8u * pScratchBuffer, NppiBorderType eBorderType);

/**
 * 1 channel 16-bit signed grayscale per source image descriptor window location with source image border control 
 * to per descriptor window destination floating point histogram of gradients. Requires first calling nppiHistogramOfGradientsBorderGetBufferSize function
 * call to get required scratch (host) working buffer size and nppiHistogramOfGradientsBorderGetDescriptorsSize() function call to get
 * total size for nLocations of output histogram block descriptor windows.  
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param hpLocations Host pointer to array of NppiPoint source pixel starting locations of requested descriptor windows. Important: hpLocations is a 
 *        <em>host pointer.</em>
 * \param nLocations Number of NppiPoint in pLocations array. 
 * \param pDstWindowDescriptorBuffer Output device memory buffer pointer of size hpDescriptorsSize bytes to first of nLoc descriptor windows (see nppiHistogramOfGradientsBorderGetDescriptorsSize() above).
 * \param oSizeROI \ref roi_specification of source image.
 * \param oHOGConfig Requested HOG configuration parameters structure.
 * \param pScratchBuffer Device memory buffer pointer of size hpBufferSize bytes to scratch memory buffer (see nppiHistogramOfGradientsBorderGetBufferSize() above).
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiHistogramOfGradientsBorder_16s32f_C1R(const Npp16s * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, 
                                          const NppiPoint * hpLocations, int nLocations, Npp32f * pDstWindowDescriptorBuffer, 
                                          NppiSize oSizeROI, const NppiHOGConfig oHOGConfig, Npp8u * pScratchBuffer, NppiBorderType eBorderType);


/**
 * 3 channel 16-bit signed color per source image descriptor window location with source image border control 
 * to per descriptor window destination floating point histogram of gradients. Requires first calling nppiHistogramOfGradientsBorderGetBufferSize function
 * call to get required scratch (host) working buffer size and nppiHistogramOfGradientsBorderGetDescriptorsSize() function call to get
 * total size for nLocations of output histogram block descriptor windows.  
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param hpLocations Host pointer to array of NppiPoint source pixel starting locations of requested descriptor windows. Important: hpLocations is a 
 *        <em>host pointer.</em>
 * \param nLocations Number of NppiPoint in pLocations array. 
 * \param pDstWindowDescriptorBuffer Output device memory buffer pointer of size hpDescriptorsSize bytes to first of nLoc descriptor windows (see nppiHistogramOfGradientsBorderGetDescriptorsSize() above).
 * \param oSizeROI \ref roi_specification of source image.
 * \param oHOGConfig Requested HOG configuration parameters structure.
 * \param pScratchBuffer Device memory buffer pointer of size hpBufferSize bytes to scratch memory buffer (see nppiHistogramOfGradientsBorderGetBufferSize() above).
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiHistogramOfGradientsBorder_16s32f_C3R(const Npp16s * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, 
                                          const NppiPoint * hpLocations, int nLocations, Npp32f * pDstWindowDescriptorBuffer, 
                                          NppiSize oSizeROI, const NppiHOGConfig oHOGConfig, Npp8u * pScratchBuffer, NppiBorderType eBorderType);

/**
 * 1 channel 32-bit floating point grayscale per source image descriptor window location with source image border control 
 * to per descriptor window destination floating point histogram of gradients. Requires first calling nppiHistogramOfGradientsBorderGetBufferSize function
 * call to get required scratch (host) working buffer size and nppiHistogramOfGradientsBorderGetDescriptorsSize() function call to get
 * total size for nLocations of output histogram block descriptor windows.  
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param hpLocations Host pointer to array of NppiPoint source pixel starting locations of requested descriptor windows. Important: hpLocations is a 
 *        <em>host pointer.</em>
 * \param nLocations Number of NppiPoint in pLocations array. 
 * \param pDstWindowDescriptorBuffer Output device memory buffer pointer of size hpDescriptorsSize bytes to first of nLoc descriptor windows (see nppiHistogramOfGradientsBorderGetDescriptorsSize() above).
 * \param oSizeROI \ref roi_specification of source image.
 * \param oHOGConfig Requested HOG configuration parameters structure.
 * \param pScratchBuffer Device memory buffer pointer of size hpBufferSize bytes to scratch memory buffer (see nppiHistogramOfGradientsBorderGetBufferSize() above).
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiHistogramOfGradientsBorder_32f_C1R(const Npp32f * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, 
                                       const NppiPoint * hpLocations, int nLocations, Npp32f * pDstWindowDescriptorBuffer, 
                                       NppiSize oSizeROI, const NppiHOGConfig oHOGConfig, Npp8u * pScratchBuffer, NppiBorderType eBorderType);


/**
 * 3 channel 32-bit floating point color per source image descriptor window location with source image border control 
 * to per descriptor window destination floating point histogram of gradients. Requires first calling nppiHistogramOfGradientsBorderGetBufferSize function
 * call to get required scratch (host) working buffer size and nppiHistogramOfGradientsBorderGetDescriptorsSize() function call to get
 * total size for nLocations of output histogram block descriptor windows.  
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset The pixel offset that pSrc points to relative to the origin of the source image. 
 * \param hpLocations Host pointer to array of NppiPoint source pixel starting locations of requested descriptor windows. Important: hpLocations is a 
 *        <em>host pointer.</em>
 * \param nLocations Number of NppiPoint in pLocations array. 
 * \param pDstWindowDescriptorBuffer Output device memory buffer pointer of size hpDescriptorsSize bytes to first of nLoc descriptor windows (see nppiHistogramOfGradientsBorderGetDescriptorsSize() above).
 * \param oSizeROI \ref roi_specification of source image.
 * \param oHOGConfig Requested HOG configuration parameters structure.
 * \param pScratchBuffer Device memory buffer pointer of size hpBufferSize bytes to scratch memory buffer (see nppiHistogramOfGradientsBorderGetBufferSize() above).
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiHistogramOfGradientsBorder_32f_C3R(const Npp32f * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, 
                                       const NppiPoint * hpLocations, int nLocations, Npp32f * pDstWindowDescriptorBuffer, 
                                       NppiSize oSizeROI, const NppiHOGConfig oHOGConfig, Npp8u * pScratchBuffer, NppiBorderType eBorderType);

/** @} HistogramOfOrientedGradientsBorder */

/** @} fixed_filters */

/** @} image_filtering_functions */

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* LW_NPPI_FILTERING_FUNCTIONS_H */
