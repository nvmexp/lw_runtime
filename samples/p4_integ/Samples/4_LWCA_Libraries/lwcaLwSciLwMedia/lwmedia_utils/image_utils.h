/* Copyright (c) 2022, LWPU CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of LWPU CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROLWREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef _LWMEDIA_TEST_IMAGE_UTILS_H_
#define _LWMEDIA_TEST_IMAGE_UTILS_H_

#ifdef __cplusplus
extern "C" {
#endif

#include "misc_utils.h"
#include "lwmedia_core.h"
#include "lwmedia_surface.h"
#include "lwmedia_image.h"
#include "cmdline.h"

#if (LW_IS_SAFETY == 1)
#include "lwmedia_image_internal.h"
#endif

#define PACK_RGBA(R, G, B, A)  (((uint32_t)(A) << 24) | ((uint32_t)(B) << 16) | \
                                ((uint32_t)(G) << 8) | (uint32_t)(R))
#define DEFAULT_ALPHA   0x80




//  ReadImage
//
//    ReadImage()  Read image from file
//
//  Arguments:
//
//   filename
//      (in) Input file name
//
//   frameNum
//      (in) Frame number to read. Use for stream input files.
//
//   width
//      (in) Surface width
//
//   height
//      (in) Surface height
//
//   image
//      (out) Pointer to pre-allocated output surface
//
//   uvOrderFlag
//      (in) Flag for UV order. If true - UV; If false - VU;
//
//   bytesPerPixel
//      (in) Bytes per pixel. Nedded for RAW image types handling.
//         RAW8 - 1 byte per pixel
//         RAW10, RAW12, RAW14 - 2 bytes per pixel
//
//   pixelAlignment
//      (in) Alignment of bits in pixel.
//         0 - LSB Aligned
//         1 - MSB Aligned

LwMediaStatus
ReadImage(
    char *fileName,
    uint32_t frameNum,
    uint32_t width,
    uint32_t height,
    LwMediaImage *image,
    LwMediaBool uvOrderFlag,
    uint32_t bytesPerPixel,
    uint32_t pixelAlignment);

//  InitImage
//
//    InitImage()  Init image data to zeros
//
//  Arguments:
//
//   image
//      (in) image to initialize
//
//   width
//      (in) Surface width
//
//   height
//      (in) Surface height

LwMediaStatus
InitImage(
    LwMediaImage *image,
    uint32_t width,
    uint32_t height);

LwMediaStatus
AllocateBufferToWriteImage(
    Blit2DTest *ctx,
    LwMediaImage *image,
    LwMediaBool uvOrderFlag,
    LwMediaBool appendFlag);

//  WriteImageToBuffer
//
//    WriteImageToBuffer()  Save RGB or YUV image
//
LwMediaStatus
WriteImageToAllocatedBuffer(
    Blit2DTest *ctx,
    LwMediaImage *image,
    LwMediaBool uvOrderFlag,
    LwMediaBool appendFlag,
    uint32_t bytesPerPixel);

#ifdef __cplusplus
}
#endif

#endif /* _LWMEDIA_TEST_IMAGE_UTILS_H_ */
