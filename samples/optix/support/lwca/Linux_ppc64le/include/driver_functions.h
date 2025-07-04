/*
 * Copyright 1993-2018 LWPU Corporation.  All rights reserved.
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

#if !defined(__DRIVER_FUNCTIONS_H__)
#define __DRIVER_FUNCTIONS_H__

#include "builtin_types.h"
#include "crt/host_defines.h"
#include "driver_types.h"

/**
 * \addtogroup LWDART_MEMORY
 *
 * @{
 */

/**
 * \brief Returns a lwdaPitchedPtr based on input parameters
 *
 * Returns a ::lwdaPitchedPtr based on the specified input parameters \p d,
 * \p p, \p xsz, and \p ysz.
 *
 * \param d   - Pointer to allocated memory
 * \param p   - Pitch of allocated memory in bytes
 * \param xsz - Logical width of allocation in elements
 * \param ysz - Logical height of allocation in elements
 *
 * \return
 * ::lwdaPitchedPtr specified by \p d, \p p, \p xsz, and \p ysz
 *
 * \sa make_lwdaExtent, make_lwdaPos
 */
static __inline__ __host__ struct lwdaPitchedPtr make_lwdaPitchedPtr(void *d, size_t p, size_t xsz, size_t ysz) 
{
  struct lwdaPitchedPtr s;

  s.ptr   = d;
  s.pitch = p;
  s.xsize = xsz;
  s.ysize = ysz;

  return s;
}

/**
 * \brief Returns a lwdaPos based on input parameters
 *
 * Returns a ::lwdaPos based on the specified input parameters \p x,
 * \p y, and \p z.
 *
 * \param x - X position
 * \param y - Y position
 * \param z - Z position
 *
 * \return
 * ::lwdaPos specified by \p x, \p y, and \p z
 *
 * \sa make_lwdaExtent, make_lwdaPitchedPtr
 */
static __inline__ __host__ struct lwdaPos make_lwdaPos(size_t x, size_t y, size_t z) 
{
  struct lwdaPos p;

  p.x = x;
  p.y = y;
  p.z = z;

  return p;
}

/**
 * \brief Returns a lwdaExtent based on input parameters
 *
 * Returns a ::lwdaExtent based on the specified input parameters \p w,
 * \p h, and \p d.
 *
 * \param w - Width in elements when referring to array memory, in bytes when referring to linear memory
 * \param h - Height in elements
 * \param d - Depth in elements
 *
 * \return
 * ::lwdaExtent specified by \p w, \p h, and \p d
 *
 * \sa make_lwdaPitchedPtr, make_lwdaPos
 */
static __inline__ __host__ struct lwdaExtent make_lwdaExtent(size_t w, size_t h, size_t d) 
{
  struct lwdaExtent e;

  e.width  = w;
  e.height = h;
  e.depth  = d;

  return e;
}

/** @} */ /* END LWDART_MEMORY */

#endif /* !__DRIVER_FUNCTIONS_H__ */
