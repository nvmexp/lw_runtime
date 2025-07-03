/*
 * SPDX-FileCopyrightText: Copyright (c) 2013-2021 LWPU CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#pragma once

#include <lwtypes.h>
#if defined(_MSC_VER)
#pragma warning(disable:4324)
#endif

//
// This file was generated with FINN, an LWPU coding tool.
// Source file: ctrl/ctrl2080/ctrl2080boardobj.finn
//
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
//
// Please edit the original FINN IDL file to create desired edits in this header
// See https://confluence.lwpu.com/display/CORERM/FINN for more info on how to
// edit FINN.
//
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)



#if (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)

#include "ctrl/ctrl2080/ctrl2080base.h"
#include "ctrl/ctrl2080/ctrl2080boardobjgrpclasses.h"

/*!
 * @file    ctrl2080boardobj.h
 *
 * @brief   LW20_SUBDEVICE_XX BOARDOBJ-related control commands and parameters.
 *
 * Base structures in RMCTRL equivalent to BOARDOBJ/BOARDOBJGRP in RM. LW2080
 * structs in this file carry info w.r.t BOARDOBJ/BOARDOBJGRP.
 */

/*!
 * @brief Type for representing an index of a BOARDOBJ within a
 * BOARDOBJGRP.  This type can also represent the number of elements
 * within a BOARDOBJGRP or the number of bits in a BOARDOBJGRPMASK.
 */
typedef LwU16 LwBoardObjIdx;

/*!
 * @brief Type for representing an index into a mask element within a
 * BOARDOBJGRPMASK to a @ref LW2080_CTRL_BOARDOBJGRP_MASK_PRIMITIVE.
 */
typedef LwU16 LwBoardObjMaskIdx;

/*!
 * @brief   Primitive type which a BOARDOBJGRPMASK is composed of.
 *
 * For example, a 32 bit mask will have one of these elements and a 256 bit
 * mask will have eight.
 */
typedef LwU32 LW2080_CTRL_BOARDOBJGRP_MASK_PRIMITIVE;

/*!
 * @brief   Min value a single BOARDOBJGRPMASK element can hold.
 *
 * @note    Must be kept in sync with @ref
 *          LW2080_CTRL_BOARDOBJGRP_MASK_PRIMITIVE.
 */
#define LW2080_CTRL_BOARDOBJGRP_MASK_ELEMENT_MIN           LW_U32_MIN

/*!
 * @brief   Max value a single BOARDOBJGRPMASK element can hold.
 *
 * @note    Must be kept in sync with @ref
 *          LW2080_CTRL_BOARDOBJGRP_MASK_PRIMITIVE.
 */
#define LW2080_CTRL_BOARDOBJGRP_MASK_ELEMENT_MAX           LW_U32_MAX

/*!
 * @brief   Number of bits in a the LW2080_CTRL_BOARDOBJGRP_MASK_PRIMITIVE type.
 *
 * This exists to eliminate the assumption that 32-bits is the width of
 * LW2080_CTRL_BOARDOBJGRP_MASK primitive element.
 *
 * @note    Left shift by 3 (multiply by 8) colwerts the sizeof in bytes to the
 * number of bits in our primitive/essential mask type.
 */
#define LW2080_CTRL_BOARDOBJGRP_MASK_MASK_ELEMENT_BIT_SIZE 32
/*!
 * @brief   Base structure which describes a BOARDOBJ in RMCTRL.
 *
 * All Objects which implement BOARDOBJ should extend this structure in their
 * RMCTRLs.
 */
typedef struct LW2080_CTRL_BOARDOBJ {
    /*!
     * @brief   BOARDOBJ type.
     *
     * This should be a unique value within the class that the BOARDOBJ belongs.
     */
    LwU8 type;
} LW2080_CTRL_BOARDOBJ;

/*!
 * @brief   Base structure which describes a BOARDOBJ_INTERFACE in RMCTRL.
 *
 * All objects which implement BOARDOBJ_INTERFACE should extend this structure
 * in their RMCTRLs.
 */
typedef struct LW2080_CTRL_BOARDOBJ_INTERFACE {
    /*!
     * @brief   Reserved for future use cases.
     */
    LwU8 rsvd;
} LW2080_CTRL_BOARDOBJ_INTERFACE;
typedef struct LW2080_CTRL_BOARDOBJ_INTERFACE *PLW2080_CTRL_BOARDOBJ_INTERFACE;

/*!
 * @brief   Type reserved for @ref LW2080_CTRL_BOARDOBJGRP_TYPE_ENUM
 *          enumerations.
 */
typedef LwU8 LW2080_CTRL_BOARDOBJGRP_TYPE;

/*!
 * @defgroup LW2080_CTRL_BOARDOBJGRP_TYPE_ENUM
 *
 * Enumeration of BOARDOBJGRP types. Of type @ref LW2080_CTRL_BOARDOBJGRP_TYPE.
 *
 * @{
 */
#define LW2080_CTRL_BOARDOBJGRP_TYPE_ILWALID          0x00U
#define LW2080_CTRL_BOARDOBJGRP_TYPE_E32              0x01U
#define LW2080_CTRL_BOARDOBJGRP_TYPE_E255             0x02U
#define LW2080_CTRL_BOARDOBJGRP_TYPE_E512             0x03U
#define LW2080_CTRL_BOARDOBJGRP_TYPE_E1024            0x04U
#define LW2080_CTRL_BOARDOBJGRP_TYPE_E2048            0x05U
/*!@}*/

/*!
 * @brief   Maximum number of BOARDOBJs for different BOARDOBJGRP classes.
 */
#define LW2080_CTRL_BOARDOBJGRP_E32_MAX_OBJECTS       (32U)
#define LW2080_CTRL_BOARDOBJGRP_E255_MAX_OBJECTS      (255U)
#define LW2080_CTRL_BOARDOBJGRP_E512_MAX_OBJECTS      (512U)
#define LW2080_CTRL_BOARDOBJGRP_E1024_MAX_OBJECTS     (1024U)
#define LW2080_CTRL_BOARDOBJGRP_E2048_MAX_OBJECTS     (2048U)

/*!
 * @brief   Value for an invalid Board Object index.
 *
 * This value should only be used directly for input to and output
 * from BOARDOBJ and BOARDOBJGRP code.
 *
 * @note This define should not be referenced directly in any
 * implementing object code.  Instead, each object should define it's
 * own IDX_ILWALID macro and alias it to whatever size fits their
 * specific index storage type.
 * For example, many objects still store indexes as LwU8 (because the
 * GRPs are either _E32 or _E255) while others store as LwBoardObjIdx
 * (lwrrently aliased to LwU16), so they should alias to a correct
 * type.
 */
#define LW2080_CTRL_BOARDOBJ_IDX_ILWALID              LW_U16_MAX

/*!
 * @brief   Value for an invalid Board Object index.
 *
 * This value encodes an invalid/unsupported BOARDOBJ index for an
 * 8-bit value.  This should be used within by any legacy appcode
 * implementing BOARDOBJGRP which stores/encodes indexes as 8-bit
 * values.
 *
 * All new groups should use @ref LW2080_CTRL_BOARDOBJ_IDX_ILWALID.
 *
 * @note This define should not be referenced directly in any
 * implementing object code.  Instead, each object should define it's
 * own IDX_ILWALID macro and alias it to whatever size fits their
 * specific index storage type.
 * For example, many objects still store indexes as LwU8 (because the
 * GRPs are either _E32 or _E255) while others store as LwU16 (for
 * GRPs larger than _E255), so they should alias to a correct type.
 */
#define LW2080_CTRL_BOARDOBJ_IDX_ILWALID_8BIT         LW_U8_MAX

/*!
 * @deprecated  Temporary define for existing references which have yet to be
 *              switched. Make use of LW2080_CTRL_BOARDOBJGRP_E32_MAX_OBJECTS
 *              instead.
 */
#define LW2080_CTRL_BOARDOBJ_MAX_BOARD_OBJECTS        LW2080_CTRL_BOARDOBJGRP_E32_MAX_OBJECTS

/*!
 * @brief   Computes the array index of a LW2080_CTRL_BOARDOBJGRP_MASK element
 *          storing requested bit.
 *
 * @note    Designed to be used in conjunction with @ref
 *          LW2080_CTRL_BOARDOBJGRP_MASK_MASK_ELEMENT_OFFSET.
 *
 * @param[in]   _bit    Index of a bit within a bit mask.
 *
 * @return  Array index of mask element containing @ref _bit.
 */
#define LW2080_CTRL_BOARDOBJGRP_MASK_MASK_ELEMENT_INDEX(_bit)                  \
    ((_bit) / LW2080_CTRL_BOARDOBJGRP_MASK_MASK_ELEMENT_BIT_SIZE)

/*!
 * @brief   Computes bit-position within LW2080_CTRL_BOARDOBJGRP_MASK element
 *          corresponding to requested bit.
 *
 * @note    Designed to be used in conjunction with @ref
 *          LW2080_CTRL_BOARDOBJGRP_MASK_MASK_ELEMENT_INDEX.
 *
 * @param[in]   _bit    Index of a bit within a bit mask.
 *
 * @return  Offset (in bits) within a mask element for @ref _bit.
 */
#define LW2080_CTRL_BOARDOBJGRP_MASK_MASK_ELEMENT_OFFSET(_bit)                 \
    ((_bit) % LW2080_CTRL_BOARDOBJGRP_MASK_MASK_ELEMENT_BIT_SIZE)

/*!
 * @brief   Computes the size of an array of LW2080_CTRL_BOARDOBJGRP_MASK
 *          elements that can store all mask's bits.
 *
 * @param[in]   _bits   Size of the mask in bits.
 *
 * @return  Number of array elements needed to store @ref _bits number of bits.
 */
#define LW2080_CTRL_BOARDOBJGRP_MASK_DATA_SIZE(_bits)                          \
    (LW2080_CTRL_BOARDOBJGRP_MASK_MASK_ELEMENT_INDEX((_bits) - 1U) + 1U)


/*!
 * @brief   Number of elements that are in the LW2080_CTRL_BOARDOBJGRP_MASK base
 *          class.
 *
 * @note    "START_SIZE" is used here to represent the size of the mask that
 *          derived classes must build up from. See @ref
 *          LW2080_CTRL_BOARDOBJGRP_MASK_E32, @ref
 *          LW2080_CTRL_BOARDOBJGRP_MASK_E255, @ref
 *          LW2080_CTRL_BOARDOBJGRP_MASK_E512, @ref
 *          LW2080_CTRL_BOARDOBJGRP_MASK_E1024, @ref
 *          LW2080_CTRL_BOARDOBJGRP_MASK_E2048.
 */
#define LW2080_CTRL_BOARDOBJGRP_MASK_ARRAY_START_SIZE 1U

/*!
 * @brief   Macro used to determine the number of LW2080_CTRL_BOARDOBJGRP_MASK
 *          elements required to extend the base number of elements in a mask,
 *          @ref LW2080_CTRL_BOARDOBJGRP_MASK_ARRAY_START_SIZE.
 *
 * @note    Used in order to avoid dynamic memory allocation and related
 *          code/data waste as well as two levels of indirection while accessing
 *          the data bits stored in an array of @ref
 *          LW2080_CTRL_BOARDOBJGRP_MASK_MASK_ELEMENT_BIT_SIZE sized words. The
 *          LW2080_CTRL_BOARDOBJGRP_MASK super-class array's size should be
 *          zero and actual data should be completely stored in child's array.
 *          Since most compilers reject structures with zero-sized arrays first
 *          element word was moved to the super-class and remaining array
 *          elements to child class.
 *
 * @param[in]   _bits   Total number of bits to be represented in the
 *                      LW2080_CTRL_BOARDOBJGRP_MASK extending mask class.
 *
 * @return  Number of additional mask elements that must be allocated in order
 *          to extend the LW2080_CTRL_BOARDOBJGRP_MASK base class.
 */
#define LW2080_CTRL_BOARDOBJGRP_MASK_ARRAY_EXTENSION_SIZE(_bits)               \
    (LW2080_CTRL_BOARDOBJGRP_MASK_DATA_SIZE(_bits) -                           \
     (LW2080_CTRL_BOARDOBJGRP_MASK_ARRAY_START_SIZE))

/*!
 * @brief   Macro to set input bit in LW2080_CTRL_BOARDOBJGRP_MASK.
 *
 * @param[in]   _pMask      PBOARDOBJGRPMASK of mask.
 * @param[in]   _bitIdx     Index of the target bit within the mask.
 */
#define LW2080_CTRL_BOARDOBJGRP_MASK_BIT_SET(_pMask, _bitIdx)                  \
    do {                                                                       \
        (_pMask)->pData[                                                       \
            LW2080_CTRL_BOARDOBJGRP_MASK_MASK_ELEMENT_INDEX(_bitIdx)] |=       \
                LWBIT_TYPE(                                                    \
                    LW2080_CTRL_BOARDOBJGRP_MASK_MASK_ELEMENT_OFFSET(_bitIdx), \
                    LW2080_CTRL_BOARDOBJGRP_MASK_PRIMITIVE);                   \
    } while (LW_FALSE)

/*!
 * @brief   Macro to clear input bit in LW2080_CTRL_BOARDOBJGRP_MASK.
 *
 * @param[in]   _pMask      PBOARDOBJGRPMASK of mask.
 * @param[in]   _bitIdx     Index of the target bit within the mask.
 */
#define LW2080_CTRL_BOARDOBJGRP_MASK_BIT_CLR(_pMask, _bitIdx)                  \
    do {                                                                       \
        (_pMask)->pData[                                                       \
            LW2080_CTRL_BOARDOBJGRP_MASK_MASK_ELEMENT_INDEX(_bitIdx)] &=       \
                ~LWBIT_TYPE(                                                   \
                    LW2080_CTRL_BOARDOBJGRP_MASK_MASK_ELEMENT_OFFSET(_bitIdx), \
                    LW2080_CTRL_BOARDOBJGRP_MASK_PRIMITIVE);                   \
    } while (LW_FALSE)

/*!
 * @brief   Macro to test input bit in LW2080_CTRL_BOARDOBJGRP_MASK.
 *
 * @param[in]   _pMask      PBOARDOBJGRPMASK of mask.
 * @param[in]   _bitIdx     Index of the target bit within the mask.
 */
#define LW2080_CTRL_BOARDOBJGRP_MASK_BIT_GET(_pMask, _bitIdx)                  \
    (((_pMask)->pData[LW2080_CTRL_BOARDOBJGRP_MASK_MASK_ELEMENT_INDEX(_bitIdx)]\
      & LWBIT_TYPE(LW2080_CTRL_BOARDOBJGRP_MASK_MASK_ELEMENT_OFFSET(_bitIdx),  \
                   LW2080_CTRL_BOARDOBJGRP_MASK_PRIMITIVE)) != 0U)

/*!
 * @brief   Not to be called directly. Helper macro allowing simple iteration
 *          over bits set in a LW2080_CTRL_BOARDOBJGRP_MASK.
 *
 * @param[in]       _maxObjects
 *     Maximum number of objects/bits in BOARDOJBGRP and its
 *     LW2080_CTRL_BOARDOBJGRP_MASK.
 * @param[in,out]   _index
 *     lvalue that is used as a bit index in the loop (can be declared
 *     as any LwU* or LwS* variable).
 *     CRPTODO - I think we need to revisit this.  Signed types of
 *     size <= sizeof(LwBoardObjIdx) can't work.
 * @param[in]       _pMask
 *     Pointer to LW2080_CTRL_BOARDOBJGRP_MASK over which to iterate.
 *
 * @note CRPTODO - Follow-on CL will add ct_assert that _index has
 * size >= sizeof(LwBoardObjIdx).
 */
#define LW2080_CTRL_BOARDOBJGRP_MASK_FOR_EACH_INDEX(_maxObjects,_index,_pMask) \
{                                                                              \
    for ((_index) = 0; (_index) < (_maxObjects); (_index)++)                   \
    {                                                                          \
        if(!LW2080_CTRL_BOARDOBJGRP_MASK_BIT_GET((_pMask), (_index)))          \
        {                                                                      \
            continue;                                                          \
        }
#define LW2080_CTRL_BOARDOBJGRP_MASK_FOR_EACH_INDEX_END                        \
    }                                                                          \
}

/*!
 * @brief   Macro allowing simple iteration over bits set in a
 *          LW2080_CTRL_BOARDOBJGRP_MASK_E32.
 *
 * Wrapper for @ref LW2080_CTRL_BOARDOBJGRP_MASK_FOR_EACH_INDEX().
 *
 * @copydetails LW2080_CTRL_BOARDOBJGRP_MASK_FOR_EACH_INDEX()
 */
#define LW2080_CTRL_BOARDOBJGRP_MASK_E32_FOR_EACH_INDEX(_index,_pMask)         \
    LW2080_CTRL_BOARDOBJGRP_MASK_FOR_EACH_INDEX(                               \
        LW2080_CTRL_BOARDOBJGRP_E32_MAX_OBJECTS,_index,_pMask)

/*!
 * @brief   Macro allowing simple iteration over bits set in a
 *          LW2080_CTRL_BOARDOBJGRP_MASK_E255.
 *
 * Wrapper for @ref LW2080_CTRL_BOARDOBJGRP_MASK_FOR_EACH_INDEX().
 *
 * @copydetails LW2080_CTRL_BOARDOBJGRP_MASK_FOR_EACH_INDEX()
 */
#define LW2080_CTRL_BOARDOBJGRP_MASK_E255_FOR_EACH_INDEX(_index,_pMask)        \
    LW2080_CTRL_BOARDOBJGRP_MASK_FOR_EACH_INDEX(                               \
        LW2080_CTRL_BOARDOBJGRP_E255_MAX_OBJECTS,_index,_pMask)

/*!
 * @brief   Macro allowing simple iteration over bits set in a
 *          LW2080_CTRL_BOARDOBJGRP_MASK_E512.
 *
 * Wrapper for @ref LW2080_CTRL_BOARDOBJGRP_MASK_FOR_EACH_INDEX().
 *
 * @copydetails LW2080_CTRL_BOARDOBJGRP_MASK_FOR_EACH_INDEX()
 */
#define LW2080_CTRL_BOARDOBJGRP_MASK_E512_FOR_EACH_INDEX(_index,_pMask)        \
    LW2080_CTRL_BOARDOBJGRP_MASK_FOR_EACH_INDEX(                               \
        LW2080_CTRL_BOARDOBJGRP_E512_MAX_OBJECTS,_index,_pMask)

/*!
 * @brief   Macro allowing simple iteration over bits set in a
 *          LW2080_CTRL_BOARDOBJGRP_MASK_E1024.
 *
 * Wrapper for @ref LW2080_CTRL_BOARDOBJGRP_MASK_FOR_EACH_INDEX().
 *
 * @copydetails LW2080_CTRL_BOARDOBJGRP_MASK_FOR_EACH_INDEX()
 */
#define LW2080_CTRL_BOARDOBJGRP_MASK_E1024_FOR_EACH_INDEX(_index,_pMask)       \
    LW2080_CTRL_BOARDOBJGRP_MASK_FOR_EACH_INDEX(                               \
        LW2080_CTRL_BOARDOBJGRP_E1024_MAX_OBJECTS,_index,_pMask)

/*!
 * @brief   Macro allowing simple iteration over bits set in a
 *          LW2080_CTRL_BOARDOBJGRP_MASK_E2048.
 *
 * Wrapper for @ref LW2080_CTRL_BOARDOBJGRP_MASK_FOR_EACH_INDEX().
 *
 * @copydetails LW2080_CTRL_BOARDOBJGRP_MASK_FOR_EACH_INDEX()
 */
#define LW2080_CTRL_BOARDOBJGRP_MASK_E2048_FOR_EACH_INDEX(_index,_pMask)       \
    LW2080_CTRL_BOARDOBJGRP_MASK_FOR_EACH_INDEX(                               \
        LW2080_CTRL_BOARDOBJGRP_E2048_MAX_OBJECTS,_index,_pMask)
/*!
 * @brief   Not to be called directly. Macro to initialize a
 *          LW2080_CTRL_BOARDOBJGRP_MASK to an empty mask.
 *
 * @param[in]   _pMask      LW2080_CTRL_BOARDOBJGRP_MASK to initialize.
 * @param[in]   _bitSize    LwU8 specifying size of the mask in bits.
 */
#define LW2080_CTRL_BOARDOBJGRP_MASK_INIT(_pMask,_bitSize)                              \
    do {                                                                                \
        LwBoardObjIdx _dataCount = LW2080_CTRL_BOARDOBJGRP_MASK_DATA_SIZE(_bitSize);    \
        LwBoardObjIdx _dataIndex;                                                       \
        for (_dataIndex = 0; _dataIndex < _dataCount; _dataIndex++)                     \
        {                                                                               \
            (_pMask)->pData[_dataIndex] = 0U;                                           \
        }                                                                               \
    } while (LW_FALSE)

/*!
 * @brief   Macro to initialize LW2080_CTRL_BOARDOBJGRP_MASK_E32 to an empty
 *          mask.
 *
 * Wrapper for @ref LW2080_CTRL_BOARDOBJGRP_MASK_INIT().
 *
 * @copydetails LW2080_CTRL_BOARDOBJGRP_MASK_INIT()
 */
#define LW2080_CTRL_BOARDOBJGRP_MASK_E32_INIT(_pMask)                          \
    LW2080_CTRL_BOARDOBJGRP_MASK_INIT(_pMask,                                  \
        LW2080_CTRL_BOARDOBJGRP_E32_MAX_OBJECTS)

/*!
 * @brief   Macro to initialize LW2080_CTRL_BOARDOBJGRP_MASK_E255 to an empty
 *          mask.
 *
 * Wrapper for @ref LW2080_CTRL_BOARDOBJGRP_MASK_INIT().
 *
 * @copydetails LW2080_CTRL_BOARDOBJGRP_MASK_INIT()
 */
#define LW2080_CTRL_BOARDOBJGRP_MASK_E255_INIT(_pMask)                         \
    LW2080_CTRL_BOARDOBJGRP_MASK_INIT(_pMask,                                  \
        LW2080_CTRL_BOARDOBJGRP_E255_MAX_OBJECTS)

/*!
 * @brief   Macro to initialize LW2080_CTRL_BOARDOBJGRP_MASK_E512 to an empty
 *          mask.
 *
 * Wrapper for @ref LW2080_CTRL_BOARDOBJGRP_MASK_INIT().
 *
 * @copydetails LW2080_CTRL_BOARDOBJGRP_MASK_INIT()
 */
#define LW2080_CTRL_BOARDOBJGRP_MASK_E512_INIT(_pMask)                         \
    LW2080_CTRL_BOARDOBJGRP_MASK_INIT(_pMask,                                  \
        LW2080_CTRL_BOARDOBJGRP_E512_MAX_OBJECTS)

/*!
 * @brief   Macro to initialize LW2080_CTRL_BOARDOBJGRP_MASK_E1024 to an empty
 *          mask.
 *
 * Wrapper for @ref LW2080_CTRL_BOARDOBJGRP_MASK_INIT().
 *
 * @copydetails LW2080_CTRL_BOARDOBJGRP_MASK_INIT()
 */
#define LW2080_CTRL_BOARDOBJGRP_MASK_E1024_INIT(_pMask)                        \
    LW2080_CTRL_BOARDOBJGRP_MASK_INIT(_pMask,                                  \
        LW2080_CTRL_BOARDOBJGRP_E1024_MAX_OBJECTS)

/*!
 * @brief   Macro to initialize LW2080_CTRL_BOARDOBJGRP_MASK_E2048 to an empty
 *          mask.
 *
 * Wrapper for @ref LW2080_CTRL_BOARDOBJGRP_MASK_INIT().
 *
 * @copydetails LW2080_CTRL_BOARDOBJGRP_MASK_INIT()
 */
#define LW2080_CTRL_BOARDOBJGRP_MASK_E2048_INIT(_pMask)                        \
    LW2080_CTRL_BOARDOBJGRP_MASK_INIT(_pMask,                                  \
        LW2080_CTRL_BOARDOBJGRP_E2048_MAX_OBJECTS)

/*!
 * @brief   Board Object Group Mask base class.
 *
 * Used to unify access to all LW2080_CTRL_BOARDOBJGRP_MASK_E** child classes.
 */
typedef struct LW2080_CTRL_BOARDOBJGRP_MASK {
    /*!
     * @brief   Start with a single element array which is enough to represent
     *          LW2080_CTRL_BOARDOBJGRP_MASK_MASK_ELEMENT_BIT_SIZE bits.
     *
     * @note    Must be the last member of this structure.
     */
    // FINN PORT: The below field is a bit vector!
    // In FINN, bit vectors are arrays of bools and each bool becomes 1 bit when used in an array
    // FINN generates an array of LwU32's on the back end for these bit vectors
    LwU32 pData[1] /* 32 bits */;
} LW2080_CTRL_BOARDOBJGRP_MASK;
typedef struct LW2080_CTRL_BOARDOBJGRP_MASK *PLW2080_CTRL_BOARDOBJGRP_MASK;

/*!
 * @brief   LW2080_CTRL_BOARDOBJGRP_MASK child class capable of storing 32 bits
 *          indexed between 0..31.
 */
typedef struct LW2080_CTRL_BOARDOBJGRP_MASK_E32 {
    /*!
     * @brief   LW2080_CTRL_BOARDOBJGRP_MASK super-class. Must be the first
     *          member of the structure.
     */
    LW2080_CTRL_BOARDOBJGRP_MASK super;
} LW2080_CTRL_BOARDOBJGRP_MASK_E32;
typedef struct LW2080_CTRL_BOARDOBJGRP_MASK_E32 *PLW2080_CTRL_BOARDOBJGRP_MASK_E32;

/*!
 * @brief   LW2080_CTRL_BOARDOBJGRP_MASK child class capable of storing 255 bits
 *          indexed between 0..254.
 */
typedef struct LW2080_CTRL_BOARDOBJGRP_MASK_E255 {
    /*!
     * @brief   LW2080_CTRL_BOARDOBJGRP_MASK super-class. Must be the first
     *          member of the structure.
     */
    LW2080_CTRL_BOARDOBJGRP_MASK super;

    /*!
     * @brief   Continuation of the array of
     *          LW2080_CTRL_BOARDOBJGRP_MASK_PRIMITIVE elements representing the
     *          bit-mask.
     *
     * @note    Must be the second member of the structure.
     */

    // FINN PORT: The below field is a bit vector!
    // In FINN, bit vectors are arrays of bools and each bool becomes 1 bit when used in an array
    // FINN generates an array of LwU32's on the back end for these bit vectors
    LwU32 pDataE255[7] /* 223 bits */;
} LW2080_CTRL_BOARDOBJGRP_MASK_E255;
typedef struct LW2080_CTRL_BOARDOBJGRP_MASK_E255 *PLW2080_CTRL_BOARDOBJGRP_MASK_E255;

/*!
 * @brief   LW2080_CTRL_BOARDOBJGRP_MASK child class capable of storing 512 bits
 *          indexed between 0..511.
 */
typedef struct LW2080_CTRL_BOARDOBJGRP_MASK_E512 {
    /*!
     * @brief   LW2080_CTRL_BOARDOBJGRP_MASK super-class. Must be the first
     *          member of the structure.
     */
    LW2080_CTRL_BOARDOBJGRP_MASK super;

    /*!
     * @brief   Continuation of the array of
     *          LW2080_CTRL_BOARDOBJGRP_MASK_PRIMITIVE elements representing the
     *          bit-mask.
     *
     * @note    Must be the second member of the structure.
     */

    // FINN PORT: The below field is a bit vector!
    // In FINN, bit vectors are arrays of bools and each bool becomes 1 bit when used in an array
    // FINN generates an array of LwU32's on the back end for these bit vectors
    LwU32 pDataE512[15] /* 480 bits */;
} LW2080_CTRL_BOARDOBJGRP_MASK_E512;
typedef struct LW2080_CTRL_BOARDOBJGRP_MASK_E512 *PLW2080_CTRL_BOARDOBJGRP_MASK_E512;

/*!
 * @brief   LW2080_CTRL_BOARDOBJGRP_MASK child class capable of storing 1024 bits
 *          indexed between 0..1023.
 */
typedef struct LW2080_CTRL_BOARDOBJGRP_MASK_E1024 {
    /*!
     * @brief   LW2080_CTRL_BOARDOBJGRP_MASK super-class. Must be the first
     *          member of the structure.
     */
    LW2080_CTRL_BOARDOBJGRP_MASK super;

    /*!
     * @brief   Continuation of the array of
     *          LW2080_CTRL_BOARDOBJGRP_MASK_PRIMITIVE elements representing the
     *          bit-mask.
     *
     * @note    Must be the second member of the structure.
     */

    // FINN PORT: The below field is a bit vector!
    // In FINN, bit vectors are arrays of bools and each bool becomes 1 bit when used in an array
    // FINN generates an array of LwU32's on the back end for these bit vectors
    LwU32 pDataE1024[31] /* 992 bits */;
} LW2080_CTRL_BOARDOBJGRP_MASK_E1024;
typedef struct LW2080_CTRL_BOARDOBJGRP_MASK_E1024 *PLW2080_CTRL_BOARDOBJGRP_MASK_E1024;

/*!
 * @brief   LW2080_CTRL_BOARDOBJGRP_MASK child class capable of storing 2048 bits
 *          indexed between 0..2047.
 */
typedef struct LW2080_CTRL_BOARDOBJGRP_MASK_E2048 {
    /*!
     * @brief   LW2080_CTRL_BOARDOBJGRP_MASK super-class. Must be the first
     *          member of the structure.
     */
    LW2080_CTRL_BOARDOBJGRP_MASK super;

    /*!
     * @brief   Continuation of the array of
     *          LW2080_CTRL_BOARDOBJGRP_MASK_PRIMITIVE elements representing the
     *          bit-mask.
     *
     * @note    Must be the second member of the structure.
     */

    // FINN PORT: The below field is a bit vector!
    // In FINN, bit vectors are arrays of bools and each bool becomes 1 bit when used in an array
    // FINN generates an array of LwU32's on the back end for these bit vectors
    LwU32 pDataE2048[63] /* 2016 bits */;
} LW2080_CTRL_BOARDOBJGRP_MASK_E2048;
typedef struct LW2080_CTRL_BOARDOBJGRP_MASK_E2048 *PLW2080_CTRL_BOARDOBJGRP_MASK_E2048;

/*!
 * @brief   BOARDOJBGRP super-class structure.
 *
 * @note    No classes should ever implement this structure directly. They
 *          should instead implement one of the LW2080_CTRL_BOARDOBJGRP_E<XYZ>
 *          structures.
 */
typedef struct LW2080_CTRL_BOARDOBJGRP {
    /*!
     * @brief   Mask of objects within this BOARDOBJGRP.
     *
     * @note    Must always be the first member in this structure to allow
     *          casting between _SUPER and _E<XYZ> classes.
     */
    LW2080_CTRL_BOARDOBJGRP_MASK objMask;
} LW2080_CTRL_BOARDOBJGRP;
typedef struct LW2080_CTRL_BOARDOBJGRP *PLW2080_CTRL_BOARDOBJGRP;

/*!
 * @brief   Base structure which describes a BOARDOBJGRP_E32 in RMCTRL.
 *
 * All Objects which implement BOARDOBJGRP_E32 should extend this structure in
 * their RMCTRLs
 */
typedef struct LW2080_CTRL_BOARDOBJGRP_E32 {
    /*!
     * @brief   Mask of objects within this BOARDOBJGRP.
     *
     * @note    Must always be the first-element in this structure to allow
     *          casting between _SUPER and _E32 classes.
     */
    LW2080_CTRL_BOARDOBJGRP_MASK_E32 objMask;
} LW2080_CTRL_BOARDOBJGRP_E32;
typedef struct LW2080_CTRL_BOARDOBJGRP_E32 *PLW2080_CTRL_BOARDOBJGRP_E32;

/*!
 * @brief   Base structure which describes a BOARDOBJGRP_E255 in RMCTRL.
 *
 * All Objects which implement BOARDOBJGRP_E255 should extend this structure in
 * their RMCTRLs
 */
typedef struct LW2080_CTRL_BOARDOBJGRP_E255 {
    /*!
     * @brief   Mask of objects within this BOARDOBJGRP.
     *
     * @note    Must always be the first-element in this structure to allow
     *          casting between _SUPER and _E255 classes.
     */
    LW2080_CTRL_BOARDOBJGRP_MASK_E255 objMask;
} LW2080_CTRL_BOARDOBJGRP_E255;
typedef struct LW2080_CTRL_BOARDOBJGRP_E255 *PLW2080_CTRL_BOARDOBJGRP_E255;

/*!
 * @brief   Base structure which describes a BOARDOBJGRP_E512 in RMCTRL.
 *
 * All Objects which implement BOARDOBJGRP_E512 should extend this structure in
 * their RMCTRLs
 */
typedef struct LW2080_CTRL_BOARDOBJGRP_E512 {
    /*!
     * @brief   Mask of objects within this BOARDOBJGRP.
     *
     * @note    Must always be the first-element in this structure to allow
     *          casting between _SUPER and _E512 classes.
     */
    LW2080_CTRL_BOARDOBJGRP_MASK_E512 objMask;
} LW2080_CTRL_BOARDOBJGRP_E512;
typedef struct LW2080_CTRL_BOARDOBJGRP_E512 *PLW2080_CTRL_BOARDOBJGRP_E512;

/*!
 * @brief   Base structure which describes a BOARDOBJGRP_E1024 in RMCTRL.
 *
 * All Objects which implement BOARDOBJGRP_E1024 should extend this structure in
 * their RMCTRLs
 */
typedef struct LW2080_CTRL_BOARDOBJGRP_E1024 {
    /*!
     * @brief   Mask of objects within this BOARDOBJGRP.
     *
     * @note    Must always be the first-element in this structure to allow
     *          casting between _SUPER and _E1024 classes.
     */
    LW2080_CTRL_BOARDOBJGRP_MASK_E1024 objMask;
} LW2080_CTRL_BOARDOBJGRP_E1024;
typedef struct LW2080_CTRL_BOARDOBJGRP_E1024 *PLW2080_CTRL_BOARDOBJGRP_E1024;

/*!
 * @brief   Base structure which describes a BOARDOBJGRP_E2048 in RMCTRL.
 *
 * All Objects which implement BOARDOBJGRP_E2048 should extend this structure in
 * their RMCTRLs
 */
typedef struct LW2080_CTRL_BOARDOBJGRP_E2048 {
    /*!
     * @brief   Mask of objects within this BOARDOBJGRP.
     *
     * @note    Must always be the first-element in this structure to allow
     *          casting between _SUPER and _E2048 classes.
     */
    LW2080_CTRL_BOARDOBJGRP_MASK_E2048 objMask;
} LW2080_CTRL_BOARDOBJGRP_E2048;
typedef struct LW2080_CTRL_BOARDOBJGRP_E2048 *PLW2080_CTRL_BOARDOBJGRP_E2048;

/*!
 * @brief   Macro to provide the BOARDOBJ type for a given (UNIT, CLASS, TYPE)
 *          combination.
 *
 * @details For arguments (FOO, BAR, BAZ), this macro will return
 *          LW2080_CTRL_FOO_BAR_TYPE_BAZ
 *
 * @param[in]   _unit   The unit.
 * @param[in]   _class  The class.
 * @param[in]   _type   The type.
 *
 * @return  BOARDOBJ object type identifier.
 */
#define LW2080_CTRL_BOARDOBJ_TYPE(_unit, _class, _type)                        \
    LW2080_CTRL_##_unit##_##_class##_TYPE_##_type

/*!
 * Type to be used for all VFE equation indices.
 * Intended for use in RMCTRL structures, RM code and data structures,
 * and RM-PMU shared structures only.
 *
 * PMU will have its own VFE index type @ref LwVfeEquIdx
 *
 * @note    This type is lwrrently typedef-ed to LwU8, which is the same
 *          as current VFE equation indices. However, once VFE internals are
 *          moved to 16-bit indices and >255 VFE entries, this will be
 *          typedefed to LwU16 (through LwBoardObjIdx)
 *
 * @note    There is a reason this define is present here. Originally
 *          intended to be placed in ctrl2080vfe.h/finn, there was
 *          an issue with cirlwlar dependency. Even though
 *          ctrl2080vfe.finn is an independent file of ctrl2080perf.finn,
 *          the FINN compiler was automatically including the file
 *          ctrl2080perf.finn inside it, and including
 *          ctrl2080vfe.h elsewhere was causing lots of
 *          cirlwlar dependency preprocessor header file issues.
 *          This is the next best option, since ctrl2080boardobj is
 *          truly independent of other files.
 *
 * */
typedef LwBoardObjIdx LW2080_CTRL_PERF_VFE_EQU_IDX;
#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)

