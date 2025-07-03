/*
 * Copyright 2007 LWPU Corporation.  All Rights Reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property and
 * proprietary rights in and to this software and related documentation.  Any
 * use, reproduction, disclosure or distribution of this software and related
 * documentation without an express license agreement from LWPU Corporation
 * is strictly prohibited.
 */

#ifndef INCLUDED_LWRM_DRF_H
#define INCLUDED_LWRM_DRF_H

/**
 *  @defgroup lwrm_drf RM DRF Macros
 *
 *  @ingroup lwddk_rm
 *
 * The following suite of macros are used for generating values to write into
 * hardware registers, or for extracting fields from read registers.  The
 * hardware headers have a RANGE define for each field in the register in the
 * form of x:y, 'x' being the high bit, 'y' the lower.  Through a clever use
 * of the C ternary operator, x:y may be passed into the macros below to
 * geneate masks, shift values, etc.
 *
 * There are two basic flavors of DRF macros, the first is used to define
 * a new register value from 0, the other is modifiying a field given a
 * register value.  An example of the first:
 *
 * reg = LW_DRF_DEF( HW, REGISTER0, FIELD0, VALUE0 )
 *     | LW_DRF_DEF( HW, REGISTER0, FIELD3, VALUE2 );
 *
 * To modify 'reg' from the previous example:
 *
 * reg = LW_FLD_SET_DRF_DEF( HW, REGISTER0, FIELD2, VALUE1, reg );
 *
 * To pass in numeric values instead of defined values from the header:
 *
 * reg = LW_DRF_NUM( HW, REGISTER3, FIELD2, 1024 );
 *
 * To read a value from a register:
 *
 * val = LW_DRF_VAL( HW, REGISTER3, FIELD2, reg );
 *
 * Some registers have non-zero reset values which may be extracted from the
 * hardware headers via LW_RESETVAL.
 */

/*
 * The LW_FIELD_* macros are helper macros for the public LW_DRF_* macros.
 */
#define LW_FIELD_LOWBIT(x)      (0?x)
#define LW_FIELD_HIGHBIT(x)     (1?x)
#define LW_FIELD_SIZE(x)        (LW_FIELD_HIGHBIT(x)-LW_FIELD_LOWBIT(x)+1)
#define LW_FIELD_SHIFT(x)       ((0?x)%32)
#define LW_FIELD_MASK(x)        (0xFFFFFFFFUL>>(31-((1?x)%32)+((0?x)%32)))
#define LW_FIELD_BITS(val, x)   (((val) & LW_FIELD_MASK(x))<<LW_FIELD_SHIFT(x))
#define LW_FIELD_SHIFTMASK(x)   (LW_FIELD_MASK(x)<< (LW_FIELD_SHIFT(x)))

/*
 * The LW_FIELD64_* macros are helper macros for the public LW_DRF64_* macros.
 */
#define LW_FIELD64_SHIFT(x) ((0?x)%64)
#define LW_FIELD64_MASK(x)  (0xFFFFFFFFFFFFFFFFULL>>(63-((1?x)%64)+((0?x)%64)))


/** LW_DRF_DEF - define a new register value.

    @ingroup lwrm_drf

    @param d register domain (hardware block)
    @param r register name
    @param f register field
    @param c defined value for the field
 */
#define LW_DRF_DEF(d,r,f,c) \
    ((d##_##r##_0_##f##_##c) << LW_FIELD_SHIFT(d##_##r##_0_##f##_RANGE))

/** LW_DRF_NUM - define a new register value.

    @ingroup lwrm_drf

    @param d register domain (hardware block)
    @param r register name
    @param f register field
    @param n numeric value for the field
 */
#define LW_DRF_NUM(d,r,f,n) \
    (((n)& LW_FIELD_MASK(d##_##r##_0_##f##_RANGE)) << \
        LW_FIELD_SHIFT(d##_##r##_0_##f##_RANGE))

/** LW_DRF_VAL - read a field from a register value.

    @ingroup lwrm_drf

    @param d register domain (hardware block)
    @param r register name
    @param f register field
    @param v register value
 */
#define LW_DRF_VAL(d,r,f,v) \
    (((v)>> LW_FIELD_SHIFT(d##_##r##_0_##f##_RANGE)) & \
        LW_FIELD_MASK(d##_##r##_0_##f##_RANGE))

/** LW_FLD_SET_DRF_NUM - modify a register field.

    @ingroup lwrm_drf

    @param d register domain (hardware block)
    @param r register name
    @param f register field
    @param n numeric field value
    @param v register value
 */
#define LW_FLD_SET_DRF_NUM(d,r,f,n,v) \
    ((v & ~LW_FIELD_SHIFTMASK(d##_##r##_0_##f##_RANGE)) | LW_DRF_NUM(d,r,f,n))

/** LW_FLD_SET_DRF_DEF - modify a register field.

    @ingroup lwrm_drf

    @param d register domain (hardware block)
    @param r register name
    @param f register field
    @param c defined field value
    @param v register value
 */
#define LW_FLD_SET_DRF_DEF(d,r,f,c,v) \
    (((v) & ~LW_FIELD_SHIFTMASK(d##_##r##_0_##f##_RANGE)) | \
        LW_DRF_DEF(d,r,f,c))

/** LW_RESETVAL - get the reset value for a register.

    @ingroup lwrm_drf

    @param d register domain (hardware block)
    @param r register name
 */
#define LW_RESETVAL(d,r)    (d##_##r##_0_RESET_VAL)


/** LW_DRF64_NUM - define a new 64-bit register value.

    @ingroup lwrm_drf

    @param d register domain
    @param r register name
    @param f register field
    @param n numeric value for the field
 */
#define LW_DRF64_NUM(d,r,f,n) \
    (((n)& LW_FIELD64_MASK(d##_##r##_0_##f##_RANGE)) << \
        LW_FIELD64_SHIFT(d##_##r##_0_##f##_RANGE))

/** LW_FLD_TEST_DRF_DEF - test a field from a register

    @ingroup lwrm_drf

    @param d register domain (hardware block)
    @param r register name
    @param f register field
    @param c defined value for the field
    @param v register value
 */
#define LW_FLD_TEST_DRF_DEF(d,r,f,c,v) \
    ((LW_DRF_VAL(d, r, f, v) == d##_##r##_0_##f##_##c))

/** LW_FLD_TEST_DRF_NUM - test a field from a register

    @ingroup lwrm_drf

    @param d register domain (hardware block)
    @param r register name
    @param f register field
    @param n numeric value for the field
    @param v register value
 */
#define LW_FLD_TEST_DRF_NUM(d,r,f,n,v) \
    ((LW_DRF_VAL(d, r, f, v) == n))

/** LW_DRF_IDX_VAL - read a field from an indexed register value.

    @ingroup lwrm_drf

    @param d register domain (hardware block)
    @param r register name
    @param f register field
    @param i register index
    @param v register value
 */
#define LW_DRF_IDX_VAL(d,r,f,i,v) \
    (((v)>> LW_FIELD_SHIFT(d##_##r##_0_##f##_RANGE(i))) & \
        LW_FIELD_MASK(d##_##r##_0_##f##_RANGE(i)))

/** LW_FLD_IDX_TEST_DRF - test a field from an indexed register value.

    @ingroup lwrm_drf

    @param d register domain (hardware block)
    @param r register name
    @param f register field
    @param i register index
    @param c defined value for the field
    @param v register value
 */
#define LW_FLD_IDX_TEST_DRF(d,r,f,i,c,v) \
    ((LW_DRF_IDX_VAL(d, r, f, i, v) == d##_##r##_0_##f##_##c))

#endif // INCLUDED_LWRM_DRF_H
