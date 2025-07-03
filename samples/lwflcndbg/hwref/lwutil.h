/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 1993-2013 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#ifndef _LWUTIL_H_
#define _LWUTIL_H_

#include "lwmisc.h"

//---------------------------------------------------------------------------
//
//  Common types.
//
//---------------------------------------------------------------------------
#ifndef VOID
#define VOID    void
#endif
#ifndef BOOL
#define BOOL    S032
#endif
#ifndef TRUE
#define TRUE    1L
#endif
#ifndef FALSE
#define FALSE   0L
#endif
#ifndef NULL
#define NULL    0L
#endif

#define BITMASK(b)              (BIT(b) - 1)
#define DR_VAL(d,r,v)           (((v)>>DRF_SHIFT(LW ## d ## r ))&DRF_MASK(LW ## d ## r ))
#define D_VAL64(d,v)            (((v)>>DRF_SHIFT64(LW ## d ))&DRF_MASK64(LW ## d ))
#define GPU_REG_WR_DRF_NUM(d,r,f,n) GPU_REG_WR32(LW ## d ## r, DRF_NUM(d,r,f,n))
#define GPU_REG_WR_DRF_DEF(d,r,f,c) GPU_REG_WR32(LW ## d ## r, DRF_DEF(d,r,f,c))
#define GPU_FLD_WR_DRF_NUM(d,r,f,n) GPU_REG_WR32(LW##d##r,(GPU_REG_RD32(LW##d##r)&~(DRF_MASK(LW##d##r##f)<<DRF_SHIFT(LW##d##r##f)))|DRF_NUM(d,r,f,n))
#define GPU_FLD_WR_DRF_DEF(d,r,f,c) GPU_REG_WR32(LW##d##r,(GPU_REG_RD32(LW##d##r)&~(DRF_MASK(LW##d##r##f)<<DRF_SHIFT(LW##d##r##f)))|DRF_DEF(d,r,f,c))
#define GPU_REG_RD_DRF(d,r,f)       (((GPU_REG_RD32(LW ## d ## r))>>DRF_SHIFT(LW ## d ## r ## f))&DRF_MASK(LW ## d ## r ## f))
#define FB_RD32_DRF(d,r,f)      (((FB_RD32(LW ## d ## r))>>DRF_SHIFT(LW ## d ## r ## f))&DRF_MASK(LW ## d ## r ## f))
#define FB_RD32_64_DRF(d,r,f)   (((FB_RD32_64(LW ## d ## r))>>DRF_SHIFT64(LW ## d ## r ## f))&DRF_MASK64(LW ## d ## r ## f))
#define DRF_BIT_MASK(d,r,f)     (DRF_MASK(LW##d##r##f)<<DEVICE_BASE(LW##d##r##f))
#define REF_VAL(drf,v)          (((v)>>DRF_SHIFT(drf))&DRF_MASK(drf))

#define GPU_REG_IDX_RD_DRF(d,r,i,f)       (((GPU_REG_RD32(LW ## d ## r(i)))>>DRF_SHIFT(LW ## d ## r ## f))&DRF_MASK(LW ## d ## r ## f))
#define GPU_REG_IDX_WR_DRF_NUM(d,r,i,f,n) GPU_REG_WR32(LW ## d ## r(i), DRF_NUM(d,r,f,n))
#define GPU_REG_IDX_WR_DRF_DEF(d,r,i,f,c) GPU_REG_WR32(LW ## d ## r(i), DRF_DEF(d,r,f,c))

#define GPU_FLD_TEST_DRF(d,r,f,c)     (FLD_TEST_DRF(d,r,f,c,GPU_REG_RD32(LW##d##r)))
// DRIF means the register is indexed. as in DR(i)_F
#define GPU_FLD_TEST_DRIF_NUM(d,r,i,f,n) (GPU_REG_IDX_RD_DRF(d,r,i,f) == n)
#define GPU_FLD_TEST_DRIF_DEF(d,r,i,f,c) (GPU_REG_IDX_RD_DRF(d,r,i,f) == LW##d##r##f##c)

#define DEV_FLD_TEST_DRF(dev,inst,d,r,f,c)     (FLD_TEST_DRF(d,r,f,c,DEV_REG_RD32(LW##d##r,dev,inst)))

#define SF_OFFSET(sf)           (((0?sf)/32)<<2)
#define SF_SHIFT(sf)            ((0?sf)&31)
#undef  SF_MASK
#define SF_MASK(sf)             (0xFFFFFFFF>>(31-(1?sf)+(0?sf)))
#define SF_DEF(s,f,c)           ((LW ## s ## f ## c)<<SF_SHIFT(LW ## s ## f))
#define SF_NUM(s,f,n)           (((n)&SF_MASK(LW ## s ## f))<<SF_SHIFT(LW ## s ## f))
#define SF_VAL(s,f,v)           (((v)>>SF_SHIFT(LW ## s ## f))&SF_MASK(LW ## s ## f))

#define SF_SHIFT64(drf)        ((0?drf) % 64)
#define SF_MASK64(drf)         (0xFFFFFFFFFFFFFFFFULL>>(63-(1?drf)+(0?drf)))
#define SF_DEF64(d,r,f,n)      ((d ## r ## f ## c)<<SF_SHIFT64(d ## r ## f))
#define SF_NUM64(d,r,f,n)      (((n)&SF_MASK64(d ## r ## f))<<SF_SHIFT64(d ## r ## f))
#define SF_VAL64(d,r,f,v)      (((v)>>SF_SHIFT64(d ## r ## f))&SF_MASK64(d ## r ## f))
#define S_VAL64(d,v)           (((v)>>SF_SHIFT64(d))&SF_MASK64(d))

//
// 64 bit operations
//
#define LO_DW64(r)               ((U032)((((LwU64)(r))<<32)>>32))
#define HI_DW64(r)               ((U032)(((LwU64)(r))>>32))
#define DW64(hi,lo)              ((((LwU64)(hi))<<32)|lo)

//
// Frame Buffer read functions
//
U032 FB_RD_FIELD32    (U032 address, U032 x, U032 y);
U032 FB_RD_FIELD32_64 (LwU64 address, U032 x, U032 y);

//
// Misc defines
//
#define LW_PAGE_SIZE                    0x1000
#define LW_PAGE_MASK                    0x0fff

#define ARRAY_ELEMENT_COUNT(_array) (sizeof(_array) / sizeof(_array[0]))

#define LW_ALIGN_PTR_DOWN(p, gran)  ((void *) LW_ALIGN_DOWN(((uintptr_t)p), (gran)))
#define LW_ALIGN_PTR_UP(p, gran)    ((void *) LW_ALIGN_UP(((uintptr_t)p), (gran)))

#define LW_PAGE_ALIGN_DOWN(value)   LW_ALIGN_DOWN((value), LW_PAGE_SIZE)
#define LW_PAGE_ALIGN_UP(value)     LW_ALIGN_UP((value), LW_PAGE_SIZE)

// Destructive operation on n32
#define ROUNDUP_POW2(n32) \
{                         \
    n32--;                \
    n32 |= n32 >> 1;      \
    n32 |= n32 >> 2;      \
    n32 |= n32 >> 4;      \
    n32 |= n32 >> 8;      \
    n32 |= n32 >> 16;     \
    n32++;                \
}

/* Pack two Signed 16-bit coordinates. Mustn't sign-extend */
#define PACK_XY(x,y)    ((V032)((((U032)(x))<<16)|(((U032)(y))&0x0000FFFF)) )

/* Pack two Unsigned 16-bit dimensions */
#define PACK_WH(w,h)    ((V032)((((U032)(w))<<16)|(((U032)(h))&0x0000FFFF)) )

//
// colors
//
/* Pack 1-bit Alpha and 5-bit R,G,B values into LW_COLOR_FORMAT_LE_X16A1R5G5B5  */
#define PACK_ARGB15(a,r,g,b)                                    \
            ((V032)((((a)?(1<<15):0))|(((U032)(r)&0x1F)<<10)|   \
            (((U032)(g)&0x1F)<<5)|((U032)(b)&0x1F)))

/* Pack 5-bit R,G,B values into LW_COLOR_FORMAT_LE_X17R5G5B5 */
#define PACK_RGB15(r,g,b)   (PACK_ARGB15(0,r,g,b))

/* Pack three 8-bit R,G,B values into LW_COLOR_FORMAT_LE_X8R8G8B8 */
#define PACK_RGB24(r,g,b)                                       \
            ((V032)((((U032)(r)&0xFF)<<16)|                     \
            (((U032)(g)&0xFF)<<8)|((U032)(b)&0xFF)))

/* Pack 8-bit Alpha and 8-bit R,G,B values into LW_COLOR_FORMAT_LE_A8R8G8B8 */
#define PACK_ARGB24(a,r,g,b)                                    \
            ((V032)((((U032)(a))<<24)|(((U032)(r)&0xFF)<<16)|   \
            (((U032)(g)&0xFF)<<8)|((U032)(b)&0xFF)))

/* Pack three 10-bit R,G,B values into LW_COLOR_FORMAT_LE_X2R10G10B10 */
#define PACK_RGB30(r,g,b)                                       \
            ((V032)((((U032)(r)&0x3FF)<<20)|                    \
            (((U032)(g)&0x3FF)<<10)|((U032)(b)&0x3FF)))

/* Pack 2-bit Alpha and 10-bit R,G,B values into LW_COLOR_FORMAT_LE_A2R10G10B10 */
#define PACK_ARGB30(a,r,g,b)                                    \
            ((V032)(((U032)(a)<<30)|(((U032)(r)&0x3FF)<<20)|    \
            (((U032)(g)&0x3FF)<<10)|((U032)(b)&0x3FF)))


//
// Utility Macros
//
#define TOUPPER(c) ((((c) >= 'a') && ((c) <= 'z')) ? ((c) - 'a' + 'A') : (c))
#define GENERIC_DELIMS " \t\n"

//
// Utility functions
//
VOID DeviceIDToString(U032 devid, char *name);
int isDelim(char ch, char* delims);
void skipDelims(char** input, char* delims);
char* struppr (char *a);

#endif // _LWUTIL_H_
