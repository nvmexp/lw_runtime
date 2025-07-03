/*
* _LWRM_COPYRIGHT_BEGIN_
*
* Copyright 2001-2010 by LWPU Corporation.  All rights reserved.  All
* information contained herein is proprietary and confidential to LWPU
* Corporation.  Any use, reproduction, or disclosure without the written
* permission of LWPU Corporation is prohibited.
*
* _LWRM_COPYRIGHT_END_
*/

#ifndef __CLA097SURF_H__
#define __CLA097SURF_H__

/*
 * cla097surf.h:  Class headers for surface descriptor structures used by 
 * compiler code for performing loads, stores, and atomics to bindless 
 * surfaces on Kepler.
 *
 * This structure is dolwmented in more detail in the "Bindless Surfaces" 
 * section of:
 *
 *    //sw/<tree>/drivers/common/lwi/spec/kepler.txt
 *
 * The architecture documentation is at:
 *
 *    https://p4viewer/get/hw/doc/gpu/kepler/kepler/design/IAS/SML1/ISA/opcodes/IAS_Visible_ISA_1_Kepler.htm#SM_surfaceAccessSequences
 * 
 * the following mapping translates between these software names and
 * the corresponding arch names:
 *
 *    LWA097_SURF_LOCATION              SUINFO_BASE_UPPER
 *    LWA097_SURF_TYPE                  SUINFO_FORMAT
 *    LWA097_SURF_CLAMP_X               SUINFO_CLIPX_DOTP
 *    LWA097_SURF_CLAMP_X_BUFFER        SUINFO_PL1D_CLIPX
 *    LWA097_SURF_CLAMP_Y               SUINFO_CLIPY
 *    LWA097_SURF_CLAMP_Z               SUINFO_CLIPZ
 *    LWA097_SURF_SIZE_X                SUINFO_BL_WD_IN_BLKS
 *    LWA097_SURF_SIZE_Y                SUINFO_BL_HT_IN_BLKS
 *    LWA097_SURF_ARRAY_PITCH           SUINFO_PLANESZ
 *    LWA097_SURF_CLAMP_Z_LAYER         SUINFO_???
 *  
 * however, arch does not impose any restrictions on the order of 
 * these words in memory, only on the layout of fields within a word.
 *
 *
 * WARNING:  THIS FILE IS HAND-GENERATED AND MUST BE MANUALLY KEPT IN SYNC 
 * WITH HARDWARE DEFINITIONS!
 */


/*
 * Interface version #define; may be useful for modular branching if we need
 * to change the descriptor structure.
 */
#define LWA097_SURF_INTERFACE_VERSION       3


/*
 * Single word structures specifying the individual word components of the
 * surface descriptor structures.
 */

typedef struct LWA097SurfaceLocationRec {
    LwU32                           addressOver256;
} LWA097SurfaceLocation;

typedef struct LWA097SurfaceTypeRec {
    unsigned int                    format          : 7;
    unsigned int                    pad1            : 1;
    unsigned int                    rScale          : 2;
    unsigned int                    log2CompCount   : 2;
    unsigned int                    pad2            : 2;
    unsigned int                    clampBehavior   : 2;
    unsigned int                    log2TexelSize   : 3;
    unsigned int                    pad3            : 12;
    unsigned int                    invalid         : 1;
} LWA097SurfaceType;

typedef struct LWA097SurfaceClampRec {
    unsigned int                    maxTexel        : 20;
    unsigned int                    pad1            : 1;
    unsigned int                    layout          : 1;
    unsigned int                    blockShift      : 4;
    unsigned int                    log2TexelSize   : 3;
    unsigned int                    log2BlockSize   : 3;
} LWA097SurfaceClamp;

typedef struct LWA097SurfaceClampBufferRec {
    LwU32                           maxTexel;
} LWA097SurfaceClampBuffer;

typedef struct LWA097SurfaceSizeRec {
    unsigned int                    size            : 26;
    unsigned int                    clampedRCType   : 2;
    unsigned int                    clampedRBType   : 1;
    unsigned int                    pad1            : 1;
    unsigned int                    clampedRAType   : 2;
} LWA097SurfaceSize;

typedef struct LWA097SurfaceArrayPitchRec {
    LwU32                           pitchOver256;
} LWA097SurfaceArrayPitch;

typedef struct LWA097SurfaceClampZLayerRec {
    unsigned int                    pad1            : 27;
    unsigned int                    log2BlockSizeZ  : 3;
    unsigned int                    pad2            : 2;
} LWA097SurfaceClampZLayer;

/*
 * Generic surface descriptor structures, usable for all target types except 
 * Layer2D (1D, 2D, 3D, Array1D, Array2D, Buffer).
 */
typedef struct LWA097SurfaceRec {
    LWA097SurfaceLocation           location;           /* word 0 */
    LWA097SurfaceType               type;               /* word 1 */
    union {
        LWA097SurfaceClamp          clampX;             /* word 2 - all but Buffer */
        LWA097SurfaceClampBuffer    clampXBuffer;       /* word 2 - Buffer */
    } u2;
    LWA097SurfaceSize               sizeX;              /* word 3 - all but Buffer, 1D, Array1D */
    LWA097SurfaceClamp              clampY;             /* word 4 - all but Buffer, 1D, Array1D */
    LWA097SurfaceArrayPitch         arrayPitch;         /* word 5 - only Array1D, Array2D */
    LWA097SurfaceClamp              clampZ;             /* word 6 - Array1D, Array2D, 3D */
    LWA097SurfaceSize               sizeY;              /* word 7 - only 3D */
} LWA097Surface;


/*
 * Specialized surface descriptor structures for each of the target types (1D,
 * 2D, Layer2D, 3D, Array1D, Array2D, Buffer).  These are simply streamlined versions of 
 * the generic descriptor.
 */
typedef struct LWA097Surface1DRec {
    LWA097SurfaceLocation           location;           /* word 0 */
    LWA097SurfaceType               type;               /* word 1 */
    LWA097SurfaceClamp              clampX;             /* word 2 */
} LWA097Surface1D;

typedef struct LWA097Surface2DRec {
    LWA097SurfaceLocation           location;           /* word 0 */
    LWA097SurfaceType               type;               /* word 1 */
    LWA097SurfaceClamp              clampX;             /* word 2 */
    LWA097SurfaceSize               sizeX;              /* word 3 */
    LWA097SurfaceClamp              clampY;             /* word 4 */
} LWA097Surface2D;

typedef struct LWA097SurfaceLayer2DRec {
    LWA097SurfaceLocation           location;           /* word 0 */
    LWA097SurfaceType               type;               /* word 1 */
    LWA097SurfaceClamp              clampX;             /* word 2 */
    LWA097SurfaceSize               sizeX;              /* word 3 */
    LWA097SurfaceClamp              clampY;             /* word 4 */
    LwU32                           unused5;            /* word 5 */
    LwU32                           unused6;            /* word 6 */
    LWA097SurfaceClampZLayer        clampZLayer;        /* word 7 */
} LWA097SurfaceLayer2D;

typedef struct LWA097Surface3DRec {
    LWA097SurfaceLocation           location;           /* word 0 */
    LWA097SurfaceType               type;               /* word 1 */
    LWA097SurfaceClamp              clampX;             /* word 2 */
    LWA097SurfaceSize               sizeX;              /* word 3 */
    LWA097SurfaceClamp              clampY;             /* word 4 */
    LwU32                           unused5;            /* word 5 */
    LWA097SurfaceClamp              clampZ;             /* word 6 */
    LWA097SurfaceSize               sizeY;              /* word 7 */
} LWA097Surface3D;

typedef struct LWA097Surface1DArrayRec {
    LWA097SurfaceLocation           location;           /* word 0 */
    LWA097SurfaceType               type;               /* word 1 */
    LWA097SurfaceClamp              clampX;             /* word 2 */
    LwU32                           unused3;            /* word 3 */
    LwU32                           unused4;            /* word 4 */
    LWA097SurfaceArrayPitch         arrayPitch;         /* word 5 */
    LWA097SurfaceClamp              clampZ;             /* word 6 */
} LWA097Surface1DArray;

typedef struct LWA097Surface2DArrayRec {
    LWA097SurfaceLocation           location;           /* word 0 */
    LWA097SurfaceType               type;               /* word 1 */
    LWA097SurfaceClamp              clampX;             /* word 2 */
    LWA097SurfaceSize               sizeX;              /* word 3 */
    LWA097SurfaceClamp              clampY;             /* word 4 */
    LWA097SurfaceArrayPitch         arrayPitch;         /* word 5 */
    LWA097SurfaceClamp              clampZ;             /* word 6 */
} LWA097Surface2DArray;

typedef struct LWA097SurfaceBufferRec {
    LWA097SurfaceLocation           location;           /* word 0 */
    LWA097SurfaceType               type;               /* word 1 */
    LWA097SurfaceClampBuffer        clampXBuffer;       /* word 2 */
} LWA097SurfaceBuffer;

typedef struct LWA097SurfaceNonBufferRec {
    /* Generic structure */
    LWA097SurfaceLocation           location;           /* word 0 */
    LWA097SurfaceType               type;               /* word 1 */
    LWA097SurfaceClamp              clampX;             /* word 2 */
    LWA097SurfaceSize               sizeX;              /* word 3 */
    LWA097SurfaceClamp              clampY;             /* word 4 */
    LwU32                           arrayPitch;         /* word 5 */
    LWA097SurfaceClamp              clampZ;             /* word 6 */
    LWA097SurfaceSize               sizeY;              /* word 7 */
} LWA097SurfaceNonBuffer;


/*
 * Word number and bitfield locations for the various surface descriptor words;
 * suitable for use in various LW bitfield manipulation macros.
 */
#define LWA097_SURF_LOCATION                                    0
#define LWA097_SURF_LOCATION_ADDRESS_ADDR_OVER_256              31:0

#define LWA097_SURF_TYPE                                        1
#define LWA097_SURF_TYPE_FORMAT                                 6:0
#define LWA097_SURF_TYPE_FORMAT_DISABLED                        0x00000000
#define LWA097_SURF_TYPE_FORMAT_RF32_GF32_BF32_AF32             2
#define LWA097_SURF_TYPE_FORMAT_RF16_GF16_BF16_AF16             12
#define LWA097_SURF_TYPE_FORMAT_RF32_GF32                       13
#define LWA097_SURF_TYPE_FORMAT_RF16_GF16                       33
#define LWA097_SURF_TYPE_FORMAT_BF10GF11RF11                    36
#define LWA097_SURF_TYPE_FORMAT_RF32                            41
#define LWA097_SURF_TYPE_FORMAT_RF16                            54
#define LWA097_SURF_TYPE_FORMAT_RU32_GU32_BU32_AU32             4
#define LWA097_SURF_TYPE_FORMAT_RU16_GU16_BU16_AU16             11
#define LWA097_SURF_TYPE_FORMAT_RU32_GU32                       15
#define LWA097_SURF_TYPE_FORMAT_AU2BU10GU10RU10                 21
#define LWA097_SURF_TYPE_FORMAT_AU8BU8GU8RU8                    28
#define LWA097_SURF_TYPE_FORMAT_RU16_GU16                       32
#define LWA097_SURF_TYPE_FORMAT_RU32                            40
#define LWA097_SURF_TYPE_FORMAT_GU8RU8                          49
#define LWA097_SURF_TYPE_FORMAT_RU16                            53
#define LWA097_SURF_TYPE_FORMAT_RU8                             58
#define LWA097_SURF_TYPE_FORMAT_RS32_GS32_BS32_AS32             3
#define LWA097_SURF_TYPE_FORMAT_RS16_GS16_BS16_AS16             10
#define LWA097_SURF_TYPE_FORMAT_RS32_GS32                       14
#define LWA097_SURF_TYPE_FORMAT_AS8BS8GS8RS8                    27
#define LWA097_SURF_TYPE_FORMAT_RS16_GS16                       31
#define LWA097_SURF_TYPE_FORMAT_RS32                            39
#define LWA097_SURF_TYPE_FORMAT_GS8RS8                          48
#define LWA097_SURF_TYPE_FORMAT_RS16                            52
#define LWA097_SURF_TYPE_FORMAT_RS8                             57
#define LWA097_SURF_TYPE_FORMAT_R16_G16_B16_A16                 8
#define LWA097_SURF_TYPE_FORMAT_A2B10G10R10                     19
#define LWA097_SURF_TYPE_FORMAT_A8B8G8R8                        24
#define LWA097_SURF_TYPE_FORMAT_A8R8G8B8                        17
#define LWA097_SURF_TYPE_FORMAT_B8G8R8A8                        71
#define LWA097_SURF_TYPE_FORMAT_R16_G16                         29
#define LWA097_SURF_TYPE_FORMAT_G8R8                            46
#define LWA097_SURF_TYPE_FORMAT_R16                             50
#define LWA097_SURF_TYPE_FORMAT_R8                              55
#define LWA097_SURF_TYPE_FORMAT_A8                              59
#define LWA097_SURF_TYPE_FORMAT_RN16_GN16_BN16_AN16             9
#define LWA097_SURF_TYPE_FORMAT_AN8BN8GN8RN8                    26
#define LWA097_SURF_TYPE_FORMAT_RN16_GN16                       30
#define LWA097_SURF_TYPE_FORMAT_GN8RN8                          47
#define LWA097_SURF_TYPE_FORMAT_RN16                            51
#define LWA097_SURF_TYPE_FORMAT_RN8                             56
#define LWA097_SURF_TYPE_RSCALE                                 9:8
#define LWA097_SURF_TYPE_LOG2_COMPONENT_COUNT                   11:10
#define LWA097_SURF_TYPE_CLAMP_BEHAVIOR                         15:14
#define LWA097_SURF_TYPE_CLAMP_BEHAVIOR_CLAMP                   0
#define LWA097_SURF_TYPE_CLAMP_BEHAVIOR_ZERO                    1
#define LWA097_SURF_TYPE_CLAMP_BEHAVIOR_TRAP                    2
#define LWA097_SURF_TYPE_LOG2_TEXEL_SIZE                        18:16
#define LWA097_SURF_TYPE_ILWALID                                31:31
#define LWA097_SURF_TYPE_ILWALID_FALSE                          0
#define LWA097_SURF_TYPE_ILWALID_TRUE                           1

#define LWA097_SURF_CLAMP_X                                     2
#define LWA097_SURF_CLAMP_X_MAX_TEXEL                           19:0
#define LWA097_SURF_CLAMP_X_LAYOUT                              21:21
#define LWA097_SURF_CLAMP_X_LAYOUT_BLOCKLINEAR                  0
#define LWA097_SURF_CLAMP_X_LAYOUT_PITCH                        1
#define LWA097_SURF_CLAMP_X_BLOCK_SHIFT                         25:22
#define LWA097_SURF_CLAMP_X_LOG2_TEXEL_SIZE                     28:26
#define LWA097_SURF_CLAMP_X_LOG2_BLOCK_SIZE                     31:29

#define LWA097_SURF_CLAMP_X_BUFFER                              2
#define LWA097_SURF_CLAMP_X_BUFFER_MAX_TEXEL                    31:0

#define LWA097_SURF_SIZE_X                                      3
#define LWA097_SURF_SIZE_X_SIZE                                 25:0
#define LWA097_SURF_SIZE_X_CLAMPED_RC_TYPE                      27:26
#define LWA097_SURF_SIZE_X_CLAMPED_RC_TYPE_32_BIT               0 /* U32 */
#define LWA097_SURF_SIZE_X_CLAMPED_RC_TYPE_24_BIT               1 /* U24 */
#define LWA097_SURF_SIZE_X_CLAMPED_RC_TYPE_16_BIT               2 /* U16H0 */
#define LWA097_SURF_SIZE_X_CLAMPED_RB_TYPE                      28:28
#define LWA097_SURF_SIZE_X_CLAMPED_RB_TYPE_24_BIT               0 /* U24 */
#define LWA097_SURF_SIZE_X_CLAMPED_RB_TYPE_16_BIT               1 /* U16H0 */
#define LWA097_SURF_SIZE_X_CLAMPED_RA_TYPE                      31:30
#define LWA097_SURF_SIZE_X_CLAMPED_RA_TYPE_32_BIT               0 /* U32 */
#define LWA097_SURF_SIZE_X_CLAMPED_RA_TYPE_24_BIT               1 /* U24 */
#define LWA097_SURF_SIZE_X_CLAMPED_RA_TYPE_16_BIT               2 /* U16H0 */

#define LWA097_SURF_CLAMP_Y                                     4
#define LWA097_SURF_CLAMP_Y_MAX_TEXEL                           19:0
#define LWA097_SURF_CLAMP_Y_LAYOUT                              21:21
#define LWA097_SURF_CLAMP_Y_LAYOUT_BLOCKLINEAR                  0
#define LWA097_SURF_CLAMP_Y_LAYOUT_PITCH                        1
#define LWA097_SURF_CLAMP_Y_BLOCK_SHIFT                         25:22
#define LWA097_SURF_CLAMP_Y_LOG2_BLOCK_SIZE                     31:29

#define LWA097_SURF_ARRAY_PITCH                                 5
#define LWA097_SURF_ARRAY_PITCH_PITCH_OVER_256                  31:0

#define LWA097_SURF_CLAMP_Z                                     6
#define LWA097_SURF_CLAMP_Z_MAX_TEXEL                           19:0
#define LWA097_SURF_CLAMP_Z_LAYOUT                              21:21
#define LWA097_SURF_CLAMP_Z_LAYOUT_BLOCKLINEAR                  0
#define LWA097_SURF_CLAMP_Z_LAYOUT_PITCH                        1
#define LWA097_SURF_CLAMP_Z_BLOCK_SHIFT                         25:22
#define LWA097_SURF_CLAMP_Z_LOG2_BLOCK_SIZE                     31:29

#define LWA097_SURF_SIZE_Y                                      7
#define LWA097_SURF_SIZE_Y_SIZE                                 25:0
#define LWA097_SURF_SIZE_Y_CLAMPED_RC_TYPE                      27:26
#define LWA097_SURF_SIZE_Y_CLAMPED_RC_TYPE_32_BIT               0 /* U32 */
#define LWA097_SURF_SIZE_Y_CLAMPED_RC_TYPE_24_BIT               1 /* U24 */
#define LWA097_SURF_SIZE_Y_CLAMPED_RC_TYPE_16_BIT               2 /* U16H0 */
#define LWA097_SURF_SIZE_Y_CLAMPED_RB_TYPE                      28:28
#define LWA097_SURF_SIZE_Y_CLAMPED_RB_TYPE_24_BIT               0 /* U24 */
#define LWA097_SURF_SIZE_Y_CLAMPED_RB_TYPE_16_BIT               1 /* U16H0 */
#define LWA097_SURF_SIZE_Y_CLAMPED_RA_TYPE                      31:30
#define LWA097_SURF_SIZE_Y_CLAMPED_RA_TYPE_32_BIT               0 /* U32 */
#define LWA097_SURF_SIZE_Y_CLAMPED_RA_TYPE_24_BIT               1 /* U24 */
#define LWA097_SURF_SIZE_Y_CLAMPED_RA_TYPE_16_BIT               2 /* U16H0 */

#define LWA097_SURF_CLAMP_Z_LAYER                               7
#define LWA097_SURF_CLAMP_Z_LAYER_LOG2_BLOCK_SIZE               29:27



#endif // #ifndef __CLA097SURF_H__
