/* Copyright 2019 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 *_LWRM_COPYRIGHT_END_
 */

// This file must be included only after dev_ctrl headers. 

#ifndef __INTR_PRIVATE_H
#define __INTR_PRIVATE_H

#define INTR_MAX_VECTOR     (REG_NUM_BITS * LW_CTRL_CPU_INTR_LEAF_ARRAY_SIZE_PER_FN) // maximum value of intr vector number

#define INTR_MAX LW_CTRL_CPU_INTR_LEAF_ARRAY_SIZE_PER_FN * REG_NUM_BITS // numbre of interrupts per gfid

#define CPU LW_FALSE
#define GSP LW_TRUE
#define GSP_ABSENT LW_FALSE
#define GSP_PRESENT LW_TRUE
#define VIRTUAL_ABSENT LW_FALSE
#define VIRTUAL_PRESENT LW_TRUE

#define INTR_LEAF_BIT_IDX(x)   ((x) % REG_NUM_BITS)
#define INTR_LEAF_IDX(x)      ((x) / REG_NUM_BITS)
#define INTR_TOP_BIT_IDX(x)    ((INTR_LEAF_IDX(x) % (LW_CTRL_LEAF_REG_PER_TOP_BIT * REG_NUM_BITS)) / LW_CTRL_LEAF_REG_PER_TOP_BIT)
#define INTR_TOP_IDX(x)       ((x) / INTR_PER_TOP)
#define INTR_GSP_TOP_IDX(x)    ((x) / INTR_PER_GSP_TOP)

#define INTR_VECT_LEAF_SET(vect, gfid, gsp)                    \
        (                                                   \
            (GPU_REG_RD32(                                  \
                (gsp) ?                                     \
                (LW_CTRL_GSP_INTR_LEAF(INTR_LEAF_IDX(vect))) :\
                (LW_CTRL_CPU_INTR_LEAF((gfid) * LW_CTRL_CPU_INTR_LEAF_ARRAY_SIZE_PER_FN + INTR_LEAF_IDX(vect))) \
            )                                               \
            & (0x1 << (INTR_LEAF_BIT_IDX(vect)))) > 0          \
        )

#define INTR_VECT_TOP_SET(vect, gfid)                          \
        (                                                   \
            (GPU_REG_RD(                                    \
                ((gfid) >= 0) ?                             \
                (LW_CTRL_CPU_INTR_TOP((gfid) * (LW_CTRL_CPU_INTR_TOP__SIZE_1 / MAX_GPUS) + INTR_TOP_IDX(vect))) : \
                (LW_CTRL_GSP_INTR_TOP(INTR_GSP_TOP_IDX(vect))) \
            )                                               \
            & (0x1 << (INTR_TOP_BIT_IDX(vect)))) > 0           \
        )


#define INTR_LEAF_ENABLE_SET_ADDR(vect, gfid, gsp)              \
        (                                                   \
            (!(gsp)) ?                                      \
            (LW_CTRL_CPU_INTR_LEAF_EN_SET((gfid) * LW_CTRL_CPU_INTR_LEAF_ARRAY_SIZE_PER_FN + INTR_LEAF_IDX(vect))) : \
            (LW_CTRL_GSP_INTR_LEAF_EN_SET(INTR_LEAF_IDX(vect))) \
        )

#define INTR_LEAF_ENABLE_CLEAR_ADDR(vect, gfid, gsp)            \
        (                                                   \
            (!(gsp)) ?                                      \
            (LW_CTRL_CPU_INTR_LEAF_EN_CLEAR((gfid) * LW_CTRL_CPU_INTR_LEAF_ARRAY_SIZE_PER_FN + INTR_LEAF_IDX(vect))) : \
            (LW_CTRL_GSP_INTR_LEAF_EN_CLEAR(INTR_LEAF_IDX(vect))) \
        )

#define INTR_TOP_ENABLE_SET_ADDR(vect, gfid, gsp)               \
        (                                                   \
            (!(gsp)) ?                                      \
            (LW_CTRL_CPU_INTR_TOP_EN_SET((gfid) * INTR_TOPS_IMPLEMENTED_PER_FN + INTR_TOP_IDX(vect))) : \
            (LW_CTRL_GSP_INTR_TOP(INTR_GSP_TOP_IDX(vect)))     \
        )

#define INTR_LEAF_ENABLED(vect, gfid, gsp)                    \
        (                                                   \
      (GPU_REG_RD32(INTR_LEAF_ENABLE_SET_ADDR((vect), (gfid), (gsp))) \
      & (0x1 << INTR_LEAF_BIT_IDX(vect))) > 0            \
        )

#define INTR_TOP_ENABLED(vect, gfid, gsp)                     \
        (                                                   \
            (GPU_REG_RD32(INTR_TOP_ENABLE_SET_ADDR((vect), (gfid), (gsp))) \
            & (0x1 << INTR_TOP_BIT_IDX(vect))) > 0             \
        )

#endif
