/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2021  by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#ifndef _LED_TCA6507_H_
#define _LED_TCA6507_H_

#include "lwtypes.h"
#include "lwmisc.h"

#define LED_TCA6507_REG_SELECT0             0
#define LED_TCA6507_REG_SELECT1             1
#define LED_TCA6507_REG_SELECT2             2
#define LED_TCA6507_SELECT_REG_NUM          3

#define LED_TCA6507_LED_STATE_OFF           0
#define LED_TCA6507_LED_STATE_OFF_ALT       1
#define LED_TCA6507_LED_STATE_ON_PWM0       2
#define LED_TCA6507_LED_STATE_ON_PWM1       3
#define LED_TCA6507_LED_STATE_ON_MAX        4
#define LED_TCA6507_LED_STATE_ON_1_SHOT     5
#define LED_TCA6507_LED_STATE_BLINK_BANK0   6
#define LED_TCA6507_LED_STATE_BLINK_BANK1   7

#define LED_TCA6507_SELECT_PORT(v, idx)     (!!((v) & LWBIT(idx)))
#define LED_TCA6507_LED_STATE(sel, idx)     (LED_TCA6507_SELECT_PORT((sel)[2], (idx)) << 2 | \
                                             LED_TCA6507_SELECT_PORT((sel)[1], (idx)) << 1 | \
                                             LED_TCA6507_SELECT_PORT((sel)[0], (idx)) << 0)
#define LED_TCA6507_LED_SET_STATE(sel, idx, state)                \
    do {                                                          \
        unsigned    _idx;                                         \
        for (_idx = 0; _idx < LED_TCA6507_SELECT_REG_NUM; ++_idx) \
        {                                                         \
            if (((state) & LWBIT(_idx)) != 0)                     \
            {                                                     \
                (sel)[_idx] |= LWBIT(idx);                        \
            }                                                     \
            else                                                  \
            {                                                     \
                (sel)[_idx] &= ~LWBIT(idx);                       \
            }                                                     \
        }                                                         \
    } while (0)

// Used for blink control 

#define LED_TCA6507_REG_FADE_ON_TIME               3
#define LED_TCA6507_REG_FULLY_ON_TIME              4
#define LED_TCA6507_REG_FADE_OFF_TIME              5
#define LED_TCA6507_REG_FIRST_FULLY_OFF_TIME       6
#define LED_TCA6507_REG_SECOND_FULLY_OFF_TIME      7
#define LED_TCA6507_TIME_REG_NUM                   5
#define LED_TCA6507_REG_TIME_BANK0               3:0
#define LED_TCA6507_REG_TIME_BANK1               7:4
#define LED_TCA6507_REG_TIME_IDX(reg)            ((reg) - LED_TCA6507_REG_FADE_ON_TIME)
#define LED_TCA6507_REG_SET_TIME(regs, reg, bank, code)       \
        do {                                                  \
            unsigned    _idx;                                 \
            _idx = LED_TCA6507_REG_TIME_IDX(reg);             \
            LWSWITCH_ASSERT(_idx < LED_TCA6507_TIME_REG_NUM); \
            (regs)[_idx] &= ~DRF_SHIFTMASK(bank);             \
            (regs)[_idx] |= REF_NUM(bank, code);              \
        } while (0)

#endif //_LED_TCA6507_H_
