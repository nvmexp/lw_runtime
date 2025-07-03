/*
 * Copyright (c) 2019-2021, LWPU CORPORATION. All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#include "lwscicommon_arch.h"

/* Returns the current system timer counter value using ARM virtual counter.
 * In case of Hypervisor, the offset of virtual counter to physical counter
 * is expected ot be 0.
 * This operation never fails.
 */
static inline uint64_t get_timer_counter(void)
{
    return 0;
}

/* Returns the system timer frequency in hertz.
 * This operation never fails.
 */
static inline uint64_t get_timer_freq(void)
{
    uint64_t value;

    value = 1U;
    return value;
}

uint64_t LwSciCommonGetTimeUS(void)
{
    const uint64_t usToSFactor = 1000000U;
    return (get_timer_counter() * usToSFactor) / get_timer_freq();
}

void LwSciCommonDMB(void)
{
    asm volatile("mfence": : :"memory");
}
