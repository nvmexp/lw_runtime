/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2011-2012 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#ifndef __DRF_MANUAL_H__
#define __DRF_MANUAL_H__

#include "drf.h"
#include "drf_register.h"

int drf_manual_lookup_by_regular_expression(drf_state_t *sate,
        const char *regex, drf_register_t ***regs, uint32_t *n_regs);
int drf_manual_lookup_by_name(drf_state_t *state,
        const char *name, drf_register_t ***regs, uint32_t *n_regs);
int drf_manual_lookup_by_address(drf_state_t *state,
        uint32_t address, drf_register_t ***regs, uint32_t *n_regs);
int drf_manual_lookup_by_address_range(drf_state_t *state,
        uint32_t start_address, uint32_t end_address, drf_register_t ***regs,
        uint32_t *n_regs);

#endif /* __DRF_MANUAL_H__ */
