/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2011-2012 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#ifndef __DRF_MCPP_H__
#define __DRF_MCPP_H__

#include <mcpp_lib.h>

void drf_mcpp_run_prologue(void);
void drf_mcpp_run_epilogue(void);

void drf_mcpp_parse_mem_buffer(const char *buffer, unsigned int size,
        const CALLBACKS *callbacks);
void drf_mcpp_parse_header_file(const char *path, unsigned int max_lines,
        const CALLBACKS *callbacks);

#endif /* __DRF_MCPP_H__ */
