/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2011-2012 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#ifndef __DRF_PARSER_H__
#define __DRF_PARSER_H__

#include <drf_types.h>

void drf_parse_replacement(const char *repl, drf_macro_type *macro_type,
        uint32_t *a, uint32_t *b);

#endif /* __DRF_PARSER_H__ */
