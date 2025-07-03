/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2012 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#include <stdint.h>
#include <stdio.h>
#include <errno.h>
#include <stdlib.h>

#include "drf_util.h"

int drf_parse_integer(const char *s, uint32_t *integer)
{
    char *endptr, *nptr = (char *)s;

    if (*nptr != '\0') {
        *integer = strtoul(nptr, &endptr, 0);
        if (*endptr == '\0')
            return 0;
    }

    return -EILWAL;
}

int drf_parse_range(const char *s, uint32_t *start_address,
        uint32_t *end_address)
{
    char *endptr, *nptr = (char *)s;

    if (*nptr != '\0') {
        *start_address = strtoul(nptr, &endptr, 0);
        if (*endptr == '-') {
            nptr = ++endptr;
            if (*nptr != '\0') {
                *end_address = strtoul(nptr, &endptr, 0);
                if (*endptr == '\0')
                    return 0;
            }
        }
    }

    return -EILWAL;
}
