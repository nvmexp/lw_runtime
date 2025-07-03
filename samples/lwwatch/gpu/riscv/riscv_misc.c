/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2017-2019 by LWPU Corporation.  All rights reserved.  All information
 * contained herein is proprietary and confidential to LWPU Corporation.  Any
 * use, reproduction, or disclosure without the written permission of LWPU
 * Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */
#include <ctype.h>
#include <string.h>
#include <stdio.h>
#include <lwtypes.h>
#include <print.h>

#include "riscv_prv.h"
#include "riscv_porting.h"

//returns 0 if EOF, updates str
static int skipWS(const char **str)
{
    const char *p;

    if (!str)
        return 0;
    if (!*str)
        return 0;

    p = *str;

    while (isspace((unsigned char)(*p)))
    {
        p++;
        if (!*p)
            goto out_nfound;
    }
    *str = p;
    return 1;
out_nfound:
    *str = p;
    return 0;
}

const char * riscvGetToken(const char *arg, const char **tok_start, int *tok_len)
{
    if (!tok_start || !arg || !tok_len)
        return NULL;

    *tok_len = 0;
    *tok_start = 0;

    if (!skipWS(&arg))
        return NULL;

    *tok_start = arg;
    while ( (!isspace((unsigned char)(*arg))) && (*arg != '\0'))
    {
        (*tok_len)++;
        arg++;
        if (!*arg)
            break;
    }

    // just to skip trailing ws
    skipWS(&arg);

    return arg;
}

void riscvDumpHex(const char *data, unsigned size, LwU64 offs)
{
    unsigned i, last_start=0;
    char line[256];

    for (i=0; i<size; ++i)
    {
        char *lp = line;
        line[0]=0;

        if (i % 16 == 0) {// newline
            lp += sprintf(lp, LwU64_FMT": ", i+offs);
            last_start = i;
        }
        lp += sprintf(lp, "%02x ", (unsigned)data[i] & 0xff);
        if (((i % 16) == 15) || ((i + 1) == size)) // end of line or data
        {
            unsigned j;
            lp += sprintf(lp, " ");
            for (j = last_start; j <= i; j+=4)
            {
                lp += sprintf(lp, "%08x ", ((const LwU32*)data)[j/4]);
            }
            lp += sprintf(lp, " ");
            for (j = last_start; j <= i; ++j)
            {
                if (isprint((unsigned char)data[j]))
                    lp += sprintf(lp, "%c", data[j]);
                else
                    lp += sprintf(lp, ".");
            }
            lp += sprintf(lp, "\n");
        }
        dprintf("%s", line);
    }
    if (size % 16 != 0)
        dprintf("\n");
}

void riscvDelay(unsigned msec)
{
    PLATFORM_DELAY_MS(msec);
}
