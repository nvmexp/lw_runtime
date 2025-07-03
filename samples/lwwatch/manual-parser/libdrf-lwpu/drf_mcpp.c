/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2011-2012 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#include <stdio.h>
#include <stdarg.h>
#include <mcpp_lib.h>

#define elementsof(x) \
    (sizeof(x) / sizeof((x)[0]))

#include <drf_types.h>
#include "drf_mcpp.h"

static int drf_mcpp_fputc(int c, OUTDEST out_dest)
{
#ifdef DEBUG
    switch (out_dest) {
        case OUT:
        case DBG:
            c = fputc(c, stdout);
            break;
        default:
            c = fputc(c, stderr);
            break;
    }
#endif
    return 1;
}

static int drf_mcpp_fputs(const char *s, OUTDEST out_dest)
{
#ifdef DEBUG
    switch (out_dest) {
        case OUT:
        case DBG:
            fputs(s, stdout);
            break;
        default:
            fputs(s, stderr);
            break;
    }
#endif
    return 1;
}

static int drf_mcpp_fprintf(OUTDEST out_dest, const char *format, ...)
{
    int n = 0;
#ifdef DEBUG
    va_list ap;

    va_start(ap, format);
    switch (out_dest) {
        case OUT:
        case DBG:
            n = vfprintf(stdout, format, ap);
            break;
        default:
            n = vfprintf(stderr, format, ap);
            break;
    }
#endif
    return n;
}

void drf_mcpp_run_prologue(void)
{
    mcpp_set_out_func(drf_mcpp_fputc, drf_mcpp_fputs, drf_mcpp_fprintf);
    mcpp_set_persistence(1);
}

void drf_mcpp_run_epilogue(void)
{
    mcpp_set_in_mem_buffer(0, NULL, 0);
    mcpp_set_callbacks(0, NULL);
    mcpp_set_persistence(0);
    mcpp_reset_def_out_func();
}

void drf_mcpp_parse_mem_buffer(const char *buffer, unsigned int size,
        const CALLBACKS *callbacks)
{
    char *argv[] = { "mcpp", "-P", "-V", "199901L" };
    int argc = elementsof(argv);

    mcpp_set_in_mem_buffer(1, buffer, size);
    mcpp_set_callbacks(1, (CALLBACKS *)callbacks);
    mcpp_lib_main(argc, argv);
}

void drf_mcpp_parse_header_file(const char *path, unsigned int max_lines,
        const CALLBACKS *callbacks)
{
    char *argv[] = { "mcpp", "-P", "-V", "199901L", NULL, "-n", NULL };
    int argc = elementsof(argv);
    char n_lines[16];

    argv[argc-3] = (char *)path;
    if (max_lines > 0) {
        snprintf(n_lines, sizeof(n_lines), "%u", max_lines);
        argv[argc-1] = n_lines;
    } else
        argc -= 2;
    mcpp_set_callbacks(1, (CALLBACKS *)callbacks);
    mcpp_lib_main(argc, argv);
}
