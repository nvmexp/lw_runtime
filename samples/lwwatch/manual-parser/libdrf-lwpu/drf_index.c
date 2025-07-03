/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2011-2012 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#define _GNU_SOURCE
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#if LWOS_IS_UNIX
#include <libgen.h> // for basename(3)
#endif
#include <mcpp_lib.h>
#include "uthash.h"

#define HASH_ADD_STRPTR(head, str, el) \
    HASH_ADD_KEYPTR(hh, head, str, strlen(str), el)

#include "drf.h"
#include "drf_macro.h"
#include "drf_state.h"
#include "drf_parser.h"
#include "drf_mcpp.h"
#include "drf_index.h"

static __drf_state_t *state;

#define DRF_INDEX_SET_GLOBAL_STATE(__state) (state = (__state))
#define DRF_INDEX_RESET_GLOBAL_STATE() (state = NULL)

static const char *exclusions[] = {
    "LW_BRIDGE",
    "LW_CIO",
    "LW_CONFIG",
    "LW_CTXSW",
    "LW_DIO",
    "LW_EXPROM",
    "LW_FPCI",
    "LW_IGRAPH",
    "LW_IO",
    "LW_IOBAR",
    "LW_IOBARSPACE",
    "LW_IOCTRL",
    "LW_ISPACE",
    "LW_MEMORY",
    "LW_MSPACE",
    "LW_LWLIPT",
    "LW_PDFB",
    "LW_PFAULT",
    "LW_PFBIN",
    "LW_PFBM",
    "LW_PMD",
    "LW_PLWENC",
    "LW_PLWL",
    "LW_PLWM",
    "LW_PRAM",
    "LW_PRAMIN",
    "LW_PREMAP",
    "LW_PROM",
    "LW_RAMIN",
    "LW_RSPACE",
    "LW_SFBHUB",
    "LW_SPACE",
    "LW_SSE",
    "LW_UDMA",
    "LW_UREMAP",
    "LW_VIO"
};

#define EXCLUSIONS \
    (sizeof(exclusions) / sizeof(exclusions[0]))

static const char *inclusions[] = {
    "LW_AZA_PRI",
    "LW_IOCTRLMIF_RX",
    "LW_IOCTRLMIF_TX",
    "LW_LWLTLC_RX",
    "LW_PCHIPLET_PWR",
    "LW_PDISP_ISOHUB",
    "LW_PERF_PMASYS",
    "LW_PERF_PMMFBP",
    "LW_PERF_PMMFBPROUTER",
    "LW_PERF_PMMGPC",
    "LW_PERF_PMMGPCROUTER",
    "LW_PERF_PMMSYS",
    "LW_PERF_PMMSYSROUTER",
    "LW_PFB_FBPA",
    "LW_PFB_FBPA_0",
    "LW_PFB_FBPA_1",
    "LW_PFB_FBPA_2",
    "LW_PFB_FBPA_3",
    "LW_PFB_FBPA_4",
    "LW_PFB_FBPA_5",
    "LW_PFB_FBPA_6",
    "LW_PFB_FBPA_7",
    "LW_PFB_FBPA_8",
    "LW_PFB_FBPA_9",
    "LW_PFB_FBPA_A",
    "LW_PFB_FBPA_B",
    "LW_PFB_FBPA_C",
    "LW_PFB_FBPA_D",
    "LW_PFB_FBPA_E",
    "LW_PFB_FBPA_F",
    "LW_PFB_FBPA_MC0",
    "LW_PFB_FBPA_MC_0",
    "LW_PFB_FBPA_MC1",
    "LW_PFB_FBPA_MC_1",
    "LW_PFB_FBPA_MC2",
    "LW_PFB_FBPA_MC_2",
    "LW_PFB_HSHUB",
    "LW_PFB_HSMMU",
    "LW_PIOCTRLMIF_RX0",
    "LW_PIOCTRLMIF_RX1",
    "LW_PIOCTRLMIF_RX2",
    "LW_PIOCTRLMIF_RX3",
    "LW_PIOCTRLMIF_RX4",
    "LW_PIOCTRLMIF_RX5",
    "LW_PIOCTRLMIF_RX_MULTICAST",
    "LW_PIOCTRLMIF_SYS0",
    "LW_PIOCTRLMIF_SYS1",
    "LW_PIOCTRLMIF_SYS2",
    "LW_PIOCTRLMIF_SYS3",
    "LW_PIOCTRLMIF_SYS4",
    "LW_PIOCTRLMIF_SYS5",
    "LW_PIOCTRLMIF_SYS_MULTICAST",
    "LW_PIOCTRLMIF_TX0",
    "LW_PIOCTRLMIF_TX1",
    "LW_PIOCTRLMIF_TX2",
    "LW_PIOCTRLMIF_TX3",
    "LW_PIOCTRLMIF_TX4",
    "LW_PIOCTRLMIF_TX5",
    "LW_PIOCTRLMIF_TX_MULTICAST",
    "LW_PMGR_MUTEX",
    "LW_PMINION_FALCON",
    "LW_PMINION_SCP",
    "LW_PLWL0_BIST0",
    "LW_PLWL0_BIST1",
    "LW_PLWL0_BR0",
    "LW_PLWL0_SL0",
    "LW_PLWL0_SL1",
    "LW_PLWL1_BIST0",
    "LW_PLWL1_BIST1",
    "LW_PLWL1_BR0",
    "LW_PLWL1_SL0",
    "LW_PLWL1_SL1",
    "LW_PLWL2_BIST0",
    "LW_PLWL2_BIST1",
    "LW_PLWL2_BR0",
    "LW_PLWL2_SL0",
    "LW_PLWL2_SL1",
    "LW_PLWL3_BIST0",
    "LW_PLWL3_BIST1",
    "LW_PLWL3_BR0",
    "LW_PLWL3_SL0",
    "LW_PLWL3_SL1",
    "LW_PLWL4_BIST0",
    "LW_PLWL4_BIST1",
    "LW_PLWL4_BR0",
    "LW_PLWL4_SL0",
    "LW_PLWL4_SL1",
    "LW_PLWL5_BIST0",
    "LW_PLWL5_BIST1",
    "LW_PLWL5_BR0",
    "LW_PLWL5_SL0",
    "LW_PLWL5_SL1",
    "LW_PLWL_BIST0",
    "LW_PLWL_BIST1",
    "LW_PLWL_BR0",
    "LW_PLWL_MULTICAST",
    "LW_PLWL_MULTICAST_BIST0",
    "LW_PLWL_MULTICAST_BIST1",
    "LW_PLWL_MULTICAST_BR0",
    "LW_PLWL_MULTICAST_SL0",
    "LW_PLWL_MULTICAST_SL1",
    "LW_PLWL_SL0",
    "LW_PLWL_SL1",
    "LW_PLWLTLC_RX0",
    "LW_PLWLTLC_RX1",
    "LW_PLWLTLC_RX2",
    "LW_PLWLTLC_RX3",
    "LW_PLWLTLC_RX4",
    "LW_PLWLTLC_RX5",
    "LW_PLWLTLC_RX_MULTICAST",
    "LW_PLWLTLC_TX0",
    "LW_PLWLTLC_TX1",
    "LW_PLWLTLC_TX2",
    "LW_PLWLTLC_TX3",
    "LW_PLWLTLC_TX4",
    "LW_PLWLTLC_TX5",
    "LW_PLWLTLC_TX_MULTICAST",
    "LW_PPRIV_FBP",
    "LW_PPRIV_GPC",
    "LW_PPRIV_MASTER",
    "LW_PPRIV_SYS",
    "LW_PPRIV_SYSC",
    "LW_PTRIM_GPC",
    "LW_PTRIM_GPC_BC",
    "LW_XBAR_CXBAR_CQ_PRI_SYS0_HXI",
    "LW_XBAR_MXBAR_CS_PRI_SYS0_HXI",
    "LW_XBAR_MXBAR_CS_PRI_XSU",
    "LW_XBAR_MXBAR_PRI_GPC0_GNIC",
    "LW_XBAR_MXBAR_PRI_GPC1_GNIC",
    "LW_XBAR_MXBAR_PRI_GPC2_GNIC",
    "LW_XBAR_MXBAR_PRI_GPC3_GNIC",
    "LW_XBAR_MXBAR_PRI_GPC4_GNIC",
    "LW_XBAR_MXBAR_PRI_GPC5_GNIC",
    "LW_XBAR_WXBAR_CS_PRI_SYS0_HXI",
    "LW_XBAR_WXBAR_CS_PRI_XSU"
};

#define INCLUSIONS \
    (sizeof(inclusions) / sizeof(inclusions[0]))

static const char *overlays[][2] = {
    { "LW_PCFG",  "LW_XVE"  },
    { "LW_PCFG1", "LW_XVE1" }
};

#define OVERLAYS \
    (sizeof(overlays) / sizeof(overlays[0]))

static drf_macro_t *hash_table;

static int sort_by_address(drf_macro_t *a, drf_macro_t *b)
{
    if (a->overlay)
        a = a->overlay;
    if (b->overlay)
        b = b->overlay;
    if (a == b)
        return 0;
    else if (a->b < b->b) {
        if (state->debug && (a->a >= b->b)) {
            printf("%s (%08x:%08x) and %s (%08x:%08x) overlap.\n",
                a->name, a->a, a->b, b->name, b->a, b->b);
        }
        return -1;
    } else if (a->b > b->b) {
        if (state->debug && (b->a >= a->b)) {
            printf("%s (%08x:%08x) and %s (%08x:%08x) overlap.\n",
                a->name, a->b, a->a, b->name, b->b, b->a);
        }
        return 1;
    } else {
        if (state->debug) {
            printf("%s (%08x:%08x) and %s (%08x:%08x) overlap.\n",
                a->name, a->b, a->a, b->name, b->b, b->a);
            return ((a->a < b->a) ? 1 : -1);
        }
        return 0;
    }
}

static void macro_defined_callback(const char *name, short nargs,
        const char *parmnames, const char *repl,
        const char *fname, long mline)
{
    uint32_t i, a, b;
    drf_macro_type macro_type;
    char *file, *str;
    drf_macro_t *macro;

    if (strstr(name, "LW") != name)
        return;
    else if (strchr((name + 3), '_')) {
        for (i = 0; i < INCLUSIONS; i++) {
            if (!strcmp(name, inclusions[i]))
                break;
        }
        if (i == INCLUSIONS)
            return;
    } else {
        for (i = 0; i < EXCLUSIONS; i++) {
            if (!strcmp(name, exclusions[i]))
                return;
        }
    }
    if (nargs > 0) {
        /*
         *                     WARNING WARNING WARNING
         *
         * We use own parser (evaluate_macro) to figure out ranges for subsystem.
         * It is very primitive and can parse only simplest macros (range/const)
         * Some IP blocks (LWDEC) have manuals where multiple instances are
         * defined in single macro, such as:
            0x00833fff+(dev)*16384:0x00830000+(dev)*16384
         *
         * It is hard to patch it properly, therefore as a workaround we set
         * whole BAR0 range as valid range for that subsystem.
         *
         * It seems it has no visible drawbacks but my (mkulikowski) knowledge
         * of this code is *very* limited.
         *
         * For more details on why it was changed see http://lwbugs/2022308
         */
        if (state->debug)
            fprintf(stderr,
                    "drf_parser: Block %s has parametric range. Can't parse that, using whole PRIV range for it.\n",
                    name);
        macro_type = MACRO_TYPE_RANGE;
        a = 0x1000000;
        b = 0;
    } else {
        drf_parse_replacement(repl, &macro_type, &a, &b);
        if (macro_type != MACRO_TYPE_RANGE) {
            drf_set_errno(EILWAL);
            return;
        }
    }
    file = strdup(fname);
    if (!file) {
        if (state->debug)
            fprintf(stderr, "strdup() failed (%s)!\n", strerror(errno));
        drf_set_errno(errno);
        return;
    }
    str = strcasestr(file, "_addendum.h");
    if (str)
        strcpy(str, ".h");
    macro = calloc(1, (sizeof(*macro) + strlen(name)));
    if (!macro) {
        if (state->debug)
            fprintf(stderr, "calloc() failed (%s)!\n", strerror(errno));
        free(file);
        drf_set_errno(errno);
        return;
    }
    strcpy((char *)&macro->name[0], name);
    macro->macro_type = macro_type;
    macro->a = a;
    macro->b = b;
    macro->fname = file;
    HASH_ADD_STRPTR(hash_table, macro->name, macro);
}

static CALLBACKS macro_callbacks = {
    macro_defined_callback,
    NULL,
    NULL,
    NULL
};

static int drf_index_discover_devices(__drf_state_t *state)
{
    drf_device_t **devices = NULL;
    drf_device_t *device;
    drf_macro_t *macro, *overlay, *iterator;
    char *manual, *addendum, *str;
    uint32_t i, k, j = 0;

    drf_mcpp_run_prologue();
    if (state->manuals) {
        for (i = 0; state->manuals[i]; i++) {
            manual = state->manuals[i];
            addendum = malloc(strlen(manual) + 10);
            if (!addendum) {
                if (state->debug)
                    fprintf(stderr, "malloc() failed (%s)!\n", strerror(errno));
                drf_mcpp_run_epilogue();
                drf_set_errno(errno);
                return -1;
            }
            strcpy(addendum, manual);
            drf_mcpp_parse_header_file(state->manuals[i], 100, &macro_callbacks);
            str = strcasestr(addendum, ".h");
            if (str) {
                strcpy(str, "_addendum.h");
                str = basename(addendum);
                if (state->addendums) {
                    for (k = 0; state->addendums[k]; k++) {
                        if (strcasestr(state->addendums[k], str)) {
                            drf_mcpp_parse_header_file(state->addendums[k], 100,
                                    &macro_callbacks);
                        }
                    }
                }
            }
            free(addendum);
        }
    }
    if (state->mem_buffers) {
        for (i = 0; state->mem_buffers[i].data; i++) {
            drf_mcpp_parse_mem_buffer(state->mem_buffers[i].data,
                    state->mem_buffers[i].data_size,
                    &macro_callbacks);
        }
    }
    drf_mcpp_run_epilogue();
    if (drf_errno)
        goto failed;
    HASH_ITER(hh, hash_table, macro, iterator) {
        for (i = 0; i < OVERLAYS; i++) {
            if (strcmp(macro->name, overlays[i][0]))
                continue;
            HASH_FIND_STR(hash_table, overlays[i][1], overlay);
            if (!overlay)
                continue;
            HASH_DEL(hash_table, macro);
            overlay->overlay = macro;
            break;
        }
    }
    HASH_SORT(hash_table, sort_by_address);
    state->n_devices = HASH_COUNT(hash_table);
    devices = calloc(1, (sizeof(*device) * state->n_devices));
    if (!devices) {
        if (state->debug)
            fprintf(stderr, "calloc() failed (%s)!\n", strerror(errno));
        drf_set_errno(errno);
        goto failed;
    }
    state->devices = devices;
    HASH_ITER(hh, hash_table, macro, iterator) {
        device = calloc(1, (sizeof(*device) + strlen(macro->name)));
        if (!device) {
            if (state->debug)
                fprintf(stderr, "calloc() failed (%s)!\n", strerror(errno));
            drf_set_errno(errno);
            goto failed;
        }
        HASH_DEL(hash_table, macro);
        devices[j++] = device;
        strcpy((char *)&device->name[0], macro->name);
        device->fname = macro->fname;
        if (!macro->overlay) {
            device->base = macro->b;
            device->extent = macro->a;
        } else {
            device->base = macro->overlay->b;
            device->extent = macro->overlay->a;
        }
        device->initial_base = macro->b;
        device->initial_extent = macro->a;
    }
    return 0;
failed:
    HASH_ITER(hh, hash_table, macro, iterator) {
        HASH_DEL(hash_table, macro);
        free(macro);
    }
    return -1;
}

int drf_index_get_devices(drf_state_t *__state, drf_device_t ***devices,
        uint32_t *n_devices)
{
    int ret = 0;

    DRF_INDEX_SET_GLOBAL_STATE(__state);
    do {
        __drf_state_t *state = __state;
        if (!state->devices) {
            ret = drf_index_discover_devices(state);
            if (ret < 0)
                goto failed;
        }
        *devices = state->devices;
        *n_devices = state->n_devices;
    } while (0);
failed:
    DRF_INDEX_RESET_GLOBAL_STATE();
    return ret;
}
