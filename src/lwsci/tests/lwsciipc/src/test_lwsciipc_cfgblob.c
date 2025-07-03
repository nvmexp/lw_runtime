/*
 * Copyright (c) 2019-2020, LWPU Corporation.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <errno.h>
#include <sys/types.h>
#include <unistd.h>
#include <ctype.h>
#include <libgen.h>

#include <sys/neutrino.h>
#include <sys/procmgr.h>
#include <sys/syspage.h>
#include <sys/stat.h>
#include <inttypes.h>

#include <lwsciipc_internal.h>
#include <lwscierror.h>

typedef struct {
    char epname[LWSCIIPC_MAX_ENDPOINT_NAME];    /* endpoint name */
    uid_t uid;
    gid_t gid;
    char *prgname;
} test_params;

static test_params s_params;

void print_usage(char *argv[]);


void print_usage(char *argv[])
{
    fprintf(stderr, "Usage: %s [OPTION]...\n", argv[0]);
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "\t -h               : "
            "Print this help screen\n");
    fprintf(stderr, "\t -e <endpoint_name>: "
            "name of LwSciIpc endpoint\n");
    fprintf(stderr, "\t -u <uid:gid>     : "
            "UID and GID setting for test\n");
}

static void print_uid_gid(char *str)
{
    printf("[%s] UID: %d, GID: %d\n", str, getuid(), getgid());
}

static char * get_backend_str(uint32_t backend)
{
    switch(backend) {
        case LWSCIIPC_BACKEND_ITC:
            return "INTER-THREAD";
        case LWSCIIPC_BACKEND_IPC:
            return "INTER-PROCESS";
        case LWSCIIPC_BACKEND_IVC:
            return "INTER-VM";
        case LWSCIIPC_BACKEND_C2C:
            return "INTER-CHIP";
        default:
            return "UNKNOWN";
    }
}

static char * get_IPAtype_str(uint32_t ipa_type)
{
    switch(ipa_type) {
        case LWSCIIPC_TRAP_IPA:
            return "TRAP";
        case LWSCIIPC_MSI_IPA:
            return "MSI";
        default:
            return "UNKNOWN";
    }
}

int main(int argc, char *argv[])
{
    test_params *tp = &s_params;
    struct LwSciIpcEndpointAccessInfo info;
    int retval;
    LwSciError ret;
    int opt;

    printf("[%20s] enter LwSciIpc test\n", argv[0]);

    while ((opt = getopt(argc, argv, "e:hu:")) != -1)
    {
        switch (opt)
        {
            case 'e':
                strncpy(tp->epname, optarg, sizeof(tp->epname));
                break;
            case 'h':
                print_usage(argv);
                return 0;
            case 'u':
                /* Set UID:GID */
                {
                    char* token = strtok(optarg, ":");
                    if (token != NULL) {
                        tp->uid = (uid_t)strtoul(token, NULL, 10);
                        token = strtok(NULL, ":");
                        if (token != NULL) {
                            tp->gid = (gid_t)strtoul(token, NULL, 10);
                            retval = 0;
                        }
                    }
                    if (retval != 0) {
                        printf("%s: UID/GID not specified correctly\n", argv[0]);
                        return -EILWAL;
                    }
                    break;
                }
            default:
                abort();
        }
    }

    tp->prgname = basename(argv[0]);
    if (!tp->epname[0]) {
        fprintf(stderr, "need to give LwSciIpc endpoint name as input\n");
        print_usage(argv);
        return -1;
    }

    print_uid_gid(tp->prgname);

    ret = LwSciIpcOpenCfgBlob();
    switch(ret) {
        case LwSciError_AccessDenied :
            printf("Access denied to CfgBlob\n");
            goto fail;
        case LwSciError_Success :
            break;
        default :
            printf("LwSciIpcOpenCfgBlob failure: 0x%x\n", ret);
            goto fail;
    }

    ret = LwSciIpcGetEndpointAccessInfo(tp->epname, &info);
    switch(ret) {
        case LwSciError_NoSuchEntry :
            printf("Entry Not found: %s\n", tp->epname);
            goto fail;
        case LwSciError_Success :
            printf("------------------------------------------------------\n");
            printf("epname:%s, GID:%d, VUID:0x%lx\n",
                tp->epname, info.gid, info.vuid);
            printf("backend:%s(%d), id:%d\n",
                get_backend_str(info.backend), info.backend, info.id);
            printf("phyAddr:0x%lx, phySize:0x%lx, irq:%d\n",
                info.phyAddr, info.phySize, info.irq);
            printf("notiIPA:0x%lx, notiIPAsize:0x%lx, notiIPAtype:%s(%d)\n",
                info.notiIpa, info.notiIpaSize,
                get_IPAtype_str(info.notiIpaType),
                info.notiIpaType);
            printf("------------------------------------------------------\n");
            break;
        default :
            printf("LwSciIpcGetEndpointAccessInfo failure: 0x%x\n", ret);
            goto fail;
    }

    LwSciIpcCloseCfgBlob();

    printf("[%20s] : test PASSED\n", tp->prgname);

    return EXIT_SUCCESS;

fail:
    LwSciIpcCloseCfgBlob();
    printf("[%20s] : test FAILED\n", tp->prgname);

    return EXIT_FAILURE;
}

