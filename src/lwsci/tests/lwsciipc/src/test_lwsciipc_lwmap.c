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
#include <sys/stat.h>
#include <unistd.h>
#include <ctype.h>
#include <libgen.h>
#include <fcntl.h>

#ifdef __QNX__
#include <sys/neutrino.h>
#include <sys/procmgr.h>
#include <sys/syspage.h>
#include <sys/stat.h>
#include <inttypes.h>
#include <sys/slog2.h>
#if defined(LW_IS_TRACER_ENABLED)
#include <tracer.h>
#endif
#endif /* __QNX__ */

#ifdef LINUX
#include <sys/ioctl.h>
#include <stdint.h>
#include <signal.h>
#endif


#include <lwsciipc_internal.h>
#include <lwscierror.h>

#ifdef __QNX__
#include "test_lwsciipc_resmgr.h"
#include <search.h>
#endif /* __QNX__ */

#define MAX_GID_NUM 255

typedef struct {
    char epname[LWSCIIPC_MAX_ENDPOINT_NAME];    /* endpoint name */
    LwSciIpcEndpoint h;
    LwSciIpcEndpointVuid vuid;
    LwSciIpcEndpointAuthToken authToken;
    uint32_t range_lo;
    uint32_t range_hi;
    bool negative;
    uid_t uid;
    gid_t gid[MAX_GID_NUM]; /* gid[0]=GID, gid[1..255]=SGIDs */
    uint32_t num_gid;
    char *prgname;
    bool pause;
} test_params;

static uint32_t s_Stop;
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
    fprintf(stderr, "\t -a <lo:hi>       : "
            "set lo/hi limit on range of ability (hexadecimal)\n");
    fprintf(stderr, "\t -n               : "
            "negative test\n");
    fprintf(stderr, "\t -@               : "
            "no exit\n");
}

#ifdef __QNX__
/*
 * can be use for dropping root privilege
 * 0000:0000 : root
 * 1000:1000 : lwpu
 * 2000:2000 : lwsciipc
 */
static int32_t drop_privileges(void)
{
    test_params *tp = (test_params *)&s_params;
    int32_t abilityId;
    int32_t ret;

    if ((tp->gid[0] != 0) && (tp->uid != 0)) {
        abilityId = procmgr_ability_lookup("LwSciIpcEndpoint");
        /* MEM_PHYS and INTERRUPTEVENT are inter-VM specific */
        ret = procmgr_ability (0,
            PROCMGR_ADN_NONROOT | PROCMGR_AOP_ALLOW | PROCMGR_AID_MEM_PHYS,
            PROCMGR_ADN_NONROOT | PROCMGR_AOP_ALLOW | PROCMGR_AID_INTERRUPTEVENT,
            PROCMGR_ADN_NONROOT | PROCMGR_AOP_ALLOW | PROCMGR_AID_PUBLIC_CHANNEL,
            PROCMGR_ADN_NONROOT | PROCMGR_AOP_ALLOW | PROCMGR_AID_CONNECTION,
            PROCMGR_ADN_NONROOT | PROCMGR_AOP_ALLOW | PROCMGR_AOP_SUBRANGE | PROCMGR_AOP_LOCK | abilityId,
            (uint64_t)tp->range_lo, (uint64_t)tp->range_hi,
            PROCMGR_ADN_NONROOT | PROCMGR_AID_EOL);
        if (ret != EOK) {
            printf("%s: Unable to reserve procmgr abilities: %d\n",
                __func__, ret);
            return ret;
        }
    }

    /* set sub group ids */
    if (tp->num_gid > 1) {
        ret = setgroups(tp->num_gid-1, &tp->gid[1]);
        if (ret == -1) {
            return ret;
        }
    }

    /* if gid is not root */
    if (tp->gid[0] != 0) {
        ret = setregid(tp->gid[0], tp->gid[0]);
        if (ret != EOK) {
            printf("Failed to set GID to %d: %d\n", tp->gid[0], ret);
            return ret;
        }
    }

    /* if uid is not root */
    if (tp->uid != 0) {
        ret = setreuid(tp->uid, tp->uid);
        if (ret != EOK) {
            printf("Failed to set UID to %d: %d\n", tp->uid, ret);
            return ret;
        }
    }

    return EOK;
}

static void print_uid_gid(char *str)
{
    printf("[%s] UID: %d, GID: %d\n", str, getuid(), getgid());
}
#endif /* __QNX__ */

static void release_resources(void)
{
    test_params *tp = &s_params;

    printf("[%20s] closing LwSciIpc endpoint\n", tp->prgname);
    LwSciIpcCloseEndpoint(tp->h);

    LwSciIpcDeinit();

    if (s_Stop) {
        exit(1);
    }
}


static void sig_handler(int sig_num)
{
    s_Stop = 1;
    release_resources();
}

static void setup_termination_handlers(void)
{
    signal(SIGINT, sig_handler);
    signal(SIGTERM, sig_handler);
    signal(SIGHUP, sig_handler);
    signal(SIGQUIT, sig_handler);
    signal(SIGABRT, sig_handler);
}

int main(int argc, char *argv[])
{
    test_params *tp = &s_params;
    LwSciError ret;
    uint32_t errcnt = 0;
    int retval;
    int opt;
#ifdef __QNX__
    int32_t fd;
    LwSciIpcTestLwMap msg;
#endif /* __QNX__ */

    tp->negative = false;
    tp->pause = false;
    printf("[%20s] enter LwSciIpc test\n", argv[0]);

    while ((opt = getopt(argc, argv, "a:e:hnu:@")) != -1)
    {
        switch (opt)
        {
            case 'a':
                /* Set ability subrange */
                {
                    char* token = strtok(optarg, ":");
                    if (token != NULL) {
                        tp->range_lo = (int32_t)strtoul(token, NULL, 16);
                        token = strtok(NULL, ":");
                        if (token != NULL) {
                            tp->range_hi = (int32_t)strtoul(token, NULL, 16);
                            ret = 0;
                        }
                    }
                    if (ret != 0) {
                        printf("%s: range_lo/range_hi not specified correctly\n", argv[0]);
                        return -EILWAL;
                    }
                    break;
                }
            case 'e':
                strncpy(tp->epname, optarg, sizeof(tp->epname));
                break;
            case 'h':
                print_usage(argv);
                return 0;
            case 'n':
                tp->negative = true;
                break;
            case 'u':
                /* Set UID:GID */
                {
                    char* token = strtok(optarg, ":");
                    if (token != NULL) {
                        tp->uid = (uid_t)strtoul(token, NULL, 10);
                        while ((token = strtok(NULL, ":,")) != NULL) {
                            tp->gid[tp->num_gid] = (gid_t)strtoul(token, NULL, 10);
                            tp->num_gid++;
                            retval = 0;
                        }
                    }
                    if (retval != 0) {
                        printf("%s: UID/GID not specified correctly\n", argv[0]);
                        return -EILWAL;
                    }
                    break;
                }
            case '@':
                /* noexit */
                tp->pause = true;
                break;
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

    /* setup sig handler */
    setup_termination_handlers();

#if __QNX__
    ret = drop_privileges();
    if (ret != LwSciError_Success) {
        errcnt++;
    }
    print_uid_gid(tp->prgname);
#endif /* __QNX__ */

    ret = LwSciIpcInit();
    if (ret != LwSciError_Success) {
        printf("%s:LwSciIpcInit: fail (%d)\n", __func__, ret);
        errcnt++;
        goto fail;
    }

    ret = LwSciIpcOpenEndpoint(tp->epname, &tp->h);
    if (ret != LwSciError_Success) {
        printf("LwSciIpcOpenEndpoint failure: 0x%x\n", ret);
        errcnt++;
        goto fail;
    }

    ret = LwSciIpcEndpointGetVuid(tp->h, &tp->vuid);
    if (ret != LwSciError_Success) {
        printf("LwSciIpcEndpointGetVuid failure: 0x%x\n", ret);
        errcnt++;
        goto fail;
    }
#ifdef __QNX__
    printf("LwSciIpcEndpointGetVuid: 0x%" PRIx64 "\n", tp->vuid);
#endif /* __QNX__ */

#ifdef LINUX
    printf("LwSciIpcEndpointGetVuid: 0x%lx\n", tp->vuid);
#endif

    ret = LwSciIpcEndpointGetAuthToken(tp->h, &tp->authToken);
    if (ret != LwSciError_Success) {
        printf("LwSciIpcEndpointGetAuthToken failure: 0x%x\n", ret);
        errcnt++;
        goto fail;
    }
#ifdef __QNX__
    printf("LwSciIpcEndpointGetAuthToken: 0x%" PRIx64 "\n", tp->authToken);
#endif /* __QNX__ */
#ifdef LINUX
    printf("LwSciIpcEndpointGetAuthToken: 0x%lx\n", tp->authToken);
#endif

#ifdef __QNX__
    /* resource manager access */
    /* open test resmgr */
    fd = open(LWSCIIPC_LWMAPDEV, O_RDWR);
    if (fd == -1) {
        printf("%s open failure: (errno: %d)\n", LWSCIIPC_LWMAPDEV, errno);
        errcnt++;
    }

    if (tp->negative) {
        /* send incorrect VUID to resource manager */
        tp->authToken++;
        tp->vuid++;
        printf("NEGATIVE TEST: authtoken(0x%lx), vuid(0x%lx)\n",
            tp->authToken, tp->vuid);
    }
    msg.authToken = tp->authToken;
    msg.vuid = tp->vuid;
    /* call devctl */
    ret = devctl(fd, DCMD_LWSCIIPC_TEST_LWMAP, &msg, sizeof(msg), NULL);
    if (ret != 0) {
        printf("%s devctl failure: (errno: %d)\n", LWSCIIPC_LWMAPDEV, ret);
        errcnt++;
    }
    delay(500); /* wait 500msec */

    /* close test resmgr */
    (void)close(fd);
#endif /* __QNX__ */

fail:
    release_resources();

    if (errcnt > 0) {
        printf("[%20s] : test FAILED\n", tp->prgname);
        retval = EXIT_FAILURE;
    }
    else {
        printf("[%20s] : test PASSED\n", tp->prgname);
        retval = EXIT_SUCCESS;
    }

    while(tp->pause){
        pause();
    }

    return retval;
}

