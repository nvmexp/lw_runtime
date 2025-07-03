/*
 * Copyright (c) 2019, LWPU CORPORATION. All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software and related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */


#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <errno.h>
#include <string.h>
#include <fcntl.h>
#include <devctl.h>
#include <libgen.h>
#include <sys/procmgr.h>
#include <sys/dispatch.h>
#include <sys/iofunc.h>
#include <sys/slog2.h>

#if defined(LW_IS_TRACER_ENABLED)
#include <tracer.h>
#endif

#include <lwqnx_common.h>
#include <lwsciipc_internal.h>
#include "test_lwsciipc_resmgr.h"

#define MAX_GID_NUM 255

typedef struct {
    char *prgname;
    uid_t uid;
    gid_t gid[MAX_GID_NUM]; /* gid[0]=GID, gid[1..255]=SGIDs */
    uint32_t num_gid;
} test_params;

volatile static int32_t s_Terminate = 0; /* Flag for any other registered signal */
static test_params s_params;

/* LwSciIpc I/O functions */
static int32_t lwsciipc_test_devctl(resmgr_context_t *ctp, io_devctl_t *msg,
    iofunc_ocb_t *ocb);
int32_t LwMapRmTest(resmgr_context_t *ctp, iofunc_ocb_t *ocb,
    LwSciIpcTestLwMap *msg);

int32_t LwMapRmTest(resmgr_context_t *ctp, iofunc_ocb_t *ocb,
    LwSciIpcTestLwMap *msg)
{
    test_params *tp = (test_params *)&s_params;
    LwSciIpcTopoId peerTopoId;
    LwSciIpcEndpointVuid peerVuId;
    LwSciIpcEndpointVuid vuid;
    uint32_t errcnt = 0;
    LwSciError ret;

    ret = LwSciIpcEndpointValidateAuthTokenQnx(ctp, msg->authToken, &vuid);
    if (ret != LwSciError_Success) {
        printf("LwSciIpcEndpointValidateAuthTokenQnx FAILED: %d(0x%x)\n", ret, ret);
        errcnt++;
    }
    else {
        printf("LwSciIpcEndpointValidateAuthTokenQnx PASSED\n");
    }
    printf("localUserVuid: 0x%" PRIx64 "\n", vuid);
    if (vuid != msg->vuid) {
        printf("VUID mismatch error: resmgr(0x%" PRIx64 "), client(0x%" PRIx64 ")\n",
            vuid, msg->vuid);
        errcnt++;
    }

    ret = LwSciIpcEndpointMapVuid(vuid, &peerTopoId, &peerVuId);
    if (ret != LwSciError_Success) {
        printf("LwSciIpcEndpointMapVuid FAILED: %d(0x%x)\n", ret, ret);
        errcnt++;
    }
    else {
        printf("LwSciIpcEndpointMapVuid PASSED\n");
    }
    printf("peerTopoId.SocId: 0x%x, peerTopoId.VmId: 0x%x\n", peerTopoId.SocId, peerTopoId.VmId);
    printf("peerVuId: 0x%" PRIx64 "\n", peerVuId);

    if (errcnt > 0) {
        printf("[%20s] : test FAILED\n", tp->prgname);
        return EPERM;
    }
    else {
        printf("[%20s] : test PASSED\n", tp->prgname);
        return EOK;
    }
}

static int32_t lwsciipc_test_devctl(resmgr_context_t *ctp, io_devctl_t *msg,
    iofunc_ocb_t *ocb)
{
    uint32_t nbytes = 0U;
    int32_t status;
    LwSciIpcTestLwMap *rxMsg;
    int32_t ret = -1;

    /* Verify the size encoded in the devctl call */
    if(lw_iofunc_devctl_verify(ctp, msg, ocb,
        _IO_DEVCTL_VERIFY_MSG_LEN | _IO_DEVCTL_VERIFY_LEN) != 0 ) {
       (void)printf("%s: Invalid size requested in devctl by %d\n",
            __func__, ctp->info.pid);
       return EILWAL;
    }

    if ((status = iofunc_devctl_default(ctp, msg, ocb)) !=
         _RESMGR_DEFAULT) {
        return(status);
    }

    rxMsg = _DEVCTL_DATA((msg->i));

    switch (msg->i.dcmd) {
        case DCMD_LWSCIIPC_TEST_LWMAP:
            {
                ret = LwMapRmTest(ctp, ocb, rxMsg);
                nbytes = 0U;
                if (ret != 0) {
                    goto fail;
                }
            }
            break;

        default:
            printf("%s: Not supported CMD: 0x%x", __func__,  msg->i.dcmd);
            ret = ENOTTY;
            goto fail;
    }

    if (memset_s(&msg->o, sizeof(msg->o), 0, sizeof(msg->o)) != EOK) {
        msg->o.nbytes = 0;
        msg->o.ret_val = (int32_t)EILWAL;
    } else {
        msg->o.nbytes = nbytes;
        msg->o.ret_val = (int32_t)ret;
    }

    return _RESMGR_PTR(ctp, &msg->o, sizeof(msg->o) + nbytes);

fail:
    return ret;
}

static void sig_handler(int32_t sig_num)
{
    s_Terminate = 1;
}

static void setup_termination_handlers(void)
{
    /*
     * Not returning an error if setting signal handler fails since it can only
     * affect graceful termination. Just printing an error message in that case.
     */
    if (signal(SIGINT, sig_handler) == SIG_ERR) {
        printf("Failed to set handler for SIGINT\n");
    }
    if (signal(SIGTERM, sig_handler) == SIG_ERR) {
        printf("Failed to set handler for SIGTERM\n");
    }
    if (signal(SIGHUP, sig_handler) == SIG_ERR) {
        printf("Failed to set handler for SIGHUP\n");
    }
    if (signal(SIGQUIT, sig_handler) == SIG_ERR) {
        printf("Failed to set handler for SIGQUIT\n");
    }
    if (signal(SIGABRT, sig_handler) == SIG_ERR) {
        printf("Failed to set handler for SIGABRT\n");
    }
}

static int32_t LwSciIpcServerRun(void)
{
    test_params *tp = (test_params *)&s_params;
    int32_t ret = -1;
    int32_t status;
    resmgr_attr_t resmgr_attr;
    dispatch_t *dpp = NULL;
    dispatch_context_t *ctp = NULL;
    /* test RM */
    resmgr_connect_funcs_t   test_connect_funcs;
    resmgr_io_funcs_t        test_io_funcs;
    iofunc_attr_t            test_iofunc_attr;


    /* put io-lwsciipc into the background */
    status = procmgr_daemon(0,
                            PROCMGR_DAEMON_NODEVNULL |
                            PROCMGR_DAEMON_NOCHDIR   |
                            PROCMGR_DAEMON_NOCLOSE);

    if (status == -1) {
        (void)printf("%s: Cannot put io-lwsciipc to background!", __func__);
        goto fail;
    }

    /* initialize dispatch interface */
    dpp = dispatch_create();
    if (dpp == NULL) {
        (void)printf("%s: Cannot intialize dispatch interface!", __func__);
        goto fail;
    }

    /* initialize resource manager attributes */
    if (memset_s(&resmgr_attr, sizeof(resmgr_attr), 0, sizeof(resmgr_attr)) != EOK) {
        goto fail;
    }

    /* lwsciipc_test node handler */
    iofunc_func_init(_RESMGR_CONNECT_NFUNCS, &test_connect_funcs,
            _RESMGR_IO_NFUNCS, &test_io_funcs);
    test_io_funcs.devctl = (void *)lwsciipc_test_devctl;

    if (memset_s(&test_iofunc_attr, sizeof(iofunc_attr_t), 0, sizeof(iofunc_attr_t)) != EOK) {
        goto fail;
    }

    iofunc_attr_init(&test_iofunc_attr,
        S_IFNAM | S_IRUSR | S_IWUSR |
        S_IRGRP | S_IWGRP, NULL, NULL);
    if (tp->uid != 0) {
        test_iofunc_attr.uid = tp->uid;
    }
    if (tp->gid[0] != 0) {
        test_iofunc_attr.gid = tp->gid[0];
    }

    status = resmgr_attach(dpp, &resmgr_attr, LWSCIIPC_LWMAPDEV,
            _FTYPE_ANY, 0, &test_connect_funcs,
            &test_io_funcs, &test_iofunc_attr);
    if (status == -1) {
        (void)printf("%s: resmgr_attach error lwsciipc_test: %d\n",
            __func__, errno);
        goto fail;
    }

    /* Register signal handlers for graceful termination */
    setup_termination_handlers();

    /* Drop all abilities given while launching */
    status = procmgr_ability(0,
        PROCMGR_ADN_NONROOT | PROCMGR_AOP_DENY | PROCMGR_AID_PATHSPACE,
        PROCMGR_ADN_NONROOT | PROCMGR_AOP_DENY | PROCMGR_AID_PUBLIC_CHANNEL,
        PROCMGR_ADN_NONROOT | PROCMGR_AOP_LOCK | PROCMGR_AID_EOL);
    if (status != EOK) {
        fprintf(stderr, "Failed to drop root privileges: %d\n", status);
        goto fail;
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

    /* allocate the context structure */
    ctp = dispatch_context_alloc(dpp);
    if (ctp == NULL) {
        (void) printf("%s: dispatch_context_alloc failed\n", __func__);
        goto fail;
    }

    /* start the resource manager message loop */
    while (s_Terminate == 0) {
        ctp = dispatch_block(ctp);
        if (ctp == NULL) {
            if (s_Terminate == 1) {
                break;
            } else {
                perror("LwSciIpcServerRun: dispatch_block error");
                ret = errno;
                goto fail;
            }
        }
        (void)dispatch_handler(ctp);
    }

    return 0;

fail:
    if (ctp != NULL) {
        /* Ignoring return type since it can be handled during de-init */
        (void)dispatch_context_free(ctp);
    }
    if (dpp != NULL) {
        /* Ignoring return type since it can be handled during de-init */
        (void)dispatch_destroy(dpp);
    }

    return ret;
}

static void print_usage(char *name)
{
    fprintf(stderr, "%s:\n", name);
    fprintf(stderr, " -u <uid:gid,sgid,...> : UID and GID of the driver.\n");
}

int main(int argc, char **argv)
{
    test_params *tp = (test_params *)&s_params;
    struct stat st;
    int32_t ret = -1;
    int32_t opt;
    LwSciError err;

    while ((opt = getopt(argc, argv, "hu:")) != -1)
    {
        switch (opt)
        {
            case 'u':
                /* Set UID:GID,SGIDs.. */
                {
                    char* token = strtok(optarg, ":");
                    if (token != NULL) {
                        tp->uid = (uid_t)strtoul(token, NULL, 10);
                        while ((token = strtok(NULL, ":,")) != NULL) {
                            tp->gid[tp->num_gid] = (gid_t)strtoul(token, NULL, 10);
                            tp->num_gid++;
                            ret = 0;
                        }
                    }
                    if (ret != 0) {
                        printf("%s: UID/GID not specified correctly\n", argv[0]);
                        return -EILWAL;
                    }
                    break;
                }
            case 'h':
                print_usage(argv[0]);
                return 0;
        }
    }

    tp->prgname = basename(argv[0]);

    /* TODO: check respawn */
    if (stat(LWSCIIPC_LWMAPDEV, &st) == 0) {
        (void)printf("resmgr already running!\n");
        ret = EXIT_FAILURE;
        goto fail;
    }

    err = LwSciIpcInit();
    if (err != LwSciError_Success) {
        printf("LwSciIpcInit is failed: %d\n", err);
        ret = EXIT_FAILURE;
        goto fail;
    }
    ret = LwSciIpcServerRun();
    LwSciIpcDeinit();

fail:
    return ret;
}

