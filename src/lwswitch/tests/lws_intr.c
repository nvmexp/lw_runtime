/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2017 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

/*
 * lws_intr - lwswitch interrupt testtest.
 *
 * This test injects interrupts into a lwswitch via the ioctl register write
 * interface.  It uses the _INJECT registers.
 *
 * This causes the interrupt serve routines to be exercised.  Right the debug
 * spew must be read by a human to see if things look reasonable.  There should
 * also be no hangs or stuck interrupts.
 *
 * The next generation of the test plans to read back information from the error
 * logging system to verify interrupt delivery programatically.
 *
 * Usage: lws_intr [--test X] [--switch N] [--rtl|--ortl] [--skinny]
 *
 * With no args all tests are run in order on /dev/lwswitch0.
 *  - Other switch devices can be specified like '--switch 2' to correspond to /dev/lwswitch2.
 *  - A single test may be selected with --test X.
 *  - --rtl and --otl tune the test to the reduced configs on RTL sim.
 *  - --skinny attempts to test just the first instance instead of all
 *
 * This test causes many interrupts, and exercises the interrupt handlers.  Many of the errors
 * are fatal, and even though the errors don't actually occur, the driver thinks they did. This
 * will require the fabric to restart after the test completes.
 *
 * It should be run w/o FM running and no traffic flowing. Running with FM should work today, but
 * as the test improves to use the error reporting interfaces, it won't be able to at the same
 * time as FM.
 *
 * HW BUG 1848387 Egress sending poisoned packed - This bug states EGRESS can taint traffic
 * while using _INJECT.  Since we don't expect to do this, SW can ignore the issue.
 *
 */
#include <stdio.h>
#include <fcntl.h>
#include <stdlib.h>
#include <errno.h>
#include <getopt.h>
#include <sys/ioctl.h>

#include "lwtypes.h"
#include "lwlink.h"
#include "ctrl/ctrl2080/ctrl2080lwlink.h"
#include "ctrl_dev_lwswitch.h"
#include "ioctl_lwswitch.h"
#include "ioctl_dev_lwswitch.h"

#include "lwswitch/svnp01/dev_lws.h"
#include "lwswitch/svnp01/dev_lws_master.h"
#include "lwswitch/svnp01/dev_lwltlc_ip.h"
#include "lwswitch/svnp01/dev_route_ip.h"
#include "lwswitch/svnp01/dev_ingress_ip.h"
#include "lwswitch/svnp01/dev_ftstate_ip.h"
#include "lwswitch/svnp01/dev_egress_ip.h"
#include "lwswitch/svnp01/dev_lwlctrl_ip.h"
#include "lwswitch/svnp01/dev_lwl_ip.h"

static const struct option ioctl_opts[] =
{
    { "test", required_argument, NULL, 't' },
    { "switch", required_argument, NULL, 's' },
    { "rtl", 0, NULL, 'r' },        // RTL sim
    { "ortl", 0, NULL, 'o' },       // Old RTL sim with more of the chip
    { "skinny", 0, NULL, 'S' },     // only test unit 0 for speed
    { 0, 0, 0, 0 }
};

#define MAX_LINKS 18

// Some blocks are logical link based, some are physical
#define FIRST_PHYS_LINK()   (rtlsim ? 16 : 0)
#define FIRST_LOG_LINK()    0
#define NUM_LINKS()         (rtlsim ? 2 : (skinny ? 1 : MAX_LINKS))
#define FIRST_IOCTRL()      (rtlsim ? 8 : 0)
#define NUM_IOCTRL()        ((rtlsim || skinny) ? 1 : (MAX_LINKS/2))

//
// register write wrapper to ioctl
//
int reg_wr(int fd, LwU32 engine, LwU32 instance, LwU32 offset, LwU32 val)
{
    LWSWITCH_REGISTER_WRITE wr;

    wr.engine = engine;
    wr.instance = instance;
    wr.offset = offset;
    wr.bcast = 0;
    wr.val = val;

    return ioctl(fd, IOCTL_LWSWITCH_REGISTER_WRITE, &wr);
}

int reg_rd(int fd, LwU32 engine, LwU32 instance, LwU32 offset, LwU32 *val)
{
    LWSWITCH_REGISTER_READ rd;
    int rc;

    rd.engine = engine;
    rd.instance = instance;
    rd.offset = offset;
    rd.val = 0;

    rc = ioctl(fd, IOCTL_LWSWITCH_REGISTER_READ, &rd);
    *val = rd.val;

    return rc;
}

void wait_for_service(int fd)
{
    LwU32 status = 1;
    LwU32 val;

    while (status)
    {
        status = 0;

        sleep(1);

        reg_rd(fd, REGISTER_RW_ENGINE_RAW, 0, LW_PSMC_INTR_LEGACY, &val);
        status |= val;
        reg_rd(fd, REGISTER_RW_ENGINE_RAW, 0, LW_PSMC_INTR_FATAL, &val);
        status |= val;
        reg_rd(fd, REGISTER_RW_ENGINE_RAW, 0, LW_PSMC_INTR_CORRECTABLE, &val);
        status |= val;
    }
}

main(int argc, char **argv)
{
    char file[64];
    int fd, rc;
    int tests = 0;
    int run = 0;
    int errors = 0;
    int opt, long_idx;
    int intrs = 0;

#define ALL_TESTS -1
    int test = ALL_TESTS;
    int minor = 0;
#define RTL_SIM         1
#define OLD_RTL_SIM     2
    int rtlsim = 0;
    int skinny = 0;

    while (1)
    {
        opt = getopt_long(argc, argv, "t:s:or", ioctl_opts, &long_idx);
        if (opt == -1)
            break;
        switch (opt)
        {
            case 't':
                test  = atoi(optarg);
                break;
            case 's':
                minor  = atoi(optarg);
                break;
            case 'S':
                skinny = 1;
                break;
            case 'r':
                rtlsim = RTL_SIM;
                break;
            case 'o':
                rtlsim = OLD_RTL_SIM;
                break;
            default:
                printf("usage: lws_ioctl [--switch n] [--test n] [--rtl|--ortl]\n");
                exit(1);
        }
    }

    sprintf(file, "/dev/lwswitch%d", minor);

    fd = open(file, O_RDONLY);
    if (fd < 0)
    {
        printf("open of %s failed!\n", file);
        exit(1);
    }

#define RUN_TEST    \
    (((++tests == test) || (test == ALL_TESTS)) && ++run)

#define IOCTL(fd, cmd, params, rc) \
    printf("%d: ioctl: " #cmd "\n", tests); \
    rc = ioctl(fd, cmd, params)

#define CHECKV(rc, str, ...) \
    if (!(rc)) \
    { \
        fflush(stdout); \
        perror("    *** failed!"); \
        errors++; \
    }
#define CHECK(rc)   CHECKV(rc, "%s", "")

    if (RUN_TEST)
    {
        LwU32 val;

        printf("PBUS and PRI:\n");

        printf("  SW interrupt\n");
        rc = reg_wr(fd, REGISTER_RW_ENGINE_RAW, 0, LW_PBUS_SW_INTR_0,
                DRF_DEF(_PBUS, _SW_INTR_0, _SET, _PENDING));
        CHECK(rc == 0);
        intrs++;

        printf("  PRI write timeout\n");
        rc = reg_wr(fd, REGISTER_RW_ENGINE_RAW, 0, 0x1050, 0xff);
        CHECK(rc == 0);
        intrs++;

        wait_for_service(fd);

        printf("  PRI read timeout\n");
        rc = reg_rd(fd, REGISTER_RW_ENGINE_RAW, 0, 0x1050, &val);
        CHECK(rc == 0);
        intrs++;

        wait_for_service(fd);

#if 0 // TODO old location that worked on RTL sim no longer works for this error
        printf("  PRI FECSERR read timeout\n");
        rc = reg_rd(fd, REGISTER_RW_ENGINE_NPG, 0, 0x534, &val);
        CHECK(rc == 0);
        intrs++;

        wait_for_service(fd);
#endif

        printf("  PRI ring write error\n");
        rc = reg_wr(fd, REGISTER_RW_ENGINE_NPG, 0, 0x534, 1);
        CHECK(rc == 0);
        intrs++;

        wait_for_service(fd);
    }

#define NPG_TEST(name, num, reg, bits)

    if (RUN_TEST)
    {
        int start = FIRST_IOCTRL();
        int end = start + NUM_IOCTRL();
        int i;

        printf("ROUTE:\n");

        for (i = start; i < end; i++)
        {
            printf("  NPORT%d route - inject 5 interrupts\n", i);
            rc = reg_wr(fd, REGISTER_RW_ENGINE_NPORT, i,
                    LW_ROUTE_ERR_INJECT_0,
                    DRF_DEF(_ROUTE, _ERR_INJECT_0, _ROUTEBUFERR, _DETECTED) |
                    DRF_DEF(_ROUTE, _ERR_INJECT_0, _NOPORTDEFINEDERR, _DETECTED) |
                    DRF_DEF(_ROUTE, _ERR_INJECT_0, _ILWALIDROUTEPOLICYERR, _DETECTED) |
                    DRF_DEF(_ROUTE, _ERR_INJECT_0, _ECCLIMITERR, _DETECTED) |
                    DRF_DEF(_ROUTE, _ERR_INJECT_0, _TRANSDONERESVERR, _DETECTED));
            CHECK(rc == 0);
            intrs += 5;

            wait_for_service(fd);
        }
    }

    if (RUN_TEST)
    {
        int start = FIRST_LOG_LINK();
        int end = start + NUM_LINKS();
        int rc;
        int i;

        printf("INGRESS:\n");

        for (i = start; i < end; i++)
        {
            printf("  NPORT%d INGRESS - inject 9 interrupts\n", i);
            rc = reg_wr(fd, REGISTER_RW_ENGINE_NPORT, i, LW_INGRESS_ERR_INJECT_0,
                    DRF_DEF(_INGRESS, _ERR_INJECT_0, _CMDDECODEERR, _INSERT) |
                    DRF_DEF(_INGRESS, _ERR_INJECT_0, _BDFMISMATCHERR, _INSERT) |
                    DRF_DEF(_INGRESS, _ERR_INJECT_0, _BUBBLEDETECT, _INSERT) |
                    DRF_DEF(_INGRESS, _ERR_INJECT_0, _ACLFAIL, _INSERT) |
                    DRF_DEF(_INGRESS, _ERR_INJECT_0, _PKTPOISONSET, _INSERT) |
                    DRF_DEF(_INGRESS, _ERR_INJECT_0, _ECCSOFTLIMITERR, _INSERT) |
                    DRF_DEF(_INGRESS, _ERR_INJECT_0, _ECCHDRDOUBLEBITERR, _INSERT) |
                    DRF_DEF(_INGRESS, _ERR_INJECT_0, _ILWALIDCMD, _INSERT) |
                    DRF_DEF(_INGRESS, _ERR_INJECT_0, _ILWALIDVCSET, _INSERT));
            CHECK(rc == 0);
            intrs += 9;

            wait_for_service(fd);
        }
    }

    if (RUN_TEST)
    {
        int start = FIRST_LOG_LINK();
        int end = start + NUM_LINKS();
        int rc;
        int i;

        printf("FLUSHSTATE:\n");

        for (i = start; i < end; i++)
        {
            printf("  NPORT%d FS - inject 8 interrupts\n", i);
            rc = reg_wr(fd, REGISTER_RW_ENGINE_NPORT, i, LW_FSTATE_ERR_INJECT_0,
                    DRF_NUM(_FSTATE, _ERR_INJECT_0, _TAGPOOLBUFERR, 1) |
                    DRF_NUM(_FSTATE, _ERR_INJECT_0, _CRUMBSTOREBUFERR, 1) |
                    DRF_NUM(_FSTATE, _ERR_INJECT_0, _SINGLEBITECCLIMITERR_CRUMBSTORE, 1) |
                    DRF_NUM(_FSTATE, _ERR_INJECT_0, _UNCORRECTABLEECCERR_CRUMBSTORE, 1) |
                    DRF_NUM(_FSTATE, _ERR_INJECT_0, _SINGLEBITECCLIMITERR_TAGSTORE, 1) |
                    DRF_NUM(_FSTATE, _ERR_INJECT_0, _UNCORRECTABLEECCERR_TAGSTORE, 1) |
                    DRF_NUM(_FSTATE, _ERR_INJECT_0, _SINGLEBITECCLIMITERR_FLUSHREQSTORE, 1) |
                    DRF_NUM(_FSTATE, _ERR_INJECT_0, _UNCORRECTABLEECCERR_FLUSHREQSTORE, 1));
            CHECK(rc == 0);
            intrs += 8;

            wait_for_service(fd);
        }
    }

    if (RUN_TEST)
    {
        int start = FIRST_LOG_LINK();
        int end = start + NUM_LINKS();
        int rc;
        int i;

        printf("TAGSTATE:\n");

        for (i = start; i < end; i++)
        {
            printf("  NPORT%d TS - inject 6 interrupts\n", i);
            rc = reg_wr(fd, REGISTER_RW_ENGINE_NPORT, i, LW_TSTATE_ERR_INJECT_0,
                    DRF_NUM(_TSTATE, _ERR_INJECT_0, _TAGPOOLBUFERR, 1) |
                    DRF_NUM(_TSTATE, _ERR_INJECT_0, _CRUMBSTOREBUFERR, 1) |
                    DRF_NUM(_TSTATE, _ERR_INJECT_0, _SINGLEBITECCLIMITERR_CRUMBSTORE, 1) |
                    DRF_NUM(_TSTATE, _ERR_INJECT_0, _UNCORRECTABLEECCERR_CRUMBSTORE, 1) |
                    DRF_NUM(_TSTATE, _ERR_INJECT_0, _SINGLEBITECCLIMITERR_TAGSTORE, 1) |
                    DRF_NUM(_TSTATE, _ERR_INJECT_0, _UNCORRECTABLEECCERR_TAGSTORE, 1));
            CHECK(rc == 0);
            intrs += 6;

            wait_for_service(fd);
        }
    }

    if (RUN_TEST)
    {
        int start = FIRST_LOG_LINK();
        int end = start + NUM_LINKS();
        int rc;
        int i;

        printf("EGRESS:\n");

        // See dislwssion of BUG 1848387 above.
        for (i = start; i < end; i++)
        {
            printf("  NPORT%d egress - inject 15 interrupts\n", i);
            rc = reg_wr(fd, REGISTER_RW_ENGINE_NPORT, i, LW_EGRESS_ERR_INJECT_0,
                    DRF_NUM(_EGRESS, _ERR_INJECT_0, _EGRESSBUFERR, 1) |
                    DRF_NUM(_EGRESS, _ERR_INJECT_0, _PKTROUTEERR, 1) |
                    DRF_NUM(_EGRESS, _ERR_INJECT_0, _ECCSINGLEBITLIMITERR0, 1) |
                    DRF_NUM(_EGRESS, _ERR_INJECT_0, _ECCHDRDOUBLEBITERR0, 1) |
                    DRF_NUM(_EGRESS, _ERR_INJECT_0, _ECCDATADOUBLEBITERR0, 1) |
                    DRF_NUM(_EGRESS, _ERR_INJECT_0, _ECCSINGLEBITLIMITERR1, 1) |
                    DRF_NUM(_EGRESS, _ERR_INJECT_0, _ECCHDRDOUBLEBITERR1, 1) |
                    DRF_NUM(_EGRESS, _ERR_INJECT_0, _ECCDATADOUBLEBITERR1, 1) |
                    DRF_NUM(_EGRESS, _ERR_INJECT_0, _NCISOCHDRCREDITOVFL, 1) |
                    DRF_NUM(_EGRESS, _ERR_INJECT_0, _NCISOCDATACREDITOVFL, 1) |
                    DRF_NUM(_EGRESS, _ERR_INJECT_0, _ADDRMATCHERR, 1) |
                    DRF_NUM(_EGRESS, _ERR_INJECT_0, _TAGCOUNTERR, 1) |
                    DRF_NUM(_EGRESS, _ERR_INJECT_0, _FLUSHRSPERR, 1) |
                    DRF_NUM(_EGRESS, _ERR_INJECT_0, _DROPNPURRSPERR, 1) |
                    DRF_NUM(_EGRESS, _ERR_INJECT_0, _POISONERR,1));
            CHECK(rc == 0);
            intrs += 15;

            wait_for_service(fd);
        }
    }

    if (RUN_TEST)
    {
        int start = FIRST_IOCTRL();
        int end = start + NUM_IOCTRL();
        int rc;
        int i;

        printf("CLKCROSS:\n");

        for (i = start; i < end; i++)
        {
            printf("  SIOCTRL%d clkcross0 - inject 8 interrupts\n", i);
            rc = reg_wr(fd, REGISTER_RW_ENGINE_SIOCTRL, i,
                    LW_LWLCTRL_CLKCROSS_0_ERR_INJECT_0,
                    DRF_NUM(_LWLCTRL, _CLKCROSS_0_ERR_INJECT_0, _INGRESSECCSOFTLIMITERR, 1) |
                    DRF_NUM(_LWLCTRL, _CLKCROSS_0_ERR_INJECT_0, _INGRESSECCHDRDOUBLEBITERR, 1) |
                    DRF_NUM(_LWLCTRL, _CLKCROSS_0_ERR_INJECT_0, _INGRESSECCDATADOUBLEBITERR, 1) |
                    DRF_NUM(_LWLCTRL, _CLKCROSS_0_ERR_INJECT_0, _INGRESSBUFFERERR, 1) |
                    DRF_NUM(_LWLCTRL, _CLKCROSS_0_ERR_INJECT_0, _EGRESSECCSOFTLIMITERR, 1) |
                    DRF_NUM(_LWLCTRL, _CLKCROSS_0_ERR_INJECT_0, _EGRESSECCHDRDOUBLEBITERR, 1) |
                    DRF_NUM(_LWLCTRL, _CLKCROSS_0_ERR_INJECT_0, _EGRESSECCDATADOUBLEBITERR, 1) |
                    DRF_NUM(_LWLCTRL, _CLKCROSS_0_ERR_INJECT_0, _EGRESSBUFFERERR, 1));
            CHECK(rc == 0);
            intrs += 8;

            wait_for_service(fd);

            printf("  SIOCTRL%d clkcross1 - inject 8 interrupts\n", i);
            rc = reg_wr(fd, REGISTER_RW_ENGINE_SIOCTRL, i,
                    LW_LWLCTRL_CLKCROSS_1_ERR_INJECT_0,
                    DRF_NUM(_LWLCTRL, _CLKCROSS_1_ERR_INJECT_0, _INGRESSECCSOFTLIMITERR, 1) |
                    DRF_NUM(_LWLCTRL, _CLKCROSS_1_ERR_INJECT_0, _INGRESSECCHDRDOUBLEBITERR, 1) |
                    DRF_NUM(_LWLCTRL, _CLKCROSS_1_ERR_INJECT_0, _INGRESSECCDATADOUBLEBITERR, 1) |
                    DRF_NUM(_LWLCTRL, _CLKCROSS_1_ERR_INJECT_0, _INGRESSBUFFERERR, 1) |
                    DRF_NUM(_LWLCTRL, _CLKCROSS_1_ERR_INJECT_0, _EGRESSECCSOFTLIMITERR, 1) |
                    DRF_NUM(_LWLCTRL, _CLKCROSS_1_ERR_INJECT_0, _EGRESSECCHDRDOUBLEBITERR, 1) |
                    DRF_NUM(_LWLCTRL, _CLKCROSS_1_ERR_INJECT_0, _EGRESSECCDATADOUBLEBITERR, 1) |
                    DRF_NUM(_LWLCTRL, _CLKCROSS_1_ERR_INJECT_0, _EGRESSBUFFERERR, 1));
            CHECK(rc == 0);
            intrs += 8;

            wait_for_service(fd);
        }
    }

    if (RUN_TEST)
    {
        int start = FIRST_PHYS_LINK();
        int end = start + NUM_LINKS();
        int rc;
        int i;

        printf("LWLTLC:\n");

        for (i = start; i < end; i++)
        {
            printf("  LWLTLC%d TX lwltlc - inject 25 interrupts\n", i);
            rc = reg_wr(fd, REGISTER_RW_ENGINE_LWLTLC, i,
                    LW_LWLTLC_TX_ERR_INJECT_0,
                    DRF_NUM(_LWLTLC_TX, _ERR_INJECT_0, _TXHDRCREDITOVFERR, 1) |
                    DRF_NUM(_LWLTLC_TX, _ERR_INJECT_0, _TXDATACREDITOVFERR, 1) |
                    DRF_NUM(_LWLTLC_TX, _ERR_INJECT_0, _TXDLCREDITOVFERR, 1) |
                    DRF_NUM(_LWLTLC_TX, _ERR_INJECT_0, _TXDLCREDITPARITYERR, 1) |
                    DRF_NUM(_LWLTLC_TX, _ERR_INJECT_0, _TXRAMHDRPARITYERR, 1) |
                    DRF_NUM(_LWLTLC_TX, _ERR_INJECT_0, _TXRAMDATAPARITYERR, 1) |
                    DRF_NUM(_LWLTLC_TX, _ERR_INJECT_0, _TXUNSUPVCOVFERR, 1) |
                    DRF_NUM(_LWLTLC_TX, _ERR_INJECT_0, _TXSTOMPDET, 1) |
                    DRF_NUM(_LWLTLC_TX, _ERR_INJECT_0, _TXPOISONDET, 1) |
                    DRF_NUM(_LWLTLC_TX, _ERR_INJECT_0, _TARGETERR, 1) |
                    DRF_NUM(_LWLTLC_TX, _ERR_INJECT_0, _UNSUPPORTEDREQUESTERR, 1));
            CHECK(rc == 0);
            intrs += 25;

            wait_for_service(fd);

            printf("  LWLTLC%d RX0 lwltlc - inject 24 interrupts\n", i);
            rc = reg_wr(fd, REGISTER_RW_ENGINE_LWLTLC, i,
                    LW_LWLTLC_RX_ERR_INJECT_0,
                    DRF_NUM(_LWLTLC_RX, _ERR_INJECT_0, _RXDLHDRPARITYERR, 1) |
                    DRF_NUM(_LWLTLC_RX, _ERR_INJECT_0, _RXDLDATAPARITYERR , 1) |
                    DRF_NUM(_LWLTLC_RX, _ERR_INJECT_0, _RXDLCTRLPARITYERR, 1) |
                    DRF_NUM(_LWLTLC_RX, _ERR_INJECT_0, _RXRAMDATAPARITYERR, 1) |
                    DRF_NUM(_LWLTLC_RX, _ERR_INJECT_0, _RXRAMHDRPARITYERR, 1) |
                    DRF_NUM(_LWLTLC_RX, _ERR_INJECT_0, _RXILWALIDAEERR, 1) |
                    DRF_NUM(_LWLTLC_RX, _ERR_INJECT_0, _RXILWALIDBEERR, 1) |
                    DRF_NUM(_LWLTLC_RX, _ERR_INJECT_0, _RXILWALIDADDRALIGNERR, 1) |
                    DRF_NUM(_LWLTLC_RX, _ERR_INJECT_0, _RXPKTLENERR, 1) |
                    DRF_NUM(_LWLTLC_RX, _ERR_INJECT_0, _RSVCMDENCERR, 1) |
                    DRF_NUM(_LWLTLC_RX, _ERR_INJECT_0, _RSVDATLENENCERR, 1) |
                    DRF_NUM(_LWLTLC_RX, _ERR_INJECT_0, _RSVADDRTYPEERR, 1) |
                    DRF_NUM(_LWLTLC_RX, _ERR_INJECT_0, _RSVRSPSTATUSERR, 1) |
                    DRF_NUM(_LWLTLC_RX, _ERR_INJECT_0, _RSVPKTSTATUSERR, 1) |
                    DRF_NUM(_LWLTLC_RX, _ERR_INJECT_0, _RSVCACHEATTRPROBEREQERR, 1) |
                    DRF_NUM(_LWLTLC_RX, _ERR_INJECT_0, _RSVCACHEATTRPROBERSPERR, 1) |
                    DRF_NUM(_LWLTLC_RX, _ERR_INJECT_0, _DATLENGTATOMICREQMAXERR, 1) |
                    DRF_NUM(_LWLTLC_RX, _ERR_INJECT_0, _DATLENGTRMWREQMAXERR, 1) |
                    DRF_NUM(_LWLTLC_RX, _ERR_INJECT_0, _DATLENLTATRRSPMINERR, 1) |
                    DRF_NUM(_LWLTLC_RX, _ERR_INJECT_0, _ILWALIDCACHEATTRPOERR, 1) |
                    DRF_NUM(_LWLTLC_RX, _ERR_INJECT_0, _ILWALIDCRERR, 1) |
                    DRF_NUM(_LWLTLC_RX, _ERR_INJECT_0, _RXRESPSTATUSTARGETERR, 1) |
                    DRF_NUM(_LWLTLC_RX, _ERR_INJECT_0, _RXRESPSTATUSUNSUPPORTEDREQUESTERR, 1));
            CHECK(rc == 0);
            intrs += 24;

            printf("  LWLTLC%d RX1 lwltlc - inject 22 interrupts\n", i);
            rc = reg_wr(fd, REGISTER_RW_ENGINE_LWLTLC, i,
                    LW_LWLTLC_RX_ERR_INJECT_1,
                    DRF_NUM(_LWLTLC_RX, _ERR_INJECT_1, _RXHDROVFERR, 0xff) |
                    DRF_NUM(_LWLTLC_RX, _ERR_INJECT_1, _RXDATAOVFERR, 0xff) |
                    DRF_NUM(_LWLTLC_RX, _ERR_INJECT_1, _STOMPDETERR, 1) |
                    DRF_NUM(_LWLTLC_RX, _ERR_INJECT_1, _RXPOISONERR, 1) |
                    DRF_NUM(_LWLTLC_RX, _ERR_INJECT_1, _CORRECTABLEINTERNALERR, 1) |
                    DRF_NUM(_LWLTLC_RX, _ERR_INJECT_1, _RXUNSUPVCOVFERR, 1) |
                    DRF_NUM(_LWLTLC_RX, _ERR_INJECT_1, _RXUNSUPLWLINKCREDITRELERR, 1) |
                    DRF_NUM(_LWLTLC_RX, _ERR_INJECT_1, _RXUNSUPNCISOCCREDITRELERR, 1));
            CHECK(rc == 0);
            intrs += 22;

            wait_for_service(fd);
        }
    }

    if (RUN_TEST)
    {
        int start = FIRST_PHYS_LINK();
        int end = start + NUM_LINKS();
        int rc;
        int i;

        // No DL on RTL
        if (!rtlsim)
        {
            printf("DL:\n");

            for (i = start; i < end; i++)
            {
                printf("  DL%d ILA - inject 1 interrupt\n", i);
                rc = reg_wr(fd, REGISTER_RW_ENGINE_DLPL, i,
                        LW_PLWL_SL1_DBG_A,
                        DRF_NUM(_PLWL_SL1, _DBG_A, _TRIGGER_ILA, 1));
                CHECK(rc == 0);
                intrs += 1;
            }
        }
    }

    printf("\ndone: %d/%d tests run with %d errors - target %d interrupts\n", run, tests, errors, intrs);

    exit(errors);
}
