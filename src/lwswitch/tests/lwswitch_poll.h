/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2019 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#ifndef _LWSWITCH_POLL_H_
#define _LWSWITCH_POLL_H_

#include "lwswitch.h"
#include "lwmisc.h"

#include "svnp01/dev_egress_ip.h"

using namespace lwswitch;

class LWSwitchDeviceTestPoll : public LWSwitchDeviceTest
{
public:
    void injectFatalError()
    {
        LwU32 val;

        val = DRF_NUM(_EGRESS, _ERR_INJECT_0, _ECCHDRDOUBLEBITERR0, 1);
        regWrite(REGISTER_RW_ENGINE_NPORT, 0, LW_EGRESS_ERR_INJECT_0, val);
    }

    void injectNonFatalError()
    {
        LwU32 val;

        val = DRF_NUM(_EGRESS, _ERR_INJECT_0, _ECCSINGLEBITLIMITERR0, 1);
        regWrite(REGISTER_RW_ENGINE_NPORT, 0, LW_EGRESS_ERR_INJECT_0, val);
    }

    bool skipTest()
    {
        if (!isRegWritePermitted())
        {
            std::cout <<
            "[  SKIPPED ] Register writes are disabled! Re-run with a debug/develop driver build."
            << std::endl;
            return true;
        }

        if (getArch() > LWSWITCH_GET_INFO_INDEX_ARCH_SV10)
        {
            std::cout <<
            "[  SKIPPED ] Unsupported ARCH detected. Please add the ARCH support to the test."
            << std::endl;
            return true;
        }

        return false;
    }
};

#endif // _LWSWITCH_POLL_H_
