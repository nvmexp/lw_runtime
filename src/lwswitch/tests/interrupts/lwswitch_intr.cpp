/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2019 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#include "interrupts/lwswitch_intr.h"

std::string getTestNameString(testing::TestParamInfo<LWSwitchIntrTestParams> info)
{
    std::stringstream ss;

    ss << info.param.testName;

    return ss.str();
}

