/*******************************************************************************
    Copyright (c) 2018-2020 LWPU Corporation

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to
    deal in the Software without restriction, including without limitation the
    rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
    sell copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

        The above copyright notice and this permission notice shall be
        included in all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
    THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
    DEALINGS IN THE SOFTWARE.

*******************************************************************************/
#ifndef _LWSWITCH_INTR_H
#define _LWSWITCH_INTR_H

#include "lwswitch.h"

using namespace lwswitch;

class LWSwitchIntrTestParams
{
public:
    LwU32 regEngine;
    LwU32 regVal, regAddr;
    lwswitch_err_type errorType;
    LwBool bUnbindOnTeardown;
    std::string testName;
    LwU32 regInstance;

    LWSwitchIntrTestParams(LwU32 regEngine, LwU32 regVal, LwU32 regAddr,
                            lwswitch_err_type errorType, LwBool bUnbind, std::string testName,
                            LwU32 regInstance = 0)
        :regEngine(regEngine), regVal(regVal), regAddr(regAddr), errorType(errorType),
            bUnbindOnTeardown(bUnbind), testName(testName), regInstance(regInstance)
    {}

    LWSwitchIntrTestParams()
    {}
};

class LWSwitchDeviceTestIntr
    : public LWSwitchDeviceTestBase, public testing::TestWithParam<LWSwitchIntrTestParams>
{
private:
    // Default to invalid arch
    LwU32 testingArch = 0;
public:
    LWSwitchIntrTestParams injParam;
    LwBool bTestDisabledInterrupt;

    LWSwitchDeviceTestIntr(LwU32 arch)
    {
        testingArch = arch;
    }

    void SetUp()
    {
        injParam = GetParam();

        bTestDisabledInterrupt = LW_FALSE;

        if (injParam.bUnbindOnTeardown)
        {
            setUnbindOnTeardown();
        }

        LWSwitchDeviceTestBase::SetUp();
    }

    void TearDown()
    {
        LWSwitchDeviceTestBase::TearDown();
    }

    void setDisabledIntrTestMode()
    {
        bTestDisabledInterrupt = LW_TRUE;
    }

    void injectErrorAndWaitForEvent()
    {
        LwU32 events[2] = { LWSWITCH_DEVICE_EVENT_FATAL, LWSWITCH_DEVICE_EVENT_NONFATAL };
        const LwU32 timeout = 100;
        lwswitch_event *event = NULL;

        // Register event with Kernel mode driver.
        createEvent(events, 2, &event);

        // Inject Error - This triggers event
        regWrite(injParam.regEngine, injParam.regInstance, injParam.regAddr, injParam.regVal);

        // Wait for event
        waitForEvent(&event, timeout, bTestDisabledInterrupt);
    }

    int validateError()
    {
        LWSWITCH_GET_ERRORS_PARAMS data = { };
        LwU32 errorType;

        memset(&data, 0, sizeof(data));

        for (errorType = 0; errorType < LWSWITCH_ERROR_SEVERITY_MAX; errorType++)
        {
            data.errorType = errorType;
            getErrors(&data);
            if (data.errorCount > 0)
            {
                break;
            }
        }

        // If interrupt is disabled we do not expect any error logging either
        if (bTestDisabledInterrupt)
        {
            if (data.errorCount != 0)
            {
                return -1;
            }
        }
        else
        {
            if (data.error[0].error_value != injParam.errorType)
            {
                return -1;
            }
        }

        return 0;
    }

    bool skipTest()
    {
        bool ret = false;

        if (testingArch != getArch())
        {
            std::cout <<
            "[  SKIPPED ] This test is not supported on " <<
            getArchString() <<
            std::endl;
            ret = true;
        }

        if (!isRegWritePermitted())
        {
            std::cout <<
            "[  SKIPPED ] Register writes are disabled! Re-run with a debug/develop driver build." <<
            std::endl;
            ret = true;
        }

        // Skip interrupt tests on lr10 fmodel
        if (isArchLr10() && isFmodel())
        {
            std::cout <<
            "[  SKIPPED ] Interrupts are not supported on lr10 fmodel." <<
            std::endl;
            ret = true;
        }

        if (ret)
        {
            // Set skip unbind state if the test isn't supported.
            skipUnbindOnTeardown();
        }

        return ret;
    }
};

// Function prototypes of helpers
std::string getTestNameString(testing::TestParamInfo<LWSwitchIntrTestParams> info);

#endif //_LWSWITCH_INTR_H
