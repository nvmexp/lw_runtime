/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2018-2020 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#include "lwswitch_poll.h"

#include "ioctl_dev_lwswitch.h"
#include "ioctl_dev_internal_lwswitch.h"

using namespace lwswitch;

int main(int argc, char **argv);

static LwU32 device_event_fatal[]    = { LWSWITCH_DEVICE_EVENT_FATAL };
static LwU32 device_event_nonfatal[] = { LWSWITCH_DEVICE_EVENT_NONFATAL };

TEST_F(LWSwitchDeviceTestPoll, PollErrors)
{
    LW_STATUS status;
    lwswitch_event *event = NULL;
    const LwU32 timeout = 100;
    LwU32 signaled[2] = { 0 };
    LwU32 numSignaled;

    if (skipTest())
    {
        return;
    }

    // Ensure device unbind/rebind after test due to fatal error injection below.
    setUnbindOnTeardown();

    status = lwswitch_api_create_event(getDevice(), device_event_fatal, 1, &event);
    ASSERT_EQ(status, LW_OK);

    // Inject non-fatal error and poll for fatal. We should hit timeout.
    injectNonFatalError();

    status = lwswitch_api_wait_events(&event, 1, timeout);
    ASSERT_EQ(status, LW_ERR_TIMEOUT);

    lwswitch_api_free_event(&event);

    status = lwswitch_api_create_event(getDevice(), device_event_nonfatal, 1, &event);
    ASSERT_EQ(status, LW_OK);

    // Inject non-fatal error and poll for nonfatal.
    injectNonFatalError();

    status = lwswitch_api_wait_events(&event, 1, timeout);
    ASSERT_EQ(status, LW_OK);

    status = lwswitch_api_get_signaled_events(event, signaled, &numSignaled);
    ASSERT_EQ(numSignaled, 1);
    ASSERT_EQ(signaled[0], LWSWITCH_DEVICE_EVENT_NONFATAL);

    lwswitch_api_free_event(&event);

    status = lwswitch_api_create_event(getDevice(), device_event_fatal, 1, &event);
    ASSERT_EQ(status, LW_OK);

    // Inject fatal error and poll for fatal.
    injectFatalError();

    status = lwswitch_api_wait_events(&event, 1, timeout);
    ASSERT_EQ(status, LW_OK);

    status = lwswitch_api_get_signaled_events(event, signaled, &numSignaled);
    ASSERT_EQ(numSignaled, 1);
    ASSERT_EQ(signaled[0], LWSWITCH_DEVICE_EVENT_FATAL);

    // Unbind the device and make sure the current event is stale.
    unbindRebindDevice();

    status = lwswitch_api_wait_events(&event, 1, timeout);
    ASSERT_EQ(status, LW_ERR_ILWALID_OBJECT);

    lwswitch_api_free_event(&event);
}
