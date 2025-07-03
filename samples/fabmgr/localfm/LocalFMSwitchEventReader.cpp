/*
 *  Copyright 2018-2022 LWPU Corporation.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 *
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sstream>
#include <stdexcept>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/poll.h>

#include "fm_log.h"
#include "LocalFMSwitchEventReader.h"

// list of LWSwitch driver events interested and subscribed by FM
uint32 LocalFMSwitchEventReader::mSubscribedEvtList[] = {
                                                        LWSWITCH_DEVICE_EVENT_FATAL,
                                                        LWSWITCH_DEVICE_EVENT_NONFATAL,
                                                        LWSWITCH_DEVICE_EVENT_INBAND_DATA
                                                        };

/*****************************************************************************/

LocalFMSwitchEventReader::LocalFMSwitchEventReader(LocalFabricManagerControl *pLfm,
                                                   LocalFMLWLinkDevRepo *linkDevRepo)
{
    FMIntReturn_t retVal;

    mpLfm = pLfm;
    // create corresponding event handler classes
    mErrorReporter = new LocalFMErrorReporter(pLfm, linkDevRepo);
    mInbandEventHndlr = new LocalFMInbandEventHndlr(pLfm);

    // allocate and subscribe for desired LWSwitch events    
    uint32 numSubscribedEvts = sizeof(mSubscribedEvtList) / sizeof(mSubscribedEvtList[0]);
    memset(mSwitchEvents, 0, sizeof(mSwitchEvents));
    LwU32 numSwitches = mpLfm->getNumLwswitchInterface();

    for (uint32 idx = 0; idx < numSwitches; idx++) {
        LW_STATUS lwStatus;
        LocalFMSwitchInterface *pSwitchInterface;
        pSwitchInterface = mpLfm->switchInterfaceAtIndex(idx);
        if (NULL == pSwitchInterface) {
            std::ostringstream ss;
            ss << "failed to get LWSwitch driver interface object to register for events";
            FM_LOG_ERROR("%s", ss.str().c_str());
            throw std::runtime_error(ss.str());
        }

        // register for all the desired LWSwitch events
        lwStatus = lwswitch_api_create_event(pSwitchInterface->mpLWSwitchDev, mSubscribedEvtList,
                                             numSubscribedEvts, &mSwitchEvents[idx]);
        if (LW_OK != lwStatus) {
            std::ostringstream ss;
            FMPciInfo_t pciInfo = pSwitchInterface->getSwtichPciInfo();
            ss << "request to register events failed for LWSwitch index:" << pSwitchInterface->getSwitchDevIndex()
               << " pci bus id:" << pciInfo.busId << " with error:" << lwstatusToString(lwStatus);
            FM_LOG_ERROR("%s", ss.str().c_str());
            throw std::runtime_error(ss.str());
        }
    }
}
 
LocalFMSwitchEventReader::~LocalFMSwitchEventReader()
{
    // worst case wait for 1 second more than the time used in poll() timeout
    StopAndWait((SWITCH_EVENT_POLLING_TIMEOUT + 1) * 1000);

    // free all the LWSwitch event monitoring
    LwU32 numSwitches = mpLfm->getNumLwswitchInterface();
    for (uint32 idx = 0; idx < numSwitches; idx++) {
        lwswitch_api_free_event(&mSwitchEvents[idx]);
    }

    if (mInbandEventHndlr) {
        delete mInbandEventHndlr;
        mInbandEventHndlr = NULL;
    }

    if (mErrorReporter) {
        delete mErrorReporter;
        mErrorReporter = NULL;
    }
}

void
LocalFMSwitchEventReader::run()
{
    // number of events is same as switch device count.
    LwU32 numEvents = mpLfm->getNumLwswitchInterface();

    // report if a device or port reset is required before getting into the thread main loop
    LwU32 numSwitches = mpLfm->getNumLwswitchInterface();
    for (uint32 idx = 0; idx < numSwitches; idx++) {
        LocalFMSwitchInterface *pSwitchInterface;
        pSwitchInterface = mpLfm->switchInterfaceAtIndex(idx);
        mErrorReporter->reportSwitchFatalErrorScope((FMUuid_t&)pSwitchInterface->getUuid());
    }

    while(!ShouldStop()) {
        LW_STATUS lwStatus;
        // wait for events from LWSwitch driver
        lwStatus = lwswitch_api_wait_events(&mSwitchEvents[0], numEvents, ( SWITCH_EVENT_POLLING_TIMEOUT * 1000 ));
        //
        // wait event returned (due to either events oclwrred or specified timeout happened) 
        // break if we have an outstanding request to exit the thread
        //
        if (ShouldStop()) {
            break;
        }

        // treat anything other than timeout and LW_OK return values as error
        if ((LW_ERR_TIMEOUT != lwStatus) && (LW_OK != lwStatus)) {
            std::ostringstream ss;
            ss << "LWSwitch event processing loop received unexpected error:" << lwstatusToString(lwStatus)
               << " from driver and exiting.";
            FM_LOG_ERROR("%s", ss.str().c_str());
            FM_SYSLOG_ERR("%s", ss.str().c_str());
            break;
        }

        if (LW_ERR_TIMEOUT == lwStatus) {
            // no events oclwrred, continue our waiting
            continue;
        }

        //
        // read the events if the status is not timeout. we only know that some event is signaled. iterate through
        // all the events to see which device generated the event
        //
        dispatchLWSwitchEvents();
    
#ifdef SIM_BUILD
        lwosThreadYield();
#endif
    }
}

void
LocalFMSwitchEventReader::dispatchLWSwitchEvents()
{
    LwU32 numSwitches = mpLfm->getNumLwswitchInterface();

    for (uint32 idx = 0; idx < numSwitches; idx++) {
        LwU32 eventTypes[LWSWITCH_DEVICE_EVENT_COUNT] = {0};
        LwU32 eventTypeCount = 0;
        lwswitch_api_get_signaled_events(mSwitchEvents[idx], eventTypes, &eventTypeCount);
        if (0 != eventTypeCount) {
            //
            // this device signaled some events, read and report the same. The only device information
            // returned by the shim layer is uuid. So we need to find our switch interface class using that.
            //
            lwswitch_event_info eventInfo;
            FMUuid_t switchUuid;
            // get the device which generated the event
            lwswitch_api_get_event_info(mSwitchEvents[idx], &eventInfo);
            memset(switchUuid.bytes, 0, FM_UUID_BUFFER_SIZE);
            lwswitch_uuid_to_string(&eventInfo.uuid, switchUuid.bytes, FM_UUID_BUFFER_SIZE);
            for (LwU32 j = 0; j < eventTypeCount; j++) {
                switch(eventTypes[j]) {
                    case LWSWITCH_DEVICE_EVENT_FATAL:
                        mErrorReporter->processLWSwitchFatalErrorEvent(switchUuid);
                        break;
                    case LWSWITCH_DEVICE_EVENT_NONFATAL:
                        mErrorReporter->processLWSwitchNonFatalErrorEvent(switchUuid);
                        break;
                    case LWSWITCH_DEVICE_EVENT_INBAND_DATA:
                        mInbandEventHndlr->processLWSwitchInbandEvent(switchUuid);
                        break;
                    default:
                        FM_LOG_WARNING("received unsupported event type %d from LWSwitch driver", eventTypes[j]);
                        break;
                }
            }
        }
    }
}
