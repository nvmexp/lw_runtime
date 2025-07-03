//! \file
//! \brief LwSciStream automatic pool class declaration.
//!
//! \copyright
//! Copyright (c) 2021 LWPU Corporation. All rights reserved.
//!
//! LWPU Corporation and its licensors retain all intellectual property and
//! proprietary rights in and to this software, related documentation and any
//! modifications thereto. Any use, reproduction, disclosure or distribution
//! of this software and related documentation without an express license
//! agreement from LWPU Corporation is strictly prohibited.

#include <thread>
#include "automaticpool.h"
#include "lwscibuf.h"

namespace LwSciStream {

//! <b>Sequence of operations</b>
//!  - This constructor ilwokes Pool constructor with three packets
//!  - Spwans a dispatch thread to handle pool events
LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A3_1_1), "Bug 2758535")
AutomaticPool::AutomaticPool(void) noexcept :
LWCOV_ALLOWLIST_LINE(LWCOV_CERTC(EXP39_C), "Bug 200662929")
    Pool(numPackets),
    dispatchThread(),
    newPackets()
{
    if (!dispatchThread.joinable()) {
        LWCOV_ALLOWLIST_BEGIN(LWCOV_CERTCPP(ERR50_CPP), "LwSciStream-ADV-CERTCPP-002")
        LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A15_5_2), "Approved TID-482")
        dispatchThread = std::thread(&AutomaticPool::handlePoolEvents, this);
        LWCOV_ALLOWLIST_END(LWCOV_CERTCPP(ERR50_CPP))
    }
}
LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A3_1_1))

//! <b>Sequence of operations</b>
//!  - Terminate dispatch thread
LWCOV_ALLOWLIST_BEGIN(LWCOV_CERTC(EXP39_C), "Bug 200662929")
AutomaticPool::~AutomaticPool(void) noexcept
{
    if (dispatchThread.joinable()) {
        dispatchThread.join();
    }
}
LWCOV_ALLOWLIST_END(LWCOV_CERTC(EXP39_C))

//! <b>Sequence of operations</b>
//!  - Call Block::getEvent() to retrieve the next pending event.
//!  - Call handlePacketCreate() to handle Elements event.
//!  - Call handlePacketsStatus() to handle PacketStatus event
//!    when receiving all packet status for all packets.
//!  - Call Pool::disconnect() to disconnect stream and termiantes
//!    this dispatch thread if receiving Disconnect, Error or
//!    unexpected event.
void
AutomaticPool::handlePoolEvents(void) noexcept
{
    // TODO: When the application provides an LwSciEventService to use
    // for the C2C block in place of an internal thread, then this
    // service should also be passed to the pool block, which will use
    // it in place of its thread too.

    // If there's any error oclwrs in the automatic pool,
    // no way to send the error to the application.

    uint32_t numPacketReady{ 0U };
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_1_1), "TID-300")
    while (true) {
        LwSciStreamEventType event;
        LwSciError err{
            getEvent(INFINITE_TIMEOUT, event)
        };
        bool ret{ true };
        if (LwSciError_Success == err) {
            switch (event) {
            case LwSciStreamEventType_Connected:
            case LwSciStreamEventType_SetupComplete:
                // Does nothing
                break;
            case LwSciStreamEventType_Elements:
                ret = handlePacketCreate();
                break;
            case LwSciStreamEventType_PacketStatus:
                ++numPacketReady;
                if (numPackets == numPacketReady) {
                    ret = handlePacketsStatus();
                }
                break;
            default:
                // Triggers disconnect and terminate the thread if
                // receiving Disconnect, Error or unexpected event.
                ret = false;
                break;
            }
        } else {
            ret = false;
        }

        if (!ret) {
            disconnect();
            return;
        }
    }
}

//TODO: split it into smaller functions to reduce complexity
//! <b>Sequence of operations</b>
//!  - Call Pool::apiElementCountGet() to query the element count from
//!    producer and consumer and take the producer element count as
//!    the allocated element count.
//!  - Call Pool::apiElementTypeGet() and Pool::apiElementAttrGet() to
//!    query the type and attributes of each element from producer and
//!    take them as the allocated element info.
//!  - Iterate the consumer elements and call Pool::apiElementAttrGet()
//!    to find the element with the same type as the producer's.
//!  - If found, call Pool::apiElementAttrGet() to query the consumer
//!    attributes and call LwSciBufAttrListValidateReconciled() to
//!    validate the reconciled attributes from the producer.
//!  - Call Pool::apiSetupStatusSet() to indicate element import done.
//!  - Call Pool::apiPacketCreate() to create numPackets packets.
//!  - For each packet,
//!  -- Call LwSciBufObjAlloc() to allocate LwSciBufObj for each element
//!     with the reconciled attributes and creates a LwSciWrap::BufAttr()
//!     which owns the LwSciBufObj.
//!  -- Call Pool::apiPacketBuffer() to insert each LwSciWrap::BufAttr
//!     into the packet.
//!  -- Call Pool::apiPacketComplete() to indicate the packet is done.
bool
AutomaticPool::handlePacketCreate(void) noexcept
{
    LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A5_1_1), "TID-300")
    // The packet layout from producer process is determined by
    // the primary pool, which will be the final layout.

    // Query producer count as the element count
    uint32_t elemCount;
    LwSciError err{
        apiElementCountGet(LwSciStreamBlockType_Producer, elemCount)
    };
    if (LwSciError_Success != err) {
        return false;
    }

    // Query consumer element count
    uint32_t consElemCount;
    err = apiElementCountGet(LwSciStreamBlockType_Consumer, consElemCount);
    if (LwSciError_Success != err) {
        return false;
    }

    // Validate element count
    if (elemCount < consElemCount) {
        return false;
    }

    std::vector<LwSciWrap::BufAttr> allocatedAttrs;

    try {
        allocatedAttrs.resize(elemCount);
    } catch (...) {
        return false;
    }

    for (uint32_t i{ 0U }; elemCount > i; ++i) {
        // Query final element type
        uint32_t type;
        err = apiElementTypeGet(LwSciStreamBlockType_Producer, i, type);
        if (LwSciError_Success != err) {
            return false;
        }

        // Query final element attributes determined by the primary pool
        LwSciBufAttrList attrList{ nullptr };
        err = apiElementAttrGet(LwSciStreamBlockType_Producer,
                                i,
                                attrList);
        if (LwSciError_Success != err) {
            return false;
        }

        // Reconciled attr list should be set for all elements now.
        // TODO: Only import the reconciled attr list for the elements
        // used by this consumer.
        if (nullptr == attrList) {
            return false;
        }
        allocatedAttrs[i] = LwSciWrap::BufAttr{ attrList, true };
        if (LwSciError_Success != allocatedAttrs[i].getErr()) {
            return false;
        }

        // Following is to validate the reconciled attr list with
        // the consumer attr list.

        // Look for the element with the same type in consumer elements
        for (uint32_t j{ 0U }; consElemCount > j; ++j) {
            uint32_t consType;
            err = apiElementTypeGet(LwSciStreamBlockType_Consumer,
                                    j,
                                    consType);
            if (LwSciError_Success != err) {
                return false;
            }

            if (type == consType) {
                LwSciBufAttrList consAttrList;
                err = apiElementAttrGet(LwSciStreamBlockType_Consumer,
                                        j,
                                        consAttrList);
                if (LwSciError_Success != err) {
                    return false;
                }
                if (nullptr == consAttrList) {
                    break;
                }
                LwSciWrap::BufAttr consBufAttr{ consAttrList, true };

                bool isReconcileListValid{ false };
                err = LwSciBufAttrListValidateReconciled(attrList,
                                                         &consAttrList,
                                                         ONE,
                                                         &isReconcileListValid);
                if (LwSciError_Success != err) {
                    return false;
                }
                if (!isReconcileListValid) {
                    return false;
                }

                break;
            }
        }
    }

    err = apiSetupStatusSet(LwSciStreamSetup_ElementImport);
    if (LwSciError_Success != err) {
        return false;
    }

    // Create packets
    for (uint32_t i{ 0U }; numPackets > i; ++i) {
        LwSciStreamCookie cookie{ i + 1U };
        err = apiPacketCreate(cookie, newPackets[i]);
        if (LwSciError_Success != err) {
            return false;
        }

        // Create buffers for all elements in the packet
        for (uint32_t e{ 0U }; elemCount > e; ++e) {
            // Allocate a buffer object
            LwSciBufObj bufObj;
            err = LwSciBufObjAlloc(allocatedAttrs[e].viewVal(), &bufObj);
            if (LwSciError_Success != err) {
                return false;
            }
            LwSciWrap::BufObj const bufObjWrap{ bufObj, true };

            // Insert the buffer in the packet
            err = apiPacketBuffer(newPackets[i], e, bufObjWrap);
            if (LwSciError_Success != err) {
                return false;
            }
        }

        // Indicate packet setup is complete
        err = apiPacketComplete(newPackets[i]);
        if (LwSciError_Success != err) {
            return false;
        }
    }

    // Indicate export of packet definitions is complete
    err = apiSetupStatusSet(LwSciStreamSetup_PacketExport);
    if (LwSciError_Success != err) {
        return false;
    }

    return true;
    LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A5_1_1))
}

//! <b>Sequence of operations</b>
//!  - Call Pool::apiPacketStatusAcceptGet() to query the packet acceptance
//!    status.
//!  - Call pool::apiPacketDelete() to remove the packet if rejected by
//!    endpoints.
//!  - Call Pool::apiSetupStatusSet() to indicate the packet setup is done.
bool
AutomaticPool::handlePacketsStatus(void) noexcept
{
    LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A5_1_1), "TID-300")
    LwSciError err;
    uint32_t numRejectedPkts{ 0U };

    // Check each packet
    for (uint32_t p{ 0U }; numPackets > p; ++p) {
        // Check packet acceptance
        bool accept;
        err = apiPacketStatusAcceptGet(newPackets[p], accept);
        if (LwSciError_Success != err) {
            return false;
        }

        // Remove the packet rejected by endpoints
        if (!accept) {
            numRejectedPkts++;
            apiPacketDelete(newPackets[p]);
        }
    }

    // No packet is available
    if (numPackets == numRejectedPkts) {
        return false;
    }

    // Indicate import of packet status is complete
    err = apiSetupStatusSet(LwSciStreamSetup_PacketImport);
    if (LwSciError_Success != err) {
        return false;
    }

    return true;
    LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A5_1_1))
}

} // namespace LwSciStream
