//! \file
//! \brief LwSciStream Block APIs interface.
//!
//! \copyright
//! Copyright (c) 2018-2022 LWPU Corporation. All rights reserved.
//!
//! LWPU Corporation and its licensors retain all intellectual property and
//! proprietary rights in and to this software, related documentation and any
//! modifications thereto. Any use, reproduction, disclosure or distribution
//! of this software and related documentation without an express license
//! agreement from LWPU Corporation is strictly prohibited.

#ifndef APIBLOCKINTERFACE_H
#define APIBLOCKINTERFACE_H
#include <cstdint>
#include "covanalysis.h"
#include "lwscistream_common.h"
#include "lwscievent.h"

namespace LwSciStream {

/**
 * @defgroup lwscistream_blanket_statements LwSciStream blanket statements.
 * Generic statements applicable for LwSciStream interfaces.
 * @{
 */

/**
 * \page lwscistream_page_blanket_statements LwSciStream blanket statements
 * \section lwscistream_return_values Return values
 * - Any APIBlockInterface which is only allowed for certain block types
 *   will return LwSciError_NotSupported for any blocks for which it is
 *   not allowed.
 * - Any APIBlockInterface called on a block which is not part of a fully
 *   connected stream or before the LwSciStreamEventType_Connected event
 *   has been queried by calling the LwSciStreamBlockEventQuery() interface
 *   will return LwSciError_StreamNotConnected error code, with the following
 *   exceptions:
 *   -- getOutputConnectPoint()
 *   -- getInputConnectPoint()
 *   -- getEvent()
 *   -- Consumer::apiPayloadReturn()
 *   -- disconnect()
 *   -- eventDefaultSetup()
 *   -- eventNotifierSetup()
 *
 * \section lwscistream_input_parameters Input parameters
 * - Any APIBlockInterface which takes an LwSciStreamPacket as input
 *   will return an LwSciError_StreamBadPacket error if the handle
 *   is not valid.
 * - In blocks which support creation of LwSciStreamPackets through
 *   APIBlockInterface, an LwSciStreamPacket is valid as an input parameter
 *   to APIBlockInterface calls if it is returned from a successful call
 *   to the apiPacketCreate() interface and has not yet been deleted by a
 *   call to the apiPacketDelete() interface.
 * - In other blocks, an LwSciStreamPacket is valid as an input parameter to
 *   APIBlockInterface calls if it was received from a call to
 *   LwSciStreamBlockPacketNewHandleGet() and was accepted in a call to
 *   LwSciStreamBlockPacketStatusSet(), and its cookie has not yet been
 *   received from a call to LwSciStreamBlockPacketOldCookieGet().
 */
/**
 * @}
 */

//! \brief Set of block interfaces which are declared as pure virtual functions.
//!  These interfaces are overridden by derived classes as required. Actual
//!  implementations of these interfaces are called by element level interfaces.
//!
//! \implements{18699864}
class APIBlockInterface
{
public:

    //
    // Connection definition functions
    //

    //! \brief Blocks return pointer to destination block. Producer
    //!   block returns pointer to pool block if initialization
    //!   is successful, pool and queue blocks return error, all the
    //!   other blocks return pointer to their own instances.
    //!
    //! \param [in,out] paramBlock: pointer to destination block.
    //!
    //! \return LwSciError, the completion code of this operation.
    //! * LwSciError_Success: If successful.
    //! * LwSciError_NotInitialized: Producer block returns this
    //!   error if the initialization is not successful.
    //! * LwSciError_AccessDenied: Pool and queue blocks always
    //!   return this error as they don't allow connection through
    //!   public API.
    virtual LwSciError getOutputConnectPoint(
                        BlockPtr& paramBlock) const noexcept = 0;

    //! \brief Blocks return pointer to source block. Consumer
    //!   block returns pointer to queue block if initialization
    //!   is successful, pool and queue blocks return error, all
    //!   the other blocks return pointer to their own instances.
    //!
    //! \param [in,out] paramBlock: pointer to source block.
    //!
    //! \return LwSciError, the completion code of this operation.
    //! * LwSciError_Success: If successful.
    //! * LwSciError_NotInitialized: Consumer block returns this
    //!   error if the initialization is not successful.
    //! * LwSciError_AccessDenied: Pool and queue blocks always
    //!   return this error as they don't allow connection
    //!   through public API.
    virtual LwSciError getInputConnectPoint(
                        BlockPtr& paramBlock) const noexcept = 0;

    //! \brief After the stream is fully connected, queries the number
    //!   of consumers in the subtree below this block.
    //!
    //! \param [in,out] numConsumers: Location in which to return count.
    //!
    //! \return LwSciError, the completion code of this operation.
    //! * LwSciError_Success: Count queried successfully.
    virtual LwSciError apiConsumerCountGet(
                        uint32_t& numConsumers) const noexcept = 0;

    //! \brief Indicate group of setup operations indicated by @a setupType
    //!   are complete.
    //!
    //! \param [in] setupType: Identifies the group of operations.
    //!
    //! \return LwSciError, the completion code of this operation.
    //! * LwSciError_Success: The operation was successful.
    //! * LwSciError_NotYetAvailable: Prerequisites for the group's setup
    //!   operations have not yet oclwrred.
    //! * LwSciError_AlreadyDone: Group was already marked as complete.
    //! * For ElementExport:
    //! ** LwSciError_InconsistentData: For secondary pools, element list
    //!    does not match the primary.
    //! ** Any error returned by Elements::mapDone().
    //! ** Any error returned by Elements::dataSend().
    //! ** Any error returned by Elements::dataClear().
    //! * For ElementImport:
    //! ** Any error returned by Elements::dataClear().
    //! * For PacketExport:
    //! ** LwSciError_InsufficientData: The number of Packets created by
    //!    the pool does not meet the expected value.
    //! * For WaiterAttrExport:
    //! ** Any error returned by Waiters::doneSet().
    //! * For SignalsAttrExport:
    //! ** Any error returned by Signals::doneSet().
    virtual LwSciError apiSetupStatusSet(
                        LwSciStreamSetup const setupType) noexcept = 0;

    //
    // Element definition functions
    //

    //! \brief Add element with the given @a elemType and @a elemBufAttr
    //!   to the list of user-specified elements.
    //!
    //! \param [in] elemType: User-defined type to identify the element.
    //! \param [in] elemBufAttr: Wrapper containing an LwScibufAttrList.
    //!
    //! \return LwSciError, the completion code of this operation.
    //! * LwSciError_Success: The operation was successful.
    //! * LwSciError_BadParameter: LwSciBufAttrList contained by
    //!   @a elemBufAttr is NULL.
    //! * LwSciError_NotYetAvailable: Block has not yet received prerequisite
    //!   information to allow element specification to begin.
    //! * LwSciError_InconsistentData: For secondary pool, @a elemType does
    //!   not match that provided by primary pool.
    //! * Any error returned by Elements::mapAdd().
    LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(M7_1_2), "Bug 3258479")
    LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A8_4_8), "Bug 2745784")
    virtual LwSciError apiElementAttrSet(
                        uint32_t const elemType,
                        LwSciWrap::BufAttr const& elemBufAttr) noexcept = 0;
    LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A8_4_8))
    LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(M7_1_2))

    //! \brief Query the number of elements received at the block from a
    //!   specified source.
    //!   Only supported for producer, consumer, and pool blocks.
    //!
    //! \param [in] queryBlockType: The element source to query
    //!   When called on producer or consumer block, the only valid value
    //!   is LwSciStreamBlockType_Pool. When called on pool block, the
    //!   valid values are LwSciStreamBlockType_Producer and
    //!   LwSciStreamBlockType_Consumer.
    //! \param [in,out] numElements: Location in which to store the count.
    //!
    //! \return LwSciError, the completion code of this operation.
    //! * LwSciError_Success: The operation was successful.
    //! * LwSciError_NotYetAvailable: The element list has not yet arrived.
    //! * LwSciError_BadParameter: The @a queryBlockType is not valid for
    //!   this block.
    //! * Any error returned by Elements::sizeGet().
    virtual LwSciError apiElementCountGet(
                        LwSciStreamBlockType const queryBlockType,
                        uint32_t& numElements) noexcept = 0;

    //! \brief Query the user-specified type assigned to an indexed element
    //!   received from a specified source.
    //!   Only supported for producer, consumer, and pool blocks.
    //!
    //! \param [in] queryBlockType: The element source to query
    //!   When called on producer or consumer block, the only valid value
    //!   is LwSciStreamBlockType_Pool. When called on pool block, the
    //!   valid values are LwSciStreamBlockType_Producer and
    //!   LwSciStreamBlockType_Consumer.
    //! \param [in] elemIndex: The index of the element to query.
    //! \param [in,out] userType: Location in which to store the type.
    //!
    //! \return LwSciError, the completion code of this operation.
    //! * LwSciError_Success: The operation was successful.
    //! * LwSciError_NotYetAvailable: The element list has not yet arrived.
    //! * LwSciError_BadParameter: The @a queryBlockType is not valid for
    //!   this block.
    //! * Any error returned by Elements::typeGet().
    virtual LwSciError apiElementTypeGet(
                        LwSciStreamBlockType const queryBlockType,
                        uint32_t const elemIndex,
                        uint32_t& userType) noexcept = 0;

    //! \brief Query a copy of the LwSciBufAttrList assigned to an indexed
    //!   element received from a specified source. The received attribute
    //!   list is owned by the caller and must be freed when no longer needed.
    //!   Only supported for producer, consumer, and pool blocks.
    //!
    //! \param [in] queryBlockType: The element source to query
    //!   When called on producer or consumer block, the only valid value
    //!   is LwSciStreamBlockType_Pool. When called on pool block, the
    //!   valid values are LwSciStreamBlockType_Producer and
    //!   LwSciStreamBlockType_Consumer.
    //! \param [in] elemIndex: The index of the element to query.
    //! \param [in,out] bufAttrList: Location in which to store the attributes.
    //!
    //! \return LwSciError, the completion code of this operation.
    //! * LwSciError_Success: The operation was successful.
    //! * LwSciError_NotYetAvailable: The element list has not yet arrived.
    //! * LwSciError_BadParameter: The @a queryBlockType is not valid for
    //!   this block.
    //! * Any error returned by Elements::attrGet().
    virtual LwSciError apiElementAttrGet(
                        LwSciStreamBlockType const queryBlockType,
                        uint32_t const elemIndex,
                        LwSciBufAttrList& bufAttrList) noexcept = 0;

    //! \brief Indicate whether an indexed element in the list of allocated
    //!   elements will be used by the block. LwSciStream may make use of
    //!   this to optimize sharing of buffers. By default, all elements are
    //!   assumed to be used.
    //!   Only supported for consumer blocks.
    //!
    //! \param [in] elemIndex: The index of the element to specify.
    //! \param [in] used: Flag indicating whether or not the element is used.
    //!
    //! \return LwSciError, the completion code of this operation.
    //! * LwSciError_Success: The operation was successful.
    //! * LwSciError_NotYetAvailable: The allocated element list has not yet
    //!   arrived.
    //! * LwSciError_NoLongerAvailable: The element import phase has completed.
    //! * LwSciError_BadParameter: The @a elemIndex is out of range for the
    //!   element list.
    //! * Any error returned by Waiters::setUsed().
    virtual LwSciError apiElementUsageSet(
                        uint32_t const elemIndex,
                        bool const used) noexcept = 0;

    //
    // Packet definition functions
    //

    //! \brief Creates a new Packet instance, assigning the @a cookie
    //!   value, and returns the LwSciStreamPacket in @a handle.
    //!   Only supported by pool blocks.
    //!
    //! \param [in] cookie: LwSciStreamCookie that the block should use for
    //!   events associated with this packet. Valid value: 1 to UINT32_MAX.
    //! \param [out] handle: Created LwSciStreamPacket.
    //!
    //! \return LwSciError, the completion code of this operation.
    //! * LwSciError_Success: The operation completed successfully.
    //! * LwSciError_NotYetAvailable: The export of elements has not yet
    //!   completed.
    //! * LwSciError_StreamBadCookie: The @a cookie is not valid.
    //! * LwSciError_Overflow: Pool has reached its limit for the
    //!   maximum number of packets it can create.
    //! * Any error returned by Block::pktCreate().
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A8_4_8), "Approved TID-281")
    virtual LwSciError apiPacketCreate(
                        LwSciStreamCookie const cookie,
                        LwSciStreamPacket& handle) noexcept = 0;

    //! \brief Stores the provided @a elemBufObj as entry @a elemIndex
    //!   of the Packet instance referenced by @a packetHandle.
    //!   Only supported by pool blocks.
    //!
    //! \param [in] packetHandle: LwSciStreamPacket.
    //! \param [in] elemIndex: Index in the list of packet elements.
    //!   Valid value: 0 to consolidated packet element count - 1.
    //! \param [in] elemBufObj: Wrapper containing an LwSciBufObj.
    //!
    //! \return  LwSciError, the completion code of this operation.
    //! * LwSciError_Success: The operation completed successfully.
    //! * LwSciError_BadParameter: If LwSciBufObj contained by
    //!   @a elemBufObj is NULL.
    //! * Any error returned by Packet::bufferSet().
    LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(M7_1_2), "Bug 3258479")
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A8_4_8), "Bug 2745784")
    virtual LwSciError apiPacketBuffer(
                        LwSciStreamPacket const packetHandle,
                        uint32_t const elemIndex,
                        LwSciWrap::BufObj const& elemBufObj) noexcept = 0;
    LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(M7_1_2))

    //! \brief Indicates the definition of the Packet instance referenced
    //!   by @a packetHandle is completed, sending the defintion up and down
    //!   the stream.
    //!   Only supported by pool blocks.
    //!
    //! \param [in] packetHandle: LwSciStreamPacket.
    //!
    //! \return  LwSciError, the completion code of this operation.
    //! * LwSciError_Success: The operation completed successfully.
    //! * Any error returned by Packet::defineDone().
    virtual LwSciError apiPacketComplete(
                        LwSciStreamPacket const handle) noexcept = 0;

    //! \brief Schedules the Packet instance referenced by the given
    //!   @a handle for deletion. If the packet is lwrrenly in the pool,
    //!   sends the deletion message to the rest of the stream.
    //!   Only supported by pool blocks.
    //!
    //! \param [in] handle: LwSciStreamPacket.
    //!
    //! \return LwSciError, the completion code of this operation.
    //! * LwSciError_Success: The operation completed successfully.
    virtual LwSciError apiPacketDelete(
                        LwSciStreamPacket const handle) noexcept = 0;

    //! \brief Retrieves the handle for a new Packet instance after
    //!   receiving an LwSciStreamEventType_PacketCreate event from
    //!   LwSciStreamBlockEventQuery().
    //!   Only supported by producer and consumer blocks.
    //!
    //! \param [in,out] handle: Location in which to store LwSciStreamPacket.
    //!
    //! \return LwSciError, the completion code of this operation.
    //! * LwSciError_Success: The operation completed successfully.
    //! * LwSciError_NotYetAvailable: Block has not yet completed importing
    //!   all element information.
    //! * LwSciError_NoLongerAvailable: Block has already completed importing
    //!   all packets.
    //! * LwSciError_NoStreamPacket: There is no pending new packet handle.
    virtual LwSciError apiPacketNewHandleGet(
                        LwSciStreamPacket& handle) noexcept = 0;

    //! \brief Retrieves the LwSciBufObj for the element with index
    //!   @a elemIndexhandle from the Packet instance referenced by @a handle.
    //!   The recipient owns the object and should free it when it is no
    //!   longer needed.
    //!   Only supported by producer and consumer blocks.
    //!
    //! \param [in] handle: The handle of the packet queried.
    //! \param [in] elemIndex: The index of the element queried.
    //! \param [in,out] bufObjWrap: Wrapper in which to store the retrieved
    //!   buffer.
    //!
    //! \return LwSciError, the completion code of this operation.
    //! * LwSciError_Success: The operation completed successfully.
    //! * LwSciError_NotYetAvailable: Block has not yet completed importing
    //!   all element information.
    //! * LwSciError_NoLongerAvailable: Block has already completed importing
    //!   all packets.
    //! * Any error returned by Packet::bufferGet().
    virtual LwSciError apiPacketBufferGet(
                        LwSciStreamPacket const handle,
                        uint32_t const elemIndex,
                        LwSciWrap::BufObj& bufObjWrap) noexcept = 0;

    //! \brief Retrieves the cookie for a deleted Packet instance after
    //!   receiving an LwSciStreamEventType_PacketDelete event from
    //!   LwSciStreamBlockEventQuery(), and frees associated resources.
    //!   Only supported by producer and consumer blocks.
    //!
    //! \param [in,out] cookie: Location in which to store LwSciStreamCookie.
    //!
    //! \return LwSciError, the completion code of this operation.
    //! * LwSciError_Success: The operation completed successfully.
    //! * LwSciError_NotYetAvailable: Block has not yet completed importing
    //!   all element information.
    //! * LwSciError_NoStreamPacket: There is no pending deleted packet cookie.
    virtual LwSciError apiPacketOldCookieGet(
                        LwSciStreamCookie& cookie) noexcept = 0;

    //! \brief Accepts or rejects the Packet instance referenced by @a handle,
    //!   assigning the @a cookie on acceptance, and informing the rest of
    //!   the stream.
    //!   Only supported by producer and consumer blocks.
    //!
    //! \param [in] handle: Handle for the Packet.
    //! \param [in] cookie: Cookie to assign on acceptance.
    //! \param [in] status: Status value.
    //!
    //! \return LwSciError, the completion code of this operation.
    //! * LwSciError_Success: The operation completed successfully.
    //! * LwSciError_AlreadyInUse: Cookie has already been assigned to
    //!   another Packet.
    //! * Any error returned by Packet::status{Prod|Cons}Set().
    virtual LwSciError apiPacketStatusSet(
                        LwSciStreamPacket const handle,
                        LwSciStreamCookie const cookie,
                        LwSciError const status) noexcept = 0;

    //! \brief Checks whether the Packet instance referenced by
    //!   @a handle was accepted by the producer and consumers.
    //!   Only supported by pool blocks.
    //!
    //! \param [in] handle: Handle for the Packet.
    //! \param [in,out] accepted: Location in which to store the acceptance.
    //!
    //! \return LwSciError, the completion code of this operation.
    //! * LwSciError_Success: The operation completed successfully.
    //! * LwSciError_NotYetAvailable: Packet status has not yet arrived.
    //! * LwSciError_NoLongerAvailable: Packet export phase has completed.
    virtual LwSciError apiPacketStatusAcceptGet(
                        LwSciStreamPacket const handle,
                        bool& accepted) noexcept = 0;

    //! \brief Retrieves the status value for the Packet instance referenced
    //!   by @a handle from the endpoint referenced by @a queryBlockType
    //!   and @a queryBlockIndex.
    //!   Only supported by pool blocks.
    //!
    //! \param [in] handle: Handle for the Packet.
    //! \param [in] queryBlockType: Either LwSciStreamBlockType_Producer or
    //!   LwSciStreamBlockType_Consumer, depending on the endpoint queried.
    //! \param [in] queryBlockIndex: The index of the endpoint of the given
    //!   type to query.
    //! \param [in,out] status: The status value from the endpoint.
    //!
    //! \return LwSciError, the completion code of this operation.
    //! * LwSciError_Success: The operation completed successfully.
    //! * LwSciError_NoLongerAvailable: Packet export phase has completed.
    //! * LwSciError_NotYetAvailable: Status has not yet arrived.
    //! * LwSciError_BadParameter: @a queryBlockType does not indicate
    //!   producer or consumer.
    //! * Any error returned by Packet::status{Prod|Cons}Get().
    virtual LwSciError apiPacketStatusValueGet(
                        LwSciStreamPacket const handle,
                        LwSciStreamBlockType const queryBlockType,
                        uint32_t const queryBlockIndex,
                        LwSciError& status) noexcept = 0;

    //
    // Sync waiter functions
    //

    //! \brief Specify LwSciSync requirements for indexed element
    //!   when the calling block waits for fences from the opposing
    //!   block(s).
    //!
    //! \param [in] elemIndex: Index of element to set.
    //! \param [in] syncAttr: Wrapper for LwSciSyncAttrList with waiter
    //!   requirements for this element. If it contains NULL, element is
    //!   used synchronously.
    //!
    //! \return LwSciError, the completion code of this operation.
    //! * LwSciError_Success: The operation completed successfully.
    //! * LwSciError_NotYetAvailable: Element information has not yet
    //!   arrived.
    //! * LwSciError_NoLongerAvailable: Waiter information was already
    //!   marked complete.
    //! * Any error returned by Waiters::attrSet().
    virtual LwSciError apiElementWaiterAttrSet(
                        uint32_t const elemIndex,
                        LwSciWrap::SyncAttr const& syncAttr) noexcept = 0;

    //! \brief Retrieve LwSciSync requirements for indexed element
    //!   when the opposing block(s) wait for fences from the calling
    //!   block. The caller owns the received LwSciSyncAttrList and
    //!   should free it when it is no longer needed.
    //!
    //! \param [in] elemIndex: Index of element to set.
    //! \param [in,out] syncAttr: Wrapper in which to store LwSciSyncAttrList
    //!   with waiter requirements for this element. If it contains NULL,
    //!   at least one of the opposing blocks uses the element synchronously.
    //!
    //! \return LwSciError, the completion code of this operation.
    //! * LwSciError_Success: The operation completed successfully.
    //! * LwSciError_NotYetAvailable: Waiter information has not yet arrived.
    //! * LwSciError_NoLongerAvailable: Waiter information import was already
    //!   marked complete.
    //! * Any error returned by Waiters::attrGet().
    virtual LwSciError apiElementWaiterAttrGet(
                        uint32_t const elemIndex,
                        LwSciWrap::SyncAttr& syncAttr) noexcept = 0;

    //
    // Sync signaller functions
    //

    //! \brief Specify endpoint's LwSciSyncObj for signalling when it is
    //!   done writing to or reading from the indexed element.
    //!
    //! \param [in] elemIndex: Index in the list of elements.
    //!   Valid values: [0, elemCount-1]
    //! \param [in] syncObj: Wrapper for LwSciSyncObj used to signal this
    //!   this element. If it contains NULL, element is used synchronously.
    //!
    //! \return LwSciError, the completion code of this operation.
    //! * LwSciError_Success: The operation completed successfully.
    //! * LwSciError_NotYetAvailable: Waiter requirements from other endpoint
    //!   have not yet arrived.
    //! * LwSciError_NoLongerAvailable: Signal information was already
    //!   marked complete.
    //! * Any error returned by Signals::syncSet().
    virtual LwSciError apiElementSignalObjSet(
                        uint32_t const elemIndex,
                        LwSciWrap::SyncObj const& syncObj) noexcept = 0;

    //! \brief Retrieve indexed opposing endpoint's LwSciSyncObj for
    //!   signalling when it is done writing to or reading from the
    //!   indexed element.
    //!
    //! \param [in] queryBlockIndex: Index within list of opposing endpoints.
    //!   Valid values: 0 when called from consumer.
    //!                 [0, consumerCount-1] when called from producer.
    //! \param [in] elemIndex: Index in the list of elements.
    //!   Valid values: [0, elemCount-1]
    //! \param [in,out] syncObj: Wrapper in which to store the opposing
    //!   endpoint's LwSciSyncObj. If it contains NULL, the endpoint either
    //!   does not use the element, or uses it synchronously.
    //!
    //! \return LwSciError, the completion code of this operation.
    //! * LwSciError_Success: The operation completed successfully.
    //! * LwSciError_NotYetAvailable: Signal information has not yet arrived.
    //! * LwSciError_NoLongerAvailable: Signal information import was already
    //!   marked complete.
    //! * Any error returned by Signals::syncGet().
    virtual LwSciError apiElementSignalObjGet(
                        uint32_t const queryBlockIndex,
                        uint32_t const elemIndex,
                        LwSciWrap::SyncObj& syncObj) noexcept = 0;

    //
    // Payload functions
    //

    //! \brief Obtains the next available packet in the pool or queue,
    //!   depending on whether called from the producer or consumer block.
    //!
    //! \param[in,out] cookie: Location in which to return packet's cookie.
    //!
    //! \return LwSciError, the completion code of this operation:
    //! * LwSciError_Success: The operation completed successfully.
    //! * LwSciError_NoStreamPacket: No packet is available.
    //! * LwSciError_StreamInternalError: An error oclwrred in the bookkeeping
    //!   for tracking packets.
    //! * Any error returned by Packet::fenceConsCopy().
    virtual LwSciError apiPayloadObtain(
                        LwSciStreamCookie& cookie) noexcept = 0;

    //! \brief Presents or releases the referenced packet, depending on
    //!   whether called from the producer or consumer block.
    //!
    //! \param[in] handle: LwSciStreamPacket of packet to present.
    //!
    //! \return LwSciError, the completion code of this operation:
    //! * LwSciError_Success: The operation completed successfully.
    //! * LwSciError_StreamBadPacket: The packet handle is not recognized.
    //! * LwSciError_StreamPacketInaccessible: The packet is not lwrrently
    //!   held by the application.
    virtual LwSciError apiPayloadReturn(
                        LwSciStreamPacket const handle) noexcept = 0;

    //! \brief Specify postfence to indicate when the endpoint will be
    //!   done operating on the indexed element after the referenced
    //!   packet is presented/released to the stream.
    //!
    //! \param [in] handle: Handle for the Packet.
    //! \param [in] elemIndex: Index in the list of elements.
    //!   Valid values: [0, elemCount-1]
    //! \param [in] postfence: Wrapper for LwSciSyncFence for this element.
    //!   The list of postfences is cleared when the endpoint receives the
    //!   packet, so any fences not specified are left empty.
    //!
    //! \return LwSciError, the completion code of this operation.
    //! * LwSciError_Success: The operation completed successfully.
    //! * LwSciError_StreamBadPacket: The packet handle is not recognized.
    //! * LwSciError_StreamPacketInaccessible: The packet is not lwrrently
    //!   held by the application.
    //! * Any error returned by Packet::fenceProdSet() or
    //!   Packet::fenceConsSet().
    virtual LwSciError apiPayloadFenceSet(
                        LwSciStreamPacket const handle,
                        uint32_t const elemIndex,
                        LwSciWrap::SyncFence const& postfence) noexcept = 0;

    //! \brief Retrieve prefence indicating when the specified opposing
    //!   endpoint will be done operating on the indexed element of the
    //!   referenced packet.
    //!
    //! \param [in] handle: Handle for the Packet.
    //! \param [in] queryBlockIndex: Index within list of opposing endpoints.
    //! \param [in] elemIndex: Index in the list of elements.
    //!   Valid values: [0, elemCount-1]
    //! \param [in,out] prefence: Wrapper to return LwSciSyncFence for element
    //!   @a elemIndex from endpoint @a queryBlockIndex.
    //!
    //! \return LwSciError, the completion code of this operation.
    //! * LwSciError_Success: The operation completed successfully.
    //! * LwSciError_StreamBadPacket: The packet handle is not recognized.
    //! * LwSciError_StreamPacketInaccessible: The packet is not lwrrently
    //!   held by the application.
    //! * Any error returned by Packet::fenceProdGet() or
    //!   Packet::fenceConsGet().
    virtual LwSciError apiPayloadFenceGet(
                        LwSciStreamPacket const handle,
                        uint32_t const queryBlockIndex,
                        uint32_t const elemIndex,
                        LwSciWrap::SyncFence& prefence) noexcept = 0;

    //! \brief Blocks disconnect the source and destination blocks. Producer
    //!  block disconnects only the destination block as it is the upstream endpoint
    //!  of the stream. Consumer block disconnects only the source block as it is the
    //!  downstream endpoint of the stream.
    //!
    //! \return LwSciError, the completion code of this operation:
    //! - always LwSciError_Success.
    virtual LwSciError disconnect(void) noexcept = 0;

    //! \brief Retrieves next LwSciStreamEventType of the block, waiting
    //!   if necessary and requested.
    //!
    //! \param [in] timeout_usec: Number of microseconds to wait if positive.
    //!  Wait forever if negative.
    //! \param [out] event: Location to return LwSciStreamEventType.
    // TODO: Autosar doesn't like output parameters
    //!
    //! \return LwSciError, the completion code of this operation:
    //!  - LwSciError_Success: If event is queried successfully.
    //!  - LwSciError_Timeout: If no pending event to query even
    //!    after waiting for @a timeout_usec.
    //!  - LwSciError_BadParamter: If the block is set up to use
    //!    LwSciEventService, and @a timeout_usec is not zero.
    //!  - Any error that info::fCopy() can generate while duplicating an LwSci
    //!    object.
    virtual LwSciError getEvent(
                        int64_t const timeout_usec,
                        LwSciStreamEventType& event) noexcept = 0;

    //! \brief Retrieves the error code for an error event.
    //!
    //! \return LwSciError, the error code set for an error event.
    virtual LwSciError apiErrorGet(void) noexcept = 0;

    //! \brief Sets up the block to use the default event-notifying mode, if
    //!  it hasn't been set up to use LwSciEventService.
    //!
    //! \return void
    virtual void eventDefaultSetup(void) noexcept = 0;

    //! \brief Sets up the block to use LwSciEventService for event signaling,
    //!  and returns the pointer to the LwSciEventNotifier bound to the block.
    //!
    //! \param [in,out] eventService: An LwSciEventService object from which
    //!  the LwSciEventNotifier is created.
    //!
    //! \return An instance of EventSetupRet, containing the pointer to the
    //!   LwSciEventNotifier bound to the block and LwSciError which indicates
    //!   the completion code of the operation:
    //! * LwSciError_Success: If the operation is successful.
    //! * LwSciError_IlwalidState: Either default event-notifying mode or
    //!   LwSciEventService already configured for the block.
    virtual EventSetupRet eventNotifierSetup(
                        LwSciEventService& eventService) noexcept = 0;

    //! \brief Specify user-defined information with @a userType.
    //!
    //! \param [in] userType: User-defined type to identify the info.
    //! \param [in] info: Endpoint info referenced by the shared pointer.
    //!
    //! \return LwSciError, the completion code of this operation.
    //! * LwSciError_Success: The operation completed successfully.
    //! * LwSciError_NoLongerAvailable: The configuration of the block instance
    //!   is already done.
    //! * Any error returned by EndInfo::infoSet().
    virtual LwSciError apiUserInfoSet(
        uint32_t const userType,
        InfoPtr const& info) noexcept = 0;

    //! \brief Retrieve endpoint information with @a userType from the source
    //!   identified by @a queryBlockType and @a queryBlockIndex after the
    //!   stream is fully connected.
    //!
    //! \param [in] queryBlockType: Indicates whether to query information
    //!                             from producer or consumer endpoint.
    //! \param [in] queryBlockIndex: Index of the endpoint block to query.
    //! \param [in] userType: User-defined type to query.
    //! \param [in,out] info: Location to store the queried endpoint info.
    //!
    //! \return LwSciError, the completion code of this operation.
    //! * LwSciError_Success: The operation completed successfully.
    //! * LwSciError_NoLongerAvailable: Setup of the stream is completed.
    //! * LwSciError_BadParameter: The @a queryBlockType is not valid.
    //! * LwSciError_IndexOutOfRange: The @a queryBlockIndex is invalid
    //! * for the @a queryBlockType.
    //! * Any error returned by EndInfo::infoGet().
    virtual LwSciError apiUserInfoGet(
        LwSciStreamBlockType const queryBlockType,
        uint32_t const queryBlockIndex,
        uint32_t const userType,
        InfoPtr& info) noexcept = 0;

    //! \brief Default destructor of APIBlockInterface.
    virtual ~APIBlockInterface(void) = default;

protected:
    //! \brief Declared constructor of abstract class protected.
    APIBlockInterface(void) noexcept = default;
    APIBlockInterface(APIBlockInterface const&) noexcept = default;
    APIBlockInterface& operator=(APIBlockInterface const&) & noexcept = default;
    APIBlockInterface(APIBlockInterface&&) noexcept = default;
    APIBlockInterface& operator=(APIBlockInterface&&) & noexcept = default;
};

} // namespace LwSciStream

#endif // APIBLOCKINTERFACE_H
