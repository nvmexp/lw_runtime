//! \file
//! \brief LwSciStream Block declaration.
//!
//! \copyright
//! Copyright (c) 2018-2022 LWPU Corporation. All rights reserved.
//!
//! LWPU Corporation and its licensors retain all intellectual property and
//! proprietary rights in and to this software, related documentation and any
//! modifications thereto. Any use, reproduction, disclosure or distribution
//! of this software and related documentation without an express license
//! agreement from LWPU Corporation is strictly prohibited.

#ifndef BLOCK_H
#define BLOCK_H
#include <cstdint>
#include <utility>
#include <atomic>
#include <array>
#include <bitset>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <chrono>
#include "covanalysis.h"
#include "lwscievent.h"
#include "lwscistream_common.h"
#include "apiblockinterface.h"
#include "srcblockinterface.h"
#include "dstblockinterface.h"
#include "safeconnection.h"
#include "enumbitset.h"
#include "packet.h"
#include "endinfo.h"
#include "elements.h"
#include "syncwait.h"
#include "lwscistream_panic.h"

namespace LwSciStream {

/**
* @defgroup lwscistream_blanket_statements LwSciStream blanket statements.
* Generic statements applicable for LwSciStream interfaces.
* @{
*/

/**
* \page lwscistream_page_blanket_statements LwSciStream blanket statements
*
* \section lwscistream_entrypoint_validations Entry point validations
* - On entry point of overridden interfaces from
*   LwSciStream::APIBlockInterface, the blocks validate the possible states
*   of the stream (ValidateCtrl::Complete, ValidateCtrl::CompleteQueried).
* - On entry point of overridden interfaces from
*   LwSciStream::DstBlockInterface and LwSciStream::SrcBlockInterface, all
*   the blocks validate the input parameters such as srcIndex (source
*   connection index), dstIndex (destination connection index).
* - For above mentioned validations, blocks create an instance of EnumBitSet on
*   entry point of every APIBlockInterface, SrcBlockInterface and
*   DstBlockInterface interfaces, then initializes it with the ValidateCtrl
*   enum values of the items to be validated. And this instance will be passed
*   to the Block::validateWithError() or Block::validateWithEvent() interfaces
*   which will get the items to be validated from the EnumBitSet instance
*   and perform the required validation.
*
* \section lwscistream_in_out_params Input/Output parameters
* - All the non-const passed-by-reference parameters to an interface are
*   considered input/output parameters. For such parameters, the validity of
*   the input is covered by a blanket statement or by the interface
*   specification if applicable. The output is valid if the error code
*   returned by the interface if any is LwSciError_Success. If the interface
*   doesn't perfom successfully, these parameters will not be modified.
*
* \section lwscistream_input_parameters Input parameters
* - All the parameters passed to Block's default implementation of
*   APIBlockInterface other than getOutputConnectPoint(),
*   getInputConnectPoint(), getEvent(), and eventNotifierSetup() are not used.
*
* - All the parameters passed to Block's default implementation of
*   SrcBlockInterface other than dstNotifyConnection() and
*   dstXmitNotifyConnection() are not used.
*
* - All the parameters passed to Block's default implementation of
*   DstBlockInterface other than srcNotifyConnection() and
*   srcXmitNotifyConnection() are not used.
*
* - All blocks have fixed numbers of source and destination connections
*   specified at the block instance creation. For each direction, if this
*   number is non-zero, then the range of valid connection indices is
*   [0, count-1]. If the number is zero, no index is valid.
*
* - LwSciStreamBlock is valid if a block with that handle is successfully
*   registered by a call to Block::registerBlock() interface and has not yet
*   been unregistered by a call to Block::removeRegisteredBlock() interface.
*
* \section lwscistream_return_values Return values
* - Block's default implementation of any APIBlockInterface other than
*   getOutputConnectPoint(), getInputConnectPoint(), getEvent(), apiErrorGet(),
*   eventDefaultSetup(), and eventNotifierSetup() will always return
*   LwSciError_NotImplemented error code.
* - Any failure in calling Block::blkMutexLock() interface will result in
*   panic.
* - Any internal error oclwrs, LwSciStream triggers the error events by calling
*   Block::setErrorEvent().
*
* \section lwscistream_class_destructors Class destructors
* - It is understood that, when the class objects are destroyed, the data
*   members of the objects will also be cleaned up by ilwoking their destructors.
*   This flow is not explicitly mentioned in the interface specification of
*   the units.
* - Default destructors for the class objects are NOT included in the interface
*   specification.
*
* \section lwscistream_smart_pointers Smart pointers
* - Smart pointers (std::shared_ptr) to block instances (BlockPtr) returned
*   by a successful call to Block::getRegisteredBlock() interface will be
*   destroyed when it goes out of scope or it is reset.
* - Smart pointers (std::shared_ptr) to packet instances (PacketPtr) returned
*   by a successful call to Block::pktFindByHandle() or Block::pktFindByCookie()
*   interfaces will be destroyed when it goes out of scope or it is reset.
*
* \section lwscistream_conlwrrency Conlwrrency
* - Block::blkMutexLock() is called to lock the block mutex whenever
*   a block's non-atomic members that could be simultaneously modified by other
*   threads are accessed.
* - Block::blkMutexLock(), Block::setErrorEvent(), Block::pktFindByCookie(),
*   and Block::pktFindByHandle() have parameter 'locked' whose default value is
*   false. To avoid relwrsive locking, if the block mutex has been locked in
*   the thread, the caller calls these functions with parameter 'locked' set to
*   true, indicating the block mutex is already locked, and these functions
*   will not attempt to lock the block mutex. The situation where parameter
*   'locked' is true is called out explictly in the interface specification of
*   the units.
* - The block mutex is released before a block calls into functions of another
*   block.
*/

/**
* @}
*/

//! \brief Constant to represent invalid block handle
constexpr LwSciStreamBlock ILWALID_BLOCK_HANDLE  {0U};

// Object maps
//! \brief Unordered map of LwSciStreamBlock and BlockPtr as key-value pairs.
//!
//! \implements{19728555}
using HandleBlockPtrMap = std::unordered_map<LwSciStreamBlock, BlockPtr>;

//! \brief Iterator of HandleBlockPtrMap.
//!
//! \implements{19728558}
using HandleBlockPtrMapIterator =
    std::unordered_map<LwSciStreamBlock, BlockPtr>::iterator;

//! \brief Block is a parent class for all concrete blocks, which inherits
//!    from three base interfaces (SrcBlockInterface, DstBlockInterface and
//!    APIBlockInterface) and contains utility functions and common data for
//!    all derived blocks.
//!
//!  * Provides default implementation of interfaces from SrcBlockInterface,
//!    DstBlockInterface and APIBlockInterface.
//!  * Maintains a static block registry and assigns LwSciStreamBlock to
//!    each block instance, which will be returned to the application while
//!    creating the concrete block.
//!  * Provides public interfaces to establish the connection between the
//!    blocks and maintains the upstream and downstream connections.
//!  * Provides utility functions to signal/wait for and retrieve the next
//!    event.
//!  * Provides utility functions to manage the packet map, including packet
//!    creation, deletion and packet retrieval with the given LwSciStreamPacket
//!    or LwSciStreamCookie.
//!  * Provides standardized validation of stream states defined in
//!    ValidateCtrl enum, which are commonly checked at numerous entry points.
//!  * Provides lock to prevent conlwrrent access of member data.
//!  * Provides SafeConnection::Access to SrcBlockInterface of the connected
//!    source block and SafeConnection::Access to DstBlockInterface of the
//!    connected destination block if any.
//!  * Provides utility functions to access member data by the derived blocks.
//!
//! \if TIER4_SWAD
//! \implements{18793185}
//! \endif
//!
//! \if TIER4_SWUD
//! \implements{21128640}
//! \endif
class Block :
    public APIBlockInterface,
    public SrcBlockInterface,
    public DstBlockInterface
{
protected:
    //
    // Connection type queries
    //

    // TODO: Should this be doxygenated in some global info?
    // For many operations, some block types don't need to interact with
    //   the incoming information themselves, and just pass it through to
    //   neighboring blocks. To avoid writing duplicate functions, we
    //   provide this pass-through as the default behavior in the base
    //   block. But we would like to catch cases where the information
    //   can't be trivially passed through and the block must provide an
    //   override. So we define a set of queries that determine whether
    //   the pass-through operation will work. There may still be block
    //   types that satisfy these conditions but still require an override,
    //   but this covers the basics.

    //! \brief Checks whether this block is a direct link which provides a
    //!  one-to-many connection from src block to dst block(s).
    bool isSrcOneToMany(void) const noexcept
    {
        return (!(connSrcRemote || connDstRemote)) &&
               ((ONE == connSrcTotal) && (ONE <= connDstTotal));
    };

    //! \brief Checks whether this block is a direct link which provides a
    //!  one-to-many connection from dst block to src block(s).
    // Note: Since there is lwrrently at most one src block, this is equivalent
    //       to a one-to-one check, but for symmetry it is handled this way
    //       so no upates are required if we introduce multi-producer streams.
    bool isDstOneToMany(void) const noexcept
    {
        return (!(connSrcRemote || connDstRemote)) &&
               ((ONE <= connSrcTotal) && (ONE == connDstTotal));
    };

public:
    //
    // Connection definition functions
    //

    // Non-virtual public facing functions to establish connections
    //! \brief Finds and initializes the available source SafeConnection of the
    //!   block.
    //!   If successful, this call must be followed by a call either to
    //!   Block::connSrcComplete() or Block::connSrcCancel() interface, depending
    //!   on whether or not the corresponding call to Block::connDstInitiate()
    //!   interface of the source block is successful.
    //!
    //! \param [in] srcPtr: pointer to the source block.
    //!
    //! \return An instance of IndexRet, containing LwSciError which indicates
    //!   the completion code of the operation and connection index to use
    //!   which is valid only if the LwSciError is LwSciError_Success.
    //! * LwSciError_Success: if source SafeConnection is initialized
    //!   successfully.
    //! * LwSciError_NotSupported: if source connections are not applicable
    //!   to the given BlockType.
    //! * LwSciError_InsufficientResource: if no source SafeConnection is
    //!   available.
    //!
    //! \if TIER4_SWAD
    //! \implements{19727136}
    //! \endif
    IndexRet connSrcInitiate(
        BlockPtr const& srcPtr) noexcept;

    //! \brief Completes the source SafeConnection of the block and checks for
    //!   complete paths to the producer and the consumer(s).
    //!
    //! \note This interface assumes that the @a srcIndex and @a dstIndex are
    //!  already validated by the caller functions, so these are not validated
    //!  again by this interface.
    //!
    //!  <b>Preconditions</b>
    //!   - Successful calls to Block::connSrcInitiate() in this block and
    //!     Block::connDstInitiate() in the source block.
    //!
    //! \param [in] srcIndex: Index in the list of source SafeConnections.
    //!   Valid value: [0 to connSrcTotal-1]
    //! \param [in] dstIndex: The connection index within the list of source
    //!   block's destination SafeConnections.
    //!   Valid value: [0 to connDstTotal-1]
    //!
    //! \return void
    //!
    //! \if TIER4_SWAD
    //! \implements{19727139}
    //! \endif
    void connSrcComplete(
        uint32_t const srcIndex,
        uint32_t const dstIndex) noexcept;

    //! \brief Cancels the source SafeConnection of the block.
    //!
    //! \note This interface assumes that the @a srcIndex is already validated
    //!  by the caller functions, so this is not validated again by this interface.
    //!
    //!  <b>Preconditions</b>
    //!   - A successful call to Block::connSrcInitiate() in this block and
    //!     a failed call to Block::connDstInitiate() in the source block.
    //!
    //! \param [in] srcIndex: Index in the list of source SafeConnections.
    //!   Valid value: [0 to connSrcTotal-1]
    //!
    //! \return void
    //!
    //! \if TIER4_SWAD
    //! \implements{19727142}
    //! \endif
    void connSrcCancel(
        uint32_t const srcIndex) noexcept;

    //! \brief Finds and initializes the available destination SafeConnection
    //!   of the block.
    //!   If successful, this call must be followed by a call
    //!   either to Block::connDstComplete() or Block::connDstCancel() interface,
    //!   depending on whether or not the corresponding call to
    //!   Block::connSrcInitiate() interface of the destination block is
    //!   successful.
    //!
    //! \param [in] dstPtr: pointer to the destination block
    //!
    //! \return An instance of IndexRet, containing LwSciError which indicates
    //!   the completion code of the operation and connection index to use
    //!   which is valid only if the LwSciError is LwSciError_Success.
    //! * LwSciError_Success: if destination SafeConnection is initialized
    //!   successfully.
    //! * LwSciError_NotSupported: if destination connections are not applicable
    //!   to the given BlockType.
    //! * LwSciError_InsufficientResource: if no destination SafeConnection is
    //!   available.
    //!
    //! \if TIER4_SWAD
    //! \implements{19727145}
    //! \endif
    IndexRet connDstInitiate(
        BlockPtr const& dstPtr) noexcept;

    //! \brief Completes the destination SafeConnection of the block and checks
    //!   for complete paths to the producer and the consumer(s).
    //!
    //! \note This interface assumes that the @a srcIndex and @a dstIndex are
    //!  already validated by the caller functions, so these are not validated
    //!  again by this interface.
    //!
    //!  <b>Preconditions</b>
    //!   - Successful calls to Block::connDstInitiate() in this block and
    //!     Block::connSrcInitiate() in the destination block.
    //!
    //! \param [in] dstIndex: Index in the list of destination SafeConnections.
    //!   Valid value: [0 to connDstTotal-1]
    //! \param [in] srcIndex: The connection index within the list of
    //!   destination block's source SafeConnections.
    //!   Valid value: [0 to connSrcTotal-1]
    //!
    //! \return void
    //!
    //! \if TIER4_SWAD
    //! \implements{19727148}
    //! \endif
    void connDstComplete(
        uint32_t const dstIndex,
        uint32_t const srcIndex) noexcept;

    //! \brief Cancels the destination SafeConnection of the block.
    //!
    //! \note This interface assumes that the @a dstIndex is already validated
    //!  by the caller functions, so this is not validated again by this interface.
    //!
    //!  <b>Preconditions</b>
    //!   - A successful call to Block::connDstInitiate() in this block and
    //!     a failed call to Block::connSrcInitiate() in the destination block.
    //!
    //! \param [in] dstIndex: Index in the list of destination SafeConnections.
    //!   Valid value: [0 to connDstTotal-1]
    //!
    //! \return void
    //!
    //! \if TIER4_SWAD
    //! \implements{19727151}
    //! \endif
    void connDstCancel(
        uint32_t const dstIndex) noexcept;

    //! \brief Finalize the block configuration at its first connection with
    //!   other block.
    //!
    //! \return void
    virtual void finalizeConfigOptions(void) noexcept;

    // Functions inherited from APIBlockInterface
    //! \brief Default implementation of
    //!   APIBlockInterface::getOutputConnectPoint,
    //!   which gets the pointer to its own instance and returns it.
    //!
    //! \param [out] paramBlock: pointer to its own instance
    //!
    //! \return LwSciError, always LwSciError_Success
    //!
    //! \if TIER4_SWAD
    //! \implements{19727154}
    //! \endif
    LwSciError getOutputConnectPoint(
        BlockPtr& paramBlock) const noexcept override;

    //! \brief Default implementation of
    //!   APIBlockInterface::getInputConnectPoint,
    //!   which gets the pointer to its own instance and returns it.
    //!
    //! \param [out] paramBlock: pointer to its own instance
    //!
    //! \return LwSciError, always LwSciError_Success
    //!
    //! \if TIER4_SWAD
    //! \implements{19727157}
    //! \endif
    LwSciError getInputConnectPoint(
        BlockPtr& paramBlock) const noexcept override;

    //! \brief Continue flow of producer info downstream if appropriate.
    //!
    //! \return void
    void prodInfoFlow(void) noexcept;

    //! \brief Continue flow of consumer info upstream if appropriate.
    //!
    //! \return void
    void consInfoFlow(void) noexcept;

    //! \copydoc LwSciStream::APIBlockInterface::apiConsumerCountGet
    LwSciError apiConsumerCountGet(
        uint32_t& numConsumers) const noexcept final;

    //! \brief Default implementation of
    //!   APIBlockInterface::apiSetupStatusSet(),
    //!   which always returns LwSciError_NotSupported error code.
    LwSciError apiSetupStatusSet(
        LwSciStreamSetup const setupType) noexcept override;

    //
    // Element definition functions
    //

    //! \brief Default implementation of
    //!   APIBlockInterface::apiElementAttrSet(),
    //!   which always returns LwSciError_NotSupported error code.
    LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A8_4_8), "Bug 2745784")
    LwSciError apiElementAttrSet(
        uint32_t const elemType,
        LwSciWrap::BufAttr const& elemBufAttr) noexcept override;
    LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A8_4_8))

    //! \brief Default implementation of
    //!   APIBlockInterface::apiElementCountGet(),
    //!   which always returns LwSciError_NotSupported error code.
    LwSciError apiElementCountGet(
        LwSciStreamBlockType const queryBlockType,
        uint32_t& numElements) noexcept override;

    //! \brief Default implementation of
    //!   APIBlockInterface::apiElementTypeGet(),
    //!   which always returns LwSciError_NotSupported error code.
    LwSciError apiElementTypeGet(
        LwSciStreamBlockType const queryBlockType,
        uint32_t const elemIndex,
        uint32_t& userType) noexcept override;

    //! \brief Default implementation of
    //!   APIBlockInterface::apiElementAttrGet(),
    //!   which always returns LwSciError_NotSupported error code.
    LwSciError apiElementAttrGet(
        LwSciStreamBlockType const queryBlockType,
        uint32_t const elemIndex,
        LwSciBufAttrList& bufAttrList) noexcept override;

    //! \brief Default implementation of
    //!   APIBlockInterface::apiElementUsageSet(),
    //!   which always returns LwSciError_NotSupported error code.
    LwSciError apiElementUsageSet(
        uint32_t const elemIndex,
        bool const used) noexcept override;

    //! \brief Default implementation of
    //!   DstBlockInterface::srcRecvSupportedElements, which always
    //!   triggers LwSciError_NotImplemented error event.
    void srcRecvSupportedElements(
        uint32_t const srcIndex,
        Elements const& inElements) noexcept override;

    //! \brief Default implementation of
    //!   SrcBlockInterface::dstRecvSupportedElements, which passes
    //!   the information through for direct one-to-many links and
    //!   triggers LwSciError_NotImplemented error event otherwise.
    //!
    //! \copydetails LwSciStream::SrcBlockInterface::dstRecvSupportedElements
    //!
    //! \return void
    void dstRecvSupportedElements(
        uint32_t const dstIndex,
        Elements const& inElements) noexcept override;

    //! \brief Default implementation of
    //!   DstBlockInterface::srcRecvAllocatedElements, which passes
    //!   the information through for direct one-to-many links and
    //!   triggers LwSciError_NotImplemented error event otherwise.
    //!
    //! \copydetails LwSciStream::SrcBlockInterface::srcRecvAllocatedElements
    //!
    //! \return void
    void srcRecvAllocatedElements(
        uint32_t const srcIndex,
        Elements const& inElements) noexcept override;

    //! \brief Default implementation of
    //!   SrcBlockInterface::dstRecvAllocatedElements, which always
    //!   triggers LwSciError_NotImplemented error event.
    void dstRecvAllocatedElements(
        uint32_t const dstIndex,
        Elements const& inElements) noexcept override;

    //
    // Packet definition functions
    //

    //! \brief Default implementation of
    //!   APIBlockInterface::apiPacketCreate,
    //!   which always returns LwSciError_NotSupported error code.
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A8_4_8), "Approved TID-281")
    LwSciError apiPacketCreate(
        LwSciStreamCookie const cookie,
        LwSciStreamPacket& handle) noexcept override;

    //! \brief Default implementation of
    //!   APIBlockInterface::apiPacketBuffer,
    //!   which always returns LwSciError_NotSupported error code.
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A8_4_8), "Bug 2745784")
    LwSciError apiPacketBuffer(
        LwSciStreamPacket const packetHandle,
        uint32_t const elemIndex,
        LwSciWrap::BufObj const& elemBufObj) noexcept override;

    //! \brief Default implementation of
    //!   APIBlockInterface::apiPacketComplete,
    //!   which always returns LwSciError_NotSupported error code.
    LwSciError apiPacketComplete(
        LwSciStreamPacket const handle) noexcept override;

    //! \brief Default implementation of
    //!   APIBlockInterface::apiPacketNewHandleGet,
    //!   which always returns LwSciError_NotSupported error code.
    LwSciError apiPacketNewHandleGet(
        LwSciStreamPacket& handle) noexcept override;

    //! \brief Default implementation of
    //!   APIBlockInterface::apiPacketBufferGet,
    //!   which always returns LwSciError_NotSupported error code.
    LwSciError apiPacketBufferGet(
        LwSciStreamPacket const handle,
        uint32_t const elemIndex,
        LwSciWrap::BufObj& bufObjWrap) noexcept override;

    //! \brief Default implementation of
    //!   APIBlockInterface::apiPacketOldCookieGet,
    //!   which always returns LwSciError_NotSupported error code.
    LwSciError apiPacketOldCookieGet(
        LwSciStreamCookie& cookie) noexcept override;

    //! \brief Default implementation of
    //!   DstBlockInterface::srcRecvPacketCreate, which allocates
    //!   a new Packet if needed and passes on the original Packet
    //!   for direct one-to-many links and triggers
    //!   LwSciError_NotImplemented error event otherwise.
    void srcRecvPacketCreate(
        uint32_t const dstIndex,
        Packet const& origPacket) noexcept override;

    //! \brief Default implementation of
    //!   SrcBlockInterface::dstRecvPacketCreate,
    //!   which always triggers LwSciError_NotImplemented error event.
    void dstRecvPacketCreate(
        uint32_t const srcIndex,
        Packet const& origPacket) noexcept override;

    //! \brief Default implementation of
    //!   DstBlockInterface::srcRecvPacketsComplete, which passes the
    //!   information through for direct one-to-many links and triggers
    //!   LwSciError_NotImplemented error event otherwise.
    void srcRecvPacketsComplete(
        uint32_t const srcIndex) noexcept override;

    //! \brief Default implementation of
    //!   SrcBlockInterface::dstRecvPacketsComplete,
    //!   which always triggers LwSciError_NotImplemented error event.
    void dstRecvPacketsComplete(
        uint32_t const dstIndex) noexcept override;

    //
    // Packet status functions
    //

    //! \brief Default implementation of
    //!   APIBlockInterface::apiPacketStatusSet,
    //!   which always returns LwSciError_NotSupported error code.
    LwSciError apiPacketStatusSet(
        LwSciStreamPacket const handle,
        LwSciStreamCookie const cookie,
        LwSciError const status) noexcept override;

    //! \brief Default implementation of
    //!   APIBlockInterface::apiPacketStatusAcceptGet,
    //!   which always returns LwSciError_NotSupported error code.
    LwSciError apiPacketStatusAcceptGet(
        LwSciStreamPacket const handle,
        bool& accepted) noexcept override;

    //! \brief Default implementation of
    //!   APIBlockInterface::apiPacketStatusValueGet,
    //!   which always returns LwSciError_NotSupported error code.
    LwSciError apiPacketStatusValueGet(
        LwSciStreamPacket const handle,
        LwSciStreamBlockType const queryBlockType,
        uint32_t const queryBlockIndex,
        LwSciError& status) noexcept override;

    //! \brief Default implementation of
    //!   DstBlockInterface::srcRecvPacketStatus,
    //!   which always triggers LwSciError_NotImplemented error event.
    void srcRecvPacketStatus(
        uint32_t const srcIndex,
        Packet const& origPacket) noexcept override;

    //! \brief Default implementation of
    //!   SrcBlockInterface::dstRecvPacketStatus, which passes the information
    //!   through for direct one-to-many links and triggers
    //!   LwSciError_NotImplemented error event otherwise.
    void dstRecvPacketStatus(
        uint32_t const dstIndex,
        Packet const& origPacket) noexcept override;

    //
    // Packet teardown functions
    //

    //! \brief Default implementation of
    //!   APIBlockInterface::apiPacketDelete,
    //!   which always returns LwSciError_NotSupported error code.
    LwSciError apiPacketDelete(
        LwSciStreamPacket const handle) noexcept override;

    //! \brief Default implementation of
    //!   DstBlockInterface::srcRecvPacketDelete, which deletes the Packet
    //!   from the map and passes the information through for direct
    //!   one-to-many links and triggers LwSciError_NotImplemented
    //!   error event otherwise.
    void srcRecvPacketDelete(
        uint32_t const srcIndex,
        LwSciStreamPacket const handle) noexcept override;

    //! \brief Default implementation of
    //!   SrcBlockInterface::dstRecvPacketDelete,
    //!   which always triggers LwSciError_NotImplemented error event.
    void dstRecvPacketDelete(
        uint32_t const dstIndex,
        LwSciStreamPacket const handle) noexcept override;

    //
    // Sync waiter functions
    //

    //! \brief Default implementation of
    //!   APIBlockInterface::apiElementWaiterAttrSet,
    //!   which always returns LwSciError_NotSupported error code.
    LwSciError apiElementWaiterAttrSet(
        uint32_t const elemIndex,
        LwSciWrap::SyncAttr const& syncAttr) noexcept override;

    //! \brief Default implementation of
    //!   APIBlockInterface::apiElementWaiterAttrGet,
    //!   which always returns LwSciError_NotSupported error code.
    LwSciError apiElementWaiterAttrGet(
        uint32_t const elemIndex,
        LwSciWrap::SyncAttr& syncAttr) noexcept override;

    //! \brief Default implementation of
    //!   DstBlockInterface::srcRecvSyncWaiter, which passes the information
    //!   through for direct one-to-many links and triggers
    //!   LwSciError_NotImplemented error event otherwise.
    void srcRecvSyncWaiter(
        uint32_t const srcIndex,
        Waiters const& syncWaiter) noexcept override;

    //! \brief Default implementation of
    //!   SrcBlockInterface::dstRecvSyncWaiter, which passes the information
    //!   through for direct one-to-many links and triggers
    //!   LwSciError_NotImplemented error event otherwise.
    void dstRecvSyncWaiter(
        uint32_t const dstIndex,
        Waiters const& syncWaiter) noexcept override;

    //
    // Sync signaller functions
    //

    //! \brief Default implementation of
    //!   APIBlockInterface::apiElementSignalObjSet,
    //!   which always returns LwSciError_NotSupported error code.
    LwSciError apiElementSignalObjSet(
        uint32_t const elemIndex,
        LwSciWrap::SyncObj const& syncObj) noexcept override;

    //! \brief Default implementation of
    //!   APIBlockInterface::apiElementSignalObjGet,
    //!   which always returns LwSciError_NotSupported error code.
    LwSciError apiElementSignalObjGet(
        uint32_t const queryBlockIndex,
        uint32_t const elemIndex,
        LwSciWrap::SyncObj& syncObj) noexcept override;

    //! \brief Default implementation of
    //!   DstBlockInterface::srcRecvSyncSignal, which passes the information
    //!   through for direct one-to-many links and triggers
    //!   LwSciError_NotImplemented error event otherwise.
    void srcRecvSyncSignal(
        uint32_t const srcIndex,
        Signals const& syncSignal) noexcept override;

    //! \brief Default implementation of
    //!   SrcBlockInterface::dstRecvSyncSignal, which passes the information
    //!   through for direct one-to-many links and triggers
    //!   LwSciError_NotImplemented error event otherwise.
    void dstRecvSyncSignal(
        uint32_t const dstIndex,
        Signals const& syncSignal) noexcept override;

    //
    // Payload functions
    //

    //! \brief Default implementation of
    //!   APIBlockInterface::apiPayloadObtain,
    //!   which always returns LwSciError_NotSupported error code.
    LwSciError apiPayloadObtain(
        LwSciStreamCookie& cookie) noexcept override;

    //! \brief Default implementation of
    //!   APIBlockInterface::apiPayloadReturn,
    //!   which always returns LwSciError_NotSupported error code.
    LwSciError apiPayloadReturn(
        LwSciStreamPacket const handle) noexcept override;

    //! \brief Default implementation of
    //!   APIBlockInterface::apiPayloadFenceSet,
    //!   which always returns LwSciError_NotSupported error code.
    LwSciError apiPayloadFenceSet(
        LwSciStreamPacket const handle,
        uint32_t const elemIndex,
        LwSciWrap::SyncFence const& postfence) noexcept override;

    //! \brief Default implementation of
    //!   APIBlockInterface::apiPayloadFenceGet,
    //!   which always returns LwSciError_NotSupported error code.
    LwSciError apiPayloadFenceGet(
        LwSciStreamPacket const handle,
        uint32_t const queryBlockIndex,
        uint32_t const elemIndex,
        LwSciWrap::SyncFence& prefence) noexcept override;

    //! \brief Default implementation of
    //!   DstBlockInterface::srcRecvPayload, which passes the information
    //!   through for direct one-to-many links and triggers
    //!   LwSciError_NotImplemented error event otherwise.
    void srcRecvPayload(
        uint32_t const srcIndex,
        Packet const& prodPayload) noexcept override;

    //! \brief Default implementation of
    //!   SrcBlockInterface::dstRecvPayload, which passes the information
    //!   through for direct one-to-many links and triggers
    //!   LwSciError_NotImplemented error event otherwise.
    void dstRecvPayload(
        uint32_t const dstIndex,
        Packet const& consPayload) noexcept override;

    //! \brief Default implementation of
    //!   DstBlockInterface::srcDequeuePayload, which always sets
    //!   a LwSciError_NotImplemented error and returns an empty pointer.
    PacketPtr srcDequeuePayload(
        uint32_t const srcIndex) noexcept override;

    //! \brief Default implementation of
    //!   SrcBlockInterface::dstDequeuePayload, which always sets
    //!   a LwSciError_NotImplemented error and returns an empty pointer.
    PacketPtr dstDequeuePayload(
        uint32_t const dstIndex) noexcept override;

    //! \brief Default implementation of
    //!   APIBlockInterface::disconnect,
    //!   which always returns LwSciError_NotImplemented error code.
    //!
    //! \implements{19727202}
    LwSciError disconnect(void) noexcept override;

    //! \copydoc LwSciStream::APIBlockInterface::getEvent
    //!
    //! \if TIER4_SWAD
    //! \implements{19727205}
    //! \endif
    LwSciError getEvent(
        int64_t const timeout_usec,
        LwSciStreamEventType& event) noexcept override;

    //! \copydoc LwSciStream::APIBlockInterface::apiErrorGet
    LwSciError apiErrorGet(void) noexcept final;

    //! \copydoc LwSciStream::APIBlockInterface::eventDefaultSetup
    //!
    //! \if TIER4_SWAD
    //! \implements{21698661}
    //! \endif
    void eventDefaultSetup(void) noexcept final;

    //! \copydoc LwSciStream::APIBlockInterface::eventNotifierSetup
    //!
    //! \if TIER4_SWAD
    //! \implements{21698704}
    //! \endif
    EventSetupRet eventNotifierSetup(
        LwSciEventService& eventService) noexcept final;

    //! \brief Default implementation of
    //!   APIBlockInterface::apiUserInfoSet,
    //!   which always returns LwSciError_NotSupported error code.
    LwSciError apiUserInfoSet(
        uint32_t const userType,
        InfoPtr const& info) noexcept override;

    //! \copydoc LwSciStream::APIBlockInterface::apiUserInfoGet
    LwSciError apiUserInfoGet(
        LwSciStreamBlockType const queryBlockType,
        uint32_t const queryBlockIndex,
        uint32_t const userType,
        InfoPtr& info) noexcept final;

    // Functions inherited from SrcBlockInterface

    //! \brief Default implementation of
    //!   SrcBlockInterface::dstRecvConsInfo, which passes
    //!   the information through for direct one-to-many links and
    //!   triggers LwSciError_NotImplemented error event otherwise.
    //!
    //! \copydetails LwSciStream::SrcBlockInterface::dstRecvConsInfo
    //!
    //! \return void
    void dstRecvConsInfo(
        uint32_t const dstIndex,
        EndInfoVector const& info) noexcept override;

    //! \brief Default implementation of
    //!   SrcBlockInterface::dstDisconnect,
    //!   which always triggers LwSciError_NotImplemented error event.
    //!
    //! \implements{19727247}
    void dstDisconnect(
        uint32_t const dstIndex) noexcept override;

    // Functions inherited from DstBlockInterface

    //! \brief Default implementation of
    //!   DstBlockInterface::srcRecvProdInfo, which passes
    //!   the information through for direct one-to-many links and
    //!   triggers LwSciError_NotImplemented error event otherwise.
    //!
    //! \copydetails LwSciStream::DstBlockInterface::srcRecvProdInfo
    //!
    //! \return void
    void srcRecvProdInfo(
        uint32_t const srcIndex,
        EndInfoVector const& info) noexcept override;

    //! \brief Default implementation of
    //! DstBlockInterface::srcDisconnect,
    //!   which always triggers LwSciError_NotImplemented error event.
    //!
    //! \implements{19727289}
    void srcDisconnect(
        uint32_t const srcIndex) noexcept override;

    // Utility Functions
    //! \brief Returns the type of block.
    //!
    //! \return BlockType
    //!
    //! \implements{19727292}
    BlockType getBlockType(void) const noexcept;

    //! \brief Returns the status of a block instance creation.
    //!   If block creation failed, the block is not usable and should not be
    //!   returned to the application.
    //!
    //! \return bool
    //! * true: if the creation of a block instance succeeded.
    //! * false: if the creation of a block instance failed.
    //!
    //! \implements{19727295}
    inline bool isInitSuccess(void) const noexcept
    {
        return initSuccess;
    }

    //! \brief Returns the handle mapped to this block instance.
    //!
    //! \return LwSciStreamBlock.
    //!
    //! \implements{19727298}
    LwSciStreamBlock getHandle(void) const noexcept;

    //! \brief Registers a new block instance by adding an entry in
    //!   blockRegistry, with LwSciStreamBlock as the key and BlockPtr as the
    //!   mapped value. If the blockRegistry has not yet been created, create it.
    //!
    //! \param [in] blkPtr: pointer to the block instance
    //!
    //! \return bool
    //! * true: if the BlockPtr is registered successfully.
    //! * false: if
    //!      - fail to allocate blockRegistry.
    //!      - Mutex locking fails or inserting to blockRegistry fails.
    //!
    //! \implements{19727316}
    static bool registerBlock(BlockPtr const& blkPtr) noexcept;

    //! \brief Returns the BlockPtr corresponding to the given
    //!   LwSciStreamBlock from the blockRegistry.
    //!
    //! \param [in] handle: Handle to the block
    //!
    //! \return BlockPtr: pointer to the block instance if found or
    //!                   null pointer if not found.
    //! * Panics if locking registryMutex failed.
    //!
    //! \if TIER4_SWAD
    //! \implements{19727319}
    //! \endif
    static BlockPtr getRegisteredBlock(LwSciStreamBlock const handle) noexcept;

    //! \brief Unregisters a block instance by removing the entry
    //!   corresponding to the input LwSciStreamBlock from blockRegistry.
    //!   If blockRegistry contained the last pointer to the block instance,
    //!   the block instance will be destroyed. Otherwise it will be destroyed
    //!   when the last function referencing it returns.
    //!
    //! \param [in] handle: LwSciStreamBlock
    //!
    //! \return void
    //! * Panics if locking registryMutex failed.
    //!
    //! \if TIER4_SWAD
    //! \implements{19727322}
    //! \endif
    static void removeRegisteredBlock(LwSciStreamBlock const handle) noexcept;

    //! \brief Destroys the Block instance. Only the smart pointers to the
    //!  Block instance should call it.
    //!
    //! \if TIER4_SWAD
    //! \implements{21698806}
    //! \endif
    ~Block(void) noexcept override;

protected:
    Block(void) noexcept                       = delete;
    Block(const Block&) noexcept               = delete;
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A8_4_8), "Bug 2745784")
    Block(Block&&) noexcept                    = delete;
    Block& operator=(const Block &) & noexcept = delete;
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A8_4_8), "Bug 2745784")
    Block& operator=(Block &&) & noexcept      = delete;

    //! \brief Parameterized constructor of Block.
    //!
    //! \param [in] type: BlockType
    //! \param [in] paramConnSrcTotal: maximum number of source connections
    //!   possible for this block.
    //!   Valid value: 0 to MAX_SRC_CONNECTIONS, default value: 1.
    //! \param [in] paramConnDstTotal: maximum number of destination
    //!   connections possible for this block.
    //!   Valid value: 0 to MAX_DST_CONNECTIONS, default value: 1.
    //! \param [in] paramConnSrcRemote: flag to indicate whether the block has
    //!   any possible remote source block (present in other process).
    //!   Default value: false.
    //! \param [in] paramConnDstRemote: flag to indicate whether the block has
    //!   any possible remote destination block (present in other process).
    //!   Default value: false.
    //!
    //! \return None
    //!
    //! \if TIER4_SWAD
    //! \implements{19727328}
    //! \endif
    Block(BlockType const type,
          uint32_t const paramConnSrcTotal=1U,
          uint32_t const paramConnDstTotal=1U,
          bool const paramConnSrcRemote=false,
          bool const paramConnDstRemote=false) noexcept;

    //! \cond TIER4_SWAD
    //! \brief Alias for std::unique_lock<std::mutex>
    //!
    using Lock = std::unique_lock<std::mutex>;
    //! \endcond

    //! \brief Creates an instance of Lock (std::unique_lock<std::mutex>) by
    //!   passing the blkMutex as an argument to its constructor. Before returning
    //!   the Lock instance back to the caller, locks blkMutex if it is not yet
    //!   locked.
    //!   This instance is expected to be used in a transitory fashion.
    //!   Callers create it, perform the operation and destroy it. When the
    //!   instance is destroyed, the destructor will also release the blkMutex.
    //!
    //! \param [in] locked: Indicates if blkMutex is alredy locked.
    //!
    //! \return An instance of Lock (std::unique_lock<std::mutex>)
    //! * Panics if locking blkMutex failed.
    //!
    //! \if TIER4_SWAD
    //! \implements{19727331}
    //! \endif
    Lock blkMutexLock(bool const locked=false) noexcept;

    //! \brief Sets the block initialization as failed,
    //!   used by deriving blocks to mark construction failures.
    //!
    //! \return void
    //!
    //! \implements{19727340}
    void setInitFail(void) noexcept;

    //! \cond TIER4_SWAD
    //! \brief Alias for clock time to wait for new event.
    //!
    using ClockTime = std::chrono::steady_clock::time_point;
    //! \endcond

    // Signal availability of new event
    //! \brief Wakes up any threads waiting for an event.
    //!
    //! \param [in] locked: Indicates whether blkMutex is held by the caller.
    //!
    //! \return void
    //! * Panics if fails to wake up any threads waiting on the associated
    //!   LwSciEventNotifier instance.
    //!
    //! \if TIER4_SWAD
    //! \implements{19727343}
    //! \endif
    void eventPost(bool const locked) noexcept;

    // Wait for new event signal (requires lock)
    //! \brief Waits for a new event only if the block is not bound
    //!  with a LwSciEventService for event signaling. If @a timeout is
    //!  NULL, it waits forever, otherwise it waits for the given @a timeout
    //!  period.
    //!
    //! \param [in,out] blockLock: Lock instance held by caller.
    //! \param [in] timeout: Time to wait for a new event.
    //!
    //! \return error code
    //! * true: if there's a new event.
    //! * false: if the block uses LwSciEventService for event signaling,
    //!   or it reaches @a timeout and no new event.
    //!
    //! \if TIER4_SWAD
    //! \implements{19727346}
    //! \endif
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A8_4_8), "Bug 2745784")
    bool eventWait(Lock& blockLock, ClockTime const* const timeout) noexcept;

    // Block type specific query for next pending event
    // Note: Lock is held throughout this call, so blocking operations
    //       and interactions with other blocks are not allowed.
    //! \brief Block-specific query for the next pending event.
    //!   Block class provides the default implementation for this
    //!   interface which always returns false. The derived classes
    //!   can override this interface as required.
    //!
    //! \param [out] event: Not used
    // TODO: Autosar doesn't like output parameters
    //!
    //! \return bool, always false
    //!
    //! \implements{19727349}
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A8_4_8), "Bug 2745784")
    virtual bool pendingEvent(LwSciStreamEventType& event) noexcept;

    //! \brief Triggers error events for any internal error.
    //!
    //! \param [in] err: LwSciError value
    //! \param [in] locked: Indicates whether blkMutex is held by the caller.
    //!
    //! \return void
    //!
    //! \if TIER4_SWAD
    //! \implements{19727352}
    //! \endif
    void setErrorEvent(LwSciError const err, bool const locked=false) noexcept;

    //! \brief Retrieve consumer info
    //!
    //! \return EndInfoVector: reference to vector of info structures
    //!   received from all consumers
    EndInfoVector const& consInfoGet(void) const noexcept
    {
        return consInfo;
    };

    //! \brief Retrieve producer info
    //!
    //! \return EndInfoVector: reference to vector of info structures
    //!   received from all producers
    EndInfoVector const& prodInfoGet(void) const noexcept
    {
        return prodInfo;
    };

    //! \brief Sets the consumer info vector and other associated information.
    //!
    //! Caller is expected to provide thread safety.
    //!
    //! \param [in] info: Info vector to copy.
    //!
    //! \return LwSciError, the completion code of this operation.
    //! - LwSciError_Success: Operation completed successfully.
    //! - LwSciError_AlreadyDone: Info has already been set.
    //! - LwSciError_InsufficientMemory: Unable to allocate memory for copy.
    LwSciError consInfoSet(
        EndInfoVector const& info) noexcept;

    //! \brief Sets the producer info vector and other associated information.
    //!
    //! Caller is expected to provide thread safety.
    //!
    //! \param [in] info: Info vector to copy.
    //!
    //! \return LwSciError, the completion code of this operation.
    //! - LwSciError_Success: Operation completed successfully.
    //! - LwSciError_AlreadyDone: Info has already been set.
    //! - LwSciError_InsufficientMemory: Unable to allocate memory for copy.
    LwSciError prodInfoSet(
        EndInfoVector const& info) noexcept;

    //! \brief List of possible things to be validated at entry points of
    //!   block interfaces (APIBlockInterface, SrcBlockInterface and
    //!   DstBlockInterface).
    //!
    //! \implements{18794841}
    enum class ValidateCtrl : uint8_t {
        //! \brief To validate whether the stream is in setup phase.
        SetupPhase,
        //! \brief To validate whether the stream is in safety phase.
        SafetyPhase,
        //! \brief To validate whether the stream is fully connected.
        Complete,
        //! \brief To validate whether LwSciStreamEventType_Connected
        //!   event is queried or not.
        CompleteQueried,
        //! \brief To validate the provided source connection index argument.
        SrcIndex,
        //! \brief To validate the provided destination connection index
        //!   argument.
        DstIndex,
    };

    //! \if TIER4_SWAD
    //! \brief Static const value to represent the default connection index
    //!   for the block instances with single source or destination
    //!   SafeConnection.
    static constexpr uint32_t singleConn { 0U };
    //! \endif

    //! \cond TIER4_SWAD
    //! \brief Alias for EnumBitset<ValidateCtrl>.
    //!
    using ValidateBits = EnumBitset<ValidateCtrl>;
    //! \endcond

    //! \brief Validates all requested states.
    //!   Provides standardized validation of states listed in ValidateCtrl
    //!   enum commonly checked at numerous entry points. This interface is
    //!   used directly in public-facing entry points.
    //!
    //! \param [in] bits: an instance of EnumBitset class.
    //! \param [in] connIndex: Optional connection index within the list of
    //!   connected block's list of SafeConnections.
    //!   Default value: 0.
    //!
    //! \return Error value
    //! * LwSciError_StreamNotConnected: the block is not fully connected to
    //!   producer and consumer(s) but trying to do an operation that can only
    //!   be performed when the stream is fully connected.
    //! * LwSciError_StreamBadSrcIndex: if an invalid connection index to a
    //!   source block was passed to this block instance.
    //! * LwSciError_StreamBadDstIndex: if an invalid connection index to a
    //!   destination block was passed to this block instance.
    //!
    //! \implements{19727355}
    LwSciError validateWithError(
                    ValidateBits const& bits,
                    uint32_t const connIndex=singleConn) const noexcept;

    //! \brief Performs requested validation.
    //!   Provides standardized validation of states listed by ValidateCtrl
    //!   enum commonly checked at numerous entry points. This interface is
    //!   used for inter-block entry points, wrapping Block::validateWithError()
    //!   interface.
    //!
    //! <b>Parameters</b>
    //!  - As specified for LwSciStream::Block::validateWithError
    //!
    //! \return bool
    //! * true: if Block::validateWithError() interface returns LwSciError_Success
    //! * false: if Block::validateWithError() interface returns error code other
    //!   than LwSciError_Success.
    //!
    //! \if TIER4_SWAD
    //! \implements{19727358}
    //! \endif
    bool validateWithEvent(
            ValidateBits const& bits,
            uint32_t const connIndex=singleConn) noexcept;

    // TODO: Rename these disconnect functions to make them distinguish from
    // 'srcDisconnect' and 'dstDisconnect'.

    //! \brief Disconnects the source block.
    //!
    //! \if TIER4_SWUD
    //! <b>Sequence of operations</b>
    //!  - Calls SafeConnection::disconnect() interface of the corresponding
    //!    source SafeConnection and sets the flag which indicates a complete
    //!    stream to false.
    //! \endif
    //!
    //! \note This interface assumes that the @a srcIndex is already validated
    //!  by the caller functions, so this is not validated again by this interface.
    //!
    //! \param [in] srcIndex: Index in the list of source SafeConnections.
    //!   Valid value: [0 to connSrcTotal-1]
    //!
    //! \return void
    //!
    //! \if TIER4_SWAD
    //! \implements{19727361}
    //! \endif
    //!
    //! \if TIER4_SWUD
    //! \implements{18794937}
    //! \endif
    void disconnectSrc(uint32_t const srcIndex=singleConn) noexcept
    {
        LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A5_2_2), "Proposed TID-1251")
        assert(srcIndex < connSrcTotal);
        assert(!connSrcRemote);
        LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A5_2_2))
        LWCOV_ALLOWLIST_BEGIN(LWCOV_CERTCPP(CTR50_CPP), "TID-1397")
        LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A4_7_1), "Bug 2738489")
        connSrc[srcIndex].disconnect();
        LWCOV_ALLOWLIST_END(LWCOV_CERTCPP(CTR50_CPP))
        LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_1_1), "TID-300")
        streamDone = false;
    };

    //! \brief Disconnects the destination block.
    //!
    //! \if TIER4_SWUD
    //! <b>Sequence of operations</b>
    //!  - Calls SafeConnection::disconnect() interface of the corresponding
    //!   destination SafeConnection and sets the flag which indicates a
    //!   complete stream to false.
    //! \endif
    //!
    //! \note This interface assumes that the @a dstIndex is already validated
    //!  by the caller functions, so this is not validated again by this interface.
    //!
    //! \param [in] dstIndex: Index in the list of destination SafeConnections.
    //!   Valid value: [0 to connDstTotal-1]
    //! \param [in] allOutputsdisconnected: Indicates that all downstream outputs
    //!   are disconnected. By default set to true.
    //!
    //! \return void
    //!
    //! \if TIER4_SWAD
    //! \implements{19727364}
    //! \endif
    //!
    //! \if TIER4_SWUD
    //! \implements{18794940}
    //! \endif
    void disconnectDst(uint32_t const dstIndex=singleConn,
         bool const allOutputsdisconnected=true) noexcept
    {
        LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A5_2_2), "Proposed TID-1251")
        assert(dstIndex < connDstTotal);
        assert(!connDstRemote);
        LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A5_2_2))
        LWCOV_ALLOWLIST_BEGIN(LWCOV_CERTCPP(CTR50_CPP), "TID-1397")
        LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A4_7_1), "Bug 2738489")
        connDst[dstIndex].disconnect();
        LWCOV_ALLOWLIST_END(LWCOV_CERTCPP(CTR50_CPP))
        LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_1_1), "TID-300")
        if (allOutputsdisconnected) {
            streamDone = false;
        }
    };

    //! \brief Schedules a LwSciStreamEventType_Disconnected event if not
    //!   already done and wakes up any threads waiting for event.
    //!
    //! \if TIER4_SWUD
    //! <b>Sequence of operations</b>
    //!  - Calls Block::eventPost() interface to wake up threads waiting for
    //!    event.
    //! \endif
    //!
    //! \return void
    //!
    //! \if TIER4_SWAD
    //! \implements{19727367}
    //! \endif
    //!
    //! \if TIER4_SWUD
    //! \implements{18794943}
    //! \endif
    void disconnectEvent(void) noexcept
    {
        if (!disconnectDone) {
            bool expected { false };
            LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A5_1_1), "TID-300")
            if (discEvent.compare_exchange_strong(expected, true)) {
                disconnectDone = true;
                eventPost(false);
            }
            LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A5_1_1))
        }
    }

    //! \brief Gets a new instance of SrcAccess.
    //!
    //! \if TIER4_SWUD
    //! <b>Sequence of operations</b>
    //!  - Calls SafeConnection::getAccess() interface of the source
    //!   SafeConnection at @a srcIndex through which the interface(s)
    //!   of BlockWrap<SrcBlockInterface> can be accessed.
    //! \endif
    //!
    //! \note This interface assumes that the @a srcIndex is already validated
    //!  by the caller functions, so this is not validated again by this interface.
    //!
    //! \param [in] srcIndex: Index of source SafeConnection.
    //!   Valid value: [0 to connSrcTotal-1]
    //!
    //! \return
    //! * SrcAccess: Instance of SafeConnection<SrcBlockInterface>::Access.
    //!
    //! \if TIER4_SWAD
    //! \implements{19727370}
    //! \endif
    //!
    //! \if TIER4_SWUD
    //! \implements{18794946}
    //! \endif
    SrcAccess getSrc(uint32_t const srcIndex=singleConn) noexcept
    {
        LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A5_2_2), "Proposed TID-1251")
        assert(srcIndex < connSrcTotal);
        assert(!connSrcRemote);
        LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A5_2_2))
        LWCOV_ALLOWLIST_BEGIN(LWCOV_CERTCPP(CTR50_CPP), "TID-1397")
        LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A4_7_1), "Bug 2738489")
        return connSrc[srcIndex].getAccess();
        LWCOV_ALLOWLIST_END(LWCOV_CERTCPP(CTR50_CPP))
    };

    //! \brief Gets a new instance of DstAccess.
    //!
    //! \if TIER4_SWUD
    //! <b>Sequence of operations</b>
    //!  - Calls SafeConnection::getAccess() interface of the destination
    //!   SafeConnection at @a dstIndex through which the interface(s) of
    //!   BlockWrap<DstBlockInterface> can be accessed.
    //! \endif
    //!
    //! \note This interface assumes that the @a dstIndex is already validated
    //!  by the caller functions, so this is not validated again by this interface.
    //!
    //! \param [in] dstIndex: Index of the destination SafeConnection.
    //!   Valid value: [0 to connDstTotal-1]
    //!
    //! \return
    //! * DstAccess: Instance of SafeConnection<DstBlockInterface>::Access.
    //!
    //! \if TIER4_SWAD
    //! \implements{19727373}
    //! \endif
    //!
    //! \if TIER4_SWUD
    //! \implements{18794949}
    //! \endif
    DstAccess getDst(uint32_t const dstIndex=singleConn) noexcept
    {
        LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A5_2_2), "Proposed TID-1251")
        assert(dstIndex < connDstTotal);
        assert(!connDstRemote);
        LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A5_2_2))
        LWCOV_ALLOWLIST_BEGIN(LWCOV_CERTCPP(CTR50_CPP), "TID-1397")
        LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A4_7_1), "Bug 2738489")
        return connDst[dstIndex].getAccess();
        LWCOV_ALLOWLIST_END(LWCOV_CERTCPP(CTR50_CPP))
    };

    //! \brief Returns the connection status of this block instance.
    //!
    //! \return bool
    //!  * true: if the block has complete connection paths to producer and
    //!    consumer(s).
    //!  * false: if connections are pending
    //!
    //! \implements{19727376}
    bool connComplete(void) const noexcept
    {
        return streamDone;
    };

    //! \brief Returns the configuration status of the block instance.
    //!
    //! \return bool
    //!  * true: if the configuration options are locked.
    //!  * false otherwise.
    bool configComplete(void) const noexcept
    {
        return configOptLocked;
    };

private:

    bool assignHandle(void) noexcept;

    //! \brief Checks if the blockRegistry has been allocated.
    //!   If not, allocates it.
    //!
    //! \return bool
    //! * true: if blockRegistry is ready for use.
    //! * false: if blockRegistry is unavailable.
    static bool prepareRegistry(void) noexcept;

private:
    //! \cond TIER4_SWAD

    //! \brief Type of the block. It is initialized to one of the types
    //!   defined by BlockType enum when a block instance is created.
    BlockType const blkType;

    //! \brief Handle to a block instance. It is initialized to the current
    //!   value of the nextHandle member when a block instance is created.
    LwSciStreamBlock blkHandle;

    //! \brief Mutex to protect conlwrrent access to block data. It is
    //!   initialized when a block instance is created.
    std::mutex blkMutex;

    //! \brief Vector of endpoint information from the consumer(s). It is
    //!   initialized to empty and populated as connections to the consumer(s)
    //!   are established. It is emptied once the initialization phase is done.
    EndInfoVector   consInfo;

    //! \brief Vector of endpoint information from the producer. It is
    //!   initialized to empty and populated after consumer information has
    //!   all made its way to the producer and the producer sends its
    //!   information downstream. It is emptied once initialization phase
    //!   is done.
    EndInfoVector   prodInfo;

    //! \brief Flag indicating whether consumer info is pending to send
    //!   upstream. Initialized at construction to false and set to true
    //!   when the info is set.
    bool            consInfoMsg;

    //! \brief Flag indicating whether producer info is pending to send
    //!   downstream. Initialized at construction to false and set to true
    //!   when the info is set.
    bool            prodInfoMsg;

    // TODO: These will be replaced with vectors. Not all blocks have
    //       both an input and an output. Some will have more than one.
    //! \brief Array of source SafeConnection(s) to other blocks. These
    //!   connections are initialized by successful calls to
    //!   Block::connSrcInitiate() and Block::connSrcComplete() interfaces.
    //!   These connections are de-initialized by a successful call to
    //!   Block::connSrcCancel() interface.
    std::array<SrcConnection,MAX_SRC_CONNECTIONS> connSrc;

    //! \brief Array of destination SafeConnection(s) to other blocks. These
    //!   connections are initialized by successful calls to
    //!   Block::connDstInitiate() and Block::connDstComplete() interfaces.
    //!   These connections are de-initialized by a successful call to
    //!   Block::connDstCancel() interface.
    std::array<DstConnection,MAX_DST_CONNECTIONS> connDst;

    //! \brief Total number of applicable source connections for a block
    //!   instance. It is initialized as one for all the block types except
    //!   for producer when a block instance is created. For producer block,
    //!   it is initialized as 0.
    uint32_t connSrcTotal;

    //! \brief Total number of applicable destination connections for a block
    //!   instance. It is initialized as one for all the block types except
    //!   for multicast and consumer when a block instance is created. For
    //!   multicast block, it is initialized to the number of consumers in
    //!   the stream and it cannot be more than MAX_DST_CONNECTIONS. For
    //!   consumer block, it is initialized as 0.
    uint32_t connDstTotal;

    //! \brief Flag to indicate whether a block instance has a source
    //!   connection from another process. It is initialized as false for
    //!   all the block types except for ipcDst when a block instance is
    //!   created. For ipcDst block, this flag is initialized as true and
    //!   connSrc member is never used.
    bool connSrcRemote;

    //! \brief Flag to indicate whether a block instance has a destination
    //!   connection from another process. It is initialized as false for
    //!   all the block types except for ipcSrc when a block instance is
    //!   created. For ipcSrc block, this flag is initialized as true and
    //!   connDst member is never used.
    bool connDstRemote;
    //! \endcond

    //! \cond TIER4_SWAD
    //! \brief Flag indicating the block configuration options are locked
    //!   when the first connection is triggered. Initialized to false at
    //!   block creation and set to true in Block::finalizeConfigOptions().
    std::atomic<bool> configOptLocked;

    //! \brief Flag indicating stream is fully connected and operations
    //!   may proceed. It is initialized as false when a block instance is
    //!   created. It is set to true only when this block instance has
    //!   complete paths to the producer and all consumer(s). It is reset
    //!   to false when disconnecton is initiated in Block::disconnectSrc()
    //!   or Block::disconnectDst() interface.
    bool streamDone;
    //! \endcond

    //! \cond
    // Not exported to doxygen. See Readme-doxygen-criteria.txt for details.
    //! \brief Flag indicating disconnect has been signaled.
    //! It is initialized as false. It is set to true once and only once
    //! in Block::disconnectEvent() interface when
    //! LwSciStreamEventType_Disconnected is scheduled.
    bool disconnectDone;
    //! \endcond

    //! \cond TIER4_SWUD
    //! \brief Conditional variable to wait for event to arrive.
    std::condition_variable  eventCond;
    //! \endcond

    //! \cond
    // Not exported to doxygen. See Readme-doxygen-criteria.txt for details.
    //! \brief When internal error events are triggered using
    //! Block::setErrorEvent() interface, it holds the error code of the first
    //! such error. All subsequent errors are ignored until the value is
    //! cleared when the error code is queried, with the assumption that they
    //! are side effects of the first one. It is initialized to
    //! LwSciError_Success when a block instance is created.
    LwSciError               internalErr;

    //! \brief Flag to indicate LwSciStreamEventType_Error event is pending.
    //! It is initialized as false. It is set to true when internalErr
    //! transitions from LwSciError_Success to anything else. It's cleared
    //! when LwSciStreamEventType_Error event is retrieved.
    std::atomic<bool>        errEvent;

    // Not exported to doxygen. See Readme-doxygen-criteria.txt for details.
    //! \brief Flag to indicate LwSciStreamEventType_Connected event is pending.
    //! It is initialized as false. It is set to true when
    //! LwSciStreamEventType_Connected event is ready for query. And it is
    //! reset to false when LwSciStreamEventType_Connected event is queried.
    // TODO: Can probably avoid use of atomic
    std::atomic<bool>        connEvent;

    // Not exported to doxygen. See Readme-doxygen-criteria.txt for details.
    //! \brief Flag to indicate LwSciStreamEventType_DisConnected event is pending.
    //! It is initialized as false. It is set to true when
    //! LwSciStreamEventType_DisConnected event is ready for query. And it is
    //! reset to false when LwSciStreamEventType_DisConnected event is queried.
    // TODO: Can probably avoid use of atomic
    std::atomic<bool>        discEvent;
    //! \endcond

    //! \cond TIER4_SWAD
    //! \brief Flag to indicate whether a block instance is initialized
    //!   successfully or not. It is initialized to true when a block
    //!   instance is created if all constructor operations succeed, and
    //!   false if EndInfoTrack::initGetError() or Block::assignHandle()
    //!   indicates a failure. It may be set to false by a derived class
    //!   with the call to Block::setInitFail() interface if the later
    //!   initialization process fails.
    bool initSuccess;
    //! \endcond

    //! \cond TIER4_SWUD
    //! \brief Enum representing the event-signaling mode for a block.
    //!
    //! \implements{21706237}
    enum class EventSignalMode : uint8_t {
        //! \brief Default value representing the initial state when a block
        //!  is created.
        None,
        //! \brief Notifies the application for a new event on the block via
        //!  LwSciEventService.
        EventService,
        //! \brief No explicit signaling to application for a new event on
        //!  the block, application will wait for new event for the given
        //!  timeout with LwSciStreamBlockEventQuery() call.
        Internal
    };

    //! \brief Flag to indicate the block's event-signaling mode.
    //!  It is initialized to EventSignalMode::None when a block instance is
    //!  created. It is set to either EventSignalMode::EventService or
    //!  EventSignalMode::Internal once the block' event-signaling mode is
    //!  decided.
    std::atomic<EventSignalMode> signalMode;

    //! \brief Pointer to the LwSciLocalEvent object via which the block
    //!  signals the associated LwSciEventNotifier.
    //!  It is initialized to null when a block instance is created. It
    //!  references an LwSciLocalEvent object after a successful call to
    //!  Block::eventNotifierSetup().
    //!  The object it references will be deleted in Block's destructor.
    LwSciLocalEvent* eventServiceEvent;
    //! \endcond

    //! \cond TIER4_SWAD
    //! \brief A global registry which is used to store the mapping of block
    //!   handles with their corresponding instances. A block's handle and
    //!   its instance will be registered to this map when a block instance
    //!   is created by calling Block::registerBlock() interface. The block
    //!   instance will be retrieved from this map by providing the block
    //!   handle when required by calling Block::getRegisteredBlock()
    //!   interface. A block instance will be removed from this map by calling
    //!   Block::removeRegisteredBlock() interface.
    static std::unique_ptr<HandleBlockPtrMap> blockRegistry;
    //! \endcond

    //! \cond TIER4_SWUD
    //! \brief A global counter variable which keeps track of handle to be
    //!   assigned to the next registered block. Initial value for this
    //!   counter is one and it will be incremented after the current handle
    //!   value is assigned to a block instance.
    static LwSciStreamBlock nextHandle;
    //! \endcond

    //! \cond TIER4_SWAD
    //! \brief A global mutex to prevent conlwrrent access to blockRegistry
    //!   and nextHandle members.
    static std::mutex registryMutex;
    //! \endcond

protected:
    //! \brief Retrieves the number of consumers in this block's subtree.
    //!   Only for use internally by functions which know the connection
    //!   is completed and therefore the value has been filled in.
    //!
    //! \return size_t, the consumer count
    size_t consumerCountGet(void) const noexcept
    {
        return consumerCount;
    };

    //! \brief Sets the number of elements determined by the pool that
    //!   serves this block.
    //!
    //! \param [in] count: The number of buffers per packet.
    //!
    //! \return void
    void elementCountSet(size_t const count) noexcept
    {
        elementCount = count;
        pktDesc.elementCount = count;
    };

    //! \brief Retrieves the number of elements determined by the pool that
    //!   serves this block.
    //!
    //! \return size_t, the element count
    size_t elementCountGet(void) const noexcept
    {
        return elementCount;
    };

    //! \brief Set the Packet::Desc to be used for creation of all Packets,
    //!   by moving the contents of the input Packet::Desc to the Block's
    //!   instance.
    //!   Only called by derived Block constructors.
    //!
    //! \param [in,out] desc: Reference to Packet::Desc object.
    //!
    //! \return void
    //!
    //! \implements{20698788}
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A8_4_8), "Bug 2745784")
    void pktDescSet(Packet::Desc&& desc) noexcept
    {
        pktDesc = std::move(desc);
        pktUsed = true;
    };

    //! \brief Creates a new Packet instance with the Block::pktDesc, copies
    //!   data from @a origPacket if provided, and inserts it into PacketMap.
    //!
    //! \param [in] handle: LwSciStreamPacket
    //! \param [in] origPacket: Packet from which to copy relevant definitions,
    //!   if provided.
    //! \param [in] cookie: LwSciStreamCookie, this argument is optional,
    //!   which will be set to LwSciStreamCookie_Ilwalid if not provided.
    //!
    //! \return error code
    //! * LwSciError_Success: If Packet is created successfully.
    //! * LwSciError_AlreadyInUse: If the @a cookie is already used for
    //!   another packet.
    //! * LwSciError_InsufficientMemory: If unable to create a new packet
    //!   instance.
    //! * LwSciError_StreamInternalError: If there's already a packet instance
    //!   in PacketMap with the same LwSciStreamPacket.
    //! * Any error encountered by Packet constructor.
    //! * Any error returned by Packet::defineCopy().
    //!
    //! \if TIER4_SWAD
    //! \implements{19727379}
    //! \endif
    LwSciError pktCreate(LwSciStreamPacket const handle,
                         Packet const* const     origPacket=nullptr,
                         LwSciStreamCookie const cookie=
                         LwSciStreamCookie_Ilwalid) noexcept;

    //! \brief Removes a packet instance with the given LwSciStreamPacket
    //!   from PacketMap under thread protection provided by
    //!   Block::blkMutexLock().
    //!
    //! \param [in] handle: LwSciStreamPacket
    //! \param [in] locked: Indicates whether blkMutex is held by the caller.
    //!
    //! \return void
    //!
    //! \implements{19727382}
    void pktRemove(LwSciStreamPacket const handle,
                   bool const locked=false) noexcept;

    //! \brief Searches PacketMap for a packet instance with the given
    //!   LwSciStreamPacket under thread protection provided by
    //!   Block::blkMutexLock() and returns the smart pointer to it if found.
    //!
    //!
    //! \param [in] handle: LwSciStreamPacket
    //! \param [in] locked: Indicates whether blkMutex is held by the caller.
    //!
    //! \return PacketPtr: Pointer to the matched packet instance if found,
    //!                    or null pointer if not found.
    //!
    //! \implements{19727385}
    PacketPtr pktFindByHandle(LwSciStreamPacket const handle,
                              bool const locked=false) noexcept;

    //! \brief Searches PacketMap for a packet instance with the given
    //!   LwSciStreamCookie and returns the smart pointer to it if found.
    //!
    //! \param [in] cookie: LwSciStreamCookie
    //! \param [in] locked: Indicates whether blkMutex is held by the caller.
    //!
    //! \return PacketPtr: Pointer to the matched packet instance if found,
    //!                    or null pointer if not found.
    //!
    //! \if TIER4_SWAD
    //! \implements{19727388}
    //! \endif
    PacketPtr pktFindByCookie(LwSciStreamCookie const cookie,
                              bool const locked=false) noexcept;

    //! \brief Retrieves the next packet with the specified criteria, if any.
    //!
    //! \param [in] criteria: Packet function to ilwoke to check criteria.
    //! \param [in] locked: Indicates whether blkMutex is held by the caller.
    //!
    //! \return PacketPtr, a packet that meets the criteria, if any.
    //!
    //! \if TIER4_SWAD
    //! \implements{19727391}
    //! \endif
    PacketPtr pktPendingEvent(Packet::Pending const criteria,
                              bool const locked) noexcept;

    //! \brief Clear all packet instances in PacketMap
    //!
    //! \return void
    void clearPacketMap(void)
    {
        pktMap.clear();
    };

    //! \brief Get the PacketMap
    //!
    //! \return PacketMap
    //!
    //! \implements{}
    PacketMap& getPacketMap(void)
    {
        return pktMap;
    };

private:
    //! \cond TIER4_SWAD
    //! \brief Number of consumers in this block's subtree.
    //!   It is initialized when connection is completed.
    size_t           consumerCount;

    //! \brief The number of elements determined by the pool that serves
    //!   this block.
    size_t           elementCount;

    //! \brief Flag indicating whether the block uses the packet map.
    //!   It is initialized to false during construction and set to true
    //!   if pktDescSet() is called.
    bool             pktUsed;

    //! \brief Packet::Desc which is used while creating packet(s) for the
    //!   block instance. It is initialized as required by calling
    //!   Block::pktDescSet() when a block instance is created.
    Packet::Desc     pktDesc;

    //! \brief Map tracking all packet instances created by the block instance.
    //!   When a packet instance is created by calling Block::pktCreate()
    //!   interface, it is added to this map with the corresponding
    //!   LwSciStreamPacket as key value. Packet instances from this map can be
    //!   looked up by calling Block::pktFindByHandle() and
    //!   Block::pktFindByCookie() interfaces. A packet instance can be removed
    //!   from this map by calling Block::pktRemove() interface.
    PacketMap        pktMap;
    //! \endcond

protected:

    //
    // Init/runtime phase transition functions
    //

    //! \brief Indicates block's packet setup is done and checks if
    //!   block is ready for runtime.
    //!
    //! \return void
    void phasePacketsDoneSet(void) noexcept
    {
        phasePacketsDone = true;
        phaseCheck();
    };

    //! \brief Indicates block's producer sync setup is done and checks if
    //!   block is ready for runtime.
    //!
    //! \return void
    void phaseProdSyncDoneSet(void) noexcept
    {
        phaseProdSyncDone = true;
        phaseCheck();
    };

    //! \brief Indicates block's consumer sync setup is done and checks if
    //!   block is ready for runtime.
    //!
    //! \return void
    void phaseConsSyncDoneSet(void) noexcept
    {
        phaseConsSyncDone = true;
        phaseCheck();
    };

    //! \brief Indicates a downstream block is ready for runtime phase
    //!   and checks if block is ready for runtime.
    //!
    //! \param [in] dstIndex: Index of output connection.
    //!
    //! \return void
    void phaseDstDoneSet(
        uint32_t const dstIndex) noexcept
    {
        static_cast<void>(dstIndex);
        // TODO: Use BranchTrack to make sure each destination only calls once,
        phaseDstRecv++;
        phaseCheck();
    };

    //! \brief Indicates an upstream block has changed to runtime phase
    //!   and switches this block to runtime phase.
    //!
    //! \param [in] srcIndex: Index of input connection.
    //!
    //! \return void
    void phaseSrcDoneSet(
        uint32_t const srcIndex) noexcept
    {
        static_cast<void>(srcIndex);
        // TODO: Need something to make sure each source only calls once,
        phaseChange();
    };

    //! \brief Checks whether all conditions for this block to be ready
    //!   for runtime phase are met, and if so informs the next block.
    //!
    //! \return void
    void phaseCheck(void) noexcept;

    //! \brief Initiates phase change in this block.
    //!
    //! \return void
    void phaseChange(void) noexcept;

    //! \brief Default function to transmit runtime readiness message.
    //!
    //! \return void
    virtual void phaseSendReady(void) noexcept;

    //! \brief Default function to transmit runtime change message.
    //!
    //! \return void
    virtual void phaseSendChange(void) noexcept;

    //! \brief Sole implementation of SrcBlockInterface::dstRecvPhaseReady,
    //!   which records the readiness message from the downstream block and
    //!   performs readiness check for this block.
    //!
    //! \copydetails LwSciStream::SrcBlockInterface::dstRecvPhaseReady
    //!
    //! \return void
    void dstRecvPhaseReady(
        uint32_t const dstIndex) noexcept final;

    //! \brief Sole implementation of DstBlockInterface::srcRecvPhaseChange,
    //!   which triggers the phase change in this block and passes the
    //!   message downstream.
    //!
    //! \copydetails LwSciStream::DstBlockInterface::srcRecvPhaseChange
    //!
    //! \return void
    void srcRecvPhaseChange(
        uint32_t const srcIndex) noexcept final;

    //! \brief Queries whether runtime phase has begun.
    //!
    //! \param [in] queried: If set, also requires phase event to have been
    //!   queried before returning true.
    //!
    //! \return bool, whether the phase change conditions are met.
    bool phaseRuntimeGet(bool const queried) const noexcept
    {
        return (phaseRuntime && !(queried && phaseEvent));
    };

private:

    //
    // Init/runtime phase transition state
    //

    //! Flag indicating block's packet setup operations are done.
    //!   Initialized to false at construction.
    bool                    phasePacketsDone;

    //! Flag indicating block's producer sync setup operations are done.
    //!   Initialized to false at construction.
    bool                    phaseProdSyncDone;

    //! Flag indicating block's consumer sync setup operations are done.
    //!   Initialized to false at construction.
    bool                    phaseConsSyncDone;

    //! Counter of number of phase change readiness messages from downstream.
    //!   Initialized to zero at construction.
    std::atomic<uint32_t>   phaseDstRecv;

    //! Flag indicating phase change readiness has been sent upstream.
    //!   Initialized to false at construction.
    std::atomic<bool>       phaseSrcSend;

    //! Flag indicating pending phase change event.
    //!   Initialized to false at construction.
    bool                    phaseEvent;

    //! Flag indicating now in runtime phase.
    //!   Initialized to false at construction.
    bool                    phaseRuntime;
};

} // namespace LwSciStream

#endif // BLOCK_H
