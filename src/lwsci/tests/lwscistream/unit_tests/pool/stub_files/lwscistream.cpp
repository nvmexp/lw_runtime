// \file
// \brief LwSciStream public APIs definition.
//
// \copyright
// Copyright (c) 2018-2021 LWPU Corporation. All rights reserved.
//
// LWPU Corporation and its licensors retain all intellectual property and
// proprietary rights in and to this software, related documentation and any
// modifications thereto. Any use, reproduction, disclosure or distribution
// of this software and related documentation without an express license
// agreement from LWPU Corporation is strictly prohibited.
#include <cstdint>
#include <array>
#include <new>
#include <unordered_map>
#include <memory>
#include <atomic>
#include <vector>
#include <functional>
#include <limits>
#include "covanalysis.h"
#include "sciwrap.h"
#include "lwscierror.h"
#include "lwscibuf.h"
#include "lwscisync.h"
#include "lwsciipc.h"
#include "lwsciipc_internal.h"
#include "lwscievent.h"
#include "lwscistream_types.h"
#include "lwscistream_api.h"
#include "apiblockinterface.h"
#include "srcblockinterface.h"
#include "dstblockinterface.h"
#include "lwscistream_common.h"
#include "lwscicommon_os.h"
#include "producer.h"
#include "consumer.h"
#include "pool.h"
#include "automaticpool.h"
#include "queue.h"
#include "multicast.h"
#include "ipcsrc.h"
#include "ipcdst.h"
#include "c2csrc.h"
#include "c2cdst.h"
#include "limiter.h"
#include "returnsync.h"
#include "presentsync.h"

using LwSciStream::Producer;
using LwSciStream::Consumer;
using LwSciStream::BlockType;
using LwSciStream::Block;
using LwSciStream::BlockPtr;
using LwSciStream::Pool;
using LwSciStream::AutomaticPool;
using LwSciStream::Mailbox;
using LwSciStream::Fifo;
using LwSciStream::MultiCast;
using LwSciStream::IpcSrc;
using LwSciStream::IpcDst;
using LwSciStream::Limiter;
using LwSciStream::C2CSrc;
using LwSciStream::C2CDst;
using LwSciStream::ReturnSync;
using LwSciStream::PresentSync;
using LwSciStream::EventSetupRet;
using LwSciStream::InfoPtr;

extern "C" {

LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A15_4_4), "Bug 3127842")
LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(M0_1_10), "LwSciStream-ADV-AUTOSARC++14-006")

// \brief Connect two stream blocks.
//
// Connects an available output of one block with an available input
// of another block.
//
// Each input and output can only have one connection. A stream is fully
// connected when all inputs and outputs of all blocks in the stream have a
// connection.
//
// <b>Preconditions</b>
//
// - The upstream block has an available output connection
// - The downstream block has an available input connection
//
// <b>Actions</b>
//
// Establish a connection between the two blocks.
//
// <b>Postconditions</b>
//
// - When the block has a complete chain of connections to the producer,
//   it will receive a ConnectUpstream event.
// - When the block has a complete chain of connections to all consumers,
//   it will receive a ConnectDownstream event.
//
// \param upstream (in) handle of the upstream block
// \param downstream (in) handle of the downstream block
//
// \return status code
// - LwSciError_Success: The connection was made successfully.
// - LwSciError_BadParameter: Upstream or downstream is not a valid block.
// - LwSciError_IlwalidState: Upstream and downstream are already connected.
// - LwSciError_AccessDenied: Upstream or downstream does not allow explicit
//     connection via LwSciStreamBlockConnect.
// - LwSciError_StreamInternalError:
//     Internal system errors such as mutex-locking failure.

/**
 *  <b>Sequence of operations</b>
 *    - Retrieves the upstream block instance referenced by the given @a upstream
 *      argument by calling Block::getRegisteredBlock() interface.
 *    - Retrieves the downstream block instance referenced by the given @a downstream
 *      argument by calling Block::getRegisteredBlock() interface.
 *    - Gets the destination block instance of the upstream block if any by calling its
 *      getOutputConnectPoint() interface and considers it as upstream block instance
 *      to be connected.
 *    - Gets the source block instance of the downstream block if any by calling
 *      its getInputConnectPoint() interface and considers it as downstream block instance
 *      to be connected.
 *    - Initializes the available destination SafeConnection of the upstream block instance
 *      by calling its Block::connDstInitiate() interface.
 *    - Initializes the available source SafeConnection of the downstream block instance
 *      by calling its Block::connSrcInitiate() interface.
 *    - If initialization is successful, completes the destination SafeConnection of the
 *      upstream block instance and source SafeConnection of the downstream block
 *      instance by calling their Block::connDstComplete() and Block::connSrcComplete()
 *      interfaces respectively.
 *    - Call Block::finalizeConfigOptions() to lock the configuration options of the
 *      upstream and downstream blocks.
 */
LwSciError LwSciStreamBlockConnect(
    LwSciStreamBlock const upstream,
    LwSciStreamBlock const downstream)
{
    LwSciError err;

    // Look up the provided handles
    BlockPtr upstreamPtr { Block::getRegisteredBlock(upstream) };
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_1_1), "TID-300")
    if (nullptr == upstreamPtr) {
        return LwSciError_BadParameter;
    }
    BlockPtr const origUpstreamPtr{ upstreamPtr };

    BlockPtr downstreamPtr { Block::getRegisteredBlock(downstream) };
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_1_1), "TID-300")
    if (nullptr == downstreamPtr) {
        return LwSciError_BadParameter;
    }
    BlockPtr const origDownstreamPtr { downstreamPtr };

    // For producer/consumer, actually connect to the fifo/queue
    err = upstreamPtr->getOutputConnectPoint(upstreamPtr);
    if (LwSciError_Success != err) {
        return err;
    }

    err = downstreamPtr->getInputConnectPoint(downstreamPtr);
    if (LwSciError_Success != err) {
        return err;
    }

    // Initiate the connections to each other, obtaining connection index
    LwSciStream::IndexRet const
        dstReserved { upstreamPtr->connDstInitiate(downstreamPtr) };

    if (LwSciError_Success != dstReserved.error) {
        return dstReserved.error;
    }

    LwSciStream::IndexRet const
        srcReserved { downstreamPtr->connSrcInitiate(upstreamPtr) };

    // If failed to initiate connection to the src, release the reservation.
    if (LwSciError_Success != srcReserved.error) {
        upstreamPtr->connDstCancel(dstReserved.index);
        return srcReserved.error;
    }

    // If successful on both ends, complete the connection, providing index
    //   of other side
    upstreamPtr->connDstComplete(dstReserved.index, srcReserved.index);
    downstreamPtr->connSrcComplete(srcReserved.index, dstReserved.index);

    // Lock the config options of the upstream and downstream blocks.
    origDownstreamPtr->finalizeConfigOptions();
    origUpstreamPtr->finalizeConfigOptions();

    // Allow connection message to continue flowing upstream if it was waiting
    origDownstreamPtr->consInfoFlow();

    return LwSciError_Success;
}

// \brief Create a stream producer block.
//
// Create a block for the producer end of a stream. All streams
// require one producer block. Producer blocks have one output
// connection and no input connections.
//
// Once the stream is fully connected, this block can be used to retrieve
// consumer requirements and finalize producer settings.
//
// <b>Postconditions</b>:
// * The block is ready to be connected to other stream blocks.
// * The block can be monitored for events.
//
// \param [out] producer: handle for new producer blok.
//
// \return status code
// - LwSciError_Success: The block was set up successfully.
// - LwSciError_BadParameter: If 'pool' is invalid or
//   if the output parameter 'producer' is a nullptr.
// - LwSciError_InsufficientMemory: The block is not created.
// - LwSciError_StreamInternalError: The block can't be initialized properly.
//     Internal system errors such as mutex-locking failure.

/**
 *  <b>Sequence of operations</b>
 *    - Retrieves the pool block instance referenced by the given LwSciStreamBlock
 *      by calling Block::getRegisteredBlock() interface.
 *    - Creates a producer block instance and checks whether the initialization
 *      is successful by calling its Block::isInitSuccess() interface.
 *    - Registers the producer block instance by calling Block::registerBlock()
 *      interface.
 *    - Connects the pool block instance with the producer block
 *      instance by calling its Producer::BindPool() interface.
 *    - Call Block::finalizeConfigOptions() to lock the configuration options of
 *      the pool block.
 *    - Retrieves the LwSciStreamBlock referencing the created producer block
 *      instance by calling its Block::getHandle() interface and returns it.
 */
LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A8_4_8), "Approved TID-281")
LwSciError LwSciStreamProducerCreate(
    LwSciStreamBlock const pool,
    LwSciStreamBlock *const producer)
{
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_1_1), "TID-300")
    if (nullptr == producer) {
        return LwSciError_BadParameter;
    }

    BlockPtr const poolPtr { Block::getRegisteredBlock(pool) };

    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_1_1), "TID-300")
    if (nullptr == poolPtr) {
        return LwSciError_BadParameter;
    }

    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A20_8_4), "Bug 3255701")
    std::shared_ptr<Producer> obj {};
    try {
        LWCOV_ALLOWLIST_BEGIN(LWCOV_CERTCPP(ERR59_CPP), "TID-1475")
        LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A15_1_5), "Proposed TID-1219")
        obj = std::make_shared<Producer>();
        LWCOV_ALLOWLIST_END(LWCOV_CERTCPP(ERR59_CPP))
    } catch (std::bad_alloc& e) {
        static_cast<void>(e);
        return LwSciError_InsufficientMemory;
    }
    if (!obj->isInitSuccess()) {
        return LwSciError_StreamInternalError;
    }
    if (!Block::registerBlock(obj)) {
        return  LwSciError_StreamInternalError;
    }

    // BindPool has to be perform after the producer block has been
    // registered.
    LwSciError const err { obj->BindPool(poolPtr) };
    if (LwSciError_Success != err) {
        Block::removeRegisteredBlock(obj->getHandle());
        return err;
    }

    // Lock config options of pool block
    poolPtr->finalizeConfigOptions();

    *producer = obj->getHandle();
    return LwSciError_Success;
}

// \brief Create a stream consumer block.
//
// Create a block for the consumer end of a stream. All streams
// require at least one consumer block. Consumer blocks have one
// input connection and no output connections.
//
// Once the stream is fully connected, this block can be used to retrieve
// producer requirements and finalize consumer settings.
//
// <b>Postconditions</b>:
// * The block is ready to be connected to other stream blocks.
// * The block can be monitored for events.
//
// \param [out] consumer: handle for new consumer block.
//
// \return status code
// - LwSciError_Success: The block was set up successfully.
// - LwSciError_BadParameter: If 'queue' is invalid or
//   if the output block 'consumer' is a nullptr.
// - LwSciError_InsufficientMemory: The block is not created.
// - LwSciError_StreamInternalError: The block can't be initialized properly.
//     Internal system errors such as mutex-locking failure.

/**
 *  <b>Sequence of operations</b>
 *    - Retrieves the queue block instance referenced by the given LwSciStreamBlock
 *      by calling Block::getRegisteredBlock() interface.
 *    - Creates a consumer block instance and checks whether the initialization
 *      is successful by calling its Block::isInitSuccess() interface.
 *    - Registers the consumer block instance by calling Block::registerBlock()
 *      interface.
 *    - Connects the queue block instance with the consumer block
 *      instance by calling its Consumer::BindQueue() interface.
 *    - Call Block::finalizeConfigOptions() to lock the configuration options of
 *      the queue block.
 *    - Retrieves the LwSciStreamBlock referencing the created consumer block
 *      instance by calling its Block::getHandle() interface and returns it.
 */
LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A8_4_8), "Approved TID-281")
LwSciError LwSciStreamConsumerCreate(
    LwSciStreamBlock const queue,
    LwSciStreamBlock *const consumer)
{
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_1_1), "TID-300")
    if (nullptr == consumer) {
        return LwSciError_BadParameter;
    }

    BlockPtr const queuePtr { Block::getRegisteredBlock(queue) };

    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_1_1), "TID-300")
    if (nullptr == queuePtr) {
        return LwSciError_BadParameter;
    }

    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A20_8_4), "Bug 3255701")
    std::shared_ptr<Consumer> obj {};
    try {
        LWCOV_ALLOWLIST_BEGIN(LWCOV_CERTCPP(ERR59_CPP), "TID-1475")
        LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A15_1_5), "Proposed TID-1219")
        obj = std::make_shared<Consumer>();
        LWCOV_ALLOWLIST_END(LWCOV_CERTCPP(ERR59_CPP))
    } catch (std::bad_alloc& e) {
        static_cast<void>(e);
        return LwSciError_InsufficientMemory;
    }
    if (!obj->isInitSuccess()) {
        return LwSciError_StreamInternalError;
    }
    if (!Block::registerBlock(obj)) {
        return  LwSciError_StreamInternalError;
    }

    // BindQueue has to be perform after the consumer block has been
    // registered.
    LwSciError const err { obj->BindQueue(queuePtr) };
    if (LwSciError_Success != err) {
        Block::removeRegisteredBlock(obj->getHandle());
        return err;
    }

    // Lock config options of queue block
    queuePtr->finalizeConfigOptions();

    *consumer = obj->getHandle();
    return LwSciError_Success;
}

// \brief Create a static stream pool block.
//
// Create a block for management of a stream's packet pool. All
// streams require one pool block.
//
// A static pool has a fixed number of packets which must be fully defined
// before streaming begins, and is intended for safety-certified usage.
//
// Pool blocks have one input connection and one output connection. Typically,
// the pool is connected just downstream of the producer.
//
// Once the stream is fully connected and the application (which may or may
// not be the same as the producer application) has determined the packet
// requirements, this block can be used to bind memory buffers to each packet.
//
// Note: Having this block separate from the producer allows for
// new types of pools with more complex behavior to be made available
// without complicating the simple case. E.g.
// - A dynamic pool for the non-safety case for which packets can be
// added and removed at any time.
// - A pool tied to a centralized buffer manager sharing a common
// set of buffers with multiple streams.
//
// <b>Preconditions</b>
//
// None
//
// <b>Actions</b>
//
// - Allocate data structures to describe packets.
// - Initialize queue of available packets.
//
// <b>Postconditions</b>
//
// - The block is ready to be connected to other stream blocks.
// - The block can be monitored for events.
//
// \param num_packets (in) number of packets
// \param pool (out) handle for new pool block
//
// \return status code
//  - LwSciError_Success: The block was set up successfully.
//  - LwSciError_BadParameter: The output parameter 'pool' is a nullptr.
//  - LwSciError_InsufficientMemory: The block is not created.
//  - LwSciError_StreamInternalError: The block can't be initialized properly.
//     Internal system errors such as mutex-locking failure.

/**
 *  <b>Sequence of operations</b>
 *    - Creates a pool block instance and checks whether the initialization
 *      is successful by calling its Block::isInitSuccess() interface.
 *    - Registers the pool block instance by calling Block::registerBlock() interface.
 *    - Retrieves the LwSciStreamBlock referencing the created pool block
 *      instance by calling its Block::getHandle() interface and returns it.
 */
LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A8_4_8), "Approved TID-281")
LwSciError LwSciStreamStaticPoolCreate(
    uint32_t const numPackets,
    LwSciStreamBlock *const pool)
{
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_1_1), "TID-300")
    if (nullptr == pool) {
        return LwSciError_BadParameter;
    }

    BlockPtr obj {};
    try {
        LWCOV_ALLOWLIST_BEGIN(LWCOV_CERTCPP(ERR59_CPP), "TID-1475")
        LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A15_1_5), "Proposed TID-1219")
        obj = std::make_shared<Pool>(numPackets);
        LWCOV_ALLOWLIST_END(LWCOV_CERTCPP(ERR59_CPP))
    } catch (std::bad_alloc& e) {
        static_cast<void>(e);
        return LwSciError_InsufficientMemory;
    }

    if (!obj->isInitSuccess()) {
        return LwSciError_StreamInternalError;
    }
    if (!Block::registerBlock(obj)) {
        return  LwSciError_StreamInternalError;
    }
    *pool = obj->getHandle();
    return LwSciError_Success;
}

// \brief Create a mailbox queue block.
//
// Create a block for tracking the next packet available to be
// acquired.
//
// All streams require one queue block.
//
// A mailbox queue holds a single packet. If a new packet arrives, the old one
// is replaced and returned to the pool for reuse without ever being acquired.
//
// This type of queue is intended for consumer applications
// which don't need to process every packet and always wish to have the
// latest input available.
//
// Queue blocks have one input connection and one output connection. Typically,
// the queue is connected just upstream of the consumer.
// Once connected, the application does not directly interact with this block.
//
// The consumer block will communicate with it to obtain new packets.
//
// Note: Having this block separate from the consumer allows for
// mailbox and fifo options to be kept separate and therefore simpler.
//
// It also allows for other less common queue operations, such as reuse
// of old frames, to be added as additional blocks between the queue
// and the consumer.
//
// <b>Preconditions</b>
//
// None
//
// <b>Actions</b>
//
// - Initialize a queue to hold a single packet for acquire.
//
// <b>Postconditions</b>
//
// - The block is ready to be connected to other stream blocks.
//
// \param queue (out) handle for new queue block
//
// \return status code
//  - LwSciError_Success: The block was set up successfully.
//  - LwSciError_BadParameter: The output parameter 'pool' is a nullptr.
//  - LwSciError_InsufficientMemory: The block is not created.
//  - LwSciError_StreamInternalError: The block can't be initialized properly.
//     Internal system errors such as mutex-locking failure.

/**
 *  <b>Sequence of operations</b>
 *    - Creates a Mailbox instance and checks whether the initialization
 *      is successful by calling its Block::isInitSuccess() interface.
 *    - Registers the Mailbox instance by calling Block::registerBlock() interface.
 *    - Retrieves the LwSciStreamBlock referencing the created Mailbox
 *      instance by calling its Block::getHandle() interface and returns it.
 */
LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A8_4_8), "Approved TID-281")
LwSciError LwSciStreamMailboxQueueCreate(
    LwSciStreamBlock *const queue)
{
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_1_1), "TID-300")
    if (nullptr == queue) {
        return LwSciError_BadParameter;
    }

    BlockPtr obj {};
    try {
        LWCOV_ALLOWLIST_BEGIN(LWCOV_CERTCPP(ERR59_CPP), "TID-1475")
        LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A15_1_5), "Proposed TID-1219")
        obj = std::make_shared<Mailbox>();
        LWCOV_ALLOWLIST_END(LWCOV_CERTCPP(ERR59_CPP))
    } catch (std::bad_alloc& e) {
        static_cast<void>(e);
        return LwSciError_InsufficientMemory;
    }

    if (!obj->isInitSuccess()) {
        return LwSciError_StreamInternalError;
    }
    if (!Block::registerBlock(obj)) {
        return  LwSciError_StreamInternalError;
    }
    *queue = obj->getHandle();
    return LwSciError_Success;
}

// \brief Create a FIFO queue block.
//
// Create a block for tracking the list of packets available to be
// acquired.
//
// All streams require one queue block.
//
// A FIFO queue holds a list of packets, which will be acquired in the order
// they were presented.
//
// If a new packet arrives, it is added to the end of the FIFO.
//
// This type of queue is intended for consumer applications which must process
// every packet that is produced.
//
// Queue blocks have one input connection and one output connection. Typically,
// the queue is connected just upstream of the consumer.
//
// Once connected, the application does not directly interact with this block.
// The consumer block will communicate with it to obtain new packets.
//
// Note: Having this block separate from the consumer allows for
// mailbox and fifo options to be kept separate and therefore simpler.
//
// It also allows for other less common queue operations, such as reuse
// of old frames, to be added as additional blocks between the queue
// and the consumer.
//
// <b>Preconditions</b>
//
// None
//
// <b>Actions</b>
//
// - Initialize a queue to manage waiting packets.
//
// <b>Postconditions</b>
//
// - The block is ready to be connected to other stream blocks.
//
// \param queue (out) handle for new queue block
//
// \return status code
//  - LwSciError_Success: The block was set up successfully.
//  - LwSciError_BadParameter: The output parameter 'pool' is a nullptr.
//  - LwSciError_InsufficientMemory: The block is not created.
//  - LwSciError_StreamInternalError: The block can't be initialized properly.
//     Internal system errors such as mutex-locking failure.

/**
 *  <b>Sequence of operations</b>
 *    - Creates a Fifo instance and checks whether the initialization
 *      is successful by calling its Block::isInitSuccess() interface.
 *    - Registers the Fifo instance by calling Block::registerBlock() interface.
 *    - Retrieves the LwSciStreamBlock referencing the created Fifo
 *      instance by calling its Block::getHandle() interface and returns it.
 */
LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A8_4_8), "Approved TID-281")
LwSciError LwSciStreamFifoQueueCreate(
    LwSciStreamBlock *const queue)
{
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_1_1), "TID-300")
    if (nullptr == queue) {
        return LwSciError_BadParameter;
    }

    BlockPtr obj {};
    try {
        LWCOV_ALLOWLIST_BEGIN(LWCOV_CERTCPP(ERR59_CPP), "TID-1475")
        LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A15_1_5), "Proposed TID-1219")
        obj = std::make_shared<Fifo>();
        LWCOV_ALLOWLIST_END(LWCOV_CERTCPP(ERR59_CPP))
    } catch (std::bad_alloc& e) {
        static_cast<void>(e);
        return LwSciError_InsufficientMemory;
    }

    if (!obj->isInitSuccess()) {
        return LwSciError_StreamInternalError;
    }
    if (!Block::registerBlock(obj)) {
        return  LwSciError_StreamInternalError;
    }
    *queue = obj->getHandle();
    return LwSciError_Success;
}

// \brief Creates a multicast block.
//
// Create a block allowing for one input and one or more outputs.
//
// Multicast block broadcasts messages sent from upstream to all of the
// downstream blocks.
//
// Multicast block aggregates messages of the same type from downstreams into
// one before sending it upstream.
//
// <b>Preconditions</b>
//
// None.
//
// <b>Actions</b>
//
// - Initialize a block that takes one input and one or more outputs.
//
// <b>Postconditions</b>
//
// - The block is ready to be connected to other stream blocks.
//
// \param[in] outputCount number of output blocks that will be connected.
// \param[out] multicast Handle for the new multicast block.
//
// \return ::LwSciError, the completion code of this operation.
//  - ::LwSciError_Success The block was set up successfully.
//  - ::LwSciError_BadParameter The output parameter @a multicast is a null
//     pointer, or @b outputCount is larger than the number allowed.
//  - ::LwSciError_InsufficientMemory The block is not created.
//  - ::LwSciError_StreamInternalError The block cannot be initialized properly.
//     An internal system error oclwrred, such as a mutex-locking failure.

/**
 *  <b>Sequence of operations</b>
 *    - Creates a multicast block instance and checks whether the initialization
 *      is successful by calling its Block::isInitSuccess() interface.
 *    - Registers the multicast block instance by calling Block::registerBlock()
 *      interface.
 *    - Retrieves the LwSciStreamBlock referencing the created multicast block
 *      instance by calling its Block::getHandle() interface and returns it.
 */
LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A8_4_8), "Approved TID-281")
LwSciError LwSciStreamMulticastCreate(
    uint32_t const outputCount,
    LwSciStreamBlock *const multicast)
{
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_1_1), "TID-300")
    if (nullptr == multicast) {
        return LwSciError_BadParameter;
    }

    if (outputCount > LwSciStream::MAX_DST_CONNECTIONS) {
        return LwSciError_BadParameter;
    }

    BlockPtr obj {};
    try {
        LWCOV_ALLOWLIST_BEGIN(LWCOV_CERTCPP(ERR59_CPP), "TID-1475")
        LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A15_1_5), "Proposed TID-1219")
        obj = std::make_shared<MultiCast>(outputCount);
        LWCOV_ALLOWLIST_END(LWCOV_CERTCPP(ERR59_CPP))
    } catch (std::bad_alloc& e) {
        static_cast<void>(e);
        return LwSciError_InsufficientMemory;
    }

    if (!obj->isInitSuccess()) {
        return LwSciError_StreamInternalError;
    }
    if (!Block::registerBlock(obj)) {
        return  LwSciError_StreamInternalError;
    }
    *multicast = obj->getHandle();
    return LwSciError_Success;
}

// \brief Create an IPC source block.
//
// Creates the upstream half of an IPC block pair which allows
// packets to be transmitted between processes.
//
// IPC source blocks have one input connection and no output connection.
//
// A IPC source block connects to downstream through the ipcEndpoint used to
// create the block.
//
// <b>Preconditions</b>
//
// None
//
// <b>Actions</b>
//
// - Ready ipc endpoint.
// - Allocate buffers for data transmission.
//
// <b>Postconditions</b>
//
// - The block is ready to be connected to other stream blocks.
//
// \param ipcEndpoint (in) ipc endpoint handle
// \param syncModule (in) LwSciSyncModule that is used to import a
//        LwSciSyncAttrList across an ipc / ivc boundary.
//
//        This must be same module that was used to create LwSciSyncAttrList
//        when specifying the associated LwSciStreamSyncAttr object
// \param bufModule (in) LwSciBufModule that is used to import a
//        LwSciBufAttrList across an ipc / ivc boundary.
//
//        This must be same module that was used to create LwSciBufAttrList
//        when specifying the associated LwSciStreamElementAttr object
// \param ipc (out) handle for new lwscistream ipc source block
//
// \return status code
//  - LwSciError_Success: The block was set up successfully.
//  - LwSciError_BadParameter: The output parameter 'pool' is a nullptr.
//  - LwSciError_InsufficientMemory: The block is not created.
//  - LwSciError_StreamInternalError: The block can't be initialized properly.
//     Internal system errors such as mutex-locking failure, or failure to
//     launch the dispatch thread.

/**
 *  <b>Sequence of operations</b>
 *    - Creates an IpcSrc block instance and checks whether the initialization
 *      is successful by calling its Block::isInitSuccess() interface.
 *    - Registers the IpcSrc block instance by calling Block::registerBlock()
 *      interface.
 *    - Retrieves the LwSciStreamBlock referencing the created IpcSrc block
 *      instance by calling its Block::getHandle() interface and returns it.
 */
LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A8_4_8), "Approved TID-281")
LwSciError LwSciStreamIpcSrcCreate(
    LwSciIpcEndpoint const ipcEndpoint,
    LwSciSyncModule const syncModule,
    LwSciBufModule const bufModule,
    LwSciStreamBlock *const ipc
)
{
    return LwSciStreamIpcSrcCreate2(ipcEndpoint,
                                    syncModule,
                                    bufModule,
                                    LwSciStream::ILWALID_BLOCK_HANDLE,
                                    ipc);
}

// \brief Create an IPC destination block.
//
// Creates the downstream half of an IPC block pair which allows
// packets to be transmitted between processes.
//
// IPC destination blocks have one output connection and no input connection.
//
// A IPC destination block connects to upstream through the ipcEndpoint used
// to create the block.
//
// <b>Preconditions</b>
//
// None
//
// <b>Actions</b>
//
// - Ready ipc endpoint.
// - Allocate buffers for data transmission.
//
// <b>Postconditions</b>
//
// - The block is ready to be connected to other stream blocks.
//
// \param ipcEndpoint (in) ipc endpoint handle.
// \param syncModule (in) LwSciSyncModule that is used to import a
//        LwSciSyncAttrList across an ipc / ivc boundary.
//
//        This must be same module that was used to create LwSciSyncAttrList
//        when specifying the associated LwSciStreamSyncAttr object
// \param bufModule (in) LwSciBufModule that is used to import a
//        LwSciBufAttrList across an ipc / ivc boundary.
//
//        This must be same module that was used to create LwSciBufAttrList
//        when specifying the associated LwSciStreamElementAttr object
// \param ipc (out) handle for new lwscistream ipc destination block
//
// \return status code
//  - LwSciError_Success: The block was set up successfully.
//  - LwSciError_BadParameter: The output parameter 'pool' is a nullptr.
//  - LwSciError_InsufficientMemory: The block is not created.
//  - LwSciError_StreamInternalError: The block can't be initialized properly.
//     Internal system errors such as mutex-locking failure, or failure to
//     launch the dispatch thread.

/**
 *  <b>Sequence of operations</b>
 *    - Creates an IpcDst block instance and checks whether the initialization
 *      is successful by calling its Block::isInitSuccess() interface.
 *    - Registers the IpcDst block instance by calling Block::registerBlock()
 *      interface.
 *    - Retrieves the LwSciStreamBlock referencing the created IpcDst block
 *      instance by calling its Block::getHandle() interface and returns it.
 */
LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A8_4_8), "Approved TID-281")
LwSciError LwSciStreamIpcDstCreate(
    LwSciIpcEndpoint const ipcEndpoint,
    LwSciSyncModule const syncModule,
    LwSciBufModule const bufModule,
    LwSciStreamBlock *const ipc)
{
    return LwSciStreamIpcDstCreate2(ipcEndpoint,
                                    syncModule,
                                    bufModule,
                                    LwSciStream::ILWALID_BLOCK_HANDLE,
                                    ipc);
}

// \brief Create a Limiter block.
//
// Create a block to limit the number of packets allowed to be sent downstream.
//
// A Limiter block can be inserted anywhere in the stream between the Producer
// and Consumer Blocks, but its primary intent is to be inserted between a
// Multicast block and a Consumer.
//
// <b>Preconditions</b>
//
// None
//
// <b>Actions</b>
//
// - Creates a new instance of Limiter block.
//
// <b>Postconditions</b>
//
// - The block is ready to be connected to other stream blocks.
//
// \param maxPackets (in) Number of packets allowed to be sent downstream.
// \param limiter (out) LwSciStreamBlock which references a new Limiter block.
//
// \return status code
//  - LwSciError_Success: The block was set up successfully.
//  - LwSciError_BadParameter: The output parameter 'limiter' is a nullptr.
//  - LwSciError_InsufficientMemory: The block is not created.
//  - LwSciError_StreamInternalError: The block can't be initialized properly.
//     Internal system errors such as mutex-locking failure.

/**
*  <b>Sequence of operations</b>
*    - Creates a Limiter block instance and checks whether the initialization
*      is successful by calling its Block::isInitSuccess() interface.
*    - Registers the Limiter block instance by calling Block::registerBlock()
*      interface.
*    - Retrieves the LwSciStreamBlock referencing the created Limiter block
*      instance by calling its Block::getHandle() interface and returns it.
*/
LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A8_4_8), "Approved TID-281")
LwSciError LwSciStreamLimiterCreate(
    uint32_t const maxPackets,
    LwSciStreamBlock *const limiter)
{
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_1_1), "TID-300")
    if (nullptr == limiter) {
        return LwSciError_BadParameter;
    }

    BlockPtr obj{};
    try {
        LWCOV_ALLOWLIST_BEGIN(LWCOV_CERTCPP(ERR59_CPP), "TID-1475")
        LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A15_1_5), "Proposed TID-1219")
        obj = std::make_shared<Limiter>(maxPackets);
        LWCOV_ALLOWLIST_END(LWCOV_CERTCPP(ERR59_CPP))
    } catch (std::bad_alloc& e) {
        static_cast<void>(e);
        return LwSciError_InsufficientMemory;
    }

    if (!obj->isInitSuccess()) {
        return LwSciError_StreamInternalError;
    }
    if (!Block::registerBlock(obj)) {
        return  LwSciError_StreamInternalError;
    }
    *limiter = obj->getHandle();
    return LwSciError_Success;
}


/**
*  <b>Sequence of operations</b>
*    - Creates a ReturnSync block instance and checks whether the initialization
*      is successful by calling its Block::isInitSuccess() interface.
*    - Registers the ReturnSync block instance by calling Block::registerBlock()
*      interface.
*    - Retrieves the LwSciStreamBlock referencing the created ReturnSync block
*      instance by calling its Block::getHandle() interface and returns it.
*/
LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A8_4_8), "Approved TID-281")
LwSciError LwSciStreamReturnSyncCreate(
    LwSciSyncModule const syncModule,
    LwSciStreamBlock *const returnSync)
{
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_1_1), "TID-300")
    if (nullptr == returnSync) {
        return LwSciError_BadParameter;
    }

    BlockPtr obj{};
    try {
        LWCOV_ALLOWLIST_BEGIN(LWCOV_CERTCPP(ERR59_CPP), "TID-1475")
        LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A15_1_5), "Proposed TID-1219")
        obj = std::make_shared<ReturnSync>(syncModule);
        LWCOV_ALLOWLIST_END(LWCOV_CERTCPP(ERR59_CPP))
    } catch (std::bad_alloc& e) {
        static_cast<void>(e);
        return LwSciError_InsufficientMemory;
    }

    if (!obj->isInitSuccess()) {
        return LwSciError_StreamInternalError;
    }
    if (!Block::registerBlock(obj)) {
        return  LwSciError_StreamInternalError;
    }
    *returnSync = obj->getHandle();
    return LwSciError_Success;
}

/**
*  <b>Sequence of operations</b>
*    - Creates a PresentSync block instance and checks whether the initialization
*      is successful by calling its Block::isInitSuccess() interface.
*    - Registers the PresentSync block instance by calling Block::registerBlock()
*      interface.
*    - Retrieves the LwSciStreamBlock referencing the created PresentSync block
*      instance by calling its Block::getHandle() interface and returns it.
*/
LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A8_4_8), "Approved TID-281")
LwSciError LwSciStreamPresentSyncCreate(
    LwSciSyncModule const syncModule,
    LwSciStreamBlock *const presentSync)
{
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_1_1), "TID-300")
    if (nullptr == presentSync) {
        return LwSciError_BadParameter;
    }

    BlockPtr obj{};
    try {
        LWCOV_ALLOWLIST_BEGIN(LWCOV_CERTCPP(ERR59_CPP), "TID-1475")
        LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A15_1_5), "Proposed TID-1219")
        obj = std::make_shared<PresentSync>(syncModule);
        LWCOV_ALLOWLIST_END(LWCOV_CERTCPP(ERR59_CPP))
    } catch (std::bad_alloc& e) {
        static_cast<void>(e);
        return LwSciError_InsufficientMemory;
    }

    if (!obj->isInitSuccess()) {
        return LwSciError_StreamInternalError;
    }
    if (!Block::registerBlock(obj)) {
        return  LwSciError_StreamInternalError;
    }
    *presentSync = obj->getHandle();
    return LwSciError_Success;
}

/**
 *  <b>Sequence of operations</b>
 *    - Retrieves the block instance referenced by the given LwSciStreamBlock
 *      by calling Block::getRegisteredBlock() interface.
 *    - Tries to set the default event-notification mode by calling
 *      its Block::eventDefaultSetup() interface.
 *    - Queries the block instance for the next pending LwSciStreamEvent if any
 *      by calling its Block::getEvent() interface and returns it.
 */
LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A8_4_8), "Approved TID-281")
LwSciError LwSciStreamBlockEventQuery(
    LwSciStreamBlock const block,
    int64_t const timeoutUsec,
    LwSciStreamEventType *const event)
{
    BlockPtr const blkPtr { Block::getRegisteredBlock(block) };
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_1_1), "TID-300")
    if (nullptr == blkPtr) {
        return LwSciError_BadParameter;
    }
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_1_1), "TID-300")
    if (nullptr == event) {
        return LwSciError_BadParameter;
    }

    // Sets the default event-notification mode of the block.
    // (If it has not been configured to use LwSciEventService.)
    blkPtr->eventDefaultSetup();

    LwSciStreamEventType localEvent { };
    LwSciError const rv { blkPtr->getEvent(timeoutUsec, localEvent) };
    if (LwSciError_Success == rv) {
        *event = localEvent;
    }
    return rv;
}


/**
*  <b>Sequence of operations</b>
*    - Retrieves the block instance referenced by the given LwSciStreamBlock
*      by calling Block::getRegisteredBlock() interface.
*    - Ilwokes apiErrorGet() interface of the block instance and stores the
*      returned value in the @a status argument.
*/
LwSciError
LwSciStreamBlockErrorGet(
    LwSciStreamBlock const block,
    LwSciError* const status)
{
    // Validate pointer to return value
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_1_1), "TID-300")
    if (nullptr == status) {
        return LwSciError_BadAddress;
    }

    // Retrieve block
    BlockPtr const blkPtr{ Block::getRegisteredBlock(block) };
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_1_1), "TID-300")
    if (nullptr == blkPtr) {
        return LwSciError_StreamBadBlock;
    }

    // Ilwoke operatioin on block
    *status = blkPtr->apiErrorGet();
    return LwSciError_Success;
}

/**
 *  <b>Sequence of operations</b>
 *    - Retrieves the block instance referenced by the given LwSciStreamBlock
 *      by calling Block::getRegisteredBlock() interface.
 *    - Makes sure provided pointer in which to store the count is valid.
 *    - Calls the block instance's Block::apiConsumerCountGet() interface to
 *      query the number of consumers downstream of it.
 */
LwSciError
LwSciStreamBlockConsumerCountGet(
    LwSciStreamBlock const block,
    uint32_t* const numConsumers)
{
    BlockPtr const blkPtr { Block::getRegisteredBlock(block) };
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_1_1), "TID-300")
    if (nullptr == blkPtr) {
        return LwSciError_StreamBadBlock;
    }
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_1_1), "TID-300")
    if (nullptr == numConsumers) {
        return LwSciError_BadAddress;
    }

    // Query the number of consumers accesssible from this block
    return blkPtr->apiConsumerCountGet(*numConsumers);
}

/**
 *  <b>Sequence of operations</b>
 *    - Retrieves the block instance referenced by the given LwSciStreamBlock
 *      by calling Block::getRegisteredBlock() interface.
 *    - Makes sure @a completed flag is true.
 *    - Calls apiSetupStatusSet() interface of the block instance with the
 *      provided @a setupType.
 */
LwSciError
LwSciStreamBlockSetupStatusSet(
    LwSciStreamBlock const block,
    LwSciStreamSetup const setupType,
    bool const completed)
{
    // Retrieve block
    BlockPtr const blkPtr { Block::getRegisteredBlock(block) };
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_1_1), "TID-300")
    if (nullptr == blkPtr) {
        return LwSciError_StreamBadBlock;
    }

    // Lwrrently we only support true for completed flag, so we just check it
    //   here, and avoid passing to the block. If and when we support dynamic
    //   reconfiguation, we'll have to pass it as an additional parameter.
    if (!completed) {
        return LwSciError_BadParameter;
    }

    // Ilwoke operation on block
    return blkPtr->apiSetupStatusSet(setupType);
}

/**
 *  <b>Sequence of operations</b>
 *    - Retrieves the block instance referenced by the given LwSciStreamBlock
 *      by calling Block::getRegisteredBlock() interface.
 *    - Constructs a LwSciWrap::BufAttr instance with the given
 *      LwSciBufAttrList.
 *    - Calls apiElementAttrSet() interface of the block instance with the
 *      LwSciWrap::BufAttr instance and @ userType arguments, to add the
 *      element to the list.
 */
LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A8_4_8), "Approved TID-281")
LwSciError
LwSciStreamBlockElementAttrSet(
    LwSciStreamBlock const block,
    uint32_t const userType,
    LwSciBufAttrList const bufAttrList)
LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A8_4_8))
{
    // Retrieve block
    BlockPtr const blkPtr { Block::getRegisteredBlock(block) };
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_1_1), "TID-300")
    if (nullptr == blkPtr) {
        return LwSciError_StreamBadBlock;
    }

    // Wrap attribute list indicating that we don't own it
    LwSciWrap::BufAttr wrapBufAttr{bufAttrList};

    // Ilwoke operation on block
    return blkPtr->apiElementAttrSet(userType, wrapBufAttr);
}

/**
 *  <b>Sequence of operations</b>
 *    - Makes sure the input pointer is valid.
 *    - Retrieves the block instance referenced by the given LwSciStreamBlock
 *      by calling Block::getRegisteredBlock() interface.
 *    - Calls apiElementCountGet() interface of the block instance, passing
 *      the @a queryBlockType and a reference to @a numElements, to retrieve
 *      the desired element count.
 */
LwSciError
LwSciStreamBlockElementCountGet(
    LwSciStreamBlock const block,
    LwSciStreamBlockType const queryBlockType,
    uint32_t* const numElements)
{
    // Validate input
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_1_1), "TID-300")
    if (nullptr == numElements) {
        return LwSciError_BadAddress;
    }

    // Retrieve block
    BlockPtr const blkPtr { Block::getRegisteredBlock(block) };
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_1_1), "TID-300")
    if (nullptr == blkPtr) {
        return LwSciError_StreamBadBlock;
    }

    // Ilwoke operation on block
    return blkPtr->apiElementCountGet(queryBlockType, *numElements);
}

/**
 *  <b>Sequence of operations</b>
 *    - Makes sure at least one of @a userType and @a bufAttrList is valid.
 *    - Retrieves the block instance referenced by the given LwSciStreamBlock
 *      by calling Block::getRegisteredBlock() interface.
 *    - If @a userType pointer is valid, calls apiElementTypeGet() interface
 *      of the block instance, passing the the @a queryBlockType, @a elemIndex,
 *      and a reference to @a userType, to retrieve the requested element's
 *      type.
 *    - If @a bufAttrList pointer is valid, calls apiElementAttrGet() interface
 *      of the block instance, passing the the @a queryBlockType, @a elemIndex,
 *      and a reference to @a bufAttrList, to retrieve the requested element's
 *      buffer attributes.
 */
LwSciError
LwSciStreamBlockElementAttrGet(
    LwSciStreamBlock const block,
    LwSciStreamBlockType const queryBlockType,
    uint32_t const elemIndex,
    uint32_t* const userType,
    LwSciBufAttrList* const bufAttrList)
{
    // Validate input
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_1_1), "TID-300")
    if ((nullptr == userType) && (nullptr == bufAttrList)) {
        return LwSciError_BadAddress;
    }

    // Retrieve block
    BlockPtr const blkPtr { Block::getRegisteredBlock(block) };
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_1_1), "TID-300")
    if (nullptr == blkPtr) {
        return LwSciError_StreamBadBlock;
    }

    // Ilwoke required operation(s) on block
    LwSciError err { LwSciError_Success };
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_1_1), "TID-300")
    if (nullptr != userType) {
        err = blkPtr->apiElementTypeGet(queryBlockType, elemIndex,
                                        *userType);
    }
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_1_1), "TID-300")
    if ((nullptr != bufAttrList) && (LwSciError_Success == err)) {
        err = blkPtr->apiElementAttrGet(queryBlockType, elemIndex,
                                        *bufAttrList);
    }
    return err;
}

/**
 *  <b>Sequence of operations</b>
 *    - Retrieves the block instance referenced by the given LwSciStreamBlock
 *      by calling Block::getRegisteredBlock() interface.
 *    - Calls apiElementUsageSet() interface of the block instance with the
 *      provided @a elemIndex and @a used values, to indicate whether the
 *      specified element is used.
 */
LwSciError
LwSciStreamBlockElementUsageSet(
    LwSciStreamBlock const block,
    uint32_t const elemIndex,
    bool const used)
{
    // Retrieve block
    BlockPtr const blkPtr { Block::getRegisteredBlock(block) };
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_1_1), "TID-300")
    if (nullptr == blkPtr) {
        return LwSciError_StreamBadBlock;
    }

    // Ilwoke operation on block
    return blkPtr->apiElementUsageSet(elemIndex, used);
}

/**
 *  <b>Sequence of operations</b>
 *    - Retrieves the pool block instance referenced by the given
 *      LwSciStreamBlock by calling Block::getRegisteredBlock() interface.
 *    - Ilwokes apiPacketCreate() interface of the pool block instance with the
 *      @a cookie and @a handle arguments.
 */
LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A8_4_8), "Approved TID-281")
LwSciError LwSciStreamPoolPacketCreate(
    LwSciStreamBlock const pool,
    LwSciStreamCookie const cookie,
    LwSciStreamPacket *const handle)
{
    // Validate pointer to return handle
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_1_1), "TID-300")
    if (nullptr == handle) {
        return LwSciError_BadAddress;
    }

    // Retrieve block
    BlockPtr const blkPtr { Block::getRegisteredBlock(pool) };
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_1_1), "TID-300")
    if (nullptr == blkPtr) {
        return LwSciError_StreamBadBlock;
    }

    // Ilwoke operatioin on block
    return blkPtr->apiPacketCreate(cookie, *handle);
}

/**
 *  <b>Sequence of operations</b>
 *    - Retrieves the pool block instance referenced by the given
 *      LwSciStreamBlock by calling Block::getRegisteredBlock() interface.
 *    - Constructs a LwSciWrap::BufObj instance with the given
 *      LwSciBufObj.
 *    - Ilwokes the apiPacketBuffer() interface of the pool block instance
 *      with the LwSciWrap::BufObj instance, @a handle and @a index arguments.
 */
LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A8_4_8), "Approved TID-281")
LwSciError LwSciStreamPoolPacketInsertBuffer(
    LwSciStreamBlock const pool,
    LwSciStreamPacket const handle,
    uint32_t const index,
    LwSciBufObj const bufObj)
{
    // Retrieve block
    BlockPtr const blkPtr { Block::getRegisteredBlock(pool) };
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_1_1), "TID-300")
    if (nullptr == blkPtr) {
        return LwSciError_StreamBadBlock;
    }

    // Wrap buffer object
    LwSciWrap::BufObj wrapBufObj{bufObj};

    // Ilwoke operatioin on block
    return blkPtr->apiPacketBuffer(handle, index, wrapBufObj);
}

/**
 *  <b>Sequence of operations</b>
 *    - Retrieves the pool block instance referenced by the given
 *      LwSciStreamBlock by calling Block::getRegisteredBlock() interface.
 *    - Ilwokes the apiPacketComplete() interface of the pool block instance.
 */
LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A8_4_8), "Approved TID-281")
LwSciError LwSciStreamPoolPacketComplete(
    LwSciStreamBlock const pool,
    LwSciStreamPacket const handle)
{
    // Retrieve block
    BlockPtr const blkPtr { Block::getRegisteredBlock(pool) };
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_1_1), "TID-300")
    if (nullptr == blkPtr) {
        return LwSciError_StreamBadBlock;
    }

    // Ilwoke operation on block
    return blkPtr->apiPacketComplete(handle);
}

/**
 *  <b>Sequence of operations</b>
 *    - Retrieves the pool block instance referenced by the given
 *      LwSciStreamBlock by calling Block::getRegisteredBlock() interface.
 *    - Ilwokes apiPacketDelete() interface of the pool block instance
 *      with the @a handle argument.
 */
LwSciError LwSciStreamPoolPacketDelete(
    LwSciStreamBlock const pool,
    LwSciStreamPacket const handle)
{
    // Retrieve block
    BlockPtr const blkPtr { Block::getRegisteredBlock(pool) };
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_1_1), "TID-300")
    if (nullptr == blkPtr) {
        return LwSciError_StreamBadBlock;
    }

    // Ilwoke operation on block
    return blkPtr->apiPacketDelete(handle);
}

/**
 *  <b>Sequence of operations</b>
 *    - Retrieves the producer or consumer block instance referenced by the
 *      given LwSciStreamBlock by calling Block::getRegisteredBlock()
 *      interface.
 *    - Ilwokes apiPacketNewHandleGet() interface of the block instance
 *      with the @a handle argument.
 */
LwSciError
LwSciStreamBlockPacketNewHandleGet(
    LwSciStreamBlock const block,
    LwSciStreamPacket* const handle)
{
    // Validate pointer to return handle
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_1_1), "TID-300")
    if (nullptr == handle) {
        return LwSciError_BadAddress;
    }

    // Retrieve block
    BlockPtr const blkPtr { Block::getRegisteredBlock(block) };
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_1_1), "TID-300")
    if (nullptr == blkPtr) {
        return LwSciError_StreamBadBlock;
    }

    // Ilwoke operation on block
    return blkPtr->apiPacketNewHandleGet(*handle);
}

/**
 *  <b>Sequence of operations</b>
 *    - Retrieves the producer or consumer block instance referenced by the
 *      given LwSciStreamBlock by calling Block::getRegisteredBlock()
 *      interface.
 *    - Ilwokes apiPacketBufferGet() interface of the block instance
 *      with an empty LwSciWrap::BufObj argument.
 *    - Ilwokes LwSciWrap::BufObj::takeVal() to retrieve the returned
 *      LwSciBufObj.
 */
LwSciError
LwSciStreamBlockPacketBufferGet(
    LwSciStreamBlock const block,
    LwSciStreamPacket const handle,
    uint32_t elemIndex,
    LwSciBufObj* const bufObj)
{
    // Validate pointer to return buffer object
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_1_1), "TID-300")
    if (nullptr == bufObj) {
        return LwSciError_BadAddress;
    }

    // Retrieve block
    BlockPtr const blkPtr { Block::getRegisteredBlock(block) };
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_1_1), "TID-300")
    if (nullptr == blkPtr) {
        return LwSciError_StreamBadBlock;
    }

    // Ilwoke operation on block
    LwSciWrap::BufObj bufObjWrap {};
    LwSciError const err
        { blkPtr->apiPacketBufferGet(handle, elemIndex, bufObjWrap) };
    if (LwSciError_Success != err) {
        return err;
    }

    // Extract buffer object
    return bufObjWrap.takeVal(*bufObj);
}

/**
 *  <b>Sequence of operations</b>
 *    - Retrieves the producer or consumer block instance referenced by the
 *      given LwSciStreamBlock by calling Block::getRegisteredBlock()
 *      interface.
 *    - Ilwokes apiPacketOldCookieGet() interface of the block instance
 *      with the @a cookie argument.
 */
LwSciError
LwSciStreamBlockPacketOldCookieGet(
    LwSciStreamBlock const block,
    LwSciStreamCookie* const cookie)
{
    // Validate pointer to return handle
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_1_1), "TID-300")
    if (nullptr == cookie) {
        return LwSciError_BadAddress;
    }

    // Retrieve block
    BlockPtr const blkPtr { Block::getRegisteredBlock(block) };
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_1_1), "TID-300")
    if (nullptr == blkPtr) {
        return LwSciError_StreamBadBlock;
    }

    // Ilwoke operation on block
    return blkPtr->apiPacketOldCookieGet(*cookie);
}

/**
 *  <b>Sequence of operations</b>
 *    - Retrieves the producer or consumer block instance referenced by the
 *      given LwSciStreamBlock by calling Block::getRegisteredBlock()
 *      interface.
 *    - Ilwokes apiPacketStatusSet() interface of the block instance
 *      with the @a handle, @a cookie, and @a status arguments.
 */
LwSciError
LwSciStreamBlockPacketStatusSet(
    LwSciStreamBlock const block,
    LwSciStreamPacket const handle,
    LwSciStreamCookie const cookie,
    LwSciError const status)
{
    // Retrieve block
    BlockPtr const blkPtr { Block::getRegisteredBlock(block) };
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_1_1), "TID-300")
    if (nullptr == blkPtr) {
        return LwSciError_StreamBadBlock;
    }

    // StreamInternalError is disallowed as an external status code
    if (LwSciError_StreamInternalError == status) {
        return LwSciError_BadParameter;
    }

    // If status is success, a cookie must be provided
    if ((LwSciError_Success == status) &&
        (LwSciStreamCookie_Ilwalid == cookie)) {
        return LwSciError_StreamBadCookie;
    }

    // Ilwoke operation on block
    return blkPtr->apiPacketStatusSet(handle, cookie, status);
}

/**
 *  <b>Sequence of operations</b>
 *    - Retrieves the pool block instance referenced by the given
 *      LwSciStreamBlock by calling Block::getRegisteredBlock() interface.
 *    - Ilwokes apiPacketStatusAcceptGet() interface of the block instance
 *      with the @a handle and @a accepted arguments.
 */
LwSciError
LwSciStreamPoolPacketStatusAcceptGet(
    LwSciStreamBlock const pool,
    LwSciStreamPacket const handle,
    bool* const accepted)
{
    // Validate pointer to return value
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_1_1), "TID-300")
    if (nullptr == accepted) {
        return LwSciError_BadAddress;
    }

    // Retrieve block
    BlockPtr const blkPtr { Block::getRegisteredBlock(pool) };
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_1_1), "TID-300")
    if (nullptr == blkPtr) {
        return LwSciError_StreamBadBlock;
    }

    // Ilwoke operation on block
    return blkPtr->apiPacketStatusAcceptGet(handle, *accepted);
}

/**
 *  <b>Sequence of operations</b>
 *    - Retrieves the pool block instance referenced by the given
 *      LwSciStreamBlock by calling Block::getRegisteredBlock() interface.
 *    - Ilwokes apiPacketStatusValueGet() interface of the block instance
 *      with the provided arguments.
 */
LwSciError
LwSciStreamPoolPacketStatusValueGet(
    LwSciStreamBlock const pool,
    LwSciStreamPacket const handle,
    LwSciStreamBlockType const queryBlockType,
    uint32_t const queryBlockIndex,
    LwSciError* const status)
{
    // Validate pointer to return value
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_1_1), "TID-300")
    if (nullptr == status) {
        return LwSciError_BadAddress;
    }

    // Retrieve block
    BlockPtr const blkPtr { Block::getRegisteredBlock(pool) };
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_1_1), "TID-300")
    if (nullptr == blkPtr) {
        return LwSciError_StreamBadBlock;
    }

    // Ilwoke operation on block
    return blkPtr->apiPacketStatusValueGet(handle,
                                           queryBlockType, queryBlockIndex,
                                           *status);
}

/**
 *  <b>Sequence of operations</b>
 *    - Retrieves the block instance referenced by the given LwSciStreamBlock
 *      by calling Block::getRegisteredBlock() interface.
 *    - Constructs a LwSciWrap::SyncAttr instance with the given
 *      LwSciSyncAttrList.
 *    - Calls apiElementWaiterAttrSet() interface of the block instance
 *      with the @a elemIndex and LwSciWrap::SyncAttr instance to set
 *      the element's waiter requirements.
 */
LwSciError
LwSciStreamBlockElementWaiterAttrSet(
    LwSciStreamBlock const block,
    uint32_t const elemIndex,
    LwSciSyncAttrList const waitSyncAttrList)
{
    // Retrieve block
    BlockPtr const blkPtr { Block::getRegisteredBlock(block) };
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_1_1), "TID-300")
    if (nullptr == blkPtr) {
        return LwSciError_StreamBadBlock;
    }

    // Wrap attribute list indicating that we don't own it
    LwSciWrap::SyncAttr const wrapSyncAttr{waitSyncAttrList};

    // Ilwoke operation on block
    return blkPtr->apiElementWaiterAttrSet(elemIndex, wrapSyncAttr);
}

/**
 *  <b>Sequence of operations</b>
 *    - Makes sure @a waitSyncAttrList is valid.
 *    - Retrieves the block instance referenced by the given LwSciStreamBlock
 *      by calling Block::getRegisteredBlock() interface.
 *    - Calls apiElementWaiterAttrGet() interface of the block instance,
 *      passing the the @a elemIndex, and a reference to an empty
 *      LwSciWrap::SyncAttr, to retrieve requested element's waiter
 *      requirements from the opposing endpoint(s).
 *    - Extract the attribute list from the wrapper.
 */
LwSciError
LwSciStreamBlockElementWaiterAttrGet(
    LwSciStreamBlock const block,
    uint32_t const elemIndex,
    LwSciSyncAttrList* const waitSyncAttrList)
{
    LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A5_1_1), "TID-300")

    // Validate input
    if (nullptr == waitSyncAttrList) {
        return LwSciError_BadAddress;
    }

    // Retrieve block
    BlockPtr const blkPtr { Block::getRegisteredBlock(block) };
    if (nullptr == blkPtr) {
        return LwSciError_StreamBadBlock;
    }

    // Ilwoke required operation on block
    LwSciWrap::SyncAttr wrapSyncAttr {};
    LwSciError const err
        { blkPtr->apiElementWaiterAttrGet(elemIndex, wrapSyncAttr) };
    if (LwSciError_Success != err) {
        return err;
    }

    return wrapSyncAttr.takeVal(*waitSyncAttrList);

    LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A5_1_1))
}

/**
 *  <b>Sequence of operations</b>
 *    - Retrieves the block instance referenced by the given LwSciStreamBlock
 *      by calling Block::getRegisteredBlock() interface.
 *    - Constructs a LwSciWrap::SyncObj instance with the given
 *      LwSciBufObj.
 *    - Calls apiElementSignalObjSet() interface of the block instance
 *      with the @a elemIndex and LwSciWrap::SyncObj instance to set
 *      the element's waiter requirements.
 */
LwSciError
LwSciStreamBlockElementSignalObjSet(
    LwSciStreamBlock const block,
    uint32_t const elemIndex,
    LwSciSyncObj const signalSyncObj)
{
    // Retrieve block
    BlockPtr const blkPtr { Block::getRegisteredBlock(block) };
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_1_1), "TID-300")
    if (nullptr == blkPtr) {
        return LwSciError_StreamBadBlock;
    }

    // Wrap attribute list indicating that we don't own it
    LwSciWrap::SyncObj wrapSyncObj { signalSyncObj };

    // Ilwoke operation on block
    return blkPtr->apiElementSignalObjSet(elemIndex, wrapSyncObj);
}

/**
 *  <b>Sequence of operations</b>
 *    - Makes sure @a signalSyncObj is valid.
 *    - Retrieves the block instance referenced by the given LwSciStreamBlock
 *      by calling Block::getRegisteredBlock() interface.
 *    - Calls apiElementSignalObjGet() interface of the block instance,
 *      passing the indices and a reference to an empty
 *      LwSciWrap::SyncObj, to retrieve requested element's signal
 *      sync object from the specified opposing endpoint.
 *    - Extract the object from the wrapper.
 */
LwSciError
LwSciStreamBlockElementSignalObjGet(
    LwSciStreamBlock const block,
    uint32_t const queryBlockIndex,
    uint32_t const elemIndex,
    LwSciSyncObj* const signalSyncObj)
{
    LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A5_1_1), "TID-300")

    // Validate input
    if (nullptr == signalSyncObj) {
        return LwSciError_BadAddress;
    }

    // Retrieve block
    BlockPtr const blkPtr { Block::getRegisteredBlock(block) };
    if (nullptr == blkPtr) {
        return LwSciError_StreamBadBlock;
    }

    // Ilwoke required operation on block
    LwSciWrap::SyncObj wrapSyncObj {};
    LwSciError const err {
        blkPtr->apiElementSignalObjGet(queryBlockIndex, elemIndex,
                                       wrapSyncObj)
    };
    if (LwSciError_Success != err) {
        return err;
    }

    return wrapSyncObj.takeVal(*signalSyncObj);

    LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A5_1_1))
}

/**
 *  <b>Sequence of operations</b>
 *    - Retrieves the producer block instance referenced by the given
 *      LwSciStreamBlock by calling Block::getRegisteredBlock() interface.
 *    - Ilwokes apiPayloadObtain() interface of the producer block instance
 *      with the cookie location.
 */
LwSciError LwSciStreamProducerPacketGet(
    LwSciStreamBlock const producer,
    LwSciStreamCookie *const cookie)
{
    // Validate input
    if (nullptr == cookie) {
        return LwSciError_BadAddress;
    }

    // Retrieve block
    BlockPtr const blkPtr { Block::getRegisteredBlock(producer) };
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_1_1), "TID-300")
    if (nullptr == blkPtr) {
        return LwSciError_StreamBadBlock;
    }

    // Ilwoke operation on block
    return blkPtr->apiPayloadObtain(*cookie);
}

/**
 *  <b>Sequence of operations</b>
 *    - Retrieves the producer block instance referenced by the given
 *      LwSciStreamBlock by calling Block::getRegisteredBlock() interface.
 *    - Ilwokes apiPayloadReturn() interface of the producer block instance
 *      with the @a handle.
 */
LwSciError LwSciStreamProducerPacketPresent(
    LwSciStreamBlock const producer,
    LwSciStreamPacket const handle)
{
    BlockPtr const blkPtr { Block::getRegisteredBlock(producer) };
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_1_1), "TID-300")
    if (nullptr == blkPtr) {
        return LwSciError_StreamBadBlock;
    }
    return blkPtr->apiPayloadReturn(handle);
}

/**
 *  <b>Sequence of operations</b>
 *    - Retrieves the consumer block instance referenced by the given
 *      LwSciStreamBlock by calling Block::getRegisteredBlock() interface.
 *    - Ilwokes apiPayloadObtain() interface of the consumer block instance
 *      with the cookie location.
 */
LwSciError LwSciStreamConsumerPacketAcquire(
    LwSciStreamBlock const consumer,
    LwSciStreamCookie *const cookie)
{
    // Validate input
    if (nullptr == cookie) {
        return LwSciError_BadAddress;
    }

    // Retrieve block
    BlockPtr const blkPtr { Block::getRegisteredBlock(consumer) };
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_1_1), "TID-300")
    if (nullptr == blkPtr) {
        return LwSciError_StreamBadBlock;
    }

    // Ilwoke operation on block
    return blkPtr->apiPayloadObtain(*cookie);
}

/**
 *  <b>Sequence of operations</b>
 *    - Retrieves the consumer block instance referenced by the given
 *      LwSciStreamBlock by calling Block::getRegisteredBlock() interface.
 *    - Ilwokes apiPayloadReturn() interface of the consumer block instance
 *      with the @a handle.
 */
LwSciError LwSciStreamConsumerPacketRelease(
    LwSciStreamBlock const consumer,
    LwSciStreamPacket const handle)
{
    BlockPtr const blkPtr { Block::getRegisteredBlock(consumer) };
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_1_1), "TID-300")
    if (nullptr == blkPtr) {
        return LwSciError_StreamBadBlock;
    }
    return blkPtr->apiPayloadReturn(handle);
}

/**
 *  <b>Sequence of operations</b>
 *    - Makes sure @a postfence is valid.
 *    - Retrieves the block instance referenced by the given LwSciStreamBlock
 *      by calling Block::getRegisteredBlock() interface.
 *    - Constructs a LwSciWrap::SyncFence instance with the given
 *      LwSciSyncFence.
 *    - Calls apiPayloadFenceSet() interface of the block instance
 *      with the @a handle, @a elemIndex and fence wrapper to set the
 *      packet's postfence.
 */
LwSciError
LwSciStreamBlockPacketFenceSet(
    LwSciStreamBlock const block,
    LwSciStreamPacket const handle,
    uint32_t const elemIndex,
    LwSciSyncFence const *const postfence)
{
    LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A5_1_1), "TID-300")

    // Validate input
    if (nullptr == postfence) {
        return LwSciError_BadAddress;
    }

    // Retrieve block
    BlockPtr const blkPtr { Block::getRegisteredBlock(block) };
    if (nullptr == blkPtr) {
        return LwSciError_StreamBadBlock;
    }

    // Wrap fence indicating that we don't own it
    LwSciWrap::SyncFence wrapFence { *postfence };

    // Ilwoke operation on block
    return blkPtr->apiPayloadFenceSet(handle, elemIndex, wrapFence);

    LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A5_1_1))
}

/**
 *  <b>Sequence of operations</b>
 *    - Makes sure @a prefence is valid.
 *    - Retrieves the block instance referenced by the given LwSciStreamBlock
 *      by calling Block::getRegisteredBlock() interface.
 *    - Calls apiPayloadFenceGet() interface of the block instance,
 *      passing the indices and a reference to an empty
 *      LwSciWrap::SyncFence, to retrieve requested packet's
 *      fence from the specified opposing endpoint.
 *    - Extract the fence from the wrapper.
 */
LwSciError
LwSciStreamBlockPacketFenceGet(
    LwSciStreamBlock const block,
    LwSciStreamPacket const handle,
    uint32_t const queryBlockIndex,
    uint32_t const elemIndex,
    LwSciSyncFence* const prefence)
{
    LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A5_1_1), "TID-300")

    // Validate input
    if (nullptr == prefence) {
        return LwSciError_BadAddress;
    }

    // Retrieve block
    BlockPtr const blkPtr { Block::getRegisteredBlock(block) };
    if (nullptr == blkPtr) {
        return LwSciError_StreamBadBlock;
    }

    // Ilwoke required operation on block
    LwSciWrap::SyncFence wrapFence {};
    LwSciError const err {
        blkPtr->apiPayloadFenceGet(handle, queryBlockIndex, elemIndex,
                                   wrapFence)
    };
    if (LwSciError_Success != err) {
        return err;
    }

    return wrapFence.takeVal(*prefence);

    LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A5_1_1))
}

// \brief Destroys a stream block.
//
// Schedules a stream block for destruction, disconnecting the stream if this
// hasn't already oclwrred.
//
// The block's handle may no longer be used for any function calls,
// and may be reassigned to a new block if more are created.
//
// Resources associated with the block may not be freed immediately.
//
// Any pending data packets downstream of the destroyed block will
// still be available for the consumer to acquire.
//
// No new packets upstream of the destroyed block can be presented. Once packets
// are released, they will be freed.
//
// <b>Preconditions</b>
//
// None.
//
// <b>Actions</b>
//
// - The block is scheduled for destruction
// - A DisconnectUpstream event is sent to all upstream blocks, if
//   they haven't received one already.
// - A DisconnectDownstream event is sent to all downstream blocks,
//   if they haven't received one already.
//
// <b>Postconditions</b>
//
// - The block handle is no longer valid
//
// \param block (in) block handle
//
// \return status code
// - LwSciError_Success: Block successfully destroyed
// - LwSciError_BadParameter: block refers to an invalid instance.

/**
 *  <b>Sequence of operations</b>
 *    - Retrieves the block instance referenced by the given LwSciStreamBlock
 *      by calling Block::getRegisteredBlock() interface.
 *    - Ilwokes disconnect() interface of the block instance.
 *    - Unregister the block instance by calling
 *      Block::removeRegisteredBlock() interface.
 */
LwSciError LwSciStreamBlockDelete(
    LwSciStreamBlock const block)
{
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A0_1_1), "Bug 2738197")
    LwSciError err { LwSciError_Success };
    BlockPtr const blkPtr { Block::getRegisteredBlock(block) };
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_1_1), "TID-300")
    if (nullptr == blkPtr) {
        err = LwSciError_BadParameter;
    } else {
        static_cast<void>(blkPtr->disconnect());

        Block::removeRegisteredBlock(block);
    }
    return err;
}

// \brief Query LwSciStream attributes
//
// Queries the value of one of the queryable LwSciStream attributes.
//
// <b>Preconditions</b>
//
// None.
//
// <b>Actions</b>
//
// LwSciStream looks up the value of the attribute.
//
// <b>Postconditions</b>
//
// None changed.
//
// \param attr (in) the attribute to query
// \param value (out) the value queried
//
// \return status code
// - LwSciError_Success: Query is successful.
// - LwSciError_BadParameter: attr is invalid or value is null.

/**
 * <b>Sequence of operations</b>
 *  - If @a attr argument is LwSciStreamQueryableAttrib_MaxElements,
 *    MAX_INT_SIZE constant value is returned.
 *  - If @a attr argument is LwSciStreamQueryableAttrib_MaxSyncObj,
 *    MAX_INT_SIZE constant value is returned.
 *  - If @a attr argument is LwSciStreamQueryableAttrib_MaxMulticastOutputs,
 *    MAX_DST_CONNECTIONS constant value is returned.
 */
LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A8_4_8), "Approved TID-281")
LwSciError LwSciStreamAttributeQuery(
    LwSciStreamQueryableAttrib const attr,
    int32_t *const value
)
{
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A0_1_1), "Bug 2738197")
    LwSciError err { LwSciError_Success };

    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_1_1), "TID-300")
    if (nullptr == value) {
        err = LwSciError_BadParameter;
    } else {
        switch(attr) {
        LWCOV_ALLOWLIST_BEGIN(LWCOV_CERTC(INT31_C), "Proposed TID-1581")
        case LwSciStreamQueryableAttrib_MaxElements:
            static_assert(
                LwSciStream::MAX_INT_SIZE <=
                static_cast<size_t>(std::numeric_limits<int32_t>::max()),
                "Max must fit in signed int for colwersion");
            *value = static_cast<int32_t>(LwSciStream::MAX_INT_SIZE);
            break;
        case LwSciStreamQueryableAttrib_MaxSyncObj:
            static_assert(
                LwSciStream::MAX_INT_SIZE <=
                static_cast<size_t>(std::numeric_limits<int32_t>::max()),
                "Max must fit in signed int for colwersion");
            *value = static_cast<int32_t>(LwSciStream::MAX_INT_SIZE);
            break;
        case LwSciStreamQueryableAttrib_MaxMulticastOutputs:
            static_assert(
                LwSciStream::MAX_DST_CONNECTIONS <=
                static_cast<uint32_t>(std::numeric_limits<int32_t>::max()),
                "Max must fit in signed int for colwersion");
            *value = static_cast<int32_t>(LwSciStream::MAX_DST_CONNECTIONS);
            break;
        LWCOV_ALLOWLIST_END(LWCOV_CERTC(INT31_C))
        default:
            err = LwSciError_BadParameter;
            break;
        }
    }
    return err;
}

// Sets up the LwSciEventService on a block referenced by the given
// LwSciStreamBlock by creating an LwSciEventNotifier to report the oclwrrence
// of any events on that block. The LwSciEventNotifier is bound to the input
// LwSciEventService and LwSciStreamBlock. Users can wait for events on the
// block using the LwSciEventService API and then retrieve event details
// using LwSciStreamBlockEventQuery(). Binding one or more blocks in a stream
// to an LwSciEventService is optional. If not bound to an LwSciEventService,
// users may instead wait for events on a block by specifying a non-zero
// timeout in LwSciStreamBlockEventQuery(). If blocks in the same stream within
// the same process are bound to different LwSciEventService, behavior is
// undefined. The user is responsible for destroying the LwSciEventNotifier when
// it's no longer needed.
//
// <b>Preconditions</b>
// - No LwSciStream API is called on the input block after its creation.
//
// <b>Actions</b>
// - Sets up the input block to use the input LwSciEventService for event
// signaling.
// - Creates an LwSciEventNotifier object and returns the pointer to the object
// via @a eventNotifier.
//
// <b>Postconditions</b>
// - LwSciStreamBlockEventQuery() calls with non-zero timeout on the block would
// return error.
//
// \param[in] block LwSciStreamBlock which references a block.
// \param[in] eventService Pointer to a LwSciEventService object.
// \param[out] eventNotifier To be filled with the pointer to the created
//  LwSciEventNotifier object.
//
// \return ::LwSciError, the completion code of this operation.
// - ::LwSciError_Success The setup is successful.
// - ::LwSciError_BadParameter @a eventService is null or @a eventNotifier
//   is null.
// - ::LwSciError_IlwalidState An LwSciStream API has already been called on the
//   block referenced by @a block.
// - Error/panic behavior of this API includes
//    - Any error/panic behavior that LwSciEventService::CreateLocalEvent()
//      can generate when @a eventService and @a eventNotifier arguments
//      are passed to it.

/**
 * <b>Sequence of operations</b>
 *    - Retrieves the block instance referenced by the given LwSciStreamBlock
 *      by calling Block::getRegisteredBlock() interface.
 *    - Ilwokes Block::eventNotifierSetup() interface of the block instance.
 */
LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A8_4_8), "Approved TID-281")
LwSciError LwSciStreamBlockEventServiceSetup(
    LwSciStreamBlock const block,
    LwSciEventService *const eventService,
    LwSciEventNotifier ** const eventNotifier
)
{
    BlockPtr const blkPtr { Block::getRegisteredBlock(block) };
    if (nullptr == blkPtr) {
        return LwSciError_BadParameter;
    }
    if (nullptr == eventService) {
        return LwSciError_BadParameter;
    }
    if (nullptr == eventNotifier) {
        return LwSciError_BadParameter;
    }

    EventSetupRet const rv { blkPtr->eventNotifierSetup(*eventService) };
    if (LwSciError_Success == rv.err) {
        *eventNotifier = rv.eventNotifier;
    }
    return rv.err;

}

/**
 *  <b>Sequence of operations</b>
 *    - Retrieves the block instance referenced by the given LwSciStreamBlock
 *      by calling Block::getRegisteredBlock() interface.
 *    - Constructs an InfoPtr instance containing a new vector, and copies the
 *      incoming @a data into the vector.
 *    - Calls apiUserInfoSet() interface of the block instance with the
 *      @a userType and InfoPtr instance to add the user-defined info into the
 *      endpoint info list.
 */
LwSciError
LwSciStreamBlockUserInfoSet(
    LwSciStreamBlock const block,
    uint32_t const userType,
    uint32_t const dataSize,
    void const* const data)
{
    // Validate the input data
    if (nullptr == data) {
        return LwSciError_BadParameter;
    }

    // Retrieve block
    BlockPtr const blkPtr{ Block::getRegisteredBlock(block) };
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_1_1), "TID-300")
    if (nullptr == blkPtr) {
        return LwSciError_StreamBadBlock;
    }

    // Set up a shared pointer containing a new vector,
    // and copies in the incoming data.
    InfoPtr info{};
    try {
        info = std::make_shared<std::vector<uint8_t>>(dataSize);
    } catch (std::bad_alloc& e) {
        static_cast<void>(e);
        return LwSciError_InsufficientMemory;
    }

    static_cast<void>(memcpy(static_cast<void*>(info->data()),
                             static_cast<void const*>(data),
                             dataSize));

    // Ilwoke operation on block
    return blkPtr->apiUserInfoSet(userType, info);
}

/**
 *  <b>Sequence of operations</b>
 *    - Makes sure @a dataSize and @a data are valid.
 *    - Retrieves the block instance referenced by the given LwSciStreamBlock
 *      by calling Block::getRegisteredBlock() interface.
 *    - Calls apiUserInfoGet() interface of the block instance, passing the
 *      @a queryBlockType, @a queryBlockIndex, @a userType and a shared pointer
 *      to an empty vector, to retrieve the requested endpoint info.
 *    - Update the @a dataSize and copy the data from the vector into the
 *      @a data if it's not NULL.
 */
LwSciError
LwSciStreamBlockUserInfoGet(
    LwSciStreamBlock const block,
    LwSciStreamBlockType const queryBlockType,
    uint32_t const queryBlockIndex,
    uint32_t const userType,
    uint32_t* const dataSize,
    void* const data)
{
    LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A5_1_1), "TID-300")

    // Validate input
    if (nullptr == dataSize) {
        return LwSciError_BadAddress;
    }

    if ((0U != *dataSize) && (nullptr == data)) {
        return LwSciError_BadAddress;
    }

    // Retrieve block
    BlockPtr const blkPtr{ Block::getRegisteredBlock(block) };
    if (nullptr == blkPtr) {
        return LwSciError_StreamBadBlock;
    }

    // Ilwoke required operation on block
    InfoPtr info;
    LwSciError const err {
        blkPtr->apiUserInfoGet(queryBlockType, queryBlockIndex,
                               userType, info)
    };
    if (LwSciError_Success != err) {
        return err;
    }

    // Set the info size
    size_t const infoSize{ info->size() };
    if ((nullptr == data) || (infoSize <= static_cast<size_t>(*dataSize))) {
        *dataSize = static_cast<uint32_t>(infoSize);
    }

    // Copy the endpoint info
    if (nullptr != data) {
        static_cast<void>(memcpy(static_cast<void*>(data),
                                 static_cast<void const*>(info->data()),
                                 static_cast<size_t>(*dataSize)));
    }
    return LwSciError_Success;

    LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A5_1_1))
}

LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(M0_1_10))
LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A15_4_4))

} // extern "C"
