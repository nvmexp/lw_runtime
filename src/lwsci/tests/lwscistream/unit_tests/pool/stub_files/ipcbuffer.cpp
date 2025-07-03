//! \file
//! \brief LwSciStream intermediate buffer for IPC.
//!
//! \copyright
//! Copyright (c) 2021 LWPU Corporation. All rights reserved.
//!
//! LWPU Corporation and its licensors retain all intellectual property and
//! proprietary rights in and to this software, related documentation and any
//! modifications thereto. Any use, reproduction, disclosure or distribution
//! of this software and related documentation without an express license
//! agreement from LWPU Corporation is strictly prohibited.

#include "lwscistream_common.h"
#include "ipcbuffer.h"

namespace LwSciStream {

//! <b>Sequence of operations</b>
//! - Initializes buffer to Idle state.
//! - Allocates data vector for buffer to be the IPC frame size.
//! - - On allocation failure, sets @ initSuccess flag to false.
LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A12_1_5), "Bug 2989775")
IpcBuffer::IpcBuffer(
    size_t const size,
    LwSciIpcEndpoint const paramIpcEndpoint,
    LwSciSyncModule const paramSyncModule,
    LwSciBufModule const paramBufModule,
    bool const paramIsC2C) noexcept :
        frameSize(size),
        ipcEndpoint(paramIpcEndpoint),
        syncModule(paramSyncModule),
        bufModule(paramBufModule),
        isC2C(paramIsC2C),
        lwrrSize(size),
        data(),
        setOffset(0U),
        getOffset(0U),
        endOffset(0U),
        mode(UserMode::Idle),
        sizeLocked(false),
        initSuccess(true)
{
    try {
        LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A15_1_5), "Proposed TID-1219")
        LWCOV_ALLOWLIST_LINE(LWCOV_CERTCPP(ERR59_CPP), "TID-1475")
        data.resize(lwrrSize);
        LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A15_1_5))
    } catch (...) {
        LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_1_1), "TID-300")
        initSuccess = false;
    }
}

//! <b>Sequence of operations</b>
//! - Check if buffer is already large enough and if so, return success.
//! - Check if buffer is lociked, and if so return failure.
//! - Attempt to grow the buffer using the vector::resize() function,
//!   reporting success or failure and updating tracked size if needed.
LwSciError
IpcBuffer::grow(
    std::size_t const size) noexcept
{
    // Success if already big enough
    if (size <= lwrrSize) {
        return LwSciError_Success;
    }

    // Fail if size is locked
    if (sizeLocked) {
        return LwSciError_MessageSize;
    }

    // Try to grow
    try {
        LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A15_1_5), "Proposed TID-1219")
        LWCOV_ALLOWLIST_LINE(LWCOV_CERTCPP(ERR59_CPP), "TID-1475")
        data.resize(size);
        LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A15_1_5))
    } catch (...) {
        return LwSciError_InsufficientMemory;
    }

    // Update size
    lwrrSize = size;
    return LwSciError_Success;
}

//! <b>Sequence of operations</b>
//! - Check if buffer is already large enough to append @ dataSize bytes
//!   after the current @ setOffset. If so, return success.
//! - Attempt to grow the buffer to have enough room for the data.
LwSciError
IpcBuffer::packSizeCheck(
    std::size_t const dataSize) noexcept
{
    // If there's enough room already, return success
    if ((lwrrSize >= dataSize) && ((lwrrSize-dataSize) >= setOffset)) {
        return LwSciError_Success;
    }

    // Try to expand the data size
    return grow(setOffset+dataSize);
}

//! <b>Sequence of operations</b>
//! - Check if buffer has @ dataSize unread bytes after the current
//!   @ getOffset. If so, return success, otherwise indicate overflow.
LwSciError
IpcBuffer::unpackSizeCheck(
    std::size_t const dataSize) const noexcept
{
    return ((endOffset >= dataSize) && ((endOffset-dataSize) >= getOffset))
        ? LwSciError_Success : LwSciError_Overflow;
}

//! <b>Sequence of operations</b>
//! - If new mode is Pack, make sure old mode was Idle. Then set @ setOffset
//!   to begin adding data after leaving room for the total size at the
//!   beginning of the buffer.
//! - If new mode is Send, do nothing if already Send, otherwise  make sure
//!   previous mode was Pack and some data has been packed. Then save the
//!   total packed size, copy it to the beginning of the buffer, and set the
//!   @ getOffset to 0 for reading.
//! - If new mode is Recv, do nothing of already Recv, otherwise make sure
//!   previous mode was Idle. Then set @ setOffset to 0 to begin receiving
//!   data at the beginning of the buffer, and set the initial expected
//!   total size to the frame size.
//! - If new mode is Unpack, make sure previous mode was Recv and some data
//!   has been received. Then set @ getOffset to the beginning of the data
//!   (after the size), to begin unpacking.
//! - If new mode is Idle, set all offsets to 0 and await next operation.
//! - On success of any mode change, save the new mode.
LwSciError
IpcBuffer::changeMode(
    UserMode const newMode) noexcept
{
    // Update state based on mode
    switch (newMode) {

    case UserMode::Pack:
        // Pack mode only follows Idle
        if (UserMode::Idle != mode) {
            return LwSciError_IlwalidState;
        }

        // Reset set offest, leaving room to save the size
        setOffset = sizeof(setOffset);
        break;

    case UserMode::Send:
        // If already sending, do nothing
        if (UserMode::Send == mode) {
            return LwSciError_Success;
        }

        // Send mode only follows Pack and a message must have been packed
        if ((UserMode::Pack != mode) || (sizeof(setOffset) >= setOffset)) {
            return LwSciError_IlwalidState;
        }

        // After packing, copy the total size of the buffer to the beginning
        //  of the buffer, so it is the first part of the message.
        static_cast<void>(
            memcpy(static_cast<void*>(data.data()),
                   static_cast<void const*>(&setOffset),
                   sizeof(setOffset)));

        // Also copy the size to the end offset so we know when to stop sending
        endOffset = setOffset;

        // Get offset starts at 0
        LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_1_1), "TID-300")
        getOffset = 0U;
        break;

    case UserMode::Recv:
        // If already receiving, do nothing
        if (UserMode::Recv == mode) {
            return LwSciError_Success;
        }

        // Recv mode only follows Idle
        if (UserMode::Idle != mode) {
            return LwSciError_IlwalidState;
        }

        // Set offset starts at 0
        LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_1_1), "TID-300")
        setOffset = 0U;
        // End offset begins as the frame size. It will be updated when
        //   the first message is received.
        endOffset = frameSize;
        break;

    case UserMode::Unpack:
        // Unpack mode only follows Recv and message must have been received
        if ((UserMode::Recv != mode) ||
            ((sizeof(setOffset) >= setOffset) || (endOffset != setOffset))) {
            return LwSciError_IlwalidState;
        }

        // Get offset starts after the size
        getOffset = sizeof(getOffset);
        break;

    case UserMode::Idle:
        // TODO: Might restrict this to only happen after Send or Unpack, and
        //       have checks to make sure previous messages completed. But
        //       its not clear if there are cases (maybe during disconnect)
        //       where we just abort previous operations.
        // For now just clear the offsets
        LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A5_1_1), "TID-300")
        setOffset = 0U;
        getOffset = 0U;
        endOffset = 0U;
        LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A5_1_1))
        break;
    }

    // Change mode
    mode = newMode;

    return LwSciError_Success;
}

//! <b>Sequence of operations</b>
//! - Checks whether @ getOffset has reached the @ endOffset.
bool
IpcBuffer::sendDone(void) const noexcept
{
    return (endOffset == getOffset);
}

//! <b>Sequence of operations</b>
//! - Makes sure buffer is in Send mode.
//! - Makes sure there is data left in the buffer to send.
//! - Returns the pointer to the next part of the buffer to send, and the
//!   amount of data left, clamped to the frame size.
LwSciError
LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A8_4_4), "Bug 3264648")
IpcBuffer::sendInfoGet(
    IpcBuffer::CBlob& blob) const noexcept
{
    // Only allowed in Send mode
    if (UserMode::Send != mode) {
        return LwSciError_IlwalidState;
    }

    // Check for overflow
    //   This shouldn't be possible but CERT will probably want it
    if (getOffset >= endOffset) {
        return LwSciError_Overflow;
    }

    // Report size and pointer
    blob.size = std::min(frameSize, endOffset-getOffset);
    blob.data = &data[getOffset];
    return LwSciError_Success;
}

//! <b>Sequence of operations</b>
//! - Makes sure buffer is in Send mode.
//! - Makes sure @ size does not exceed amount of data left to send.
//! - Updates the amount of data that has been sent.
LwSciError
IpcBuffer::sendSizeAdd(
    std::size_t const size) noexcept
{
    // Only allowed in Send mode
    if (UserMode::Send != mode) {
        return LwSciError_IlwalidState;
    }

    // Must be at least this much data left
    if ((size > endOffset) || ((endOffset - size) < getOffset)) {
        return LwSciError_Overflow;
    }

    // Advance the get offset
    getOffset += size;
    return LwSciError_Success;
}

//! <b>Sequence of operations</b>
//! - Checks whether @ setOffset has reached the @ endOffset.
bool
IpcBuffer::recvDone(void) const noexcept
{
    return (endOffset == setOffset);
}

//! <b>Sequence of operations</b>
//! - Makes sure buffer is in Recv mode.
//! - Makes sure there is data left in the message to receive.
//! - Returns the pointer to the next part of the buffer to write into, and the
//!   amount of data left in the message, clamped to the frame size.
LwSciError
LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A8_4_4), "Bug 3264648")
IpcBuffer::recvInfoGet(
    IpcBuffer::VBlob& blob) noexcept
{
    // Only allowed in Recv mode
    if (UserMode::Recv != mode) {
        return LwSciError_IlwalidState;
    }

    // Check for overflow
    //   This shouldn't be possible but CERT will probably want it
    if (setOffset >= endOffset) {
        return LwSciError_Overflow;
    }

    // Report size and pointer
    blob.size = std::min(frameSize, endOffset-setOffset);
    blob.data = &data[setOffset];
    return LwSciError_Success;
}

//! <b>Sequence of operations</b>
//! - Makes sure buffer is in Recv mode.
//! - Makes sure @ size does not exceed amount of data left to receive.
//! - After reading first part of message and retrieving total size, if the
//!   buffer is not large enough for the whole message, grow it.
//! - Updates the amount of data that has been received.
LwSciError
IpcBuffer::recvSizeAdd(
    std::size_t const size) noexcept
{
    // Only allowed in Recv mode
    if (UserMode::Recv != mode) {
        return LwSciError_IlwalidState;
    }

    // Must be at least this much data left
    if ((size > endOffset) || ((endOffset - size) < setOffset)) {
        return LwSciError_Overflow;
    }

    // Temporary variable because the IPC functions behave oddly (see below)
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A0_1_1), "Bug 2738197")
    std::size_t tmpSize { size };

    // If this was the first part of the message, update the end offset
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_1_1), "TID-300")
    if (0U == setOffset) {
        // Read value from start of buffer
        assert(sizeof(setOffset) <= size);
        static_cast<void>(memcpy(static_cast<void*>(&endOffset),
                                 static_cast<void const*>(data.data()),
                                 sizeof(endOffset)));

        // Make sure buffer is large enough
        LwSciError const err { grow(endOffset) };
        if (LwSciError_Success != err) {
            return err;
        }

        // The IPC read functions don't behave like normal read functions.
        //   If you request a full frame's worth of data, but less than
        //   that much was sent, it will still report a full frame's worth
        //   was read. A normal read function would only report the actual
        //   amount of data that was available. We therefore need to adjust
        //   the size that came in before updating the offset.
        // TODO: We don't yet have any messages larger than a frame that
        //   causes us to wrap. When we do, we need to see what is reported
        //   as the amount read when we request less than a full frame.
        //   This function may require further tweaks. Also need to make
        //   sure the IPC function doesn't expect us to read out the whole
        //   emtpy part of the frame if we only need the first bit. If so
        //   we would have to grow in increments of the frame size and
        //   always request a full frame of data even if we don't need it.
        if (size > endOffset) {
            tmpSize = endOffset;
        }
    }

    // Advance the get offset
    setOffset += tmpSize;
    return LwSciError_Success;
}

//! <b>Sequence of operations</b>
//! - Packs size of blob in buffer with packVal().
//! - Makes sure there's room for blob with packSizeCheck().
//! - Copies blob data at current offset in buffer and updates offset.
LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A8_4_7), "Bug 2812980")
LwSciError
IpcBuffer::packBlob(
    IpcBuffer::CBlob const& blob) noexcept
{
    LwSciError err;

    LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A5_1_1), "TID-300")
    // Pack size
    err = packVal(blob.size);
    if (LwSciError_Success != err) {
        return err;
    }

    // Make sure there's enough room for blob
    err = packSizeCheck(blob.size);
    if (LwSciError_Success != err) {
        return err;
    }

    // Append data and increase offset
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_1_1), "TID-300")
    if (0UL < blob.size) {
        static_cast<void>(memcpy(static_cast<void*>(&data[setOffset]),
                                 blob.data,
                                 blob.size));
        // TODO: Deviation for A4-7-1. Already checked by packSizeCheck().
        setOffset += blob.size;
    }

    return LwSciError_Success;
    LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A5_1_1))
};
LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A8_4_7))

//! <b>Sequence of operations</b>
//! - Retrieves size of blob from buffer with unpackVal().
//! - Makes sure expected blob size does not exceed data in buffer with
//!   unpackSizeCheck().
//! - Retrieves pointer to blob data and updates offset.
LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A8_4_7), "Bug 2812980")
LwSciError
LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A8_4_4), "Bug 3264648")
IpcBuffer::unpackBlob(
    IpcBuffer::CBlob& blob) noexcept
{
    LwSciError err;

     // Extract size
    err = unpackVal(blob.size);
    if (LwSciError_Success != err) {
        return err;
    }

    // Make sure there's enough room for blob
    err = unpackSizeCheck(blob.size);
    if (LwSciError_Success != err) {
        return err;
    }

    // Return pointer to data and advance offset
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_1_1), "TID-300")
    if (0UL < blob.size) {
        blob.data = static_cast<void const*>(&data[getOffset]);
        // TODO: Deviation for A4-7-1. Already checked by unpackSizeCheck().
        getOffset += blob.size;
    } else {
        blob.data = nullptr;
    }

    return LwSciError_Success;
    LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A5_1_1))
};
LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A8_4_7))

} // namespace LwSciStream
