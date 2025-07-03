//! \file
//! \brief LwSciStream intermediate buffer for IPC.
//!
//! \copyright
//! Copyright (c) 2021-2022 LWPU Corporation. All rights reserved.
//!
//! LWPU Corporation and its licensors retain all intellectual property and
//! proprietary rights in and to this software, related documentation and any
//! modifications thereto. Any use, reproduction, disclosure or distribution
//! of this software and related documentation without an express license
//! agreement from LWPU Corporation is strictly prohibited.

#ifndef IPC_BUFFER_H
#define IPC_BUFFER_H

#include <type_traits>
#include <limits>
#include <vector>
#include <map>
#include "covanalysis.h"
#include "lwscistream_common.h"
#include "lwsciipc.h"

namespace LwSciStream {

//! \brief Utility object to provide an intermediate buffer for packing
//!        and unpacking data transmitted over IPC. It ensures alignment
//!        and overflow protection, and allows messages to be broken
//!        up over multiple LwSciIpc frames.
LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(M5_2_8), "Approved TID-465")
class IpcBuffer final
{
public:
    //! \brief Enum to indicate the current allowed operations
    //!   on the buffer. In general a buffer is either used for
    //!   transmission, and toggles between Pack and Send when not
    //!   Idle, or reception, and toggles between Recv and Unpack.
    enum class UserMode : uint8_t {
        //! \brief Not lwrrently handling data
        Idle,
        //! \brief Lwrrently packing new data for transmission
        Pack,
        //! \brief Lwrrently transmitting data
        Send,
        //! \brief Lwrrently receiving data
        Recv,
        //! \brief Lwrrently unpacking received data
        Unpack
    };

    //! \brief Structure used for passing constant blob pointer with size.
    LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A9_6_1), "Proposed TID-599")
    struct CBlob
    {
        //! \brief Size of the data.
        std::size_t size;
        //! \brief Pointer to the data.
        void const* data;
    };
    LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A9_6_1))

    //! \brief Structure used for passing non-constant blob pointer with size.
    LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A9_6_1), "Proposed TID-599")
    struct VBlob
    {
        //! \brief Size of the data.
        std::size_t size;
        //! \brief Pointer to the data.
        void*       data;
    };
    LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A9_6_1))

public:
    //! \brief Constructor, which initializes the buffer to @a size.
    //!  If the allocation fails, flag initSuccess will be set to false.
    //!
    //! \param [in] size: The IPC frame size.
    //! \param [in] paramIpcEndpoint: Endpoint asssciated with this buffer
    //! \param [in] paramSyncModule: Sync module associated with this buffer
    //! \param [in] paramBufModule: Buf module associated with this buffer
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A12_1_5), "Bug 2989775")
    IpcBuffer(size_t const size,
              LwSciIpcEndpoint const paramIpcEndpoint,
              LwSciSyncModule const paramSyncModule,
              LwSciBufModule const paramBufModule,
              bool const paramIsC2C) noexcept;

    //! \brief Default destructor
    ~IpcBuffer(void) noexcept                         = default;
    IpcBuffer(void) noexcept                          = delete;
    IpcBuffer(const IpcBuffer&) noexcept              = delete;
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A8_4_8), "Bug 2745784")
    IpcBuffer(IpcBuffer&&) noexcept                   = delete;
    IpcBuffer& operator=(const IpcBuffer&) & noexcept = delete;
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A8_4_8), "Bug 2745784")
    IpcBuffer& operator=(IpcBuffer&&) & noexcept      = delete;

    //! \brief Checks if IpcBuffer was created successfully.
    //!
    //! \return bool
    //! - true: Creation successful.
    //! - false: Creation failed.
    bool isInitSuccess(void) const noexcept
    {
        return initSuccess;
    };

    //! \brief Locks buffer so no further resizes are allowed.
    //!
    //! \return void
    void lockBuffer(void) noexcept
    {
        sizeLocked = true;
    };

    //! \brief Performs any necessary operations to complete current
    //!   mode and then transitions to new one.
    //!
    //! \param [in] newMode: Operational mode to switch to
    //!
    //! \return LwSciError, the completion code of this operation.
    //! * LwSciError_Success: The mode was changed successfully.
    //! * LwSciError_Ilwalid: The current state is not compatible with
    //!   @ newMode.
    LwSciError changeMode(UserMode const newMode) noexcept;

    //! \brief Packs a value, with size checking based on type.
    //!   This is used only for basic arithmetic and enum types. More
    //!   complex types use specialized functions layered on top of
    //!   this and the blob packing function.
    //!
    //! \param [in] val: Value to be packed in the buffer.
    //!
    //! \tparam T: Any arithmetic or enum type.
    //!
    //! \return LwSciError
    //! - LwSciError_Success - If packing succeeded
    //! - LwSciError_IlwalidState - If buffer is not in Pack mode
    //! - Any error returned by packSizeCheck().
    LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A8_4_7), "Bug 2812980")
    template <typename T>
    LwSciError packVal(T const& val) noexcept
    {
        // Only allowed for basic types. More complex objects must define
        //   specialized functions on top of the base unpacking functions.
        static_assert(std::is_arithmetic<T>::value || std::is_enum<T>::value,
                      "Requires basic type");

        // Only allowed in Pack mode
        if (UserMode::Pack != mode) {
            return LwSciError_IlwalidState;
        }

        // Make sure value of this type fits, growing if neeed and allowed
        constexpr std::size_t size { sizeof(T) };
        LwSciError const err { packSizeCheck(size) };
        if (LwSciError_Success != err) {
            return err;
        }

        // Append the data and increase offset
        static_cast<void>(memcpy(static_cast<void*>(&data[setOffset]),
                                 static_cast<void const*>(&val),
                                 size));
        setOffset += size;
        return LwSciError_Success;
    }
    LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A8_4_7))

    //! \brief Unpacks a value, with size checking based on type.
    //!   This is used only for basic arithmetic and enum types. More
    //!   complex types use specialized functions layered on top of
    //!   this and the blob unpacking function.
    //!
    //! \param [in,out] val: Value unpacked from the buffer.
    //!
    //! \tparam T: Any arithmetic or enum type.
    //!
    //! \return LwSciError
    //! - LwSciError_Success - If unpacking succeeded
    //! - LwSciError_IlwalidState - If buffer is not in Unpack mode
    //! - Any error returned by unpackSizeCheck().
    LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A8_4_9), "Bug 2738197")
    template <typename T>
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A8_4_8), "Bug 3264648")
    LwSciError unpackVal(T& val) noexcept
    {
        // Only allowed for basic types. More complex objects must define
        //   specialized functions on top of the base unpacking functions.
        static_assert(std::is_arithmetic<T>::value || std::is_enum<T>::value,
                      "Requires basic type");

        // Only allowed in Unpack mode
        if (UserMode::Unpack != mode) {
            return LwSciError_IlwalidState;
        }

        // Make sure size of this type doesn't exceed reamaining size
        constexpr std::size_t size { sizeof(T) };
        LwSciError const err { unpackSizeCheck(size) };
        if (LwSciError_Success != err) {
            return err;
        }

        // Extract the value and update offset
        static_cast<void>(memcpy(static_cast<void*>(&val),
                                 static_cast<void const*>(&data[getOffset]),
                                 size));
        getOffset += size;
        return LwSciError_Success;
    }
    LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A8_4_9))

    //! \brief Packs an opaque data blob, with size checking.
    //!
    //! \param [in] blob: Description of blob to pack
    //!
    //! \return LwSciError, the completion code of this operation.
    //! * LwSciError_Success: Data was successfully packed.
    //! * Any error returned by packVal().
    //! * Any error returned by packSizeCheck().
    LwSciError packBlob(IpcBuffer::CBlob const& blob) noexcept;

    //! \brief Unpacks an opaque data blob, with size checking.
    //!
    //! \param [in] blob: Description of unpacked blob.
    //!
    //! \return LwSciError, the completion code of this operation.
    //! * LwSciError_Success: Data was successfully retrieved.
    //! * Any error returned by unpackVal().
    //! * Any error returned by unpackSizeCheck().
    LwSciError unpackBlob(IpcBuffer::CBlob& blob) noexcept;

    //! \brief Checks whether there is data left to send.
    //!
    //! \return bool
    //! - true: There is packed data in the buffer waiting to be sent.
    //! - false: There is no more packed data in the buffer to send.
    bool sendDone(void) const noexcept;

    //! \brief Retrieves a block of data to send
    //!
    //! \param [in,out] blob: Description of data array to send
    //!
    //! \return LwSciError, the completion code of this operation.
    //! * LwSciError_Success: Data was retrieved to send.
    //! * LwSciError_IlwalidState: Buffer is in incorrect mode to send data.
    //! * LwSciError_Overflow: Attempting to send more data than exists.
    LwSciError sendInfoGet(CBlob& blob) const noexcept;

    //! \brief Report amount of data successfully sent
    //!
    //! \param [in] size: Amount of data sent.
    //!
    //! \return LwSciError, the completion code of this operation.
    //! * LwSciError_Success: Data state was updated.
    //! * LwSciError_IlwalidState: Buffer is in incorrect mode to send data.
    //! * LwSciError_Overflow: Amount of data left to send is less than @ size.
    LwSciError sendSizeAdd(std::size_t const size) noexcept;

    //! \brief Check whether there is data left to receive
    //!
    //! \return bool
    //! - true: The entire message has been received.
    //! - false: The entire message has not yet been received.
    bool recvDone(void) const noexcept;

    //! \brief Retrieves a block of memory to receive data
    //!
    //! \param [in,out] blob: Description of data array in which to store data.
    //!
    //! \return LwSciError, the completion code of this operation.
    //! * LwSciError_Success: Data pointer was retrieved.
    //! * LwSciError_IlwalidState: Buffer is in incorrect mode to receive data.
    //! * LwSciError_Overflow: Attempting to receive more data than exists.
    LwSciError recvInfoGet(VBlob& blob) noexcept;

    //! \brief Report amount of data received
    //!
    //! \param [in] size: Amount of data received.
    //!
    //! \return LwSciError, the completion code of this operation.
    //! * LwSciError_Success: Data state was updated.
    //! * LwSciError_IlwalidState: Buffer is in incorrect mode to receive data.
    //! * LwSciError_Overflow: Amount of data left to receive is less than
    //!   @ size.
    //! * Any error returned by grow().
    LwSciError recvSizeAdd(std::size_t const size) noexcept;

    //! \brief Retrieves the LwSciIpcEndpoint.
    LwSciIpcEndpoint ipcEndpointGet(void) const noexcept
    {
        return ipcEndpoint;
    };

    //! \brief Retrieves the LwSciSyncModule.
    LwSciSyncModule syncModuleGet(void) const noexcept
    {
        return syncModule;
    };

    //! \brief Retrieves the LwSciBufModule.
    LwSciBufModule bufModuleGet(void) const noexcept
    {
        return bufModule;
    };

    //! \brief Retrieves the C2C flag.
    bool isC2CGet(void) const noexcept
    {
        return isC2C;
    };

private:
    //! \brief Grows the buffer to at least the specified size if
    //!   necessary and allowed.
    //!
    //! \param [in] size: Desired minimum size for the buffer.
    //!
    //! \return LwSciError, the completion code of this operation.
    //! * LwSciError_Success: The buffer is at least the desired size.
    //! * LwSciError_MessageSize: The buffer is not sufficient size and
    //!   is locked so cannot be grown further.
    //! * LwSciError_InsufficientMemory: Allocating more memory for the
    //!   buffer failed.
    LwSciError grow(std::size_t const size) noexcept;

    //! \brief Checks for room to add data to the buffer, growing if
    //!   necessary and allowed.
    //!
    //! \param [in] dataSize: Amount of data to pack
    //!
    //! \return LwSciError, the completion code of this operation.
    //! * LwSciError_Success: The buffer has room to add the data.
    //! * Any error returned by grow()
    LwSciError packSizeCheck(std::size_t const dataSize) noexcept;

    //! \brief Checks for sufficient data to unpack from the buffer
    //!
    //! \param [in] dataSize: Amount of data to unpack
    //! \return LwSciError, the completion code of this operation.
    //! * LwSciError_Success: The buffer has at lease @ dataSize bytes to read.
    //! * LwSciError_Overflow: The buffer does not have enough unread bytes.
    LwSciError unpackSizeCheck(std::size_t const dataSize) const noexcept;

    //! \brief Size of IPC channel frames. It is initialized during
    //!   construction and is immutable for the life of the object.
    //!   When transmitting data, this is the largest size that will
    //!   be sent or received at once. It is used as the initial
    //!   size of the data buffer.
    std::size_t const       frameSize;
    //! \brief The IPC endpoint with which this buffer is associated.
    //!    It is needed for export/import if LwSci objects.
    //!    It is initialized by the owner during construction.
    LwSciIpcEndpoint const  ipcEndpoint;
    //! \brief The sync module with which this buffer is associated.
    //!    It is needed for export/import if LwSciSync objects.
    //!    It is initialized by the owner during construction.
    LwSciSyncModule const   syncModule;
    //! \brief The buf module with which this buffer is associated.
    //!    It is needed for export/import if LwSciBuf objects.
    //!    It is initialized by the owner during construction.
    LwSciBufModule const    bufModule;
    //! \brief Flag indicating whether the owning block uses C2C.
    //!    It is needed for certain pack/unpack functions.
    //!    It is initialized by the owner during construction.
    bool const              isC2C;
    //! \brief Current size of the buffer. It is initialized during
    //!   construction to the size of the IPC frames. It may be
    //!   increased if a message requires more room and must be broken
    //!   across multiple IPC frames.
    std::size_t          lwrrSize;
    //! \brief The buffer that holds the packed data that is sent or
    //!   received over the IPC channel. It is initalized during
    //!   construction to have the size of the IPC frames. It may
    //!   grow as needed during the initialization phase. Its size
    //!   is always lwrrSize, unless an error oclwrs.
    std::vector<uint8_t> data;
    //! \brief When packing or receiving data, the offset within the
    //!   data buffer at which new data is written.
    //!   It is initialized to 0 during construction.
    std::size_t          setOffset;
    //! \brief When unpacking or sending data, the offset within the
    //!   data buffer at which new data is read. It is initialized to 0
    //!   during construction.
    std::size_t          getOffset;
    //! \brief When reading or unpacking data, the offset of the end of
    //!   the data. It is initialized to 0 during construction.
    std::size_t          endOffset;
    //! \brief Current mode of operation for the buffer. It is initialized
    //!   to Idle during construction.
    UserMode             mode;
    //! \brief The flag indicating whether the buffer size is locked. It
    //!   is initialized to false, and set to true when runtime phase
    //!   begins. Growing the data buffer is no longer allowed once the
    //!   flag is set.
    bool                 sizeLocked;
    //! \brief The flag indicating if this object is successfully constructed.
    //!   Its value will be set to true if the constructor finishes without
    //!   error; otherwise the value will be false.
    bool                 initSuccess;
};
LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(M5_2_8))

// For all objects which will be packed into IpcBuffers, we require
//   ipcBufferPack() and ipcBufferUnpack() functions to be implemented.
//   For arithemetic and enum types, these are simple wrappers around
//   the the corresponding packVal/unpackVal functions provided by
//   IpcBuffer. For non-POD objects that LwSciStream defines, they will
//   be simple wrappers around pack/unpack member functions of the objects.
//   For POD objects or externally defined non-POD objects, the function
//   itself must handle packing of individual members.
//
// Templates are provided here for the arithemetic/enum wrappers, and
//   for a few templated std C++ objects. All others must be individually
//   declared where the objects are declared using overloaded functions.
//   To comply with AUTOSAR rules, do not specialize the templates here.
//
// TODO: With C++17, the "if constexpr()" feature could be used to make
//       these templates cover both the wrappers around the IpcBuffer
//       packVal/unpackVal functions and the non-POD object pack/unpack
//       functions, by checking the traits of the template parameter.
//       But for now, the latter functions must be handle explicitly for
//       each object.

//! \brief Wrapper to pack simple types using the IpcBuffer member function.
//!    This is needed to support templates which can operate on both simple
//!    and complex objects.
//!
//! \param [in,out] buf: Buffer in which the object will be packed.
//! \param [in] val: Value to be packed.
//!
//! \tparam T: Any arithmetic or enum type.
//!
//! \return LwSciError
//! - LwSciError_Success - Packing succeeded
//! - Any error returned by IpcBuffer::packVal().
LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A8_4_4), "Bug 3264648")
LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A8_4_7), "Bug 2812980")
LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A8_4_8), "Bug 3264648")
LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A8_4_9), "Bug 2738197")
template <typename T>
LwSciError ipcBufferPack(IpcBuffer& buf, T const& val) noexcept
LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A8_4_9))
{
    return buf.packVal(val);
}
LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A8_4_8))
LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A8_4_7))
LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A8_4_4))

//! \brief Wrapper to unpack simple types using the IpcBuffer member function.
//!    This is needed to support templates which can operate on both simple
//!    and complex objects.
//!
//! \param [in,out] buf: Buffer from which the object will be unpacked.
//! \param [in,out] val: Location in which to store the value.
//!
//! \tparam T: Any arithmetic or enum type.
//!
//! \return LwSciError
//! - LwSciError_Success - Unpacking succeeded
//! - Any error returned by IpcBuffer::unpackVal().
LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A8_4_4), "Bug 3264648")
LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A8_4_7), "Bug 2812980")
LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A8_4_8), "Bug 3264648")
LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A8_4_9), "Bug 2738197")
template <typename T>
LwSciError ipcBufferUnpack(IpcBuffer& buf, T& val) noexcept
LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A8_4_9))
{
    return buf.unpackVal(val);
}
LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A8_4_8))
LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A8_4_7))
LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A8_4_4))

//! \brief Packs a pair to an IpcBuffer.
//!
//! \param [in,out] buf: Buffer in which the object will be packed.
//! \param [in] val: Pair to be packed.
//!
//! \tparam T1: Any type which provides an ipcBufferPack() function.
//! \tparam T2: Any type which provides an ipcBufferPack() function.
// TODO: Need a means to assert this, along the lines of std C++ type_traits.
//!
//! \return LwSciError
//! - LwSciError_Success - If packing succeeded
//! - Any error returned by ipcBufferPack(T1).
//! - Any error returned by ipcBufferPack(T2).
LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A8_4_4), "Bug 3264648")
LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A8_4_7), "Bug 2812980")
LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A8_4_8), "Bug 3264648")
LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A8_4_9), "Bug 2738197")
template <typename T1, typename T2>
LwSciError ipcBufferPack(
    IpcBuffer& buf,
    std::pair<T1,T2> const& val) noexcept
LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A8_4_9))
LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A8_4_8))
{
    // Pack both values
    LwSciError err { ipcBufferPack(buf, val.first) };
    if (LwSciError_Success == err) {
        err = ipcBufferPack(buf, val.second);
    }
    return err;
}
LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A8_4_7))
LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A8_4_4))

//! \brief Unpacks a pair from an IpcBuffer.
//!
//! \param [in,out] buf: Buffer from which the pair will be unpacked.
//! \param [in,out] val: Location in which to store the pair.
//!
//! \tparam T1: Any type which provides an ipcBufferUnpack() function.
//! \tparam T2: Any type which provides an ipcBufferUnpack() function.
// TODO: Need a means to assert this, along the lines of std C++ type_traits.
//!
//! \return LwSciError
//! - LwSciError_Success - If unpacking succeeded
//! - Any error returned by ipcBufferUnpack(T1).
//! - Any error returned by ipcBufferUnpack(T2).
LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A8_4_4), "Bug 3264648")
LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A8_4_7), "Bug 2812980")
LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A8_4_8), "Bug 3264648")
LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A8_4_9), "Bug 2738197")
template <typename T1, typename T2>
LwSciError ipcBufferUnpack(
    IpcBuffer& buf,
    std::pair<T1,T2>& val) noexcept
LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A8_4_9))
LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A8_4_8))
{
    // Unpack both values
    LwSciError err { ipcBufferUnpack(buf, val.first) };
    if (LwSciError_Success == err) {
        err = ipcBufferUnpack(buf, val.second);
    }
    return err;
}
LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A8_4_7))
LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A8_4_4))

//! \brief Packs a vector to an IpcBuffer.
//!
//! \param [in,out] buf: Buffer in which the object will be packed.
//! \param [in] vec: Vector to be packed.
//!
//! \tparam T: Any type which provides an ipcBufferPack() function.
// TODO: Need a means to assert this, along the lines of std C++ type_traits.
//!
//! \return LwSciError
//! - LwSciError_Success - If packing succeeded
//! - LwSciError_Overflow - If vector size is more than 31 bits.
//! - Any error returned by ipcBufferPack(T).
LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A8_4_4), "Bug 3264648")
LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A8_4_7), "Bug 2812980")
LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A8_4_8), "Bug 3264648")
LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A8_4_9), "Bug 2738197")
template <typename T>
LwSciError ipcBufferPack(
    IpcBuffer& buf,
    std::vector<T> const& vec) noexcept
LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A8_4_9))
LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A8_4_8))
{
    // Retrieve size
    size_t const size { vec.size() };

    // All of our vectors should be kept within 31-bit size, but check
    //   just to be sure
    if (MAX_INT_SIZE < size) {
        return LwSciError_Overflow;
    }

    // Pack vector size
    LwSciError const err { buf.packVal(size) };
    if (LwSciError_Success != err) {
        return err;
    }

    // Pack all entries
    for (size_t i {0U}; size > i; i++) {
        LwSciError const err2 { ipcBufferPack(buf, vec[i]) };
        if (LwSciError_Success != err2) {
            return err2;
        }
    }

    return LwSciError_Success;
}
LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A8_4_7))
LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A8_4_4))

//! \brief Unpacks a vector from an IpcBuffer.
//!
//! \param [in,out] buf: Buffer from which the vector will be unpacked.
//! \param [in,out] vec: Location in which to store the vector.
//!
//! \tparam T: Any type which provides an ipcBufferUnpack() function.
// TODO: Need a means to assert this, along the lines of std C++ type_traits.
//!
//! \return LwSciError
//! - LwSciError_Success - If unpacking succeeded
//! - LwSciError_Overflow - If vector size is more than 31 bits.
//! - LwSciError_InconsistentData - Vector doesn't match incoming size.
//! - LwSciError_InsufficientMemory - Failed to allocate space for vector.
//! - Any error returned by ipcBufferUnpack(T).
LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A8_4_4), "Bug 3264648")
LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A8_4_7), "Bug 2812980")
LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A8_4_8), "Bug 3264648")
LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A8_4_9), "Bug 2738197")
template <typename T>
LwSciError ipcBufferUnpack(
    IpcBuffer& buf,
    std::vector<T>& vec) noexcept
LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A8_4_9))
LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A8_4_8))
{
    // Unpack vector size
    size_t size { 0U };
    LwSciError const err { buf.unpackVal(size) };
    if (LwSciError_Success != err) {
        return err;
    }

    // Check size in case of maliciously altered message
    if (MAX_INT_SIZE < size) {
        return LwSciError_Overflow;
    }

    // If the vector is already allocated, the incoming vector size must match
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_1_1), "TID-300")
    if ((vec.size() != 0UL) && (vec.size() != size)) {
        return LwSciError_InconsistentData;
    }

    // If the vector is not already allocated, do so
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_1_1), "TID-300")
    if ((vec.size() == 0UL) && (0U != size)) {
        try {
            LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A15_1_5), "Proposed TID-1219")
            LWCOV_ALLOWLIST_LINE(LWCOV_CERTCPP(ERR59_CPP), "TID-1475")
            vec.resize(size);
            LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A15_1_5))
        } catch (...) {
            return LwSciError_InsufficientMemory;
        }
    }

    // Unpack all entries
    for (size_t i {0U}; size > i; i++) {
        LwSciError const err2 { ipcBufferUnpack(buf, vec[i]) };
        if (LwSciError_Success != err2) {
            return err2;
        }
    }

    return LwSciError_Success;
}
LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A8_4_7))
LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A8_4_4))

//! \brief Unpacks a vector from an IpcBuffer, using auxiliary data.
//!   The incoming data size must be an integer multiple of that of
//!   the auxiliary data, the auxiliary vector is used as if there
//!   were multiple copies of it one after the other.
//!
//! \param [in,out] buf: Buffer from which the vector will be unpacked.
//! \param [in,out] vec: Location in which to store the vector.
//! \param [in] aux: Auxiliary data vector.
//!
//! \tparam T: Any type which provides an ipcBufferUnpack() function
//!   that uses A as auxiliary data.
// TODO: Need a means to assert this, along the lines of std C++ type_traits.
//! \tparam A: Any type.
//!
//! \return LwSciError
//! - LwSciError_Success - If unpacking succeeded
//! - LwSciError_Overflow - If vector size is more than 31 bits.
//! - LwSciError_InconsistentData - If vector size is incompatible with
//!   auxiiliary data size.
//! - LwSciError_InconsistentData - Vector doesn't match incoming size.
//! - LwSciError_InsufficientMemory - Failed to allocate space for vector.
//! - Any error returned by ipcBufferUnpack(T, A).
LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A8_4_4), "Bug 3264648")
LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A8_4_7), "Bug 2812980")
LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A8_4_8), "Bug 3264648")
LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A8_4_9), "Bug 2738197")
template <typename T, typename A>
LwSciError ipcBufferUnpack(
    IpcBuffer& buf,
    std::vector<T>& vec,
    std::vector<A> const& aux) noexcept
LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A8_4_9))
LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A8_4_8))
{
    LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A5_1_1), "TID-300")

    // Unpack vector size
    size_t size { 0U };
    LwSciError const err { buf.unpackVal(size) };
    if (LwSciError_Success != err) {
        return err;
    }

    // Check size in case of maliciously altered message
    if (MAX_INT_SIZE < size) {
        return LwSciError_Overflow;
    }

    // The size must be an integer multiple of the auxiliary data size
    size_t const auxSize { aux.size() };
    if ((0U == auxSize) || (0U != (size % auxSize))) {
        return LwSciError_InconsistentData;
    }

    // If the vector is already allocated, the incoming vector size must match
    if ((vec.size() != 0UL) && (vec.size() != size)) {
        return LwSciError_InconsistentData;
    }

    // If the vector is not already allocated, do so
    if ((vec.size() == 0UL) && (0U != size)) {
        try {
            LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A15_1_5), "Proposed TID-1219")
            LWCOV_ALLOWLIST_LINE(LWCOV_CERTCPP(ERR59_CPP), "TID-1475")
            vec.resize(size);
            LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A15_1_5))
        } catch (...) {
            return LwSciError_InsufficientMemory;
        }
    }

    // Unpack all entries, using the corresponding auxiliary data
    for (size_t i {0U}; size > i; i++) {
        LwSciError const err2 { ipcBufferUnpack(buf, vec[i], aux[i%auxSize]) };
        if (LwSciError_Success != err2) {
            return err2;
        }
    }

    return LwSciError_Success;

    LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A5_1_1))
}
LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A8_4_7))
LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A8_4_4))

//! \brief Packs a map to an IpcBuffer.
//!
//! \param [in,out] buf: Buffer in which the object will be packed.
//! \param [in] map: Map to be packed.
//!
//! \tparam TK: Any type which provides an ipcBufferPack() function.
//! \tparam TV: Any type which provides an ipcBufferPack() function.
// TODO: Need a means to assert this, along the lines of std C++ type_traits.
//!
//! \return LwSciError
//! - LwSciError_Success - If packing succeeded
//! - LwSciError_Overflow - If map size is more than 31 bits.
//! - Any error returned by ipcBufferPack(TK).
//! - Any error returned by ipcBufferPack(TV).
LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A8_4_4), "Bug 3264648")
LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A8_4_7), "Bug 2812980")
LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A8_4_8), "Bug 3264648")
LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A8_4_9), "Bug 2738197")
template <typename TK, typename TV>
LwSciError ipcBufferPack(
    IpcBuffer& buf,
    std::map<TK,TV> const& map) noexcept
LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A8_4_9))
LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A8_4_8))
{
    // Retrieve size
    size_t const size { map.size() };

    // All of our maps should be kept within 31-bit size, but check
    //   just to be sure
    if (MAX_INT_SIZE < size) {
        return LwSciError_Overflow;
    }

    // Pack map size
    LwSciError err { buf.packVal(size) };
    if (LwSciError_Success != err) {
        return err;
    }

    // Pack all entries
    for (auto const entry : map) {

        // Pack key
        err = ipcBufferPack(buf, entry.first);
        if (LwSciError_Success != err) {
            return err;
        }

        // Pack value
        err = ipcBufferPack(buf, entry.second);
        if (LwSciError_Success != err) {
            return err;
        }
    }

    return LwSciError_Success;
}
LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A8_4_7))
LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A8_4_4))

//! \brief Unpacks a map from an IpcBuffer.
//!
//! \param [in,out] buf: Buffer from which the map will be unpacked.
//! \param [in,out] map: Location in which to store the map.
//!
//! \tparam TK: Any type which provides an ipcBufferPack() function.
//! \tparam TV: Any type which provides an ipcBufferPack() function.
// TODO: Need a means to assert this, along the lines of std C++ type_traits.
//!
//! \return LwSciError
//! - LwSciError_Success - If unpacking succeeded
//! - LwSciError_AlreadyDone - If map already has data.
//! - LwSciError_Overflow - If map size is more than 31 bits.
//! - LwSciError_AlreadyInUse - If duplicate map entries are found.
//! - LwSciError_InsufficientMemory - Failed to allocate space for vector.
//! - Any error returned by ipcBufferUnpack(TK).
//! - Any error returned by ipcBufferUnpack(TV).
LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A8_4_4), "Bug 3264648")
LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A8_4_7), "Bug 2812980")
LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A8_4_8), "Bug 3264648")
LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A8_4_9), "Bug 2738197")
template <typename TK, typename TV>
LwSciError ipcBufferUnpack(
    IpcBuffer& buf,
    std::map<TK,TV>& map) noexcept
LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A8_4_9))
LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A8_4_8))
{
    // Make sure map is empty
    if (map.size() != 0U) {
        return LwSciError_AlreadyDone;
    }

    // Unpack map size
    size_t size { 0U };
    LwSciError err { buf.unpackVal(size) };
    if (LwSciError_Success != err) {
        return err;
    }

    // Check size in case of maliciously altered message
    if (MAX_INT_SIZE < size) {
        return LwSciError_Overflow;
    }

    // Unpack all entries
    for (size_t i {0U}; size > i; i++) {

        // Unpack key
        TK key;
        err = ipcBufferUnpack(buf, key);
        if (LwSciError_Success != err) {
            break;
        }

        // Unpack value
        TV val;
        err = ipcBufferUnpack(buf, val);
        if (LwSciError_Success != err) {
            break;
        }

        // Attempt to insert element in map, handling any exceptions
        try {
            LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A15_1_5), "Proposed TID-1219")
            auto const insertion
                { map.emplace(std::move(key), std::move(val)) };

            // Any failure that doesn't throw an exception should be because
            //   entry already exists, but that shouldn't happen unless the
            //   incoming data was corrupted.
            if (!insertion.second) {
                err = LwSciError_AlreadyInUse;
                break;
            }
            LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A15_1_5))
        } catch (...) {
            err = LwSciError_InsufficientMemory;
            break;
        }
    }

    // If there was a failure, we're probably doomed, but clean up the
    //   map so it can be used again.
    if (LwSciError_Success != err) {
        map.clear();
    }

    return err;
}
LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A8_4_7))
LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A8_4_4))

//! \brief Packs an object referenced by a shared pointer to an IpcBuffer.
//!
//! \param [in,out] buf: Buffer in which the object will be packed.
//! \param [in] ptr: Object referenced by a shared pointer to be packed.
//!
//! \tparam T: Any type which provides an ipcBufferPack() function.
// TODO: Need a means to assert this, along the lines of std C++ type_traits.
//!
//! \return LwSciError
//! - LwSciError_Success - If packing succeeded
//! - Any error returned by ipcBufferPack(T).
LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A8_4_4), "Bug 3264648")
LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A8_4_7), "Bug 2812980")
LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A8_4_8), "Bug 3264648")
LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A8_4_9), "Bug 2738197")
template <typename T>
LwSciError ipcBufferPack(
    IpcBuffer& buf,
    std::shared_ptr<T> const& ptr) noexcept
    LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A8_4_9))
    LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A8_4_8))
{
    // Check whether pointer is NULL
    bool const isNull{ nullptr == ptr };

    // Pack the null pointer indicator
    LwSciError const err{ buf.packVal(isNull) };
    if (LwSciError_Success != err) {
        return err;
    }

    // If pointer is NULL, nothing else to do
    if (isNull) {
        return LwSciError_Success;
    }

    // Ilwoke the function to pack the referenced object
    return ipcBufferPack(buf, *ptr);
}
LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A8_4_7))
LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A8_4_4))

//! \brief Unpacks an object referenced by a shared pointer from an IpcBuffer.
//!
//! \param [in,out] buf: Buffer from which the shared pointer will be unpacked.
//! \param [in,out] ptr: Location in which to store the object referenced by
//!                      the shared pointer.
//!
//! \tparam T: Any type which provides an ipcBufferUnpack() function.
// TODO: Need a means to assert this, along the lines of std C++ type_traits.
//!
//! \return LwSciError
//! - LwSciError_Success - If unpacking succeeded
//! - LwSciError_InsufficientMemory - Failed to allocate the new object.
//! - Any error returned by ipcBufferUnpack(T).
LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A8_4_4), "Bug 3264648")
LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A8_4_7), "Bug 2812980")
LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A8_4_8), "Bug 3264648")
LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A8_4_9), "Bug 2738197")
template <typename T>
LwSciError ipcBufferUnpack(
    IpcBuffer& buf,
    std::shared_ptr<T>& ptr) noexcept
    LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A8_4_9))
    LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A8_4_8))
{
    // Unpack null pointer indicator
    bool isNull{ false };
    LwSciError const err{ buf.unpackVal(isNull) };
    if (LwSciError_Success != err) {
        return err;
    }

    // If pointer is NULL, reset the shared pointer
    if (isNull) {
        ptr.reset();
        return LwSciError_Success;
    }

    try {
        LWCOV_ALLOWLIST_BEGIN(LWCOV_CERTCPP(ERR59_CPP), "TID-1475")
        LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A15_1_5), "Proposed TID-1219")
        ptr = std::make_shared<T>();
        LWCOV_ALLOWLIST_END(LWCOV_CERTCPP(ERR59_CPP))
    } catch (std::bad_alloc& e) {
        static_cast<void>(e);
        return LwSciError_InsufficientMemory;
    }

    // Ilwoke the function to unpack the referenced object
    LwSciError const err2{ ipcBufferUnpack(buf, *ptr) };
    if (LwSciError_Success != err2) {
        return err2;
    }

    return LwSciError_Success;
}
LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A8_4_7))
LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A8_4_4))


//! \brief Enum to identify messages sent over IPC/C2C.
// TODO: We'll eventually have a common IPC base class. This can be part of it.
enum class IpcMsg : uint32_t {

    // TODO: Separate upstream/downstream connection/phase change?

    //! \brief Default value for initializing variables.
    None                = 0x0000,
    //! \brief Stream connection.
    Connect             = 0x0100,
    //! \brief Stream disconnection.
    Disconnect          = 0x0110,
    //! \brief Switch to runtime phase.
    Runtime             = 0x0120,
    //! \brief Supported element information.
    SupportedElements   = 0x0200,
    //! \brief Allocated element information.
    AllocatedElements   = 0x0210,
    //! \brief IPC Packet definition.
    IPCPacketCreate     = 0x0220,
    //! \brief C2C Packet definition.
    C2CPacketCreate     = 0x0221,
    //! \brief IPC Packet status.
    IPCPacketStatus     = 0x0240,
    //! \brief C2C Packet status.
    C2CPacketStatus     = 0x0241,
    //! \brief IPC Packet deletion.
    IPCPacketDelete     = 0x0260,
    //! \brief C2C Packet deletion.
    C2CPacketDelete     = 0x0261,
    //! \brief IPC Packet list completion.
    IPCPacketsComplete  = 0x0280,
    //! \brief C2C Packet list completion.
    C2CPacketsComplete  = 0x0281,
    //! \brief Waiter sync attributes for IPC blocks
    IPCWaiterAttr       = 0x0300,
    //! \brief Waiter sync attributes coordination for C2C blocks
    C2CWaiterAttr       = 0x0301,
    //! \brief Signal sync attributes.
    SignalAttr          = 0x0320,
    //! \brief Waiter sync objects.
    WaiterObj           = 0x0340,
    //! \brief Signal sync objects.
    SignalObj           = 0x0360,
    //! \brief IPC Payload.
    IPCPayload          = 0x0400,
    //! \brief C2C payload.
    C2CPayload          = 0x0401
};

} // namespace LwSciStream
#endif // IPC_BUFFER_H
