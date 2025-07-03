//! \file
//! \brief LwSciStream common internal data symbols.
//!
//! \copyright
//! Copyright (c) 2019-2021 LWPU Corporation. All rights reserved.
//!
//! LWPU Corporation and its licensors retain all intellectual property and
//! proprietary rights in and to this software, related documentation and any
//! modifications thereto. Any use, reproduction, disclosure or distribution
//! of this software and related documentation without an express license
//! agreement from LWPU Corporation is strictly prohibited.

#ifndef LWSCISTREAM_COMMON_H
#define LWSCISTREAM_COMMON_H
#ifdef __cplusplus
#include <cstdint>
#else
#include <stdint.h>
#endif
#include <utility>
#include <limits>
#include <array>
#include <vector>
#include <cassert>
#include <cstring>
#include <cstdlib>
#include <unordered_map>
#include <memory>
#include "covanalysis.h"
#include "sciwrap.h"
#include "lwscistream_types.h"
#include "lwscicommon_symbol.h"
#include "lwscicommon_os.h"
#include "lwscievent.h"

namespace LwSciStream {

// Constant symbols for the compliance with AUTOSAR C++ guideline.

//! \brief Constant to represent one.
constexpr uint32_t ONE  {1U};

//! \brief Constant to represent 8-bit one.
constexpr uint8_t CHAR_ONE  {1U};

//! \brief Constant to represent 64-bit one.
constexpr uint64_t LONG_ONE  {1UL};

//! \brief Constant to represent 64-bit all ones.
constexpr uint64_t LONG_MASK {~static_cast<uint64_t>(0U)};

//! \brief Constant to represent minus one, used for infinite timeout.
constexpr int64_t INFINITE_TIMEOUT {-1L};

// std::size_t is used to send sizes over IPC, so need to ensure that it is
//   the same for all processes. The simplest approach is to assert that
//   it is 64-bits. If this ever changes, may need to update the IPC code
//   to cast to a fixed size value.
static_assert(sizeof(uint64_t) == sizeof(size_t), "size_t must be 64-bit");

//! \brief Maximum integer represented as size_t. Will be used as max size
//!   for containers, so we can do math with size_t but ensure that we can
//!   stay within 32-bit bounds imposed by public APIs.
constexpr size_t MAX_INT_SIZE
    { static_cast<size_t>(std::numeric_limits<int32_t>::max()) };

//! \brief Constant to represent the maximum number of
//!  source connections allowed for a block.
//!
//! \implements{19721997}
//!
LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A0_1_1), "Bug 2738197")
LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(M0_1_4), "Approved TID-327")
constexpr uint32_t MAX_SRC_CONNECTIONS { 1U };
LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(M0_1_4))
LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A0_1_1))

//! \brief Constant to represent the maximum number of
//! destination connections allowed for a block.
//!
//! \implements{19722009}
//!
constexpr uint32_t MAX_DST_CONNECTIONS { 64U };

//! \brief Enum used by several data tracking objects to restrict them to a
//!   subset of their supported data entry functions for specific use cases.
enum class FillMode : uint8_t {
    //! \brief Data is not used.
    None,
    //! \brief Data is filled by inserting individual entries from user.
    User,
    //! \brief Data is merged in from several other objects.
    Merge,
    //! \brief Data is collated in from several other objects.
    Collate,
    //! \brief Data is filled by copying entire data set from another object.
    Copy,
    //! \brief Data is filled by unpacking from IPC.
    IPC
};

//! \brief Structure to represent the status information of a connection
//!  operation.
//!
//! \implements{19722039}
//!
LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A9_6_1), "Proposed TID-599")
struct IndexRet {
    //! \brief Represents the completion code of connection operation.
    LwSciError error;
    //! \brief Represents the connection index to which a particular block is
    //!  connected which is valid only when error code is LwSciError_Success.
    uint32_t   index;
};
LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A9_6_1))

//! \brief Structure used to return the status and result of an arithmetic
//!  operation.
//!
//! \implements{19722042}
//!
LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A9_6_1), "Proposed TID-599")
LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A14_1_1), "LwSciStream-ADV-AUTOSARC++14-002")
template <class intType>
struct ArithRet {
    //! \brief Status of an arithmetic operation.
    bool                status;
    //! \brief Result of an arithmetic operation which is valid only
    //!  if the status is true.
    intType             result;
};
LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A9_6_1))

//! \brief Function to add two integers.
//!  Returns the sum of the two input integers if no addition overflow.
//!
//! \param [in] var1: First operand.
//! \param [in] var2: Second operand.
//! \tparam intType: Any integer datatype.
//!
//! \return ArithRet<intType>, status and result of the operation
//!   - status: true, if no addition overflow.
//!             false, otherwise.
//!   - result: output of an arithmetic operation.
//!
//! \implements{19800969}
//!
LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A2_10_5), "Bug 200695637")
template <class intType>
inline auto uintAdd (intType const var1,
                     intType const var2) noexcept -> ArithRet<intType> {
    ArithRet<intType>  rv { true, (var1 + var2) };

    if (rv.result < var1) {
        LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_1_1), "TID-300")
        rv.status = false;
    }
    return rv;
}
LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A2_10_5))

// Forward declaration
class Block;

// Set of block interfaces which are declared as pure virtual functions.
// These interfaces are overridden by derived classes as required. They are
// called by a connected destination block through SafeConnection, ilwoking
// the actual implementation. In general, these interfaces receive information
// or requests from the destination block and act accordingly.
class SrcBlockInterface;

// Set of block interfaces which are declared as pure virtual functions.
// These interfaces are overridden by derived classes as required. They are
// called by a connected source block through SafeConnection, ilwoking the
// actual implementation. In general, these interfaces receive information
// or requests from the source block and act accordingly.
class DstBlockInterface;

//! \brief Enum represents the LwSciStream Block Types.
//!
//! \implements{19722057}
//!
enum class BlockType : uint8_t {
    //! \brief Invalid Block Type.
    NONE,
    //! \brief Producer Block Type.
    PRODUCER,
    //! \brief Consumer Block Type.
    CONSUMER,
    //! \brief Pool Block Type.
    POOL,
    //! \brief Queue Block Type.
    QUEUE,
    //! \brief Multicast Block Type.
    MULTICAST,
    //! \brief IpcSrc Block Type.
    IPCSRC,
    //! \brief IpcDst Block Type.
    IPCDST,
    //! \brief Limter Block Type.
    LIMITER,
    //! \brief ReturnSync Block Type.
    RETURNSYNC,
    //! \brief PresentSync Block Type.
    PRESENTSYNC
};

// Smart pointer types
//! \brief Alias for smart pointer of Block class.
//!
//! \implements{19722060}
//!
using BlockPtr = std::shared_ptr<Block>;

//! \brief Structure to represent the result of
//!  APIBlockInterface::eventNotifierSetup() call on a block.
//!
//! \implements{21697062}
//!
LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A9_6_1), "TID-599")
struct EventSetupRet {
    //! \brief Pointer to LwSciEventNotififer when err is
    //!  LwSciError_Success.
    LwSciEventNotifier *eventNotifier;
    //! \brief Completion code of the APIBlockInterface::eventNotifierSetup()
    //!  call on a block.
    LwSciError err;
};
LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A9_6_1))

//! \brief A simple special case of a scope-exit class (aka scope-guard) that
//!  just sets an atomic variable when leaving the scope.
//!
//! \tparam  T: Any type usable with std::atomic.
//
// TODO: Std C++ lwrrently has a scope-exit as an experimental class. If it
//       ever becomes official, we can make use of it.
template <class T>
class ScopeExitSet final
{
public:
    //! \brief Constructor saves reference and value
    ScopeExitSet(std::atomic<T>& paramRef, T const paramVal) noexcept :
        ref(paramRef), val(paramVal)
    { /* Nothing else to do */ };

    //! \brief Destructor copies value into reference
    ~ScopeExitSet(void)
    {
        ref.store(val);
    };

    // Other basic operations not allowed
    ScopeExitSet(void) noexcept                           = delete;
    ScopeExitSet(const ScopeExitSet&) noexcept            = delete;
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A8_4_8), "Bug 2745784")
    ScopeExitSet(ScopeExitSet&&) noexcept                 = delete;
    ScopeExitSet& operator=(const ScopeExitSet&) noexcept = delete;
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A8_4_8), "Bug 2745784")
    ScopeExitSet& operator=(ScopeExitSet&&) noexcept      = delete;

private:
    //! \brief Reference to variable to be updated on destruction.
    std::atomic<T>& ref;
    //! \brief Value to be saved in the reference on destruction.
    T               val;
};

//! \brief Alias for shared pointer to a vector.
using InfoPtr = std::shared_ptr<std::vector<uint8_t>>;

} // namespace LwSciStream
#endif // LWSCISTREAM_COMMON_H
