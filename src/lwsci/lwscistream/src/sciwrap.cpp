//! \file
//! \brief LwSciStream C++ wrappers for objects in other LwSci libraries.
//!
//! \copyright
//! Copyright (c) 2020-2021 LWPU Corporation. All rights reserved.
//!
//! LWPU Corporation and its licensors retain all intellectual property and
//! proprietary rights in and to this software, related documentation and any
//! modifications thereto. Any use, reproduction, disclosure or distribution
//! of this software and related documentation without an express license
//! agreement from LWPU Corporation is strictly prohibited.
#include <cstdint>
#include <cassert>
#include <array>
#include <functional>
#include "covanalysis.h"
#include "lwscierror.h"
#include "lwscibuf.h"
#include "lwscisync.h"
#include "lwsciipc.h"
#include "lwscisync_internal.h"
#include "lwscibuf_internal.h"
#include "sciwrap.h"

namespace LwSciWrap {

// Note: The values for ::cIlwalid are already defined in the header, but
//       the compiler still needs a separate "definition" in a source file.
//       This is a bit weird because it looks like we're declaring it in
//       the source file and defining it in the header.


//
// LwSciSyncAttrList wrapper support
//

// Invalid sync attribute list
constexpr LwSciSyncAttrList
WrapInfo<WrapType::WTSyncAttr>::cIlwalid;

//! <b>Sequence of operations</b>
//!  - Calls LwSciSyncAttrListClone() to duplicate the given LwSciSyncAttrList.
//!
//! \implements{19789053}
LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A8_4_4), "Bug 3264648")
LwSciError
LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A8_4_8), "Bug 2745784")
WrapInfo<WrapType::WTSyncAttr>::fCopy(
    LwSciSyncAttrList& attr) noexcept
{
    LwSciSyncAttrList  newAttr;
    LwSciError const   err { LwSciSyncAttrListClone(attr, &newAttr) };
    if (LwSciError_Success == err) {
        attr = newAttr;
    } else {
        attr = cIlwalid;
    }
    return err;
}
LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A8_4_4))

//! <b>Sequence of operations</b>
//!  - Frees the given LwSciSyncAttrList by calling
//!  LwSciSyncAttrListFree().
//!
//! \implements{19789059}
void
LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A8_4_8), "Bug 2745784")
WrapInfo<WrapType::WTSyncAttr>::fFree(
    LwSciSyncAttrList& attr) noexcept
{
    LwSciSyncAttrListFree(attr);
    attr = cIlwalid;
}

// Checks whether the given LwSciSyncAttrList is NULL or not.
bool
LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A8_4_8), "Bug 2745784")
WrapInfo<WrapType::WTSyncAttr>::fValid(
    LwSciSyncAttrList const& attr) noexcept
{
    return (cIlwalid != attr);
}

//! <b>Sequence of operations</b>
//!  - Merges the two input LwSciSyncAtterLists with
//!    LwSciSyncAttrListAppendReconciled() and overwrites the output location.
LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A8_4_4), "Bug 3264648")
LwSciError
LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A8_4_8), "Bug 2745784")
WrapInfo<WrapType::WTSyncAttr>::fMerge(
    LwSciSyncAttrList const& attr1,
    LwSciSyncAttrList const& attr2,
    LwSciSyncAttrList& newAttr) noexcept
{
    std::array<LwSciSyncAttrList,2U> const attrs { attr1, attr2 };
    LwSciError const err {
        LwSciSyncAttrListAppendUnreconciled(attrs.data(), attrs.size(),
                                           &newAttr)
    };
    if (LwSciError_Success != err) {
        newAttr = cIlwalid;
    }
    return err;
}
LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A8_4_4))


//
// LwSciSyncObj wrapper support
//

// Invalid sync object
constexpr LwSciSyncObj WrapInfo<WrapType::WTSyncObj>::cIlwalid;

//! <b>Sequence of operations</b>
//!  - Calls LwSciSyncObjRef() to create a reference of LwSciSyncObj.
//!
//! \implements{19789071}
LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A8_4_9), "Bug 2738197")
LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A8_4_4), "Bug 3264648")
LwSciError
LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A8_4_8), "Bug 2745784")
WrapInfo<WrapType::WTSyncObj>::fCopy(
    LwSciSyncObj&    obj) noexcept
{
    LwSciError const err { LwSciSyncObjRef(obj) };
    if (LwSciError_Success != err) {
        obj = cIlwalid;
    }
    return err;
}
LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A8_4_4))
LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A8_4_9))

//! <b>Sequence of operations</b>
//!  - Frees the given LwSciSyncObj by calling LwSciSyncObjFree().
//!
//! \implements{19789074}
void
LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A8_4_8), "Bug 2745784")
WrapInfo<WrapType::WTSyncObj>::fFree(
    LwSciSyncObj& obj) noexcept
{
    LwSciSyncObjFree(obj);
    obj = cIlwalid;
}

// Checks whether the given LwSciSyncObj is NULL or not.
bool
LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A8_4_8), "Bug 2745784")
WrapInfo<WrapType::WTSyncObj>::fValid(
    LwSciSyncObj const& obj) noexcept
{
    return (cIlwalid != obj);
}


//
// LwSciSyncFence wrapper support
//

// Invalid sync fence
constexpr LwSciSyncFence WrapInfo<WrapType::WTSyncFence>::cIlwalid;

//! <b>Sequence of operations</b>
//!  - Duplicates the given LwSciSyncFence by calling
//!  LwSciSyncFenceDup().
//!
//! \implements{19789086}
LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A8_4_4), "Bug 3264648")
LwSciError
LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A8_4_8), "Bug 2745784")
WrapInfo<WrapType::WTSyncFence>::fCopy(
    LwSciSyncFence&  fence) noexcept
{
    LwSciSyncFence   newFence { cIlwalid };
    LwSciError const err { LwSciSyncFenceDup(&fence, &newFence) };
    if (LwSciError_Success == err) {
        fence = newFence;
    } else {
        fence = cIlwalid;
    }
    return err;
}
LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A8_4_4))

//! <b>Sequence of operations</b>
//!  - Clears the given LwSciSyncFence by calling LwSciSyncFenceClear().
//!
//! \implements{19789089}
void
LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A8_4_8), "Bug 2745784")
WrapInfo<WrapType::WTSyncFence>::fFree(
    LwSciSyncFence& ref) noexcept
{
    LwSciSyncFenceClear(&ref);
    ref = cIlwalid;
}

// Checks whether the given LwSciSyncFence is cleared or not.
bool
LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A8_4_8), "Bug 2745784")
WrapInfo<WrapType::WTSyncFence>::fValid(
    LwSciSyncFence const& fence) noexcept
{
    // Asserts to detect changes to the fence structure
    constexpr uint32_t payloadSize { 6U };
    LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A16_2_3), "TID-1405")
    LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A5_2_2), "Proposed TID-1251")
    assert(sizeof(fence.payload) == sizeof(fence));
    assert(sizeof(fence.payload) == (payloadSize * sizeof(uint64_t)));
    LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A5_2_2))
    LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A16_2_3))
    LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A5_1_1), "TID-300")
    for (uint32_t i { 0U }; payloadSize > i; ++i) {
        if (0UL != fence.payload[i]) {
            return true;
        }
    }
    return false;
    LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A5_1_1))
}


//
// LwSciBufAttrList wrapper support
//

// Invalid buffer attribute list
constexpr LwSciBufAttrList WrapInfo<WrapType::WTBufAttr>::cIlwalid;

//! <b>Sequence of operations</b>
//!  - Duplicates the given LwSciBufAttrList by calling
//!  LwSciBufAttrListClone().
//!
//! \implements{19789104}
LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A8_4_4), "Bug 3264648")
LwSciError
LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A8_4_8), "Bug 2745784")
WrapInfo<WrapType::WTBufAttr>::fCopy(
    LwSciBufAttrList& attr) noexcept
{
    LwSciBufAttrList  newAttr;
    LwSciError const  err { LwSciBufAttrListClone(attr, &newAttr) };
    if (LwSciError_Success == err) {
        attr = newAttr;
    } else {
        attr = cIlwalid;
    }
    return err;
}
LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A8_4_4))

//! <b>Sequence of operations</b>
//!  - Frees the given LwSciBufAttrList by calling
//!  LwSciBufAttrListFree().
//!
//! \implements{19789107}
void
LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A8_4_8), "Bug 2745784")
WrapInfo<WrapType::WTBufAttr>::fFree(
    LwSciBufAttrList& attr) noexcept
{
    LwSciBufAttrListFree(attr);
    attr = cIlwalid;
}

// Checks whether the given LwSciBufAttrList is NULL or not.
bool
LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A8_4_8), "Bug 2745784")
WrapInfo<WrapType::WTBufAttr>::fValid(
    LwSciBufAttrList const& attr) noexcept
{
    return (cIlwalid != attr);
}

//! <b>Sequence of operations</b>
//!  - Merges the two input LwSciBufAtterLists with
//!    LwSciBufAttrListAppendReconciled() and overwrites the output location.
LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A8_4_4), "Bug 3264648")
LwSciError
LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A8_4_8), "Bug 2745784")
WrapInfo<WrapType::WTBufAttr>::fMerge(
    LwSciBufAttrList const& attr1,
    LwSciBufAttrList const& attr2,
    LwSciBufAttrList& newAttr) noexcept
{
    std::array<LwSciBufAttrList,2U> const attrs { attr1, attr2 };
    LwSciError const err {
        LwSciBufAttrListAppendUnreconciled(attrs.data(), attrs.size(),
                                           &newAttr)
    };
    if (LwSciError_Success != err) {
        newAttr = cIlwalid;
    }
    return err;
}
LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A8_4_4))


//
// LwSciBufObj wrapper support
//

// Invalid buffer object
constexpr LwSciBufObj WrapInfo<WrapType::WTBufObj>::cIlwalid;

//! <b>Sequence of operations</b>
//!  - Creates a reference of LwSciBufObj by calling LwSciBufObjRef().
//!
//! \implements{19790043}
LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A8_4_9), "Bug 2738197")
LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A8_4_4), "Bug 3264648")
LwSciError
LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A8_4_8), "Bug 2745784")
WrapInfo<WrapType::WTBufObj>::fCopy(
    LwSciBufObj&     obj) noexcept
{
    LwSciError const err { LwSciBufObjRef(obj) };
    if (LwSciError_Success != err) {
        obj = cIlwalid;
    }
    return err;
}
LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A8_4_4))
LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A8_4_9))

//! <b>Sequence of operations</b>
//!  - Frees the given LwSciBufObj by calling LwSciBufObjFree().
//!
//! \implements{19790046}
void
LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A8_4_8), "Bug 2745784")
WrapInfo<WrapType::WTBufObj>::fFree(
    LwSciBufObj& obj) noexcept
{
    LwSciBufObjFree(obj);
    obj = cIlwalid;
}

// Checks whether the given LwSciBufObj is NULL or not.
bool
LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A8_4_8), "Bug 2745784")
WrapInfo<WrapType::WTBufObj>::fValid(
    LwSciBufObj const& obj) noexcept
{
    return (cIlwalid != obj);
}

} // namespace LwSciWrap
