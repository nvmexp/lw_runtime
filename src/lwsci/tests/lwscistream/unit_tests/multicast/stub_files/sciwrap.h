//! \file
//! \brief LwSciStream C++ wrappers for objects in other LwSci libraries.
//!
//! \copyright
//! Copyright (c) 2020-2022 LWPU Corporation. All rights reserved.
//!
//! LWPU Corporation and its licensors retain all intellectual property and
//! proprietary rights in and to this software, related documentation and any
//! modifications thereto. Any use, reproduction, disclosure or distribution
//! of this software and related documentation without an express license
//! agreement from LWPU Corporation is strictly prohibited.

// TODO: Proper doxygen formatting for this comment.
//
// Rationale:
//
// When LwSciSync/Buf handles and fences are passed in to the public
//   LwSciStream APIs, they are assumed to remain the property of the
//   caller, which is free to release their reference to them at any
//   time after the functions return. If LwSciStream (or anything it
//   communicates with) needs to preserve these objects, it must
//   duplicate or refcount them, as appropriate for the object type.
// Colwersely, when such objects are received from LwSciStream through
//   an event or payload function, we must provide ones which belongs
//   to the recipient. So if LwSciStream needs to maintain its own
//   copy, it must again duplicate or refcount them.
// Tracking ownership of these objects and determining when to duplicate
//   them and when to free them can be quite complicated. We wish to
//   avoid any unnecessary duplication, but still satisfy the requirements
//   of the interfaces. No one approach for deciding fits all use cases,
//   because of the many ways different stream blocks can be connected.
// This unit provides a set of generic wrappers for tracking LwSci objects
//   as they move through a stream, along with a set of uniform rules for
//   how to use them. Following these rules will ensure correct and
//   near-optimal behavior for all use cases.
//
// Design:
//
// The wrapper takes the form of a template parametrized by an object
//   type and an enum value. The enum is lwrrently redundant but is
//   provided for future proofing. If LwSci handles are ever changed
//   to use generic pointers/integers rather than pointers to opaque
//   structures, they would resolve to the same underlying type and
//   our templates would stop working correctly. The enum value ensures
//   we distinguish each type LwSci object.
// The wrapper contains the object and a flag indicating whether the
//   wrapper owns the object (and is therefore responsible for freeing
//   it if it is still present when the wrapper is destroyed). It also
//   has a field for tracking any errors which occur during duplication
//   of the object. We lwrrently checks this flag in only a few places,
//   but could make more extensive use of it if necessary for safety
//   certification.
// There are three situations in which we receive raw LwSci objects:
//   * An object may be passed in through a public API, in which case
//     we insert it in a wrapper with the flag indicating that we do
//     not own it.
//   * An object may be imported over IPC in which case we insert it
//     in a wrapper which does own it.
//   * An object may be received in an internal event structure, in
//     which case we again insert it in a wrapper which does own it.
// The objects pass through all Block functions as non-const references
//   to the wrapper object. It would be preferable to use rvalue references
//   rather than lvalue references (i.e. consume parameters) which would
//   better indicate that they are intended to be transitory objects and
//   blocks are encouraged to move data out of them (avoiding unnecessary
//   copies). However autosar rules forbid consume parameters that are
//   not always consumed, and we would run into violations with those
//   functions that don't need to access or pass on the contents.
// As these wrappers pass through, we may need to store, copy, or access
//   the LwSci object. These actions are handled uniformly by template
//   functions.
//   * Move constructor and operator will take possession of any object
//     in the original, copying the ownership flag as well, and clearing
//     the original so it will no longer be responsible for freeing the
//     object (if owned).
//   * Copy constructor and operator will create a new duplicate of the
//     object, which the new wrapper will always own. The original is
//     unaffected.
//   * Frequently, we need to create a new wrapper which owns its object,
//     but want to avoid an extra duplication if we already have a copy
//     that we own. For this, we use a special copy function which checks
//     the original. If the object is not owned, then a dup is made. But
//     if it is owned, then it is moved to the new wrapper.
//   * When performing an LwSci operation that needs the object as input,
//     we need to retrieve it from the wrapper without affecting ownership,
//     so we provide a getter function for this.
//   * When returning an object in an event, we need to obtain a copy
//     whose ownership can be passed with the event. We provide another
//     getter function for this which behaves as our special copy function.
//     If the wrapper owns the object, then it moves it out of the wrapper.
//     If the wrapper does not own it, it makes a duplicate.
// When a wrapper is destroyed, if it owned the object, then the object
//   is freed. So LwSci objects that are owned by wrappers that are part
//   of a Block or other structure will automatically be cleaned up when
//   that structure is destroyed. LwSci objects owned by wrappers on the
//   stack will automatically be cleaned up when they go out of scope,
//   unless a function which was ilwoked during their scope took ownership.
//   Therefore, outside of the template itself, we never need to call an
//   LwSci free operation. Meanwhile, objects passed in from an API which
//   are never owned by LwSciStream are left untouched.

#ifndef SCIWRAP_H
#define SCIWRAP_H

#include <cstdint>
#include <cstddef>
#include <cstring>
#include <cassert>
#include <utility>
#include "covanalysis.h"
#include "lwscierror.h"
#include "lwscisync_internal.h"
#include "lwscibuf_internal.h"

/**
 * @defgroup lwscistream_blanket_statements LwSciStream blanket statements.
 * Generic statements applicable for LwSciStream interfaces.
 * @{
 */

/**
 * \page lwscistream_page_blanket_statements LwSciStream blanket statements
 * \section lwscistream_lwscistream_input_parameters Input parameters
 * - LwSciWrap::SyncAttr reference passed as an input parameter to an interface
 *   is valid if it contains a valid LwSciSyncAttrList, it can also be empty if the
 *   stream is in synchronous mode (if synchronousOnly parameter for the
 *   LwSciStreamBlockSyncRequirements() API from the producer or consumer(s)
 *   is received as true).
 * - LwSciWrap::SyncObj reference passed as an input parameter to an interface
 *   is valid if it contains a valid LwSciSyncObj.
 * - LwSciWrap::BufObj reference passed as an input parameter to an interface
 *   is valid if it contains a valid LwSciBufObj.
 * - LwSciWrap::BufAttr reference passed as an input parameter to an interface
 *   is valid if it contains a valid LwSciBufAttrList.
 * - LwSciWrap::SyncFence reference passed as an input parameter to an
 *   interface is valid if it contains a valid LwSciSyncFence.
 *
 * \section lwscistream_lwsciwrap_objects LwSciWrap Objects
 * - LwSciStream public APIs creates Wrapper objects for the received LwSciSync
 *   or LwSciBuf handle with the ownership flag as false as ownership of that
 *   object remains with the application. The Wrapper objects are passed through
 *   the blocks as references, so no change in the ownership is expected when
 *   the Wrappers are passing through the blocks from public APIs. When the
 *   blocks or related utility objects need to preserve these objects, they
 *   create their own Wrapper instances by creating a duplicate or reference
 *   of the objects with the copy or move constructors/operators provided by
 *   the Wrapper class.
 * - When a wrapper is destroyed, if it owned the object, then the object is freed.
 *
 * \section lwscistream_dependency Dependency on other sub-elements
 *   LwSciStream calls the following LwSciSync interfaces:
 *    - LwSciSyncAttrListClone() to duplicate a LwSciSyncAttrList.
 *    - LwSciSyncAttrListFree() to free the duplicate of LwSciSyncAttrList.
 *    - LwSciSyncObjRef() to create a new reference of LwSciSyncObj.
 *    - LwSciSyncObjFree() to free the reference of LwSciSyncObj.
 *    - LwSciSyncFenceDup() to duplicate a LwSciSyncFence.
 *    - LwSciSyncFenceClear() to free the duplicate of LwSciSyncFence.
 * \section lwscistream_dependency Dependency on other sub-elements
 *   LwSciStream calls the following LwSciBuf interfaces:
 *    - LwSciBufAttrListClone() to duplicate a LwSciBufAttrList.
 *    - LwSciBufAttrListFree() to free the duplicate of LwSciBufAttrList.
 *    - LwSciBufObjRef() to create a new reference of LwSciBufObj.
 *    - LwSciBufObjFree() to free the reference of LwSciBufObj.
 */

/**
 * @}
 */

// In theory, these could be used in other C++ libraries too, so we give
//   them their own namespace instead of using LwSciStream.
namespace LwSciWrap {

//! \brief Enum to identify the type of LwSciBuf/LwSciSync
//!   object being wrapped by the Wrapper class. This is used
//!   instead of the wrapped object's actual type, because this is guaranteed
//!   to be unique, while the object types are not. That is, objects could
//!   be represented by opaque handles which map to integers or void*.
//!
//! \implements{19788561}
enum class WrapType : std::uint8_t
{
    //! \brief Wrapper for LwSciSyncAttrList
    WTSyncAttr,
    //! \brief Wrapper for LwSciSyncObj
    WTSyncObj,
    //! \brief Wrapper for LwSciSyncFence
    WTSyncFence,
    //! \brief Wrapper for LwSciBufAttrList
    WTBufAttr,
    //! \brief Wrapper for LwSciBufObj
    WTBufObj,
};

//! \brief Empty generic WrapInfo class template. A specialization
//!   of this class with actual contents will be provided for each WrapType.
//!
//! \tparam id: Any one of the WrapType.
//!
//! \implements{19788567}
template <WrapType id>
class WrapInfo
{
};

//! \brief Template class for managing duplicates or references of
//!   LwSciSync/LwSciBuf handles.
//!
//! \tparam  T: Any one of the following: LwSciSyncAttrList, LwSciSyncObj,
//!   LwSciSyncFence, LwSciBufAttrList, LwSciBufObj.
//! \tparam  id: WrapType corresponding to T. There will be one to one
//!   mapping as follows between T and id values:
//!   - T: LwSciSyncAttrList, id: WrapType::WTSyncAttr
//!   - T: LwSciSyncObj, id: WrapType::WTSyncObj
//!   - T: LwSciSyncFence, id: WrapType::WTSyncFence
//!   - T: LwScibufAttrList, id: WrapType::WTBufAttr
//!   - T: LwSciBufObj, id: WrapType::WTBufObj
//!
//! \if TIER4_SWAD
//! \implements{19788570}
//! \endif
//!
//! \if TIER4_SWUD
//! \implements{21231396}
//! \endif
LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A14_1_1), "LwSciStream-ADV-AUTOSARC++14-002")
template<class T, WrapType id>
class Wrapper final
{
private:
    //! \cond TIER4_SWUD
    //! \brief Shorthand for the WrapInfo<id>.
    using info = WrapInfo<id>;
    //! \endcond

    LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A8_4_8), "Bug 2745784")
    // Swap values
    static void swapMembers(Wrapper& Obj1, Wrapper& Obj2) noexcept
    {
        std::swap(Obj1.val, Obj2.val);
        std::swap(Obj1.err, Obj2.err);
        std::swap(Obj1.own, Obj2.own);
    }
    LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A8_4_8))
public:
    //! \brief Delegating constructor used by all other constructors which
    //!  initializes the LwSciSync or LwSciBuf handle managed by this wrapper
    //!  instance and ownership flag from the given arguments.
    //!
    //! \if TIER4_SWUD
    //! <b>Sequence of operations</b>
    //! - It checks whether the @a paramVal is valid or not by calling
    //!   info::fValid() interface. If @a paramVal is valid and @a paramErr is
    //!   LwSciError_Success, it copies the @a paramVal argument to LwSciSync or
    //!   LwSciBuf handle which this wrapper instance manages. If @a paramVal is
    //!   not valid or @a paramErr is not LwSciError_Success, it initializes
    //!   LwSciSync or LwSciBuf handle to info::cIlwalid, sets the ownership
    //!   flag as false and returns.
    //! - If @a paramDup argument is true, it duplicates or creates a reference
    //!   of the LwSciSync or LwSciBuf handle by calling info::fCopy() interface
    //!   then it updates the ownership flag as true if info::fCopy() is
    //!   successful, false otherwise. If @a paramDup argument is false, it just
    //!   copies the @a paramOwn argument value to ownership flag and returns.
    //! \endif
    //!
    //! \tparam T: Any one of the following:
    //!  - LwSciSyncAttrList if WrapType of this Wrapper instance is WTSyncAttr.
    //!  - LwSciSyncObj if WrapType of this Wrapper instance is WTSyncObj.
    //!  - LwSciSyncFence if WrapType of this Wrapper instance is WTSyncFence.
    //!  - LwSciBufAttrList if WrapType of this Wrapper instance is WTBufAttr.
    //!  - LwSciBufObj if WrapType of this Wrapper instance is WTBufObj.
    //!
    //! \param [in] paramVal: LwSciSync or LwSciBuf handle.
    //!  Default value for this argument is info::cIlwalid.
    //!  Valid value: Refer to template parameter T.
    //! \param [in] paramOwn: Flag to indicate whether this wrapper
    //!  will own the handle. Default value for this argument is false.
    //! \param [in] paramDup: Flag to indicate whether this wrapper
    //!  should create a duplicate or reference of the @a paramVal.
    //!  Default value for this argument is false.
    //! \param [in] paramErr: LwSciError passed from a previous failure.
    //!  Default value for this argument is LwSciError_Success.
    //!
    //! \if TIER4_SWAD
    //! \implements{20125125}
    //! \endif
    //!
    //! \if TIER4_SWUD
    //! \implements{19788861}
    //! \endif
    Wrapper(
        T const&         paramVal=info::cIlwalid,
        bool const       paramOwn=false,
        bool const       paramDup=false,
        LwSciError const paramErr=LwSciError_Success) noexcept :
            val(paramVal),
            own(paramOwn),
            err(paramErr)
    {
        if ((LwSciError_Success != err) || !info::fValid(val)) {
            // If error or invalid value, clear the value and ownership.
            //   (Validity check is needed in case we were told to dup an
            //    empty wrapper.)
            val = info::cIlwalid;
            LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_1_1), "TID-300")
            own = false;
        } else {
            if (paramDup) {
                // If duplication is required, copy/ref in place.
                LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_2_2), "Bug 3002886")
                    err = info::fCopy(val);
                own = own && (LwSciError_Success == err);
            }
        }
    };

    LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A12_1_1), "Bug 2805307")
    LWCOV_ALLOWLIST_BEGIN(LWCOV_CERTCPP(OOP58_CPP), "Proposed TID-1419")
    //! \brief Copy constructor which always duplicates the LwSciSync or
    //!  LwSciBuf handle managed by the original. The original is unaffected.
    //!
    //! \if TIER4_SWUD
    //! <b>Sequence of operations</b>
    //!  - It calls the delegating constructor with the LwSciSync or LwSciBuf
    //!  handle and LwSciError value of the original along with the paramOwn
    //!  and paramDup arguments as true.
    //! \endif
    //!
    //! \param [in] other: Reference to Wrapper instance to be copied.
    //!
    //! \if TIER4_SWAD
    //! \implements{20125134}
    //! \endif
    //!
    //! \if TIER4_SWUD
    //! \implements{19788864}
    //! \endif
    Wrapper(Wrapper const& other) noexcept :
        LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_1_1), "TID-300")
        Wrapper(other.val, true, true, other.err)
    LWCOV_ALLOWLIST_END(LWCOV_CERTCPP(OOP58_CPP))
    LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A12_1_1))
    {
    };

    LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A5_1_1), "TID-300")
    LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A12_1_1), "Bug 2805307")
    LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A8_4_8), "Bug 2745784")
    //! \brief Move constructor moves the LwSciSync or LwSciBuf handle managed
    //!  by the original.
    //!
    //! \if TIER4_SWUD
    //! <b>Sequence of operations</b>
    //!  - It calls the delegating constructor with the LwSciSync
    //!  or LwSciBuf handle, LwSciError and paramOwn value of the original and
    //!  paramDup argument set to false.
    //! \endif
    //!
    //! \param [in,out] other: Reference to wrapper to be moved.
    //!
    //! \if TIER4_SWAD
    //! \implements{20125185}
    //! \endif
    //!
    //! \if TIER4_SWUD
    //! \implements{19788867}
    //! \endif
    Wrapper(Wrapper&& other) noexcept :
        Wrapper(std::exchange(other.val, info::cIlwalid),
                std::exchange(other.own, false),
                false,
                std::exchange(other.err, LwSciError_Success))
    {
    };
    LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A8_4_8))
    LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A12_1_1))
    LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A5_1_1))

    LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A6_2_1), "Bug 3029538")
    LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A5_1_1), "TID-300")
    LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A7_1_5), "Bug 2751189")
    //! \brief Copy operator which always duplicates the LwSciSync or LwSciBuf
    //!  handle managed by the original. The original is unaffected.
    //!
    //! \param [in] other: Reference Wrapper instance to be copied.
    //!
    //! \return Current Wrapper instance.
    //!
    //! \if TIER4_SWAD
    //! \implements{20141004}
    //! \endif
    //!
    auto operator=(Wrapper const& other) noexcept -> Wrapper&
    {
        Wrapper tmp{other};
        swapMembers(*this, tmp);

        return *this;
    };
    LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A7_1_5))
    LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A5_1_1))
    LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A6_2_1))

    LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A6_2_1), "Bug 3029538")
    LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A7_1_5), "Bug 2751189")
    LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A8_4_8), "Bug 2745784")
    //! \brief Move operator copies all fields and clears the original.
    //!
    //! \param [in,out] other: Reference Wrapper instance to be moved from.
    //!
    //! \return Current Wrapper instance.
    //!
    //! \if TIER4_SWAD
    //! \implements{20141007}
    //! \endif
    auto operator=(Wrapper&& other) noexcept -> Wrapper&
    LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A8_4_8))
    LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A7_1_5))
    {
        Wrapper tmp{std::move(other)};
        swapMembers(*this, tmp);

        return *this;
    };
    LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A6_2_1))

    LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A7_1_5), "Bug 2751189")
    LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A8_4_8), "Bug 2745784")
    //! \brief Hybrid move/copy operation which always returns a new Wrapper.
    //!
    //! \if TIER4_SWUD
    //! <b>Sequence of operations</b>
    //! - If the original owns the value, its contents will be moved to the new
    //!   Wrapper instance.
    //! - Otherwise it will be copied to the new Wrapper instance.
    //! \endif
    //!
    //! \return New Wrapper.
    //!
    //! \if TIER4_SWAD
    //! \implements{20141001}
    //! \endif
    //!
    //! \if TIER4_SWUD
    //! \implements{20140659}
    //! \endif
    auto take(void) noexcept -> Wrapper
    LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A8_4_8))
    LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A7_1_5))
    {
        Wrapper newWrapper{};
        if (own) {
            newWrapper = std::move(*this);
        } else {
            newWrapper = *this;
        }
        return newWrapper;
    };

    //! \brief Destructor which frees underlying LwSciSync or LwSciBuf handle
    //!  if it is owned by the wrapper instance.
    //!
    //! \if TIER4_SWUD
    //! <b>Sequence of operations</b>
    //!  - If the LwSciSync or LwSciBuf handle is valid and owned by this
    //!  Wrapper instance, it is freed by calling the info::fFree() interface.
    //! \endif
    //!
    //! \if TIER4_SWAD
    //! \implements{20125212}
    //! \endif
    //!
    //! \if TIER4_SWUD
    //! \implements{19788873}
    //! \endif
    ~Wrapper(void) noexcept
    {
        if (own) {
            LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_2_2), "Bug 3002886")
            info::fFree(val);
        }
    };

    //! \brief Merges another Wrapper into this one. Any failures are
    //!   reflected in the Wrapper's error value.
    //!
    //! \param [in] other: Other Wrapper instance to be merged in.
    //! \param [in] ilwalidOkay: Indicates how to handle invalid values.
    //!   If true, then if one value is invalid, the other is used.
    //!   If false, then if either value is invalid, the merge is invalid.
    //!
    //! \return void
    //!
    //! \if TIER4_SWUD
    //! <b>Sequence of operations</b>
    //! - Set up a temporary Wrapper with default constructor.
    //! - If this Wrapper has an error, copy it to the temporary Wrapper.
    //! - Otherwise, if the other Wrapper has an error, copy it to the
    //!   temporary Wrapper.
    //! - Otherwise if the other Wrapper does not contain a valid value,
    //!   then if @a ilwalidOkay is true, just return, as no changes are
    //!   required to this Wrapper.
    //! - Otherwise, if this Wrapper does not contain a valid value, then
    //!   if @a ilwalidOkay is true, copy the other Wrapper to the temporary
    //!   Wrapper, since there is no original to merge with.
    //! - Otherwise, call info::fMerge to merge this Wrapper's value with
    //!   that of the other Wrapper, storing the result and any error in
    //!   the temporary Wrapper, and setting the ownership flag appropriately.
    //! - Swap this Wrapper with the temporary Wrapper, saving the new
    //!   contents and allowing the old ones to be freed when the temporary
    //!   Wrapper goes out of scope.
    //! \endif
    //
    // Note: The info::fMerge is only provided for attributes. For any others,
    //   merging is not supported, and attempting to instantiate this function
    //   will cause a compile error. This is the most straightforward way to
    //   catch improper use of this function. Hopefully that is acceptable to
    //   Coverity.
    void merge(Wrapper const& other, bool const ilwalidOkay) noexcept
    {
        Wrapper tmp {};
        if (LwSciError_Success != err) {
            tmp.err = err;
        } else if (LwSciError_Success != other.err) {
            tmp.err = other.err;
        } else if (!info::fValid(other.val)) {
            if (ilwalidOkay) {
                return;
            }
        } else if (!info::fValid(val)) {
            if (ilwalidOkay) {
                tmp = other;
            }
        } else {
            tmp.err = info::fMerge(val, other.val, tmp.val);
            tmp.own = (LwSciError_Success == tmp.err);
        }
        swapMembers(*this, tmp);
    };

    //! \brief Retrieves err value from the Wrapper instance.
    //!
    //! \returns LwSciError, err value.
    //!
    //! \implements{20125245}
    LwSciError getErr(void) const noexcept
    {
        return err;
    };

    LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A7_1_5), "Bug 2751189")
    //! \brief Retrieves the managed LwSciSync/LwSciBuf handle.
    //!  The caller does not own the value unless it explcitly makes a duplicate
    //!  itself.
    //!
    //! \returns T, LwSciSync/LwSciBuf handle.
    //!
    //! \implements{20125257}
    auto viewVal(void) const noexcept -> T const&
    LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A7_1_5))
    {
        return val;
    };

    LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A8_4_9), "Bug 2738197")
    LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A8_4_4), "Bug 3264648")
    //! \brief Updates the @a retVal with the value of LwSciSync/LwSciBuf handle
    //!  and returns the status of operation.
    //!
    //! \if TIER4_SWUD
    //! <b>Sequence of operations</b>
    //!  - If there was a previous error in the wrapper, returns it.
    //!  - Calls info::fValid() to check for valid LwSciSync/LwSciBuf handle.
    //!  - If Wrapper owns the LwSciSync/LwSciBuf handle, updates the given
    //!    @a retVal with it.
    //!  - If Wrapper doesn't own the LwSciSync/LwSciBuf handle, duplicates it
    //!    by calling info::fCopy() interface and updates @a retVal with it.
    //!  - In both the cases, it resets the LwSciSync/LwSciBuf handle of the
    //!    wrapper instance as info::cIlwalid and ownership flag as false.
    //! \endif
    //!
    //! \return LwSciError, the completion code of this operation.
    //!  - LwSciError_Success if successful.
    //!  - Error/panic behavior of this API includes any error/panic
    //!    behavior that info::fCopy() can generate.
    //!
    //! \if TIER4_SWAD
    //! \implements{20125278}
    //! \endif
    //!
    //! \if TIER4_SWUD
    //! \implements{19789044}
    //! \endif
    LwSciError takeVal(T& retVal) noexcept
    {
        if (LwSciError_Success != err) {
            return err;
        }
        LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A0_1_1), "Bug 2738197")
        retVal = std::exchange(val, info::cIlwalid);
        LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_1_1), "TID-300")
        bool const retOwn { std::exchange(own, false) };
        if (!retOwn && info::fValid(retVal)) {
            LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_2_2), "Bug 3002886")
            return info::fCopy(retVal);
        }
        return LwSciError_Success;
    };
    LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A8_4_4))
    LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A8_4_9))

private:
    //! \cond TIER4_SWAD
    //! \brief Managed LwSciSync/LwSciBuf handle. It is initialized to a valid
    //!  LwSciSync/LwSciBuf handle or to info::cIlwalid when a new Wrapper
    //!  instance is created.
    T           val;

    //! \brief Ownership flag indicating whether the managed LwSciSync/LwSciBuf
    //!  handle is owned by this Wrapper instance or not. It is initialized to
    //!  true or false when a new Wrapper instance is created.
    bool        own;

    //! \brief Any LwSciError encountered while creating a duplicate or
    //!  reference of the LwSciSync/LwSciBuf handle. It is initialized to
    //!  LwSciError resulted while creating a duplicate or reference of the
    //!  handle.
    LwSciError  err;
    //! \endcond
};


//! \brief Specialization of WrapInfo class for LwSciSyncAttrList which provides
//!  a set of static functions to duplicate, free and check validity of
//!  LwSciSyncAttrLists.
//!
//! \implements{19788573}
template <>
class WrapInfo<WrapType::WTSyncAttr>
{
public:
    //! \brief Static const value to represent the invalid LwSciSyncAttrList.
    static constexpr LwSciSyncAttrList cIlwalid { nullptr };

    //! \brief Duplicates the LwSciSyncAttrList referenced by the given @a attr.
    //!
    //! \param [in,out] attr: Input is the original LwSciSyncAttrList to duplicate.
    //! Output is the cloned value on success or cIlwalid on error.
    //!
    //! \returns LwSciError, the completion code of this operation:
    //!  - LwSciError_Success if successful.
    //!  - Error/panic behavior of this API includes any error/panic
    //!    behavior that LwSciSyncAttrListClone() can generate when
    //!    @a attr argument is passed to it.
    //!
    //! \if TIER4_SWAD
    //! \implements{20125323}
    //! \endif
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A8_4_8), "Bug 2745784")
    static LwSciError fCopy(LwSciSyncAttrList& attr) noexcept;

    //! \brief Frees the LwSciSyncAttrList referenced by the given @a attr.
    //!
    //! \param [in,out] attr: LwSciSyncAttrList to free. The value is cleared to
    //! cIlwalid on return.
    //!
    //! \returns void
    //!  - Panic behavior of this API includes the panic behavior
    //!    that LwSciSyncAttrListFree() can generate when @a attr
    //!    argument is passed to it.
    //!
    //! \if TIER4_SWAD
    //! \implements{20125326}
    //! \endif
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A8_4_8), "Bug 2745784")
    static void       fFree(LwSciSyncAttrList& attr) noexcept;


    //! \brief Merges two unreconciled LwSciSyncAttrLists, producing a new one.
    //!
    //! \param [in] attr1: The first attribute list to merge.
    //! \param [in] attr2: The second attribute list to merge.
    //! \param [in,out] newAttr: Location in which to store merged list.
    //!
    //! \returns LwSciError, the completion code of this operation:
    //!  - LwSciError_Success: if successful.
    //!  - Any error returned by LwSciSyncAttrListAppendUnreconciled().
    static LwSciError fMerge(LwSciSyncAttrList const& attr1,
                             LwSciSyncAttrList const& attr2,
                             LwSciSyncAttrList& newAttr) noexcept;

    //! \brief Checks whether the LwSciSyncAttrList referenced by the given
    //!  @a attr is NULL or not.
    //!
    //! \param [in] attr: LwSciSyncAttrList to validate
    //!
    //! \returns boolean
    //!  - true: If LwSciSyncAttrList is not NULL
    //!  - false: Otherwise
    //!
    //! \implements{20125332}
    static bool       fValid(LwSciSyncAttrList const& attr) noexcept;
};

//! \brief Alias for LwSciSyncAttrList wrapper.
//!
//! \implements{19530117}
LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(M2_10_1), "TID-547")
using SyncAttr = Wrapper<LwSciSyncAttrList, WrapType::WTSyncAttr>;


//! \brief Specialization of WrapInfo class for LwSciSyncObj which provides
//!  a set of static functions to duplicate, free and check validity of
//!  LwSciSyncObjs.
//!
//! \implements{19788579}
template <>
class WrapInfo<WrapType::WTSyncObj>
{
public:
    //! \brief Static const value to represent the invalid LwSciSyncObj.
    static constexpr LwSciSyncObj cIlwalid { nullptr };

    //! \brief Creates a reference of LwSciSyncObj.
    //!
    //! \param [in,out] obj: Input is the original LwSciSyncObj for which a
    //! reference has to be created. Output is the same LwSciSyncObj after
    //! successful referencing or cIlwalid on error.
    //!
    //! \returns LwSciError, the completion code of this operation:
    //!  - LwSciError_Success if successful.
    //!  - Error/panic behavior of this API includes any error/panic
    //!    behavior that LwSciSyncObjRef() can generate when
    //!    @a obj argument is passed to it.
    //!
    //! \if TIER4_SWAD
    //! \implements{20125344}
    //! \endif
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A8_4_8), "Bug 2745784")
    static LwSciError fCopy(LwSciSyncObj& obj) noexcept;

    //! \brief Frees the LwSciSyncObj referenced by the given @a obj.
    //!
    //! \param [in,out] obj: LwSciSyncObj to free. The value is cleared to cIlwalid
    //! on return.
    //!
    //! \returns void
    //!  - Panic behavior of this API includes the panic behavior
    //!    that LwSciSyncObjFree() can generate when @a obj
    //!    argument is passed to it.
    //!
    //! \if TIER4_SWAD
    //! \implements{20125347}
    //! \endif
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A8_4_8), "Bug 2745784")
    static void       fFree(LwSciSyncObj& obj) noexcept;

    //! \brief Checks whether the given LwSciSyncObj is NULL or not.
    //!
    //! \param [in] obj: LwSciSyncObj to validate
    //!
    //! \returns boolean
    //!  - true: If LwSciSyncObj is not NULL
    //!  - false: Otherwise
    //!
    //! \implements{20125350}
    static bool       fValid(LwSciSyncObj const& obj) noexcept;
};

//! \brief Alias for LwSciSyncObj wrapper.
//!
//! \implements{19530123}
using SyncObj = Wrapper<LwSciSyncObj, WrapType::WTSyncObj>;


//! \brief Specialization of WrapInfo class for LwSciSyncFence which provides
//!  a set of static functions to duplicate, free and check validity of
//!  LwSciSyncFences.
//!
//! \implements{19788582}
template <>
class WrapInfo<WrapType::WTSyncFence>
{
public:
// Note: We can't use LwSciSyncFenceInitializer here because it is defined
//       as a const, rather than a constexpr. So we have to directly insert
//       its value here. Perhaps we should have an ssert somewhere to ensure
//       the values don't get out of sync.
    //! \brief Static const value to represent the cleared LwSciSyncFence.
    static constexpr LwSciSyncFence cIlwalid { {0U, 0U, 0U, 0U, 0U, 0U} };

    //! \brief Duplicates the LwSciSyncFence referenced by the given @a fence.
    //!
    //! \param [in,out] fence: Input is the original LwSciSyncFence to duplicate.
    //! Output is the duped value on success or cIlwalid on error.
    //!
    //! \returns LwSciError, the completion code of this operation:
    //!  - LwSciError_Success if successful.
    //!  - Error/panic behavior of this API includes any error/panic
    //!    behavior that LwSciSyncFenceDup() can generate when
    //!    @a fence argument is passed to it.
    //!
    //! \if TIER4_SWAD
    //! \implements{20125374}
    //! \endif
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A8_4_8), "Bug 2745784")
    static LwSciError fCopy(LwSciSyncFence& fence) noexcept;

    //! \brief Clears the LwSciSyncFence referenced by the given @a ref.
    //!
    //! \param [in,out] ref: LwSciSyncFence to clear. The value is cleared to
    //! cIlwalid on return.
    //!
    //! \returns void
    //!  - Panic behavior of this API includes the panic behavior
    //!    that LwSciSyncFenceClear() can generate when @a ref
    //!    argument is passed to it.
    //!
    //! \if TIER4_SWAD
    //! \implements{20125377}
    //! \endif
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A8_4_8), "Bug 2745784")
    static void       fFree(LwSciSyncFence& ref) noexcept;

    //! \brief Checks whether the given LwSciSyncFence is cleared or not.
    //!
    //! \param [in] fence: LwSciSyncFence to validate
    //!
    //! \returns boolean
    //!  - true: If LwSciSyncFence is not cleared
    //!  - false: Otherwise
    //!
    //! \implements{20125380}
    static bool       fValid(LwSciSyncFence const& fence) noexcept;
};

//! \brief Alias for LwSciSyncFence wrapper.
//!
//! \implements{19788633}
using SyncFence = Wrapper<LwSciSyncFence, WrapType::WTSyncFence>;


//! \brief Specialization of WrapInfo class for LwSciBufAttrList which provides
//!  a set of static functions to duplicate, free and check validity of
//!  LwSciBufAttrLists.
//!
//! \implements{19788591}
template <>
class WrapInfo<WrapType::WTBufAttr>
{
public:
    //! \brief Static const value to represent the invalid LwSciBufAttrList.
    static constexpr LwSciBufAttrList cIlwalid { nullptr };

    //! \brief Duplicates the LwSciBufAttrList referenced by the given @a attr.
    //!
    //! \param [in,out] attr: Input is the original LwSciBufAttrList to duplicate.
    //! Output is the cloned value on success or cIlwalid on error.
    //!
    //! \returns LwSciError, the completion code of this operation:
    //!  - LwSciError_Success if successful.
    //!  - Error/panic behavior of this API includes any error/panic
    //!    behavior that LwSciBufAttrListClone() can generate when
    //!    @a attr argument is passed to it.
    //!
    //! \if TIER4_SWAD
    //! \implements{20125389}
    //! \endif
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A8_4_8), "Bug 2745784")
    static LwSciError fCopy(LwSciBufAttrList& attr) noexcept;

    //! \brief Frees the LwSciBufAttrList referenced by the given @a attr.
    //!
    //! \param [in,out] attr: LwSciBufAttrList to free. The value is cleared to
    //! cIlwalid on return.
    //!
    //! \returns void
    //!  - Panic behavior of this API includes the panic behavior
    //!    that LwSciBufAttrListFree() can generate when @a attr
    //!    argument is passed to it.
    //!
    //! \if TIER4_SWAD
    //! \implements{20125392}
    //! \endif
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A8_4_8), "Bug 2745784")
    static void       fFree(LwSciBufAttrList& attr) noexcept;

    //! \brief Merges two unreconciled LwSciBufAttrLists, producing a new one.
    //!
    //! \param [in] attr1: The first attribute list to merge.
    //! \param [in] attr2: The second attribute list to merge.
    //! \param [in,out] newAttr: Location in which to store merged list.
    //!
    //! \returns LwSciError, the completion code of this operation:
    //!  - LwSciError_Success: if successful.
    //!  - Any error returned by LwSciBufAttrListAppendUnreconciled().
    static LwSciError fMerge(LwSciBufAttrList const& attr1,
                             LwSciBufAttrList const& attr2,
                             LwSciBufAttrList& newAttr) noexcept;

    //! \brief Checks whether the given LwSciBufAttrList is NULL or not.
    //!
    //! \param [in] attr: LwSciBufAttrList to validate
    //!
    //! \returns boolean
    //!  - true: If LwSciBufAttrList is not NULL
    //!  - false: Otherwise
    //!
    //! \implements{20125395}
    static bool       fValid(LwSciBufAttrList const& attr) noexcept;
};

//! \brief Alias for LwSciBufAttrList wrapper.
//!
//! \implements{19530129}
using BufAttr = Wrapper<LwSciBufAttrList, WrapType::WTBufAttr>;


//! \brief Specialization of WrapInfo class for LwSciBufObj which provides
//!  a set of static functions to duplicate, free and check validity of
//!  LwSciBufObjs.
//!
//! \implements{19788594}
template <>
class WrapInfo<WrapType::WTBufObj>
{
public:
    //! \brief Static const value to represent the invalid LwSciBufObj.
    static constexpr LwSciBufObj cIlwalid { nullptr };

    //! \brief Creates a reference of LwSciBufObj.
    //!
    //! \param [in,out] obj: Input is the original LwSciBufObj for which a
    //! reference has to be created. Output is the same LwSciBufObj after
    //! successful referencing or cIlwalid on error.
    //!
    //! \returns LwSciError, the completion code of this operation:
    //!  - LwSciError_Success if successful.
    //!  - Error/panic behavior of this API includes any error/panic
    //!    behavior that LwSciBufObjRef() can generate when
    //!    @a obj argument is passed to it.
    //!
    //! \if TIER4_SWAD
    //! \implements{20125404}
    //! \endif
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A8_4_8), "Bug 2745784")
    static LwSciError fCopy(LwSciBufObj& obj) noexcept;

    //! \brief Frees the LwSciBufObj referenced by the given @a obj.
    //!
    //! \param [in,out] obj: LwSciBufObj to free. The value is cleared to cIlwalid
    //! on return.
    //!
    //! \returns void
    //!  - Panic behavior of this API includes the panic behavior
    //!    that LwSciBufObjFree() can generate when @a obj
    //!    argument is passed to it.
    //!
    //! \if TIER4_SWAD
    //! \implements{20125407}
    //! \endif
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A8_4_8), "Bug 2745784")
    static void       fFree(LwSciBufObj& obj) noexcept;

    //! \brief Checks whether the given LwSciBufObj is NULL or not.
    //!
    //! \param [in] obj: LwSciBufObj to validate
    //!
    //! \returns boolean
    //!  - true: If LwSciBufObj is not NULL
    //!  - false: Otherwise
    //!
    //! \implements{20125410}
    static bool       fValid(LwSciBufObj const& obj) noexcept;
};

//! \brief Alias for full LwSciBufObj wrapper.
//!
//! \implements{19530132}
using BufObj = Wrapper<LwSciBufObj, WrapType::WTBufObj>;

} // namespace LwSciWrap

#endif // SCIWRAP_H
