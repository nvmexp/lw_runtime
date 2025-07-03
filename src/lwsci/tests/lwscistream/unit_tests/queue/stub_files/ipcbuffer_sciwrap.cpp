//! \file
//! \brief LwSciStream IpcBuffer pack/unpack functions for LwSciWra.
//!
//! \copyright
//! Copyright (c) 2021-2022 LWPU Corporation. All rights reserved.
//!
//! LWPU Corporation and its licensors retain all intellectual property and
//! proprietary rights in and to this software, related documentation and any
//! modifications thereto. Any use, reproduction, disclosure or distribution
//! of this software and related documentation without an express license
//! agreement from LWPU Corporation is strictly prohibited.
#include <unistd.h>
#include "lwscibuf.h"
#include "lwscisync.h"
#include "lwscistream_common.h"
#include "ipcbuffer.h"
#include "ipcbuffer_sciwrap.h"

namespace LwSciStream {

// TODO: A2-10-5 violations appear to be a bug with overloaded functions
//       related to, but not quite the same as, Bug 200695637
//       Need to fully investigate and file bug, if not already done.

// TODO: A8-4-9 violations appear to be a bug which we haven't reported yet,
//       although they are similar to existing bugs. Need to investigate and
//       file one if not already done.

LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A8_4_4), "Bug 3264648")
LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A8_4_8), "Bug 3264648")

//! <b>Sequence of operations</b>
//! - Retrieve wrapper's error with Wrapper::getErr() and pack it with
//!   IpcBuffer::packVal(). If the wrapper's error was not success,
//!   skip the rest of the packing.
//! - Retrieve wrapper's attribute list with Wrapper::viewVal().
//! - If attribute list is not NULL, query whether it is reconciled with
//!   LwSciBufAttrListIsReconciled() and then export it to a blob with
//!   LwSciBufAttrListIpcExport{R|Unr}econciled().
//! - Pack the blob, if any, with IpcBuffer::packBlob().
//! - Pack the flag indicating whether or not the list is reconciled with
//!   IpcBuffer::packVal().
LwSciError
ipcBufferPack(
    IpcBuffer& buf,
    LwSciWrap::BufAttr const& val) noexcept
{
    LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A5_1_1), "TID-300")

    // Pack any error that oclwrred along the way
    LwSciError const wrapErr { val.getErr() };
    LwSciError err { buf.packVal(wrapErr) };
    if (LwSciError_Success != err) {
        return err;
    }
    if (LwSciError_Success != wrapErr) {
        return LwSciError_Success;
    }

    // Retrieve attribute list and colwert to blob
    LwSciBufAttrList const attr { val.viewVal() };
    bool isReconciled { false };
    IpcBuffer::CBlob blob { 0U, nullptr };
    void* blobData;
    if (nullptr != attr) {

        // Check whether attribute list is reconciled
        err = LwSciBufAttrListIsReconciled(attr, &isReconciled);
        if (LwSciError_Success != err) {
            return err;
        }

        // Expore with appropriate function
        err = isReconciled
            ? LwSciBufAttrListIpcExportReconciled(
                    attr,
                    buf.ipcEndpointGet(),
                    &blobData,
                    &blob.size)
            : LwSciBufAttrListIpcExportUnreconciled(
                    &attr,
                    ONE,
                    buf.ipcEndpointGet(),
                    &blobData,
                    &blob.size);
        blob.data = blobData;
        if (LwSciError_Success != err) {
            return err;
        }
    }

    // Pack the blob
    err = buf.packBlob(blob);
    if (nullptr != blob.data) {
        LwSciBufAttrListFreeDesc(blobData);
    }
    if (LwSciError_Success != err) {
        return err;
    }

    // Pack the reconciliation flag
    return buf.packVal(isReconciled);

    LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A5_1_1))
}

//! <b>Sequence of operations</b>
//! - Unpack wrapper's error value with IpcBuffer::unpackVal(). If it is
//!   not success, return a new wrapper with the error.
//! - Unpack the blob with IpcBuffer::unpackBlob().
//! - Unpack the flag indicating whether the attribute list is reconciled
//!   with IpcBuffer::unpackVal().
//! - If the blob was not empty, import it to an attribute list with
//!   LwSciBufAttrListIpcImport{R|Unr}econciled().
//! - Return the attribute list in a wrapper with ownership set to true.
LwSciError
ipcBufferUnpack(
    IpcBuffer& buf,
    LwSciWrap::BufAttr& val) noexcept
{
    LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A5_1_1), "TID-300")

    // Unpack any error that happened to the attribute itself
    LwSciError wrapErr;
    LwSciError err { buf.unpackVal(wrapErr) };
    if (LwSciError_Success != err) {
        return err;
    }

    // If there was an attribute error, fill in the wrapper
    //   This is not an error for this function
    if (LwSciError_Success != wrapErr) {
        val = LwSciWrap::BufAttr(nullptr, false, false, wrapErr);
        return LwSciError_Success;
    }

    // Unpack the blob
    IpcBuffer::CBlob blob;
    err = buf.unpackBlob(blob);
    if (LwSciError_Success != err) {
        return err;
    }

    // Unpack the reconcilation flag
    bool isReconciled;
    err = buf.unpackVal(isReconciled);
    if (LwSciError_Success != err) {
        return err;
    }

    // If there is a blob, colwert to attribute list
    //   Any failure here is put in the wrapper, rather than causing
    //   the function to fail.
    LwSciBufAttrList attr { nullptr };
    if (0U < blob.size) {
        wrapErr = isReconciled
            ? LwSciBufAttrListIpcImportReconciled(
                    buf.bufModuleGet(),
                    buf.ipcEndpointGet(),
                    blob.data,
                    blob.size,
                    nullptr,
                    0U,
                    &attr)
            : LwSciBufAttrListIpcImportUnreconciled(
                    buf.bufModuleGet(),
                    buf.ipcEndpointGet(),
                    blob.data,
                    blob.size,
                    &attr);
    }

    // Fill in wrapper
    val = LwSciWrap::BufAttr(attr, true, false, wrapErr);
    return LwSciError_Success;

    LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A5_1_1))
}

// Cast LwSciBuf descriptor size to size_t for colwenience.
constexpr size_t sciBufExportDescSize {
    static_cast<size_t>(LWSCIBUF_EXPORT_DESC_SIZE)
};

// These asserts detect any changes to the definition of the
//   LwSciBufObjIpcExportDescriptor which would require these functions
//   to be updated. The functions here assume that the structure consists
//   of a single field named "data" which is an array of uint64_t of size
//   LWSCIBUF_EXPORT_DESC_SIZE.
static_assert(sizeof(LwSciBufObjIpcExportDescriptor) ==
              sizeof(LwSciBufObjIpcExportDescriptor::data),
              "Buffer descriptor only contains data array");
static_assert(sizeof(LwSciBufObjIpcExportDescriptor::data) ==
              (sciBufExportDescSize * sizeof(uint64_t)),
              "Buffer descriptor size hasn't changed");

//! <b>Sequence of operations</b>
//! - Call Wrapper::getErr() to retrieve wrapper's error and call
//!   IpcBuffer::packVal() to pack it. If the wrapper's error was
//!   not success, skip the rest of the packing.
//! - Call Wrapper::viewVal() to retrieve the wrapper's buffer object.
//! - Call LwSciBufObjIpcExport() to export the object to a portable
//!   descriptor.
//! - Loop over the descriptor contents, calling IpcBuffer::packVal() to
//!   pack each one.
LwSciError
ipcBufferPack(
    IpcBuffer& buf,
    LwSciWrap::BufObj const& val) noexcept
{
    LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A5_1_1), "TID-300")

    // Pack any error that oclwrred along the way
    LwSciError const wrapErr { val.getErr() };
    LwSciError err { buf.packVal(wrapErr) };
    if (LwSciError_Success != err) {
        return err;
    }
    if (LwSciError_Success != wrapErr) {
        return LwSciError_Success;
    }

    // TODO: Is there any possibility it will be NULL?
    assert(nullptr != val.viewVal());

    // Export buffer to portable description
    LwSciBufObjIpcExportDescriptor bufObjDesc;
    err = bufObjExportWrapper(buf.ipcEndpointGet(),
                              bufObjDesc,
                              val.viewVal());
    if (LwSciError_Success != err) {
        return err;
    }

    // Pack individual components
    for (size_t i {0U}; sciBufExportDescSize > i; ++i) {
        err = buf.packVal(bufObjDesc.data[i]);
        if (LwSciError_Success != err) {
            return err;
        }
    }

    return LwSciError_Success;
}

//! <b>Sequence of operations</b>
//! - Call IpcBuffer::unpackVal() to retrieve wrapper's error. If it is
//!   not success, return a new wrapper with the error.
//! - Loop over the descriptor contents, calling IpcBuffer::unpackVal() to
//!   unpack each one.
//! - Call Wrapper::viewVal() to extract the auxiliary attribute list.
//! - Call LwSciBufObjIpcImport() to colwert the descriptor to an object.
//! - Create and return a new wrapper instance which owns the object.
LwSciError
ipcBufferUnpack(
    IpcBuffer& buf,
    LwSciWrap::BufObj& val,
    LwSciWrap::BufAttr const& aux) noexcept
{
    LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A5_1_1), "TID-300")

    // Unpack any error that happened to the object itself
    LwSciError wrapErr;
    LwSciError err { buf.unpackVal(wrapErr) };
    if (LwSciError_Success != err) {
        return err;
    }

    // If there was an object error, fill in the wrapper
    //   This is not an error for this function
    if (LwSciError_Success != wrapErr) {
        val = LwSciWrap::BufObj(nullptr, false, false, wrapErr);
        return LwSciError_Success;
    }

    // Unpack individual components of portable buffer description
    LwSciBufObjIpcExportDescriptor bufObjDesc;
    for (size_t i {0U}; sciBufExportDescSize > i; ++i) {
        err = buf.unpackVal(bufObjDesc.data[i]);
        if (LwSciError_Success != err) {
            return err;
        }
    }

    // Colwert to a buffer object
    LwSciBufObj bufObj;
    wrapErr = bufObjImportWrapper(buf.ipcEndpointGet(),
                                  bufObjDesc,
                                  aux.viewVal(),
                                  bufObj);

    // Fill in wrapper
    val = LwSciWrap::BufObj(bufObj, true, false, wrapErr);
    return LwSciError_Success;

    LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A5_1_1))
}

//! <b>Sequence of operations</b>
//! - Retrieve wrapper's error with Wrapper::getErr() and pack it with
//!   IpcBuffer::packVal(). If the wrapper's error was not success,
//!   skip the rest of the packing.
//! - Retrieve wrapper's attribute list with Wrapper::viewVal().
//! - If attribute list is not NULL, query whether it is reconciled with
//!   LwSciSyncAttrListIsReconciled() and then export it to a blob with
//!   LwSciSyncAttrListIpcExport{R|Unr}econciled().
//! - Pack the blob, if any, with IpcBuffer::packBlob().
//! - Pack the flag indicating whether or not the list is reconciled with
//!   IpcBuffer::packVal().
LwSciError
ipcBufferPack(
    IpcBuffer& buf,
    LwSciWrap::SyncAttr const& val) noexcept
{
    LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A5_1_1), "TID-300")

    // Pack any error that oclwrred along the way
    LwSciError const wrapErr { val.getErr() };
    LwSciError err { buf.packVal(wrapErr) };
    if (LwSciError_Success != err) {
        return err;
    }
    if (LwSciError_Success != wrapErr) {
        return LwSciError_Success;
    }

    // Retrieve attribute list and colwert to blob
    LwSciSyncAttrList const attr { val.viewVal() };
    bool isReconciled { false };
    IpcBuffer::CBlob blob { 0U, nullptr };
    void* blobData;
    if (nullptr != attr) {

        // Check whether attribute list is reconciled
        err = LwSciSyncAttrListIsReconciled(attr, &isReconciled);
        if (LwSciError_Success != err) {
            return err;
        }

        // Expore with appropriate function
        err = isReconciled
            ? LwSciSyncAttrListIpcExportReconciled(
                    attr,
                    buf.ipcEndpointGet(),
                    &blobData,
                    &blob.size)
            : LwSciSyncAttrListIpcExportUnreconciled(
                    &attr,
                    ONE,
                    buf.ipcEndpointGet(),
                    &blobData,
                    &blob.size);
        blob.data = blobData;
        if (LwSciError_Success != err) {
            return err;
        }
    }

    // Pack the blob
    err = buf.packBlob(blob);
    if (nullptr != blob.data) {
        LwSciSyncAttrListFreeDesc(blobData);
    }
    if (LwSciError_Success != err) {
        return err;
    }

    // Pack the reconciliation flag
    return buf.packVal(isReconciled);

    LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A5_1_1))
}

//! <b>Sequence of operations</b>
//! - Unpack wrapper's error value with IpcBuffer::unpackVal(). If it is
//!   not success, return a new wrapper with the error.
//! - Unpack the blob with IpcBuffer::unpackBlob().
//! - Unpack the flag indicating whether the attribute list is reconciled
//!   with IpcBuffer::unpackVal().
//! - If the blob was not empty, import it to an attribute list with
//!   LwSciSyncAttrListIpcImport{R|Unr}econciled().
//! - Return the attribute list in a wrapper with ownership set to true.
LwSciError
ipcBufferUnpack(
    IpcBuffer& buf,
    LwSciWrap::SyncAttr& val) noexcept
{
    LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A5_1_1), "TID-300")

    // Unpack any error that happened to the attribute itself
    LwSciError wrapErr;
    LwSciError err { buf.unpackVal(wrapErr) };
    if (LwSciError_Success != err) {
        return err;
    }

    // If there was an attribute error, fill in the wrapper
    //   This is not an error for this function
    if (LwSciError_Success != wrapErr) {
        val = LwSciWrap::SyncAttr(nullptr, false, false, wrapErr);
        return LwSciError_Success;
    }

    // Unpack the blob
    IpcBuffer::CBlob blob;
    err = buf.unpackBlob(blob);
    if (LwSciError_Success != err) {
        return err;
    }

    // Unpack the reconcilation flag
    bool isReconciled;
    err = buf.unpackVal(isReconciled);
    if (LwSciError_Success != err) {
        return err;
    }

    // If there is a blob, colwert to attribute list
    //   Any failure here is put in the wrapper, rather than causing
    //   the function to fail.
    LwSciSyncAttrList attr { nullptr };
    if (0U < blob.size) {
        wrapErr = isReconciled
            ? LwSciSyncAttrListIpcImportReconciled(
                    buf.syncModuleGet(),
                    buf.ipcEndpointGet(),
                    blob.data,
                    blob.size,
                    nullptr,
                    0U,
                    &attr)
            : LwSciSyncAttrListIpcImportUnreconciled(
                    buf.syncModuleGet(),
                    buf.ipcEndpointGet(),
                    blob.data,
                    blob.size,
                    &attr);
    }

    // Fill in wrapper
    val = LwSciWrap::SyncAttr(attr, true, false, wrapErr);
    return LwSciError_Success;

    LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A5_1_1))
}

//! <b>Sequence of operations</b>
//! - Retrieve wrapper's error with Wrapper::getErr() and pack it with
//!   IpcBuffer::packVal(). If the wrapper's error was not success,
//!   skip the rest of the packing.
//! - Retrieve wrapper's sync object with Wrapper::viewVal().
//! - If object is not NULL, export it to a blob with
//!   LwSciSyncIpcExportAttrListAndObj().
//! - Pack the blob, if any, with IpcBuffer::packBlob().
LwSciError
ipcBufferPack(
    IpcBuffer& buf,
    LwSciWrap::SyncObj const& val) noexcept
{
    LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A5_1_1), "TID-300")

    // Pack any error that oclwrred along the way
    LwSciError const wrapErr { val.getErr() };
    LwSciError err { buf.packVal(wrapErr) };
    if (LwSciError_Success != err) {
        return err;
    }
    if (LwSciError_Success != wrapErr) {
        return LwSciError_Success;
    }

    // Retrieve buffer object and colwert to blob
    LwSciSyncObj const obj { val.viewVal() };
    IpcBuffer::CBlob blob { 0U, nullptr };
    void* blobData;
    if (nullptr != obj) {
        err = syncAttrListAndObjExportWrapper(
                       buf.ipcEndpointGet(),
                       obj,
                       &blobData,
                       blob.size);
        blob.data = blobData;
        if (LwSciError_Success != err) {
            return err;
        }
    }

    // Pack the blob
    err = buf.packBlob(blob);
    if (nullptr != blob.data) {
        LwSciSyncAttrListAndObjFreeDesc(blobData);
    }
    return err;

    LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A5_1_1))
}

//! <b>Sequence of operations</b>
//! - Unpack wrapper's error value with IpcBuffer::unpackVal(). If it is
//!   not success, return a new wrapper with the error.
//! - Unpack the blob with IpcBuffer::unpackBlob().
//! - If the blob was not empty, import it to an object with
//!   LwSciSyncIpcImportAttrListAndObj().
//! - Return the object in a wrapper with ownership set to true.
LwSciError
ipcBufferUnpack(
    IpcBuffer& buf,
    LwSciWrap::SyncObj& val,
    LwSciWrap::SyncAttr const& aux,
    bool const ignore) noexcept
{
    LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A5_1_1), "TID-300")

    // Unpack any error that happened to the object itself
    LwSciError wrapErr;
    LwSciError err { buf.unpackVal(wrapErr) };
    if (LwSciError_Success != err) {
        return err;
    }

    // If there was an object error, fill in the wrapper
    //   This is not an error for this function
    if (LwSciError_Success != wrapErr) {
        val = LwSciWrap::SyncObj(nullptr, false, false, wrapErr);
        return LwSciError_Success;
    }

    // Unpack the blob
    IpcBuffer::CBlob blob;
    err = buf.unpackBlob(blob);
    if (LwSciError_Success != err) {
        return err;
    }

    // If there is a blob, colwert to object
    //   Any failure here is put in the wrapper, rather than causing
    //   the function to fail.
    LwSciSyncObj obj { nullptr };
    if (!ignore && (0U < blob.size)) {
        // Retrieve attribute list for validation
        LwSciSyncAttrList const auxAttr { aux.viewVal() };

        // Import if attribute list is valid
        if (nullptr != auxAttr) {
            wrapErr = syncAttrListAndObjImportWrapper(
                            buf.syncModuleGet(),
                            buf.ipcEndpointGet(),
                            blob.data,
                            blob.size,
                            auxAttr,
                            obj);
        } else {
            wrapErr = LwSciError_InconsistentData;
        }
    }

    // Fill in wrapper
    val = LwSciWrap::SyncObj(obj, true, false, wrapErr);
    return LwSciError_Success;

    LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A5_1_1))
}

// LwSciSync doesn't define a symbol for the size of the fence descriptor
constexpr size_t sciFenceExportDescSize { 7U };

// These asserts detect any changes to the definition of the
//   LwSciSyncFenceIpcExportDescriptor which would require these functions
//   to be updated. The functions here assume that the structure consists of
//   a single field named "payload" which is an array of uint64_t of size 7.
static_assert(sizeof(LwSciSyncFenceIpcExportDescriptor) ==
              sizeof(LwSciSyncFenceIpcExportDescriptor::payload),
              "Fence descriptor only contains data array");
static_assert(sizeof(LwSciSyncFenceIpcExportDescriptor::payload) ==
              (sciFenceExportDescSize * sizeof(uint64_t)),
              "Fence descriptor size hasn't changed");

//! <b>Sequence of operations</b>
//! - Call Wrapper::getErr() to retrieve wrapper's error and call
//!   IpcBuffer::packVal() to pack it. If the wrapper's error was
//!   not success, skip the rest of the packing.
//! - Call Wrapper::viewVal() to retrieve the wrapper's fence.
//! - Call LwSciSyncIpcExportFence() to export the fence to a portable
//!   descriptor.
//! - Loop over the descriptor contents, calling IpcBuffer::packVal() to
//!   pack each one.
LwSciError
ipcBufferPack(
    IpcBuffer& buf,
    LwSciWrap::SyncFence const& val) noexcept
{
    LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A5_1_1), "TID-300")

    // Pack any error that oclwrred along the way
    LwSciError const wrapErr { val.getErr() };
    LwSciError err { buf.packVal(wrapErr) };
    if (LwSciError_Success != err) {
        return err;
    }
    if (LwSciError_Success != wrapErr) {
        return LwSciError_Success;
    }

    // Export fence to portable description
    LwSciSyncFenceIpcExportDescriptor fenceDesc;
    err = LwSciSyncIpcExportFence(&val.viewVal(),
                                  buf.ipcEndpointGet(),
                                  &fenceDesc);
    if (LwSciError_Success != err) {
        return err;
    }

    // Pack individual components
    for (size_t i {0U}; sciFenceExportDescSize > i; ++i) {
        err = buf.packVal(fenceDesc.payload[i]);
        if (LwSciError_Success != err) {
            return err;
        }
    }

    return LwSciError_Success;
}

//! <b>Sequence of operations</b>
//! - Call IpcBuffer::unpackVal() to retrieve wrapper's error. If it is
//!   not success, return a new wrapper with the error.
//! - Loop over the descriptor contents, calling IpcBuffer::unpackVal() to
//!   unpack each one.
//! - Call Wrapper::viewVal() to extract the auxiliary sync object.
//! - Call LwSciSyncIpcImportFence() to colwert the descriptor to a fence.
//! - Create and return a new wrapper instance which owns the fence.
LwSciError
ipcBufferUnpack(
    IpcBuffer& buf,
    LwSciWrap::SyncFence& val,
    LwSciWrap::SyncObj const& aux) noexcept
{
    LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A5_1_1), "TID-300")

    // Unpack any error that happened to the fence itself
    LwSciError wrapErr;
    LwSciError err { buf.unpackVal(wrapErr) };
    if (LwSciError_Success != err) {
        return err;
    }

    // If there was an fence error, fill in the wrapper
    //   This is not an error for this function
    if (LwSciError_Success != wrapErr) {
        val = LwSciWrap::SyncFence(LwSciSyncFenceInitializer,
                                   false, false, wrapErr);
        return LwSciError_Success;
    }

    // Unpack individual components of portable fence description
    LwSciSyncFenceIpcExportDescriptor fenceDesc;
    for (size_t i {0U}; sciFenceExportDescSize > i; ++i) {
        err = buf.unpackVal(fenceDesc.payload[i]);
        if (LwSciError_Success != err) {
            return err;
        }
    }

    // Colwert to a fence if there is a corresponding sync object.
    //   If there isn't, then either the fence is empty or the recipient
    //   doesn't use this sync object
    LwSciSyncFence fence { LwSciSyncFenceInitializer };
    LwSciSyncObj auxSync { aux.viewVal() };
    if (nullptr != auxSync) {
        wrapErr = LwSciSyncIpcImportFence(auxSync, &fenceDesc, &fence);
    }

    // Fill in wrapper
    val = LwSciWrap::SyncFence(fence, true, false, wrapErr);
    return LwSciError_Success;

    LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A5_1_1))
}


// TODO: The functions below will be phased out.

//! \brief Specialized pack function for fence export descriptors
LwSciError
ipcBufferPack(
    IpcBuffer& buf,
    LwSciSyncFenceIpcExportDescriptor const& val) noexcept
{
    // Asserts to detect changes to the fence descriptor
    constexpr size_t payloadSize { 7U };
    LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A5_2_2), "Proposed TID-1251")
    static_assert(sizeof(val) == sizeof(val.payload),
                  "Fence descriptor only contains payload array");
    static_assert(sizeof(val.payload) == (payloadSize * sizeof(uint64_t)),
                  "Fence descriptor size hasn't changed");
    LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A5_2_2))
    // Extract individual components
    LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(M2_10_1), "TID-547")
    for (size_t i {0U}; payloadSize > i; ++i) {
        LwSciError const err { buf.packVal(val.payload[i]) };
        if (LwSciError_Success != err) {
            return err;
        }
    }
    LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(M2_10_1))
    return LwSciError_Success;
}

//! \brief Specialized unpack function for fence export descriptors
LwSciError
ipcBufferUnpack(
    IpcBuffer& buf,
    LwSciSyncFenceIpcExportDescriptor& val) noexcept
{
    // Asserts to detect changes to the fence descriptor
    constexpr size_t payloadSize { 7U };
    LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A5_2_2), "Proposed TID-1251")
    static_assert(sizeof(val) == sizeof(val.payload),
                  "Fence descriptor only contains payload array");
    static_assert(sizeof(val.payload) == (payloadSize * sizeof(uint64_t)),
                  "Fence descriptor size hasn't changed");
    LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A5_2_2))
    // Extract individual components
    LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(M2_10_1), "TID-547")
    for (size_t i {0U}; payloadSize > i; ++i) {
        LwSciError const err { buf.unpackVal(val.payload[i]) };
        if (LwSciError_Success != err) {
            return err;
        }
    }
    LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(M2_10_1))
    return LwSciError_Success;
}

// TODO:
// The IPC export/import of LwSciBufObj and LwSciSyncObj could return
// LwSciError_TryItAgain in C2C cases. The approach to retry after certain
// timeout may be violating the ARRs of DRIVE OS.
// We may need to have separate implementation for safety and non-safety
// for the below wrappers.

// Number of retries to handle LwSciError_TryItAgain error
// from IPC export/import.
constexpr uint32_t EAGAIN_MAX_RETRIES { 50U };
// Timeout of each retry when getting LwSciError_TryItAgain
// error from IPC export/import.
constexpr useconds_t EAGAIN_TIMEOUT_US_RETRIES { 50000U };

LwSciError bufObjImportWrapper(
    LwSciIpcEndpoint const ipcEndpoint,
    LwSciBufObjIpcExportDescriptor const& desc,
    LwSciBufAttrList const reconciledAttrList,
    LwSciBufObj& bufObj)
{
    LwSciError err;
    uint32_t numTry {0U};
    do {
        err = LwSciBufObjIpcImport(ipcEndpoint,
                                    &desc,
                                    reconciledAttrList,
                                    LwSciBufAccessPerm_Auto,
                                    INFINITE_TIMEOUT,
                                    &bufObj);
        if (LwSciError_TryItAgain == err) {
            usleep(EAGAIN_TIMEOUT_US_RETRIES);
            ++numTry;
        } else {
            return err;
        }
    } while (EAGAIN_MAX_RETRIES > numTry);
    return LwSciError_ResourceError;
}

LwSciError bufObjExportWrapper(
    LwSciIpcEndpoint const ipcEndpoint,
    LwSciBufObjIpcExportDescriptor& desc,
    LwSciBufObj const bufObj)
{
    LwSciError err;
    uint32_t numTry {0U};
    do {
        err = LwSciBufObjIpcExport(bufObj,
                                LwSciBufAccessPerm_Auto,
                                ipcEndpoint,
                                &desc);
        if (LwSciError_TryItAgain == err) {
            usleep(EAGAIN_TIMEOUT_US_RETRIES);
            ++numTry;
        } else {
            return err;
        }
    } while (EAGAIN_MAX_RETRIES > numTry);
    return LwSciError_ResourceError;
}

LwSciError syncAttrListAndObjImportWrapper(
    LwSciSyncModule const module,
    LwSciIpcEndpoint const ipcEndpoint,
    void const *const attrListAndObjDesc,
    size_t const attrListAndObjDescSize,
    LwSciSyncAttrList const attrList,
    LwSciSyncObj& syncObj)
{
    LwSciError err;
    uint32_t numTry {0U};
    do {
        err = LwSciSyncIpcImportAttrListAndObj(module,
                                            ipcEndpoint,
                                            attrListAndObjDesc,
                                            attrListAndObjDescSize,
                                            &attrList,
                                            ONE,
                                            LwSciSyncAccessPerm_Auto,
                                            INFINITE_TIMEOUT,
                                            &syncObj);
        if (LwSciError_TryItAgain == err) {
            usleep(EAGAIN_TIMEOUT_US_RETRIES);
            ++numTry;
        } else {
            return err;
        }
    } while (EAGAIN_MAX_RETRIES > numTry);
    return LwSciError_ResourceError;
}

LwSciError syncAttrListAndObjExportWrapper(
    LwSciIpcEndpoint const ipcEndpoint,
    LwSciSyncObj const syncObj,
    void** const attrListAndObjDesc,
    size_t& attrListAndObjDescSize)
{
    LwSciError err;
    uint32_t numTry {0U};
    do {
        err = LwSciSyncIpcExportAttrListAndObj(syncObj,
                                            LwSciSyncAccessPerm_Auto,
                                            ipcEndpoint,
                                            attrListAndObjDesc,
                                            &attrListAndObjDescSize);
        if (LwSciError_TryItAgain == err) {
            usleep(EAGAIN_TIMEOUT_US_RETRIES);
            ++numTry;
        } else {
            return err;
        }
    } while (EAGAIN_MAX_RETRIES > numTry);
    return LwSciError_ResourceError;
}

LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A8_4_8))
LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A8_4_4))

} // namespace LwSciStream
