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

#ifndef IPC_BUFFER_SCIWRAP_H
#define IPC_BUFFER_SCIWRAP_H

#include "covanalysis.h"
#include "lwscistream_common.h"
#include "ipcbuffer.h"
#include "lwsciipc.h"
#include "lwscibuf.h"
#include "lwscisync.h"
#include "sciwrap.h"


namespace LwSciStream {

// TODO: A2-10-5 violations appear to be a bug with overloaded functions
//       related to, but not quite the same as, Bug 200695637
//       Need to fully investigate and file bug, if there isn't one already.

LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A8_4_8), "Bug 3264648")
LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A2_7_3), "Temporary WAR")

//! \brief Packs a wrapped LwSciBufAttrList to an IpcBuffer.
//!
//! \param [in,out] buf: Buffer in which the wrapper will be packed.
//! \param [in] val: Wrapper to be packed.
//!
//! \return LwSciError
//! - LwSciError_Success - If packing succeeded
//! - Any error returned by IpcBuffer::packVal().
//! - Any error returned by IpcBuffer::packBlob().
//! - Any error returned by LwSciBufAttrListIsReconciled().
//! - Any error returned by LwSciBufAttrListIpcExportReconciled().
//! - Any error returned by LwSciBufAttrListIpcExportUnreconciled().
LwSciError
ipcBufferPack(
    IpcBuffer& buf,
    LwSciWrap::BufAttr const& val) noexcept;

//! \brief Unpacks a wrapped LwSciBufAttrList from an IpcBuffer.
//!
//! \param [in,out] buf: Buffer from which the wrapper will be unpacked.
//! \param [in,out] val: Location in which to store the wrapper.
//!
//! \return LwSciError
//! - LwSciError_Success - If unpacking succeeded.
//! - Any error returned by IpcBuffer::unpackVal().
//! - Any error returned by IpcBuffer::unpackBlob().
//! - Any error returned by LwSciBufAttrListIpcImportReconciled().
//! - Any error returned by LwSciBufAttrListIpcImportUnreconciled().
LwSciError
ipcBufferUnpack(
    IpcBuffer& buf,
    LwSciWrap::BufAttr& val) noexcept;

//! \brief Packs a wrapped LwSciBufObj to an IpcBuffer.
//!
//! \param [in,out] buf: Buffer in which the wrapper will be packed.
//! \param [in] val: Wrapper to be packed.
//!
//! \return LwSciError
//! - LwSciError_Success - If packing succeeded
//! - Any error returned by IpcBuffer::packVal().
//! - Any error returned by LwSciBufObjIpcExport().
LwSciError
ipcBufferPack(
    IpcBuffer& buf,
    LwSciWrap::BufObj const& val) noexcept;

//! \brief Unpacks a wrapped LwSciBufObj from an IpcBuffer.
//!
//! \param [in,out] buf: Buffer from which the wrapper will be unpacked.
//! \param [in,out] val: Location in which to store the wrapper.
//! \param [in] aux: Buffer attribute list for the object.
//!
//! \return LwSciError
//! - LwSciError_Success - If unpacking succeeded.
//! - Any error returned by IpcBuffer::unpackVal().
//! - Any error returned by LwSciBufObjIpcImport().
LwSciError
ipcBufferUnpack(
    IpcBuffer& buf,
    LwSciWrap::BufObj& val,
    LwSciWrap::BufAttr const& aux) noexcept;

//! \brief Packs a wrapped LwSciSyncAttrList to an IpcBuffer.
//!
//! \param [in,out] buf: Buffer in which the wrapper will be packed.
//! \param [in] val: Wrapper to be packed.
//!
//! \return LwSciError
//! - LwSciError_Success - If packing succeeded
//! - Any error returned by IpcBuffer::packVal().
//! - Any error returned by IpcBuffer::packBlob().
//! - Any error returned by LwSciSyncAttrListIsReconciled().
//! - Any error returned by LwSciSyncAttrListIpcExportReconciled().
//! - Any error returned by LwSciSyncAttrListIpcExportUnreconciled().
LwSciError
ipcBufferPack(
    IpcBuffer& buf,
    LwSciWrap::SyncAttr const& val) noexcept;

//! \brief Unpacks a wrapped LwSciSyncAttrList from an IpcBuffer.
//!
//! \param [in,out] buf: Buffer from which the wrapper will be unpacked.
//! \param [in,out] val: Location in which to store the wrapper.
//!
//! \return LwSciError
//! - LwSciError_Success - If unpacking succeeded.
//! - Any error returned by IpcBuffer::unpackVal().
//! - Any error returned by IpcBuffer::unpackBlob().
//! - Any error returned by LwSciSyncAttrListIpcImportReconciled().
//! - Any error returned by LwSciSyncAttrListIpcImportUnreconciled().
LwSciError
ipcBufferUnpack(
    IpcBuffer& buf,
    LwSciWrap::SyncAttr& val) noexcept;

//! \brief Packs a wrapped LwSciSyncFence to an IpcBuffer.
//!
//! \param [in,out] buf: Buffer in which the wrapper will be packed.
//! \param [in] val: Wrapper to be packed.
//!
//! \return LwSciError
//! - LwSciError_Success - If packing succeeded
//! - Any error returned by IpcBuffer::packVal().
//! - Any error returned by LwSciSyncIpcExportFence().
LwSciError
ipcBufferPack(
    IpcBuffer& buf,
    LwSciWrap::SyncFence const& val) noexcept;

//! \brief Unpacks a wrapped LwSciSyncFence from an IpcBuffer.
//!
//! \param [in,out] buf: Buffer from which the wrapper will be unpacked.
//! \param [in,out] val: Location in which to store the wrapper.
//! \param [in] aux: Sync object for the fence.
//!
//! \return LwSciError
//! - LwSciError_Success - If unpacking succeeded.
//! - Any error returned by IpcBuffer::unpackVal().
//! - Any error returned by LwSciSyncIpcImportFence().
LwSciError
ipcBufferUnpack(
    IpcBuffer& buf,
    LwSciWrap::SyncFence& val,
    LwSciWrap::SyncObj const& aux) noexcept;

//! \brief Packs a wrapped LwSciSyncObj to an IpcBuffer.
//!
//! \param [in,out] buf: Buffer in which the wrapper will be packed.
//! \param [in] val: Wrapper to be packed.
//!
//! \return LwSciError
//! - LwSciError_Success - If packing succeeded
//! - Any error returned by IpcBuffer::packVal().
//! - Any error returned by IpcBuffer::packBlob().
//! - Any error returned by LwSciSyncIpcExportAttrListAndObj().
LwSciError
ipcBufferPack(
    IpcBuffer& buf,
    LwSciWrap::SyncObj const& val) noexcept;

//! \brief Unpacks a wrapped LwSciSyncObj from an IpcBuffer.
//!
//! \param [in,out] buf: Buffer from which the wrapper will be unpacked.
//! \param [in,out] val: Location in which to store the wrapper.
//! \param [in] aux: Attribute list used to validate the object.
//! \param [in} ignore: Indicates the sync object won't be used, so there
//!   is no need to import it.
//!
//! \return LwSciError
//! - LwSciError_Success - If unpacking succeeded.
//! - Any error returned by IpcBuffer::unpackVal().
//! - Any error returned by IpcBuffer::unpackBlob().
//! - Any error returned by LwSciSyncIpcImportAttrListAndObj().
LwSciError
ipcBufferUnpack(
    IpcBuffer& buf,
    LwSciWrap::SyncObj& val,
    LwSciWrap::SyncAttr const& aux,
    bool const ignore=false) noexcept;

//
// NOTE: The overloaded pack/unpack functions defined below are temporary
//       and will be replaced by new functions associated with objects created
//       for the new API paradigm. Therefore full doxygen comments are not
//       provided.
//

LwSciError
ipcBufferPack(IpcBuffer& buf,
              LwSciSyncFenceIpcExportDescriptor const& val) noexcept;

LwSciError
ipcBufferUnpack(IpcBuffer& buf,
                LwSciSyncFenceIpcExportDescriptor& val) noexcept;

LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A2_7_3))
LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A8_4_8))

// TODO:
// The IPC export/import of LwSciBufObj and LwSciSyncObj could return
// LwSciError_TryItAgain in C2C cases. The approach to retry after certain
// timeout may be violating the ARRs of DRIVE OS.
// We may need to have separate implementation for safety and non-safety
// for the below wrappers.

LwSciError bufObjImportWrapper(
    LwSciIpcEndpoint const ipcEndpoint,
    LwSciBufObjIpcExportDescriptor const& desc,
    LwSciBufAttrList const reconciledAttrList,
    LwSciBufObj& bufObj);

LwSciError bufObjExportWrapper(
    LwSciIpcEndpoint const ipcEndpoint,
    LwSciBufObjIpcExportDescriptor& desc,
    LwSciBufObj const bufObj);

LwSciError syncAttrListAndObjImportWrapper(
    LwSciSyncModule const module,
    LwSciIpcEndpoint const ipcEndpoint,
    void const *const attrListAndObjDesc,
    size_t const attrListAndObjDescSize,
    LwSciSyncAttrList const attrList,
    LwSciSyncObj& syncObj);

LwSciError syncAttrListAndObjExportWrapper(
    LwSciIpcEndpoint const ipcEndpoint,
    LwSciSyncObj const syncObj,
    void** const attrListAndObjDesc,
    size_t& attrListAndObjDescSize);

} // namespace LwSciStream
#endif // IPC_BUFFER_SCIWRAP_H
