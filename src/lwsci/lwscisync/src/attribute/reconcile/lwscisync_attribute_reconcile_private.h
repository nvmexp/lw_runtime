/*
 * Copyright (c) 2019-2021, LWPU CORPORATION. All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#ifndef INCLUDED_LWSCISYNC_ATTRIBUTE_RECONCILE_SEMAPHORE_H
#define INCLUDED_LWSCISYNC_ATTRIBUTE_RECONCILE_SEMAPHORE_H

#include "lwscierror.h"
#include "lwscisync_attribute_core.h"
#include "lwscisync_attribute_core_cluster.h"

/** Reconcile semaphore LwSciBufAttrList */
LwSciError ReconcileSemaAttrList(
    LwSciSyncCoreAttrListObj* objAttrList,
    LwSciSyncCoreAttrListObj* newObjAttrList);

/** Validate Reconciled semaphore LwSciBufAttrList */
LwSciError ValidateReconciledSemaAttrList(
    LwSciSyncCoreAttrListObj* objAttrList,
    LwSciSyncCoreAttrListObj* newObjAttrList);

/** Determine if timestamps are supported and needed */
LwSciError ReconcileWaiterRequireTimestamps(
    const LwSciSyncCoreAttrListObj* objAttrList,
    const LwSciSyncCoreAttrListObj* newObjAttrList);

/** Reconcile SignalerTimestampInfo key */
LwSciError ReconcileSignalerTimestampInfo(
    const LwSciSyncCoreAttrListObj* objAttrList,
    const LwSciSyncCoreAttrListObj* newObjAttrList);

#ifdef LWSCISYNC_EMU_SUPPORT
/** Validate the reconciled ExternalPrimitiveInfo against unreconciled ones */
LwSciError ValidateReconciledExternalPrimitiveInfo(
    const LwSciSyncCoreAttrListObj* objAttrList,
    const LwSciSyncCoreAttrListObj* recObjAttrList);

/** Reconcile the ExternalPrimitiveInfo attributes */
LwSciError ReconcileUseExternalPrimitiveInfo(
    const LwSciSyncCoreAttrListObj* objAttrList,
    const LwSciSyncCoreAttrListObj* newObjAttrList,
    LwSciSyncAttrList newReconciledList);
#endif

/**
 * \brief Fills Signaler Timestamp Info attributes for a CPU Signaler
 *
 * This fills the Signaler Timestamp Info attribute for a CPU Signaler when
 * none was provided.
 *
 * \param[in,out] objAttrList attribute list to get the primitive info
 *
 * \return void
 */
void LwSciSyncCoreFillSignalerTimestampInfo(
    const LwSciSyncCoreAttrListObj* objAttrList);

#endif
