/*
* _LWRM_COPYRIGHT_BEGIN_
*
* Copyright 2013-2022 by LWPU Corporation.  All rights reserved.  All
* information contained herein is proprietary and confidential to LWPU
* Corporation.  Any use, reproduction, or disclosure without the written
* permission of LWPU Corporation is prohibited.
*
* _LWRM_COPYRIGHT_END_
*/

#pragma once

//
// This file was generated with FINN, an LWPU coding tool.
// Source file: ctrl/ctrl90cc.finn
//
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
//
// Please edit the original FINN IDL file to create desired edits in this header
// See https://confluence.lwpu.com/display/CORERM/FINN for more info on how to
// edit FINN.
//
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)




#include "ctrl/ctrlxxxx.h"
#include "ctrl90cc/ctrl90ccbase.h"

#include "ctrl90cc/ctrl90cchwpm.h"
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

#include "ctrl90cc/ctrl90cclwlink.h"
/* LWRM_UNPUBLISHED */
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)


#include "ctrl90cc/ctrl90ccpower.h"


/*
 * LW90CC_CTRL_CMD_PROFILER_RESERVE_HWPM
 *
 * This command is deprecated; please use LW90CC_CTRL_CMD_HWPM_RESERVE.
 *
 * This command attempts to reserve the perfmon for use by the calling client.
 * If this object was allocated as a child of a subdevice, then the
 * reservation will be global among all contexts on that subdevice. If this
 * object was allocated as a child of a channel group or a channel, then the
 * reservation will only be for the hardware context of that channel group or
 * channel.
 *
 * If the global reservation is held on a subdevice by another client, then
 * this command will fail, regardless of the parent class.
 *
 * If one or more per-context reservations are held by other clients, then
 * this command will fail if the parent object is a subdevice or another
 * client already holds the perfmon reservation for the parent context.
 *
 * This command will return LW_ERR_STATE_IN_USE for all of the failure
 * cases described above. A return status of LW_OK guarantees
 * that the client holds the perfmon reservation of the appropriate scope.
 *
 * This command does not take any parameters.
 *
 * Possible status values returned are:
 *    LW_OK
 *    LW_ERR_STATE_IN_USE
 */
#define LW90CC_CTRL_CMD_PROFILER_RESERVE_HWPM              LW90CC_CTRL_CMD_HWPM_RESERVE

/*
 * LW90CC_CTRL_CMD_PROFILER_RELEASE_HWPM
 *
 * This command is deprecated; please use LW90CC_CTRL_CMD_HWPM_RELEASE.
 *
 * This command releases an existing reservation of the perfmon for the
 * calling client. If the calling client does not lwrrently have the perfmon
 * reservation as acquired by LW90CC_CTRL_CMD_PROFILER_RESERVE_HWPM, this
 * command will return LW_ERR_ILWALID_REQUEST.
 *
 * This command does not take any parameters.
 *
 * Possible status values returned are:
 *    LW_OK
 *    LW_ERR_ILWALID_REQUEST
 */
#define LW90CC_CTRL_CMD_PROFILER_RELEASE_HWPM              LW90CC_CTRL_CMD_HWPM_RELEASE

/*
 * LW90CC_CTRL_CMD_PROFILER_GET_HWPM_RESERVATION_INFO
 *
 * This command is deprecated; please use
 * LW90CC_CTRL_CMD_HWPM_GET_RESERVATION_INFO.
 *
 * This command returns information about any outstanding perfmon
 * reservations. If this object was allocated as a child of a subdevice, then
 * this command will return information about all reservations on the
 * subdevice (global or per-context). If this object was allocated as a child
 * of a channel group or channel, then this command will only return
 * information about the per-context reservation for that context or the
 * global reservation, if one exists.
 *
 *   reservationCount
 *     This parameter returns the number of outstanding perfmon reservations
 *     in the applicable scope. If the value of the bGlobal parameter is
 *     LW_TRUE, then this parameter will have a value of 1. If this object was
 *     allocated as a child of a channel group or channel, then this parameter
 *     will have a value of either 0 or 1. If this object was allocated as a
 *     child of a subdevice and the bGlobal parameter is LW_FALSE, then this
 *     parameter will return the number of per-context reservations on the
 *     subdevice.
 *   pid
 *     This parameter returns the PID of a process that holds a reservation in
 *     the applicable scope. If the value of the bGlobal parameter is LW_TRUE,
 *     then this parameter will be the PID of the process holding the global
 *     perfmon reservation on the subdevice. Otherwise, if the value of
 *     reservationCount is greater than 0, the value of this parameter will be
 *     the PID of one of the process that holds the per-context lock in the
 *     applicable scope. If the value of the reservationCount parameter is 0,
 *     the value of this parameter is undefined.
 *   bGlobal
 *     This parameter returns whether the outstanding perfmon reservation held
 *     by any client is global or per-context. If the value of this parameter
 *     is LW_TRUE, then the value of the reservationCount parameter shoud be 1
 *     and the value of the pid parameter should be the pid of the process
 *     that holds the global perfmon reservation. The value of this parameter
 *     will be LW_FALSE when there is no global perfmon reservation.
 *
 * Possible status values returned are:
 *    LW_OK
 *    LW_ERR_ILWALID_ARGUMENT
 */
#define LW90CC_CTRL_CMD_PROFILER_GET_HWPM_RESERVATION_INFO LW90CC_CTRL_CMD_HWPM_GET_RESERVATION_INFO

typedef LW90CC_CTRL_HWPM_GET_RESERVATION_INFO_PARAMS LW90CC_CTRL_PROFILER_GET_HWPM_RESERVATION_INFO_PARAMS;

/*
 * LW90CC_CTRL_CMD_PROFILER_REQUEST_CG_CONTROLS
 *
 * This command is deprecated; please use
 * LW90CC_CTRL_CMD_POWER_REQUEST_POWER_FEATURES.
 *
 * This command attempts to enable or disable various clock-gating features of
 * the GPU on behalf of the profiler. If this command is unable to set the
 * clock-gating feature state of any of the requested features, this command
 * will fail and none of the requested features will be modified. If this
 * command fails because one or more clock-gating feature requests were
 * rejected, it will return LW_ERR_STATE_IN_USE in the globalStatus
 * parameter and the fields in the statusMask parameter for the features for
 * which the requests were rejected will have the value
 * LW90CC_CTRL_PROFILER_CG_CONTROL_MASK_*_REQUEST_REJECTED.
 * If a given feature is not supported on the GPU, the field for that clock-
 * gating feature will have the value
 * LW90CC_CTRL_PROFILER_CG_CONTROL_MASK_*_REQUEST_NOT_SUPPORTED in the
 * statusMask parameter, but this condition by itself will not cause the
 * command to fail. Even if this command fails, the field for clock-gating
 * features which would have successfully changed will have the value
 * LW90CC_CTRL_PROFILER_CG_CONTROL_MASK_*_FULFILLED in the statusMask
 * parameter.
 *
 * Each of the clock-gating features is reference-counted individually, so
 * that multiple GF100_PROFILER objects may request and rely on the same
 * settings for the features simultaneously. Each clock-gating feature is
 * locked to the requested state until the GF100_PROFILER object is freed or
 * the LW90CC_CTRL_CMD_PROFILER_RELEASE_CG_CONTROLS command is called for that
 * feature.
 *
 * Lwrrently, only GF100_PROFILER requests for power features using this
 * command are reference counted. Changes to the power feature settings made
 * either by other control commands or the RM itself may interfere with the
 * settings requested by GF100_PROFILER instances.
 *
 * This command will always return LW_OK when given valid
 * parameters. If there is any other failure that prevents the clock-gating
 * features from being set appropriately, the globalStatus parameter will
 * indicate this and the statusMask parameter will indicate which clock-gating
 * feature requests failed and why.
 *
 *   controlMask
 *     This parameter indicates which clock-gating features the request should
 *     apply to. This parameter has the following fields:
 *       LW90CC_CTRL_PROFILER_CG_CONTROL_MASK_ELCG
 *         The value of this field indicates whether this request should apply
 *         to engine-level clock-gating of the GR engine. Valid values for
 *         this field are:
 *           LW90CC_CTRL_PROFILER_CG_CONTROL_MASK_ELCG_IGNORE
 *             This value indicates that the GR engine-level clock-gating
 *             should be ignored. This will not affect the reference count for
 *             this feature.
 *           LW90CC_CTRL_PROFILER_CG_CONTROL_MASK_ELCG_DISABLE
 *             This value indicates that the GR engine-level clock-gating
 *             should be disabled.
 *           LW90CC_CTRL_PROFILER_CG_CONTROL_MASK_ELCG_ENABLE
 *             This value indicates that the GR engine-level clock-gating
 *             should be enabled.
 *       LW90CC_CTRL_PROFILER_CG_CONTROL_MASK_BLCG
 *         The value of this field indicates whether this request should apply
 *         to block-level clock-gating. Valid values for this field are:
 *           LW90CC_CTRL_PROFILER_CG_CONTROL_MASK_BLCG_IGNORE
 *             This value indicates that block-level clock-gating should be
 *             ignored. This will not affect the reference count for this
 *             feature.
 *           LW90CC_CTRL_PROFILER_CG_CONTROL_MASK_BLCG_DISABLE
 *             This value indicates that block-level clock-gating should be
 *             disabled.
 *           LW90CC_CTRL_PROFILER_CG_CONTROL_MASK_BLCG_ENABLE
 *             This value indicates that block-level clock-gating should be
 *             enabled.
 *       LW90CC_CTRL_PROFILER_CG_CONTROL_MASK_SLCG
 *         The value of this field indicates whether this request should apply
 *         to second-level clock-gating. Valid values for this field are:
 *           LW90CC_CTRL_PROFILER_CG_CONTROL_MASK_SLCG_IGNORE
 *             This value indicates that second-level clock-gating should be
 *             ignored. This will not affect the reference count for this
 *             feature.
 *           LW90CC_CTRL_PROFILER_CG_CONTROL_MASK_SLCG_DISABLE
 *             This value indicates that second-level clock-gating should be
 *             disabled.
 *           LW90CC_CTRL_PROFILER_CG_CONTROL_MASK_SLCG_ENABLE
 *             This value indicates that second-level clock-gating should be
 *             enabled.
 *       LW90CC_CTRL_PROFILER_CG_CONTROL_MASK_ELPG
 *         The value of this field indicates whether this request should apply
 *         to GR engine-level power-gating. Valid values for this field are:
 *           LW90CC_CTRL_PROFILER_CG_CONTROL_MASK_ELPG_IGNORE
 *             This value indicates that engine-level power-gating should be
 *             ignored. This will not affect the reference count for this
 *             feature.
 *           LW90CC_CTRL_PROFILER_CG_CONTROL_MASK_ELPG_DISABLE
 *             This value indicates that engine-level power-gating should be
 *             disabled.
 *           LW90CC_CTRL_PROFILER_CG_CONTROL_MASK_ELPG_ENABLE
 *             This value indicates that engine-level power-gating should be
 *             enabled.
 *         Note that this field is only temporary to allow reference counting
 *         restricted to GF100_PROFILER instances, until the time when the
 *         existing controls for this power feature can be updated to support
 *         reference counting across all clients and the RM.
 *       LW90CC_CTRL_PROFILER_CG_CONTROL_MASK_IDLE_SLOWDOWN
 *         The value of this field indicates whether this request should apply
 *         to SM idle slowdown. Valid values for this field are:
 *           LW90CC_CTRL_PROFILER_CG_CONTROL_MASK_IDLE_SLOWDOWN_IGNORE
 *             This value indicates that SM idle slowdown should be ignored.
 *             This will not affect the reference count for this feature.
 *           LW90CC_CTRL_PROFILER_CG_CONTROL_MASK_IDLE_SLOWDOWN_DISABLE
 *             This value indicates that SM idle slowdown should be disabled.
 *           LW90CC_CTRL_PROFILER_CG_CONTROL_MASK_IDLE_SLOWDOWN_ENABLE
 *             This value indicates that SM idle slowdown should be enabled.
 *         Note that this field is only temporary to allow reference counting
 *         restricted to GF100_PROFILER instances, until the time when the
 *         existing controls for this power feature can be updated to support
 *         reference counting across all clients and the RM.
  *       LW90CC_CTRL_PROFILER_CG_CONTROL_MASK_VAT
 *         The value of this field indicates whether this request should apply
 *         to VAT. Valid values for this field are:
 *           LW90CC_CTRL_PROFILER_CG_CONTROL_MASK_VAT_IGNORE
 *             This value indicates that VAT should be ignored.
 *             This will not affect the reference count for this feature.
 *           LW90CC_CTRL_PROFILER_CG_CONTROL_MASK_VAT_DISABLE
 *             This value indicates that VAT should be disabled.
 *           LW90CC_CTRL_PROFILER_CG_CONTROL_MASK_VAT_ENABLE
 *             This value indicates that VAT should be enabled.
 *         Note that this field is only temporary to allow reference counting
 *         restricted to GF100_PROFILER instances, until the time when the
 *         existing controls for this power feature can be updated to support
 *         reference counting across all clients and the RM.
 *   globalStatus
 *     This parameter returns the overall status of the requests for all
 *     clock-gating controls. If the value of this parameter is not
 *     LW_OK, none of the clock-gating controls will be set as
 *     requested. Possible values for this parameter are:
 *       LW_OK
 *         This value indicates that all of the clock-gating control requests
 *         were either fulfilled or not supported on the hardware.
 *       LW_ERR_ILWALID_REQUEST
 *         This value indicates that at least one of the clock-gating control
 *         requests were invalid given the GF100_PROFILER instance's
 *         outstanding requests.
 *       LW_ERR_STATE_IN_USE
 *         This value indicates that at least one of the clock-gating controls
 *         has already been locked to a conflicting state by another
 *         GF100_PROFILER instance or the RM itself.
 *   statusMask
 *     This parameter returns the status of the request to set each clock-
 *     gating control specified by the controlMask parameter. The fields are
 *     identical to those of the controlMask parameter. For each field for
 *     which the corresponding field in the controlMask parameter has the
 *     value LW90CC_CTRL_PROFILER_CG_CONTROL_MASK_*_IGNORE, the value is
 *     undefined. For each field for which the corresponding field in the
 *     controlMask parameter has the value
 *     LW90CC_CTRL_PROFILER_CG_CONTROL_MASK_*_REQUEST, the value will be
 *     one of the following:
 *       LW90CC_CTRL_PROFILER_CG_CONTROL_MASK_*_REQUEST_FULFILLED
 *         This value indicates that the clock-gating feature corresponding to
 *         the field in question was enabled or disabled according to the
 *         controlMask parameter, and the reference count for the feature was
 *         incremented accordingly.
 *       LW90CC_CTRL_PROFILER_CG_CONTROL_MASK_*_REQUEST_REJECTED
 *         This value indicates that the clock-gating feature corresponding to
 *         the field in question was not set to the expected state according
 *         to the controlMask parameter because another conflicting request is
 *         lwrrently outstanding for the clock-gating feature.
 *       LW90CC_CTRL_PROFILER_CG_CONTROL_MASK_*_REQUEST_FAILED
 *         This value indicates that the clock-gating feature corresponding to
 *         the field in question was not set to the expected state according
 *         to the controlMask parameter because the attempt to do so failed
 *         with an error other than a conflicting request.
 *       LW90CC_CTRL_PROFILER_CG_CONTROL_MASK_*_REQUEST_NOT_SUPPORTED
 *         This value indicates that the clock-gating feature corresponding to
 *         the field in question is not supported on this GPU. 
 *
 * Possible status values returned are:
 *    LW_OK
 *    LW_ERR_ILWALID_ARGUMENT
 */
#define LW90CC_CTRL_CMD_PROFILER_REQUEST_CG_CONTROLS LW90CC_CTRL_CMD_POWER_REQUEST_FEATURES

typedef LW90CC_CTRL_POWER_REQUEST_FEATURES_PARAMS LW90CC_CTRL_PROFILER_REQUEST_CG_CONTROLS_PARAMS;

/* valid fields for the controlMask and statusMask parameters */
#define LW90CC_CTRL_PROFILER_CG_CONTROL_MASK_ELCG                 1:0
#define LW90CC_CTRL_PROFILER_CG_CONTROL_MASK_BLCG                 3:2
#define LW90CC_CTRL_PROFILER_CG_CONTROL_MASK_SLCG                 5:4

/* 
 * The following are temporary fields for the controlMask and statusMask
 * parameters. They are required to reference count their respective features
 * until the existing RM controls can be safely updated, and the definitions
 * for these features will be removed soon after that.
 */
#define LW90CC_CTRL_PROFILER_CG_CONTROL_MASK_ELPG                 7:6
#define LW90CC_CTRL_PROFILER_CG_CONTROL_MASK_IDLE_SLOWDOWN        9:8
#define LW90CC_CTRL_PROFILER_CG_CONTROL_MASK_VAT                  11:10

/* valid values for fields in the controlMask parameter */
#define LW90CC_CTRL_PROFILER_CG_CONTROL_IGNORE                                   (0x00000000)
#define LW90CC_CTRL_PROFILER_CG_CONTROL_DISABLE                                  (0x00000001)
#define LW90CC_CTRL_PROFILER_CG_CONTROL_ENABLE                                   (0x00000002)

#define LW90CC_CTRL_PROFILER_CG_CONTROL_MASK_ELCG_IGNORE                         LW90CC_CTRL_PROFILER_CG_CONTROL_IGNORE
#define LW90CC_CTRL_PROFILER_CG_CONTROL_MASK_ELCG_DISABLE                        LW90CC_CTRL_PROFILER_CG_CONTROL_DISABLE
#define LW90CC_CTRL_PROFILER_CG_CONTROL_MASK_ELCG_ENABLE                         LW90CC_CTRL_PROFILER_CG_CONTROL_ENABLE
#define LW90CC_CTRL_PROFILER_CG_CONTROL_MASK_BLCG_IGNORE                         LW90CC_CTRL_PROFILER_CG_CONTROL_IGNORE
#define LW90CC_CTRL_PROFILER_CG_CONTROL_MASK_BLCG_DISABLE                        LW90CC_CTRL_PROFILER_CG_CONTROL_DISABLE
#define LW90CC_CTRL_PROFILER_CG_CONTROL_MASK_BLCG_ENABLE                         LW90CC_CTRL_PROFILER_CG_CONTROL_ENABLE
#define LW90CC_CTRL_PROFILER_CG_CONTROL_MASK_SLCG_IGNORE                         LW90CC_CTRL_PROFILER_CG_CONTROL_IGNORE
#define LW90CC_CTRL_PROFILER_CG_CONTROL_MASK_SLCG_DISABLE                        LW90CC_CTRL_PROFILER_CG_CONTROL_DISABLE
#define LW90CC_CTRL_PROFILER_CG_CONTROL_MASK_SLCG_ENABLE                         LW90CC_CTRL_PROFILER_CG_CONTROL_ENABLE
#define LW90CC_CTRL_PROFILER_CG_CONTROL_MASK_ELPG_IGNORE                         LW90CC_CTRL_PROFILER_CG_CONTROL_IGNORE
#define LW90CC_CTRL_PROFILER_CG_CONTROL_MASK_ELPG_DISABLE                        LW90CC_CTRL_PROFILER_CG_CONTROL_DISABLE
#define LW90CC_CTRL_PROFILER_CG_CONTROL_MASK_ELPG_ENABLE                         LW90CC_CTRL_PROFILER_CG_CONTROL_ENABLE
#define LW90CC_CTRL_PROFILER_CG_CONTROL_MASK_IDLE_SLOWDOWN_IGNORE                LW90CC_CTRL_PROFILER_CG_CONTROL_IGNORE
#define LW90CC_CTRL_PROFILER_CG_CONTROL_MASK_IDLE_SLOWDOWN_DISABLE               LW90CC_CTRL_PROFILER_CG_CONTROL_DISABLE
#define LW90CC_CTRL_PROFILER_CG_CONTROL_MASK_IDLE_SLOWDOWN_ENABLE                LW90CC_CTRL_PROFILER_CG_CONTROL_ENABLE
#define LW90CC_CTRL_PROFILER_CG_CONTROL_MASK_VAT_IGNORE                          LW90CC_CTRL_PROFILER_CG_CONTROL_IGNORE
#define LW90CC_CTRL_PROFILER_CG_CONTROL_MASK_VAT_DISABLE                         LW90CC_CTRL_PROFILER_CG_CONTROL_DISABLE
#define LW90CC_CTRL_PROFILER_CG_CONTROL_MASK_VAT_ENABLE                          LW90CC_CTRL_PROFILER_CG_CONTROL_ENABLE
/* possible values for fields in the statusMask parameter */
#define LW90CC_CTRL_PROFILER_CG_CONTROL_REQUEST_FULFILLED                        (0x00000000)
#define LW90CC_CTRL_PROFILER_CG_CONTROL_REQUEST_REJECTED                         (0x00000001)
#define LW90CC_CTRL_PROFILER_CG_CONTROL_REQUEST_NOT_SUPPORTED                    (0x00000002)
#define LW90CC_CTRL_PROFILER_CG_CONTROL_REQUEST_FAILED                           (0x00000003)

#define LW90CC_CTRL_PROFILER_CG_CONTROL_MASK_BLCG_REQUEST_FULFILLED              LW90CC_CTRL_PROFILER_CG_CONTROL_REQUEST_FULFILLED
#define LW90CC_CTRL_PROFILER_CG_CONTROL_MASK_BLCG_REQUEST_REJECTED               LW90CC_CTRL_PROFILER_CG_CONTROL_REQUEST_REJECTED
#define LW90CC_CTRL_PROFILER_CG_CONTROL_MASK_BLCG_REQUEST_NOT_SUPPORTED          LW90CC_CTRL_PROFILER_CG_CONTROL_REQUEST_NOT_SUPPORTED
#define LW90CC_CTRL_PROFILER_CG_CONTROL_MASK_BLCG_REQUEST_FAILED                 LW90CC_CTRL_PROFILER_CG_CONTROL_REQUEST_FAILED
#define LW90CC_CTRL_PROFILER_CG_CONTROL_MASK_ELCG_REQUEST_FULFILLED              LW90CC_CTRL_PROFILER_CG_CONTROL_REQUEST_FULFILLED
#define LW90CC_CTRL_PROFILER_CG_CONTROL_MASK_ELCG_REQUEST_REJECTED               LW90CC_CTRL_PROFILER_CG_CONTROL_REQUEST_REJECTED
#define LW90CC_CTRL_PROFILER_CG_CONTROL_MASK_ELCG_REQUEST_NOT_SUPPORTED          LW90CC_CTRL_PROFILER_CG_CONTROL_REQUEST_NOT_SUPPORTED
#define LW90CC_CTRL_PROFILER_CG_CONTROL_MASK_ELCG_REQUEST_FAILED                 LW90CC_CTRL_PROFILER_CG_CONTROL_REQUEST_FAILED
#define LW90CC_CTRL_PROFILER_CG_CONTROL_MASK_SLCG_REQUEST_FULFILLED              LW90CC_CTRL_PROFILER_CG_CONTROL_REQUEST_FULFILLED
#define LW90CC_CTRL_PROFILER_CG_CONTROL_MASK_SLCG_REQUEST_REJECTED               LW90CC_CTRL_PROFILER_CG_CONTROL_REQUEST_REJECTED
#define LW90CC_CTRL_PROFILER_CG_CONTROL_MASK_SLCG_REQUEST_NOT_SUPPORTED          LW90CC_CTRL_PROFILER_CG_CONTROL_REQUEST_NOT_SUPPORTED
#define LW90CC_CTRL_PROFILER_CG_CONTROL_MASK_SLCG_REQUEST_FAILED                 LW90CC_CTRL_PROFILER_CG_CONTROL_REQUEST_FAILED
#define LW90CC_CTRL_PROFILER_CG_CONTROL_MASK_ELPG_REQUEST_FULFILLED              LW90CC_CTRL_PROFILER_CG_CONTROL_REQUEST_FULFILLED
#define LW90CC_CTRL_PROFILER_CG_CONTROL_MASK_ELPG_REQUEST_REJECTED               LW90CC_CTRL_PROFILER_CG_CONTROL_REQUEST_REJECTED
#define LW90CC_CTRL_PROFILER_CG_CONTROL_MASK_ELPG_REQUEST_NOT_SUPPORTED          LW90CC_CTRL_PROFILER_CG_CONTROL_REQUEST_NOT_SUPPORTED
#define LW90CC_CTRL_PROFILER_CG_CONTROL_MASK_ELPG_REQUEST_FAILED                 LW90CC_CTRL_PROFILER_CG_CONTROL_REQUEST_FAILED
#define LW90CC_CTRL_PROFILER_CG_CONTROL_MASK_IDLE_SLOWDOWN_REQUEST_FULFILLED     LW90CC_CTRL_PROFILER_CG_CONTROL_REQUEST_FULFILLED
#define LW90CC_CTRL_PROFILER_CG_CONTROL_MASK_IDLE_SLOWDOWN_REQUEST_REJECTED      LW90CC_CTRL_PROFILER_CG_CONTROL_REQUEST_REJECTED
#define LW90CC_CTRL_PROFILER_CG_CONTROL_MASK_IDLE_SLOWDOWN_REQUEST_NOT_SUPPORTED LW90CC_CTRL_PROFILER_CG_CONTROL_REQUEST_NOT_SUPPORTED
#define LW90CC_CTRL_PROFILER_CG_CONTROL_MASK_IDLE_SLOWDOWN_REQUEST_FAILED        LW90CC_CTRL_PROFILER_CG_CONTROL_REQUEST_FAILED
#define LW90CC_CTRL_PROFILER_CG_CONTROL_MASK_VAT_REQUEST_FULFILLED               LW90CC_CTRL_PROFILER_CG_CONTROL_REQUEST_FULFILLED
#define LW90CC_CTRL_PROFILER_CG_CONTROL_MASK_VAT_REQUEST_REJECTED                LW90CC_CTRL_PROFILER_CG_CONTROL_REQUEST_REJECTED
#define LW90CC_CTRL_PROFILER_CG_CONTROL_MASK_VAT_REQUEST_NOT_SUPPORTED           LW90CC_CTRL_PROFILER_CG_CONTROL_REQUEST_NOT_SUPPORTED
#define LW90CC_CTRL_PROFILER_CG_CONTROL_MASK_VAT_REQUEST_FAILED                  LW90CC_CTRL_PROFILER_CG_CONTROL_REQUEST_FAILED

/* utility masks for the controlMask parameter for all controls */
#define LW90CC_CTRL_PROFILER_CG_CONTROL_MASK_ALL                             \
    DRF_SHIFTMASK(LW90CC_CTRL_PROFILER_CG_CONTROL_MASK_ELCG) |               \
    DRF_SHIFTMASK(LW90CC_CTRL_PROFILER_CG_CONTROL_MASK_BLCG) |               \
    DRF_SHIFTMASK(LW90CC_CTRL_PROFILER_CG_CONTROL_MASK_SLCG) |               \
    DRF_SHIFTMASK(LW90CC_CTRL_PROFILER_CG_CONTROL_MASK_ELPG) |               \
    DRF_SHIFTMASK(LW90CC_CTRL_PROFILER_CG_CONTROL_MASK_IDLE_SLOWDOWN) |      \
    DRF_SHIFTMASK(LW90CC_CTRL_PROFILER_CG_CONTROL_MASK_VAT)
#define LW90CC_CTRL_PROFILER_CG_CONTROL_MASK_ALL_IGNORE                      \
    DRF_DEF(90CC, _CTRL_PROFILER_CG_CONTROL_MASK, _ELCG, _IGNORE) |          \
    DRF_DEF(90CC, _CTRL_PROFILER_CG_CONTROL_MASK, _BLCG, _IGNORE) |          \
    DRF_DEF(90CC, _CTRL_PROFILER_CG_CONTROL_MASK, _SLCG, _IGNORE) |          \
    DRF_DEF(90CC, _CTRL_PROFILER_CG_CONTROL_MASK, _ELPG, _IGNORE) |          \
    DRF_DEF(90CC, _CTRL_PROFILER_CG_CONTROL_MASK, _IDLE_SLOWDOWN, _IGNORE) | \
    DRF_DEF(90CC, _CTRL_PROFILER_CG_CONTROL_MASK, _VAT, _IGNORE)
#define LW90CC_CTRL_PROFILER_CG_CONTROL_MASK_ALL_DISABLE                     \
    DRF_DEF(90CC, _CTRL_PROFILER_CG_CONTROL_MASK, _ELCG, _DISABLE) |         \
    DRF_DEF(90CC, _CTRL_PROFILER_CG_CONTROL_MASK, _BLCG, _DISABLE) |         \
    DRF_DEF(90CC, _CTRL_PROFILER_CG_CONTROL_MASK, _SLCG, _DISABLE) |         \
    DRF_DEF(90CC, _CTRL_PROFILER_CG_CONTROL_MASK, _ELPG, _DISABLE) |         \
    DRF_DEF(90CC, _CTRL_PROFILER_CG_CONTROL_MASK, _IDLE_SLOWDOWN, _DISABLE)| \
    DRF_DEF(90CC, _CTRL_PROFILER_CG_CONTROL_MASK, _VAT, _DISABLE)
#define LW90CC_CTRL_PROFILER_CG_CONTROL_MASK_ALL_ENABLE                      \
    DRF_DEF(90CC, _CTRL_PROFILER_CG_CONTROL_MASK, _ELCG, _ENABLE) |          \
    DRF_DEF(90CC, _CTRL_PROFILER_CG_CONTROL_MASK, _BLCG, _ENABLE) |          \
    DRF_DEF(90CC, _CTRL_PROFILER_CG_CONTROL_MASK, _SLCG, _ENABLE) |          \
    DRF_DEF(90CC, _CTRL_PROFILER_CG_CONTROL_MASK, _ELPG, _ENABLE) |          \
    DRF_DEF(90CC, _CTRL_PROFILER_CG_CONTROL_MASK, _IDLE_SLOWDOWN, _ENABLE) | \
    DRF_DEF(90CC, _CTRL_PROFILER_CG_CONTROL_MASK, _VAT, _ENABLE)

/*
 * LW90CC_CTRL_CMD_PROFILER_RELEASE_CG_CONTROLS
 *
 * This command is deprecated; please us
 * LW90CC_CTRL_CMD_POWER_RELEASE_FEATURES.
 *
 * This command releases the GF100_PROFILER's request for the given clock-
 * gating features that was previously created by the
 * LW90CC_CTRL_CMD_PROFILER_REQUEST_CG_CONTROLS command. If the GF100_PROFILER
 * object does not have an outstanding request to one or more of the given
 * clock-gating features, those features will be ignored while the other
 * feature requests will be released.
 *
 * After calling this command, the calling client may not rely on the current
 * value of any of the released clock-gating features to remain, even if
 * several identical requests for the given clock-gating features were made
 * using LW90CC_CTRL_CMD_PROFILER_REQUEST_CG_CONTROLS. The RM only reference-
 * counts the settings requested by GF100_PROFILER instances - it does not
 * reference-count multiple identical requests made using the same
 * GF100_PROFILER instance.
 *
 * All outstanding requests made using this GF100_PROFILER object are
 * implicitly released when the GF100_PROFILER is freed.
 *
 *   controlMask
 *     This parameter indicates which clock-gating features the RM should
 *     release the GF100_PROFILER's reference to. See
 *     LW90CC_CTRL_CMD_PROFILER_REQUEST_CG_CONTROLS for valid fields. Valid
 *     values for each field are:
 *       LW90CC_CTRL_PROFILER_CG_CONTROL_MASK_ELCG_IGNORE
 *       LW90CC_CTRL_PROFILER_CG_CONTROL_MASK_BLCG_IGNORE
 *       LW90CC_CTRL_PROFILER_CG_CONTROL_MASK_SLCG_IGNORE
 *       LW90CC_CTRL_PROFILER_CG_CONTROL_MASK_ELPG_IGNORE
 *       LW90CC_CTRL_PROFILER_CG_CONTROL_MASK_IDLE_SLOWDOWN_IGNORE
 *       LW90CC_CTRL_PROFILER_CG_CONTROL_MASK_VAT_IGNORE
 *         This value indicates that the clock-gating feature associated with
 *         the field should not be released, even if the GF100_PROFILER has an
 *         outstanding request for it. This will not affect the reference
 *         count for the feature.
 *       LW90CC_CTRL_PROFILER_CG_CONTROL_MASK_ELCG_RELEASE
 *       LW90CC_CTRL_PROFILER_CG_CONTROL_MASK_BLCG_RELEASE
 *       LW90CC_CTRL_PROFILER_CG_CONTROL_MASK_SLCG_RELEASE
 *       LW90CC_CTRL_PROFILER_CG_CONTROL_MASK_ELPG_RELEASE
 *       LW90CC_CTRL_PROFILER_CG_CONTROL_MASK_IDLE_SLOWDOWN_RELEASE
 *       LW90CC_CTRL_PROFILER_CG_CONTROL_MASK_VAT_RELEASE
 *         This value indicates that the GF100_PROFILER's outstanding request
 *         for the clock-gating feature associated with the field should be
 *         released. This will decrement the reference count for the feature
 *         if the GF100_PROFILER has an outstanding request for it.
 *
 * Possible status values returned are:
 *    LW_OK
 *    LW_ERR_ILWALID_ARGUMENT
 */
#define LW90CC_CTRL_CMD_PROFILER_RELEASE_CG_CONTROLS                             LW90CC_CTRL_CMD_POWER_RELEASE_FEATURES

typedef LW90CC_CTRL_POWER_RELEASE_FEATURES_PARAMS LW90CC_CTRL_PROFILER_RELEASE_CG_CONTROLS_PARAMS;

/* 
 * valid values for the controlMask parameter in addition to
 * LW90CC_CTRL_PROFILER_CG_CONTROL_MASK_*_IGNORE
 */
#define LW90CC_CTRL_PROFILER_CG_CONTROL_RELEASE                    (0x00000003)

#define LW90CC_CTRL_PROFILER_CG_CONTROL_MASK_ELCG_RELEASE          LW90CC_CTRL_PROFILER_CG_CONTROL_RELEASE
#define LW90CC_CTRL_PROFILER_CG_CONTROL_MASK_BLCG_RELEASE          LW90CC_CTRL_PROFILER_CG_CONTROL_RELEASE
#define LW90CC_CTRL_PROFILER_CG_CONTROL_MASK_SLCG_RELEASE          LW90CC_CTRL_PROFILER_CG_CONTROL_RELEASE
#define LW90CC_CTRL_PROFILER_CG_CONTROL_MASK_ELPG_RELEASE          LW90CC_CTRL_PROFILER_CG_CONTROL_RELEASE
#define LW90CC_CTRL_PROFILER_CG_CONTROL_MASK_IDLE_SLOWDOWN_RELEASE LW90CC_CTRL_PROFILER_CG_CONTROL_RELEASE
#define LW90CC_CTRL_PROFILER_CG_CONTROL_MASK_VAT_RELEASE           LW90CC_CTRL_PROFILER_CG_CONTROL_RELEASE

/* utility mask for the controlMask parameter for all fields */
#define LW90CC_CTRL_PROFILER_CG_CONTROL_MASK_ALL_RELEASE                     \
    DRF_DEF(90CC, _CTRL_PROFILER_CG_CONTROL_MASK, _ELCG, _RELEASE) |         \
    DRF_DEF(90CC, _CTRL_PROFILER_CG_CONTROL_MASK, _BLCG, _RELEASE) |         \
    DRF_DEF(90CC, _CTRL_PROFILER_CG_CONTROL_MASK, _SLCG, _RELEASE) |         \
    DRF_DEF(90CC, _CTRL_PROFILER_CG_CONTROL_MASK, _ELPG, _RELEASE) |         \
    DRF_DEF(90CC, _CTRL_PROFILER_CG_CONTROL_MASK, _IDLE_SLOWDOWN, _RELEASE)| \
    DRF_DEF(90CC, _CTRL_PROFILER_CG_CONTROL_MASK, _VAT, _RELEASE)

#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

/*
 * LW90CC_CTRL_CMD_PROFILER_RESERVE_LWLINK_COUNTERS
 *
 * This command is deprecated; please use
 * LW90CC_CTRL_CMD_LWLINK_RESERVE_COUNTERS.
 *
 * This command attempts to reserve lwlink counters for use by the calling
 * client. This object should be allocated as a child of a subdevice.
 *
 * If the reservation is held by another client, then this command will fail and
 * will return LW_ERR_STATE_IN_USE. 
 *
 * This command will fail and return LW_ERR_ILWALID_OBJECT_PARENT if this object
 * is not a child of a subdevice. A return status of LW_OK guarantees that the
 * client holds the reservation.
 *
 * This command does not take any parameters.
 *
 * Possible status values returned are:
 *    LW_OK
 *    LW_ERR_STATE_IN_USE
 *    LW_ERR_ILWALID_OBJECT_PARENT
 */
#define LW90CC_CTRL_CMD_PROFILER_RESERVE_LWLINK_COUNTERS           LW90CC_CTRL_CMD_LWLINK_RESERVE_COUNTERS

/*
 * LW90CC_CTRL_CMD_PROFILER_RELEASE_LWLINK_COUNTERS
 *
 * This command is deprecated; please use
 * LW90CC_CTRL_CMD_LWLINK_RELEASE_COUNTERS.
 *
 * This command releases an existing reservation of the lwlink counters for
 * the calling client. If the calling client does not lwrrently have the 
 * reservation as acquired by LW90CC_CTRL_CMD_PROFILER_RESERVE_LWLINK_COUNTERS,
 * this command will return LW_ERR_ILWALID_REQUEST.
 *
 * This command does not take any parameters.
 *
 * Possible status values returned are:
 *    LW_OK
 *    LW_ERR_ILWALID_REQUEST
 */
#define LW90CC_CTRL_CMD_PROFILER_RELEASE_LWLINK_COUNTERS           LW90CC_CTRL_CMD_LWLINK_RELEASE_COUNTERS
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

