/*
 * Copyright (c) 2019-2022, LWPU CORPORATION. All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#ifndef INCLUDED_LWSCIBUF_ATTR_DESC_H
#define INCLUDED_LWSCIBUF_ATTR_DESC_H

#include "lwscilist.h"
#include "lwscibuf_attr_priv.h"
#include "lwscibuf_ipc_table.h"
#include "lwscibuf_attr_validate.h"

/**
 * @brief Enum denotes status of an attribute key in LwSciBufAttrList.
 *
 * @implements{18842187}
 */
typedef enum {
    /** implies that key is never set */
    LwSciBufAttrStatus_Empty,
    /** implies that key is set */
    LwSciBufAttrStatus_SetLocked,
    /** implies that key corresponds to unreconciled LwSciBufAttrList which
     * was cloned from another unreconciled LwSciBufAttrList and status of
     * which was changed to unlocked in order to allow setting value for
     * key again. */
    LwSciBufAttrStatus_SetUnlocked,
    /** implies that key corresponding to reconciled LwSciBufAttrList whose
     * value has been reconciled */
    LwSciBufAttrStatus_Reconciled,
    /** implies that key belongs to appended unreconciled conflict
     * LwSciBufAttrList which denotes conflicted values for a key resulting
     * into reconciliation failure */
    LwSciBufAttrStatus_Conflict
} LwSciBufAttrStatus;


/**
 * @brief Enum denotes reconcile policy of an attribute key.
 *
 * @implements{18842190}
 */
typedef enum {
    LwSciBuf_MatchPolicy,
    LwSciBuf_OrPolicy,
    LwSciBuf_MaxPolicy,
    LwSciBuf_ArrayUnionPolicy,
    LwSciBuf_ListUnionPolicy,
    LwSciBuf_ArrayIntersectionPolicy,
    LwSciBuf_GpuCacheAndPolicy,
    LwSciBuf_GpuCompressionMatchPolicy,
    LwSciBuf_IlwalidPolicy,
    LwSciBuf_PolicyUpperBound
} LwSciBuf_ReconcilePolicy;

/**
 * @brief Enum denotes access type of an attribute key.
 *
 * @implements{18842193}
 */
typedef enum {
    LwSciBufKeyAccess_Input = 0x00U,
    LwSciBufKeyAccess_Output,
    LwSciBufKeyAccess_InOut,
} LwSciBufKeyAccess;

#define LW_SCI_BUF_PYRAMID_MAX_PLANES \
            LW_SCI_BUF_IMAGE_MAX_PLANES * LW_SCI_BUF_PYRAMID_MAX_LEVELS

/**
 * Maximum number of GPUs that can share the LwSciBufObj.
 * @implements{}
 */
#define LW_SCI_BUF_MAX_GPUS 16

/**
 * @defgroup LW_SCI_BUF_ATTR_KEY_DEF_GROUP, this group
 * define the desscription of public general
 * attribute key. This macro is defined in form of
 * LW_SCI_BUF_ATTR_KEY_DEF(keyname, key_datatype, max_arr_entries, is_read_only)
 * This macro laters expands to form perslotattrlist static structure and
 * attrdesclist which holds information of offset, get/set permission, type,
 * default value
 *
 * @{
 */

/**
 * @brief Describes public general attributes
 */
#define LW_SCI_BUF_PUB_GENERAL_ATTR \
    LW_SCI_BUF_ATTR_KEY_DEF( \
        LwSciBufGeneralAttrKey_Types, \
        LW_SCI_BUF_GENKEYTYPE_TYPES, \
        LwSciBufType_MaxValid, \
        LwSciBufKeyAccess_InOut, \
        LwSciBuf_ArrayUnionPolicy, \
        LwSciBufValidateBufferType, \
        LwSciBufValidateBufferType, \
        LwSciBufKeyImportQualifier_Mandatory, \
        LwSciBufKeyReconcileQualifier_Mandatory, \
        LwSciBufIpcRoute_AffinityNone, \
        LwSciBufIpcRoute_AffinityNone) \
    LW_SCI_BUF_ATTR_KEY_DEF( \
        LwSciBufGeneralAttrKey_NeedCpuAccess, \
        LW_SCI_BUF_GENKEYTYPE_NEEDCPUACCESS, \
        1, \
        LwSciBufKeyAccess_InOut, \
        LwSciBuf_MaxPolicy, \
        LwSciBufValidateBool, \
        LwSciBufValidateBool, \
        LwSciBufKeyImportQualifier_Mandatory, \
        LwSciBufKeyReconcileQualifier_Optional, \
        LwSciBufIpcRoute_OwnerAffinity, \
        LwSciBufIpcRoute_OwnerAffinity) \
    LW_SCI_BUF_ATTR_KEY_DEF( \
        LwSciBufGeneralAttrKey_RequiredPerm, \
        LW_SCI_BUF_GENKEYTYPE_REQUIREDPERM, \
        1, \
        LwSciBufKeyAccess_Input, \
        LwSciBuf_MaxPolicy, \
        LwSciBufValidatePermission, \
        LwSciBufValidatePermission, \
        LwSciBufKeyImportQualifier_Max, \
        LwSciBufKeyReconcileQualifier_Optional, \
        LwSciBufIpcRoute_RouteAffinity, \
        LwSciBufIpcRoute_RouteAffinity) \
    LW_SCI_BUF_ATTR_KEY_DEF( \
        LwSciBufGeneralAttrKey_EnableCpuCache, \
        LW_SCI_BUF_GENKEYTYPE_ENABLECPUCACHE, \
        1, \
        LwSciBufKeyAccess_InOut, \
        LwSciBuf_MaxPolicy, \
        LwSciBufValidateBool, \
        LwSciBufValidateBool, \
        LwSciBufKeyImportQualifier_Optional, \
        LwSciBufKeyReconcileQualifier_Optional, \
        LwSciBufIpcRoute_SocAffinity, \
        LwSciBufIpcRoute_SocAffinity) \
    LW_SCI_BUF_ATTR_KEY_DEF( \
        LwSciBufGeneralAttrKey_GpuId, \
        LW_SCI_BUF_GENKEYTYPE_GPUID, \
        LW_SCI_BUF_MAX_GPUS, \
        LwSciBufKeyAccess_InOut, \
        LwSciBuf_ArrayUnionPolicy, \
        LwSciBufValidateGpuId, \
        LwSciBufValidateGpuId, \
        LwSciBufKeyImportQualifier_Optional, \
        LwSciBufKeyReconcileQualifier_Optional, \
        LwSciBufIpcRoute_SocAffinity, \
        LwSciBufIpcRoute_SocAffinity) \
    LW_SCI_BUF_ATTR_KEY_DEF( \
        LwSciBufGeneralAttrKey_CpuNeedSwCacheCoherency, \
        LW_SCI_BUF_GENKEYTYPE_CPUSWCACHECOHERENCY, \
        1, \
        LwSciBufKeyAccess_Output, \
        LwSciBuf_IlwalidPolicy, \
        LwSciBufValidateBool, \
        LwSciBufValidateBool, \
        LwSciBufKeyImportQualifier_Mandatory, \
        LwSciBufKeyReconcileQualifier_Max, \
        LwSciBufIpcRoute_SocAffinity, \
        LwSciBufIpcRoute_SocAffinity) \
    LW_SCI_BUF_ATTR_KEY_DEF( \
        LwSciBufGeneralAttrKey_ActualPerm, \
        LW_SCI_BUF_GENKEYTYPE_ACTUALPERM, \
        1, \
        LwSciBufKeyAccess_Output, \
        LwSciBuf_IlwalidPolicy, \
        LwSciBufValidatePermission, \
        LwSciBufValidatePermission, \
        LwSciBufKeyImportQualifier_Mandatory, \
        LwSciBufKeyReconcileQualifier_Max, \
        LwSciBufIpcRoute_RouteAffinity, \
        LwSciBufIpcRoute_RouteAffinity) \
    LW_SCI_BUF_ATTR_KEY_DEF( \
        LwSciBufGeneralAttrKey_VidMem_GpuId, \
        LW_SCI_BUF_GENKEYTYPE_VIDMEMGPUID, \
        1, \
        LwSciBufKeyAccess_InOut, \
        LwSciBuf_MatchPolicy, \
        LwSciBufValidateGpuId, \
        LwSciBufValidateGpuId, \
        LwSciBufKeyImportQualifier_Optional, \
        LwSciBufKeyReconcileQualifier_Optional, \
        LwSciBufIpcRoute_SocAffinity, \
        LwSciBufIpcRoute_SocAffinity) \
    LW_SCI_BUF_ATTR_KEY_DEF( \
        LwSciBufGeneralAttrKey_EnableGpuCache, \
        LwSciBufAttrValGpuCache, \
        LW_SCI_BUF_MAX_GPUS, \
        LwSciBufKeyAccess_InOut, \
        LwSciBuf_GpuCacheAndPolicy, \
        LwSciBufValidateAttrValGpuCache, \
        LwSciBufValidateAttrValGpuCache, \
        LwSciBufKeyImportQualifier_Optional, \
        LwSciBufKeyReconcileQualifier_Optional, \
        LwSciBufIpcRoute_SocAffinity, \
        LwSciBufIpcRoute_SocAffinity) \
    LW_SCI_BUF_ATTR_KEY_DEF( \
        LwSciBufGeneralAttrKey_GpuSwNeedCacheCoherency, \
        LwSciBufAttrValGpuCache, \
        LW_SCI_BUF_MAX_GPUS, \
        LwSciBufKeyAccess_Output, \
        LwSciBuf_IlwalidPolicy, \
        LwSciBufValidateAttrValGpuCache, \
        LwSciBufValidateAttrValGpuCache, \
        LwSciBufKeyImportQualifier_Optional, \
        LwSciBufKeyReconcileQualifier_Optional, \
        LwSciBufIpcRoute_SocAffinity, \
        LwSciBufIpcRoute_SocAffinity) \
    LW_SCI_BUF_ATTR_KEY_DEF( \
        LwSciBufGeneralAttrKey_EnableGpuCompression, \
        LwSciBufAttrValGpuCompression, \
        LW_SCI_BUF_MAX_GPUS, \
        LwSciBufKeyAccess_InOut, \
        LwSciBuf_GpuCompressionMatchPolicy, \
        LwSciBufValidateAttrValGpuCompressionExternal, \
        LwSciBufValidateAttrValGpuCompressionInternal, \
        LwSciBufKeyImportQualifier_Optional, \
        LwSciBufKeyReconcileQualifier_Optional, \
        LwSciBufIpcRoute_SocAffinity, \
        LwSciBufIpcRoute_SocAffinity)

/**
 * @brief Describes public raw-buffer attributes
 */
#define LW_SCI_BUF_PUB_RAWBUFFER_ATTR \
    LW_SCI_BUF_ATTR_KEY_DEF( \
        LwSciBufRawBufferAttrKey_Size, \
        LW_SCI_BUF_RAWKEYTYPE_SIZE, \
        1, \
        LwSciBufKeyAccess_InOut, \
        LwSciBuf_MatchPolicy, \
        LwSciBufValidateBufferSize, \
        LwSciBufValidateBufferSize, \
        LwSciBufKeyImportQualifier_Mandatory, \
        LwSciBufKeyReconcileQualifier_Mandatory, \
        LwSciBufIpcRoute_AffinityNone, \
        LwSciBufIpcRoute_AffinityNone) \
    LW_SCI_BUF_ATTR_KEY_DEF( \
        LwSciBufRawBufferAttrKey_Align, \
        LW_SCI_BUF_RAWKEYTYPE_ALIGN, \
        1, \
        LwSciBufKeyAccess_InOut, \
        LwSciBuf_MaxPolicy, \
        LwSciBufValidateAlignmentU64, \
        LwSciBufValidateAlignmentU64, \
        LwSciBufKeyImportQualifier_Optional, \
        LwSciBufKeyReconcileQualifier_Optional, \
        LwSciBufIpcRoute_AffinityNone, \
        LwSciBufIpcRoute_AffinityNone)

/**
 * @brief LW_SCI_BUF_IMAGE_ATTR describes public image attributes
 */
#define LW_SCI_BUF_PUB_IMAGE_ATTR \
    LW_SCI_BUF_ATTR_KEY_DEF( \
        LwSciBufImageAttrKey_Layout, \
        LW_SCI_BUF_IMGKEYTYPE_LAYOUT, \
        1, \
        LwSciBufKeyAccess_InOut, \
        LwSciBuf_MatchPolicy, \
        LwSciBufValidateLayoutType, \
        LwSciBufValidateLayoutType, \
        LwSciBufKeyImportQualifier_Mandatory, \
        LwSciBufKeyReconcileQualifier_Mandatory, \
        LwSciBufIpcRoute_AffinityNone, \
        LwSciBufIpcRoute_AffinityNone) \
    LW_SCI_BUF_ATTR_KEY_DEF( \
        LwSciBufImageAttrKey_TopPadding, \
        LW_SCI_BUF_IMGKEYTYPE_TOPPADDING, \
        LW_SCI_BUF_IMAGE_MAX_PLANES, \
        LwSciBufKeyAccess_InOut, \
        LwSciBuf_MatchPolicy, \
        NULL, \
        NULL, \
        LwSciBufKeyImportQualifier_Optional, \
        LwSciBufKeyReconcileQualifier_Optional, \
        LwSciBufIpcRoute_AffinityNone, \
        LwSciBufIpcRoute_AffinityNone) \
    LW_SCI_BUF_ATTR_KEY_DEF( \
        LwSciBufImageAttrKey_BottomPadding, \
        LW_SCI_BUF_IMGKEYTYPE_BOTTOMPADDING, \
        LW_SCI_BUF_IMAGE_MAX_PLANES, \
        LwSciBufKeyAccess_InOut, \
        LwSciBuf_MatchPolicy, \
        NULL, \
        NULL, \
        LwSciBufKeyImportQualifier_Optional, \
        LwSciBufKeyReconcileQualifier_Optional, \
        LwSciBufIpcRoute_AffinityNone, \
        LwSciBufIpcRoute_AffinityNone) \
    LW_SCI_BUF_ATTR_KEY_DEF( \
        LwSciBufImageAttrKey_LeftPadding, \
        LW_SCI_BUF_IMGKEYTYPE_LEFTPADDING, \
        LW_SCI_BUF_IMAGE_MAX_PLANES, \
        LwSciBufKeyAccess_InOut, \
        LwSciBuf_MatchPolicy, \
        NULL, \
        NULL, \
        LwSciBufKeyImportQualifier_Optional, \
        LwSciBufKeyReconcileQualifier_Optional, \
        LwSciBufIpcRoute_AffinityNone, \
        LwSciBufIpcRoute_AffinityNone) \
    LW_SCI_BUF_ATTR_KEY_DEF( \
        LwSciBufImageAttrKey_RightPadding, \
        LW_SCI_BUF_IMGKEYTYPE_RIGHTPADDING, \
        LW_SCI_BUF_IMAGE_MAX_PLANES, \
        LwSciBufKeyAccess_InOut, \
        LwSciBuf_MatchPolicy, \
        NULL, \
        NULL, \
        LwSciBufKeyImportQualifier_Optional, \
        LwSciBufKeyReconcileQualifier_Optional, \
        LwSciBufIpcRoute_AffinityNone, \
        LwSciBufIpcRoute_AffinityNone) \
    LW_SCI_BUF_ATTR_KEY_DEF( \
        LwSciBufImageAttrKey_VprFlag, \
        LW_SCI_BUF_IMGKEYTYPE_VPRFLAG, \
        1, \
        LwSciBufKeyAccess_InOut, \
        LwSciBuf_OrPolicy, \
        LwSciBufValidateBool, \
        LwSciBufValidateBool, \
        LwSciBufKeyImportQualifier_Optional, \
        LwSciBufKeyReconcileQualifier_Optional, \
        LwSciBufIpcRoute_AffinityNone, \
        LwSciBufIpcRoute_AffinityNone) \
    LW_SCI_BUF_ATTR_KEY_DEF( \
        LwSciBufImageAttrKey_Size, \
        LW_SCI_BUF_IMGKEYTYPE_SIZE, \
        1, \
        LwSciBufKeyAccess_Output, \
        LwSciBuf_IlwalidPolicy, \
        LwSciBufValidateBufferSize, \
        LwSciBufValidateBufferSize, \
        LwSciBufKeyImportQualifier_Mandatory, \
        LwSciBufKeyReconcileQualifier_Max, \
        LwSciBufIpcRoute_AffinityNone, \
        LwSciBufIpcRoute_AffinityNone) \
    LW_SCI_BUF_ATTR_KEY_DEF( \
        LwSciBufImageAttrKey_Alignment, \
        LW_SCI_BUF_IMGKEYTYPE_ALIGNMENT, \
        1, \
        LwSciBufKeyAccess_Output, \
        LwSciBuf_IlwalidPolicy, \
        LwSciBufValidateAlignmentU64, \
        LwSciBufValidateAlignmentU64, \
        LwSciBufKeyImportQualifier_Mandatory, \
        LwSciBufKeyReconcileQualifier_Max, \
        LwSciBufIpcRoute_AffinityNone, \
        LwSciBufIpcRoute_AffinityNone) \
    LW_SCI_BUF_ATTR_KEY_DEF( \
        LwSciBufImageAttrKey_PlaneCount, \
        LW_SCI_BUF_IMGKEYTYPE_PLANECOUNT, \
        1, \
        LwSciBufKeyAccess_InOut, \
        LwSciBuf_MatchPolicy, \
        LwSciBufValidatePlaneCount, \
        LwSciBufValidatePlaneCount, \
        LwSciBufKeyImportQualifier_Mandatory, \
        LwSciBufKeyReconcileQualifier_Mandatory, \
        LwSciBufIpcRoute_AffinityNone, \
        LwSciBufIpcRoute_AffinityNone) \
    LW_SCI_BUF_ATTR_KEY_DEF( \
        LwSciBufImageAttrKey_PlaneColorFormat, \
        LW_SCI_BUF_IMGKEYTYPE_PLANECOLORFMT, \
        LW_SCI_BUF_IMAGE_MAX_PLANES, \
        LwSciBufKeyAccess_InOut, \
        LwSciBuf_MatchPolicy, \
        LwSciBufValidatePlaneColorFmt, \
        LwSciBufValidatePlaneColorFmt, \
        LwSciBufKeyImportQualifier_Mandatory, \
        LwSciBufKeyReconcileQualifier_Mandatory, \
        LwSciBufIpcRoute_AffinityNone, \
        LwSciBufIpcRoute_AffinityNone) \
    LW_SCI_BUF_ATTR_KEY_DEF( \
        LwSciBufImageAttrKey_PlaneColorStd, \
        LW_SCI_BUF_IMGKEYTYPE_PLANECOLORSTD, \
        LW_SCI_BUF_IMAGE_MAX_PLANES, \
        LwSciBufKeyAccess_InOut, \
        LwSciBuf_MatchPolicy, \
        LwSciBufValidatePlaneColorStd, \
        LwSciBufValidatePlaneColorStd, \
        LwSciBufKeyImportQualifier_Optional, \
        LwSciBufKeyReconcileQualifier_Optional, \
        LwSciBufIpcRoute_AffinityNone, \
        LwSciBufIpcRoute_AffinityNone) \
    LW_SCI_BUF_ATTR_KEY_DEF( \
        LwSciBufImageAttrKey_PlaneBaseAddrAlign, \
        LW_SCI_BUF_IMGKEYTYPE_PLANEADDRALIGN, \
        LW_SCI_BUF_PYRAMID_MAX_PLANES, \
        LwSciBufKeyAccess_InOut, \
        LwSciBuf_MaxPolicy, \
        LwSciBufValidateAlignmentU32, \
        LwSciBufValidateAlignmentU32, \
        LwSciBufKeyImportQualifier_Mandatory, \
        LwSciBufKeyReconcileQualifier_Optional, \
        LwSciBufIpcRoute_AffinityNone, \
        LwSciBufIpcRoute_AffinityNone) \
    LW_SCI_BUF_ATTR_KEY_DEF( \
        LwSciBufImageAttrKey_PlaneWidth, \
        LW_SCI_BUF_IMGKEYTYPE_PLANEWIDTH, \
        LW_SCI_BUF_PYRAMID_MAX_PLANES, \
        LwSciBufKeyAccess_InOut, \
        LwSciBuf_MatchPolicy, \
        LwSciBufValidateImageWidth, \
        LwSciBufValidateImageWidth, \
        LwSciBufKeyImportQualifier_Mandatory, \
        LwSciBufKeyReconcileQualifier_Mandatory, \
        LwSciBufIpcRoute_AffinityNone, \
        LwSciBufIpcRoute_AffinityNone) \
    LW_SCI_BUF_ATTR_KEY_DEF( \
        LwSciBufImageAttrKey_PlaneHeight, \
        LW_SCI_BUF_IMGKEYTYPE_PLANEHEIGHT, \
        LW_SCI_BUF_PYRAMID_MAX_PLANES, \
        LwSciBufKeyAccess_InOut, \
        LwSciBuf_MatchPolicy, \
        LwSciBufValidateImageHeight, \
        LwSciBufValidateImageHeight, \
        LwSciBufKeyImportQualifier_Mandatory, \
        LwSciBufKeyReconcileQualifier_Mandatory, \
        LwSciBufIpcRoute_AffinityNone, \
        LwSciBufIpcRoute_AffinityNone) \
    LW_SCI_BUF_ATTR_KEY_DEF( \
        LwSciBufImageAttrKey_ScanType, \
        LW_SCI_BUF_IMGKEYTYPE_SCANTYPE, \
        1, \
        LwSciBufKeyAccess_InOut, \
        LwSciBuf_MatchPolicy, \
        LwSciBufValidateScanType, \
        LwSciBufValidateScanType, \
        LwSciBufKeyImportQualifier_Mandatory, \
        LwSciBufKeyReconcileQualifier_Mandatory, \
        LwSciBufIpcRoute_AffinityNone, \
        LwSciBufIpcRoute_AffinityNone) \
    LW_SCI_BUF_ATTR_KEY_DEF( \
        LwSciBufImageAttrKey_PlaneBitsPerPixel, \
        LW_SCI_BUF_IMGKEYTYPE_PLANEBITSPERPIXEL, \
        LW_SCI_BUF_IMAGE_MAX_PLANES, \
        LwSciBufKeyAccess_Output, \
        LwSciBuf_IlwalidPolicy, \
        LwSciBufValidateBitsPerPixel, \
        LwSciBufValidateBitsPerPixel, \
        LwSciBufKeyImportQualifier_Mandatory, \
        LwSciBufKeyReconcileQualifier_Max, \
        LwSciBufIpcRoute_AffinityNone, \
        LwSciBufIpcRoute_AffinityNone) \
    LW_SCI_BUF_ATTR_KEY_DEF( \
        LwSciBufImageAttrKey_PlaneOffset, \
        LW_SCI_BUF_IMGKEYTYPE_PLANEOFFSET, \
        LW_SCI_BUF_PYRAMID_MAX_PLANES, \
        LwSciBufKeyAccess_Output, \
        LwSciBuf_IlwalidPolicy, \
        NULL, \
        NULL, \
        LwSciBufKeyImportQualifier_Mandatory, \
        LwSciBufKeyReconcileQualifier_Max, \
        LwSciBufIpcRoute_AffinityNone, \
        LwSciBufIpcRoute_AffinityNone) \
    LW_SCI_BUF_ATTR_KEY_DEF( \
        LwSciBufImageAttrKey_PlaneDatatype, \
        LW_SCI_BUF_IMGKEYTYPE_PLANEDATATYPE, \
        LW_SCI_BUF_IMAGE_MAX_PLANES, \
        LwSciBufKeyAccess_Output, \
        LwSciBuf_IlwalidPolicy, \
        LwSciBufValidatePlaneDataType, \
        LwSciBufValidatePlaneDataType, \
        LwSciBufKeyImportQualifier_Mandatory, \
        LwSciBufKeyReconcileQualifier_Max, \
        LwSciBufIpcRoute_AffinityNone, \
        LwSciBufIpcRoute_AffinityNone) \
    LW_SCI_BUF_ATTR_KEY_DEF( \
        LwSciBufImageAttrKey_PlaneChannelCount, \
        LW_SCI_BUF_IMGKEYTYPE_PLANECHANNELCOUNT, \
        LW_SCI_BUF_IMAGE_MAX_PLANES, \
        LwSciBufKeyAccess_Output, \
        LwSciBuf_IlwalidPolicy, \
        LwSciBufValidatePlaneChannelCount, \
        LwSciBufValidatePlaneChannelCount, \
        LwSciBufKeyImportQualifier_Mandatory, \
        LwSciBufKeyReconcileQualifier_Max, \
        LwSciBufIpcRoute_AffinityNone, \
        LwSciBufIpcRoute_AffinityNone) \
    LW_SCI_BUF_ATTR_KEY_DEF( \
        LwSciBufImageAttrKey_PlaneSecondFieldOffset, \
        LW_SCI_BUF_IMGKEYTYPE_PLANESECONDOFFSET, \
        LW_SCI_BUF_PYRAMID_MAX_PLANES, \
        LwSciBufKeyAccess_Output, \
        LwSciBuf_IlwalidPolicy, \
        NULL, \
        NULL, \
        LwSciBufKeyImportQualifier_Mandatory, \
        LwSciBufKeyReconcileQualifier_Max, \
        LwSciBufIpcRoute_AffinityNone, \
        LwSciBufIpcRoute_AffinityNone) \
    LW_SCI_BUF_ATTR_KEY_DEF( \
        LwSciBufImageAttrKey_PlanePitch, \
        LW_SCI_BUF_IMGKEYTYPE_PLANEPITCH, \
        LW_SCI_BUF_PYRAMID_MAX_PLANES, \
        LwSciBufKeyAccess_Output, \
        LwSciBuf_IlwalidPolicy, \
        LwSciBufValidatePlanePitch, \
        LwSciBufValidatePlanePitch, \
        LwSciBufKeyImportQualifier_Mandatory, \
        LwSciBufKeyReconcileQualifier_Max, \
        LwSciBufIpcRoute_AffinityNone, \
        LwSciBufIpcRoute_AffinityNone) \
    LW_SCI_BUF_ATTR_KEY_DEF( \
        LwSciBufImageAttrKey_PlaneAlignedHeight, \
        LW_SCI_BUF_IMGKEYTYPE_PLANEALIGNEDHEIGHT, \
        LW_SCI_BUF_PYRAMID_MAX_PLANES, \
        LwSciBufKeyAccess_Output, \
        LwSciBuf_IlwalidPolicy, \
        LwSciBufValidatePlaneAlignedHeight, \
        LwSciBufValidatePlaneAlignedHeight, \
        LwSciBufKeyImportQualifier_Mandatory, \
        LwSciBufKeyReconcileQualifier_Max, \
        LwSciBufIpcRoute_AffinityNone, \
        LwSciBufIpcRoute_AffinityNone) \
    LW_SCI_BUF_ATTR_KEY_DEF( \
        LwSciBufImageAttrKey_PlaneAlignedSize, \
        LW_SCI_BUF_IMGKEYTYPE_PLANEALIGNEDSIZE, \
        LW_SCI_BUF_PYRAMID_MAX_PLANES, \
        LwSciBufKeyAccess_Output, \
        LwSciBuf_IlwalidPolicy, \
        LwSciBufValidateBufferSize, \
        LwSciBufValidateBufferSize, \
        LwSciBufKeyImportQualifier_Mandatory, \
        LwSciBufKeyReconcileQualifier_Max, \
        LwSciBufIpcRoute_AffinityNone, \
        LwSciBufIpcRoute_AffinityNone) \
    LW_SCI_BUF_ATTR_KEY_DEF( \
        LwSciBufImageAttrKey_ImageCount, \
        LW_SCI_BUF_IMGKEYTYPE_IMAGECOUNT, \
        1, \
        LwSciBufKeyAccess_InOut, \
        LwSciBuf_MatchPolicy, \
        LwSciBufValidateImageCount, \
        LwSciBufValidateImageCount, \
        LwSciBufKeyImportQualifier_Conditional, \
        LwSciBufKeyReconcileQualifier_Conditional, \
        LwSciBufIpcRoute_AffinityNone, \
        LwSciBufIpcRoute_AffinityNone) \
    LW_SCI_BUF_ATTR_KEY_DEF( \
        LwSciBufImageAttrKey_SurfType, \
        LW_SCI_BUF_IMGKEYTYPE_SURFTYPE, \
        1, \
        LwSciBufKeyAccess_InOut, \
        LwSciBuf_MatchPolicy, \
        LwSciBufValidateSurfType, \
        LwSciBufValidateSurfType, \
        LwSciBufKeyImportQualifier_Optional, \
        LwSciBufKeyReconcileQualifier_Optional, \
        LwSciBufIpcRoute_AffinityNone, \
        LwSciBufIpcRoute_AffinityNone) \
    LW_SCI_BUF_ATTR_KEY_DEF( \
        LwSciBufImageAttrKey_SurfMemLayout, \
        LW_SCI_BUF_IMGKEYTYPE_SURFMEMLAYOUT, \
        1, \
        LwSciBufKeyAccess_InOut, \
        LwSciBuf_MatchPolicy, \
        LwSciBufValidateSurfMemLayout, \
        LwSciBufValidateSurfMemLayout, \
        LwSciBufKeyImportQualifier_Optional, \
        LwSciBufKeyReconcileQualifier_Optional, \
        LwSciBufIpcRoute_AffinityNone, \
        LwSciBufIpcRoute_AffinityNone) \
    LW_SCI_BUF_ATTR_KEY_DEF( \
        LwSciBufImageAttrKey_SurfSampleType, \
        LW_SCI_BUF_IMGKEYTYPE_SURFSAMPLETYPE, \
        1, \
        LwSciBufKeyAccess_InOut, \
        LwSciBuf_MatchPolicy, \
        LwSciBufValidateSurfSampleType, \
        LwSciBufValidateSurfSampleType, \
        LwSciBufKeyImportQualifier_Optional, \
        LwSciBufKeyReconcileQualifier_Optional, \
        LwSciBufIpcRoute_AffinityNone, \
        LwSciBufIpcRoute_AffinityNone) \
    LW_SCI_BUF_ATTR_KEY_DEF( \
        LwSciBufImageAttrKey_SurfBPC, \
        LW_SCI_BUF_IMGKEYTYPE_SURFBPC, \
        1, \
        LwSciBufKeyAccess_InOut, \
        LwSciBuf_MatchPolicy, \
        LwSciBufValidateSurfBPC, \
        LwSciBufValidateSurfBPC, \
        LwSciBufKeyImportQualifier_Optional, \
        LwSciBufKeyReconcileQualifier_Optional, \
        LwSciBufIpcRoute_AffinityNone, \
        LwSciBufIpcRoute_AffinityNone) \
    LW_SCI_BUF_ATTR_KEY_DEF( \
        LwSciBufImageAttrKey_SurfComponentOrder, \
        LW_SCI_BUF_IMGKEYTYPE_SURFCOMPONENTORDER, \
        1, \
        LwSciBufKeyAccess_InOut, \
        LwSciBuf_MatchPolicy, \
        LwSciBufValidateSurfComponentOrder, \
        LwSciBufValidateSurfComponentOrder, \
        LwSciBufKeyImportQualifier_Optional, \
        LwSciBufKeyReconcileQualifier_Optional, \
        LwSciBufIpcRoute_AffinityNone, \
        LwSciBufIpcRoute_AffinityNone) \
    LW_SCI_BUF_ATTR_KEY_DEF( \
        LwSciBufImageAttrKey_SurfWidthBase, \
        LW_SCI_BUF_IMGKEYTYPE_SURFWIDTHBASE, \
        1, \
        LwSciBufKeyAccess_InOut, \
        LwSciBuf_MatchPolicy, \
        LwSciBufValidateImageWidth, \
        LwSciBufValidateImageWidth, \
        LwSciBufKeyImportQualifier_Optional, \
        LwSciBufKeyReconcileQualifier_Optional, \
        LwSciBufIpcRoute_AffinityNone, \
        LwSciBufIpcRoute_AffinityNone) \
    LW_SCI_BUF_ATTR_KEY_DEF( \
        LwSciBufImageAttrKey_SurfHeightBase, \
        LW_SCI_BUF_IMGKEYTYPE_SURFHEIGHTBASE, \
        1, \
        LwSciBufKeyAccess_InOut, \
        LwSciBuf_MatchPolicy, \
        LwSciBufValidateImageHeight, \
        LwSciBufValidateImageHeight, \
        LwSciBufKeyImportQualifier_Optional, \
        LwSciBufKeyReconcileQualifier_Optional, \
        LwSciBufIpcRoute_AffinityNone, \
        LwSciBufIpcRoute_AffinityNone)

/**
 * @brief LW_SCI_BUF_PUB_TENSOR_ATTR describes public tensor attributes
 */
#define LW_SCI_BUF_PUB_TENSOR_ATTR \
    LW_SCI_BUF_ATTR_KEY_DEF( \
        LwSciBufTensorAttrKey_DataType, \
        LW_SCI_BUF_TENSKEYTYPE_TYPE, \
        1, \
        LwSciBufKeyAccess_InOut, \
        LwSciBuf_MatchPolicy, \
        LwSciBufValidateTensorDataType, \
        LwSciBufValidateTensorDataType, \
        LwSciBufKeyImportQualifier_Mandatory, \
        LwSciBufKeyReconcileQualifier_Mandatory, \
        LwSciBufIpcRoute_AffinityNone, \
        LwSciBufIpcRoute_AffinityNone) \
    LW_SCI_BUF_ATTR_KEY_DEF( \
        LwSciBufTensorAttrKey_NumDims, \
        LW_SCI_BUF_TENSKEYTYPE_NUMDIMS, \
        1, \
        LwSciBufKeyAccess_InOut, \
        LwSciBuf_MatchPolicy, \
        LwSciBufValidateNumDims, \
        LwSciBufValidateNumDims, \
        LwSciBufKeyImportQualifier_Mandatory, \
        LwSciBufKeyReconcileQualifier_Mandatory, \
        LwSciBufIpcRoute_AffinityNone, \
        LwSciBufIpcRoute_AffinityNone) \
    LW_SCI_BUF_ATTR_KEY_DEF( \
        LwSciBufTensorAttrKey_SizePerDim, \
        LW_SCI_BUF_TENSKEYTYPE_SIZEPERDIM, \
        LW_SCI_BUF_TENSOR_MAX_DIMS, \
        LwSciBufKeyAccess_InOut, \
        LwSciBuf_MatchPolicy, \
        LwSciBufValidateTensorSizePerDim, \
        LwSciBufValidateTensorSizePerDim, \
        LwSciBufKeyImportQualifier_Mandatory, \
        LwSciBufKeyReconcileQualifier_Mandatory, \
        LwSciBufIpcRoute_AffinityNone, \
        LwSciBufIpcRoute_AffinityNone) \
    LW_SCI_BUF_ATTR_KEY_DEF( \
        LwSciBufTensorAttrKey_AlignmentPerDim, \
        LW_SCI_BUF_TENSKEYTYPE_ALIGNMENTPERDIM, \
        LW_SCI_BUF_TENSOR_MAX_DIMS, \
        LwSciBufKeyAccess_InOut, \
        LwSciBuf_MaxPolicy, \
        LwSciBufValidateAlignmentU32, \
        LwSciBufValidateAlignmentU32, \
        LwSciBufKeyImportQualifier_Conditional, \
        LwSciBufKeyReconcileQualifier_Conditional, \
        LwSciBufIpcRoute_AffinityNone, \
        LwSciBufIpcRoute_AffinityNone) \
    LW_SCI_BUF_ATTR_KEY_DEF( \
        LwSciBufTensorAttrKey_StridesPerDim, \
        LW_SCI_BUF_TENSKEYTYPE_STRIDESPERDIM, \
        LW_SCI_BUF_TENSOR_MAX_DIMS, \
        LwSciBufKeyAccess_Output, \
        LwSciBuf_MatchPolicy, \
        LwSciBufValidateTensorStridePerDim, \
        LwSciBufValidateTensorStridePerDim, \
        LwSciBufKeyImportQualifier_Mandatory, \
        LwSciBufKeyReconcileQualifier_Max, \
        LwSciBufIpcRoute_AffinityNone, \
        LwSciBufIpcRoute_AffinityNone) \
    LW_SCI_BUF_ATTR_KEY_DEF( \
        LwSciBufTensorAttrKey_PixelFormat, \
        LW_SCI_BUF_TENSKEYTYPE_PIXELFORMAT, \
        1, \
        LwSciBufKeyAccess_InOut, \
        LwSciBuf_MatchPolicy, \
        LwSciBufValidatePlaneColorFmt, \
        LwSciBufValidatePlaneColorFmt, \
        LwSciBufKeyImportQualifier_Conditional, \
        LwSciBufKeyReconcileQualifier_Conditional, \
        LwSciBufIpcRoute_AffinityNone, \
        LwSciBufIpcRoute_AffinityNone) \
    LW_SCI_BUF_ATTR_KEY_DEF( \
        LwSciBufTensorAttrKey_BaseAddrAlign, \
        LW_SCI_BUF_TENSKEYTYPE_BASEADDRALIGN, \
        1, \
        LwSciBufKeyAccess_InOut, \
        LwSciBuf_MaxPolicy, \
        LwSciBufValidateAlignmentU64, \
        LwSciBufValidateAlignmentU64, \
        LwSciBufKeyImportQualifier_Optional, \
        LwSciBufKeyReconcileQualifier_Optional, \
        LwSciBufIpcRoute_AffinityNone, \
        LwSciBufIpcRoute_AffinityNone) \
    LW_SCI_BUF_ATTR_KEY_DEF( \
        LwSciBufTensorAttrKey_Size, \
        LW_SCI_BUF_TENSKEYTYPE_SIZE, \
        1, \
        LwSciBufKeyAccess_Output, \
        LwSciBuf_IlwalidPolicy, \
        LwSciBufValidateBufferSize, \
        LwSciBufValidateBufferSize, \
        LwSciBufKeyImportQualifier_Optional, \
        LwSciBufKeyReconcileQualifier_Max, \
        LwSciBufIpcRoute_AffinityNone, \
        LwSciBufIpcRoute_AffinityNone)

/**
 * @brief LW_SCI_BUF_PUB_ARRAY_ATTR describes public array attributes
 */
#define LW_SCI_BUF_PUB_ARRAY_ATTR \
    LW_SCI_BUF_ATTR_KEY_DEF( \
        LwSciBufArrayAttrKey_DataType, \
        LW_SCI_BUF_ARRKEYTYPE_TYPE, \
        1, \
        LwSciBufKeyAccess_InOut, \
        LwSciBuf_MatchPolicy, \
        LwSciBufValidateArrayDataType, \
        LwSciBufValidateArrayDataType, \
        LwSciBufKeyImportQualifier_Mandatory, \
        LwSciBufKeyReconcileQualifier_Mandatory, \
        LwSciBufIpcRoute_AffinityNone, \
        LwSciBufIpcRoute_AffinityNone) \
    LW_SCI_BUF_ATTR_KEY_DEF( \
        LwSciBufArrayAttrKey_Stride, \
        LW_SCI_BUF_ARRKEYTYPE_STRIDE, \
        1, \
        LwSciBufKeyAccess_InOut, \
        LwSciBuf_MatchPolicy, \
        LwSciBufValidateArrayStride, \
        LwSciBufValidateArrayStride, \
        LwSciBufKeyImportQualifier_Mandatory, \
        LwSciBufKeyReconcileQualifier_Mandatory, \
        LwSciBufIpcRoute_AffinityNone, \
        LwSciBufIpcRoute_AffinityNone) \
    LW_SCI_BUF_ATTR_KEY_DEF( \
        LwSciBufArrayAttrKey_Capacity, \
        LW_SCI_BUF_ARRKEYTYPE_CAPACITY, \
        1, \
        LwSciBufKeyAccess_InOut,\
        LwSciBuf_MatchPolicy, \
        LwSciBufValidateArrayCapacity, \
        LwSciBufValidateArrayCapacity, \
        LwSciBufKeyImportQualifier_Mandatory, \
        LwSciBufKeyReconcileQualifier_Mandatory, \
        LwSciBufIpcRoute_AffinityNone, \
        LwSciBufIpcRoute_AffinityNone) \
    LW_SCI_BUF_ATTR_KEY_DEF( \
        LwSciBufArrayAttrKey_Size, \
        LW_SCI_BUF_ARRKEYTYPE_SIZE, \
        1, \
        LwSciBufKeyAccess_Output, \
        LwSciBuf_IlwalidPolicy, \
        LwSciBufValidateBufferSize, \
        LwSciBufValidateBufferSize, \
        LwSciBufKeyImportQualifier_Mandatory, \
        LwSciBufKeyReconcileQualifier_Max, \
        LwSciBufIpcRoute_AffinityNone, \
        LwSciBufIpcRoute_AffinityNone) \
    LW_SCI_BUF_ATTR_KEY_DEF( \
        LwSciBufArrayAttrKey_Alignment, \
        LW_SCI_BUF_ARRKEYTYPE_ALIGNMENT, \
        1, \
        LwSciBufKeyAccess_Output, \
        LwSciBuf_IlwalidPolicy, \
        LwSciBufValidateAlignmentU64, \
        LwSciBufValidateAlignmentU64, \
        LwSciBufKeyImportQualifier_Mandatory, \
        LwSciBufKeyReconcileQualifier_Max, \
        LwSciBufIpcRoute_AffinityNone, \
        LwSciBufIpcRoute_AffinityNone)

/**
 * @brief LW_SCI_BUF_PUB_PYRAMID_ATTR describes public pyramid attributes
 */
#define LW_SCI_BUF_PUB_PYRAMID_ATTR \
    LW_SCI_BUF_ATTR_KEY_DEF( \
        LwSciBufPyramidAttrKey_NumLevels, \
        LW_SCI_BUF_PYRKEYTYPE_NUMLEVELS, \
        1, \
        LwSciBufKeyAccess_InOut, \
        LwSciBuf_MatchPolicy, \
        LwSciBufValidatePyramidLevels, \
        LwSciBufValidatePyramidLevels, \
        LwSciBufKeyImportQualifier_Mandatory, \
        LwSciBufKeyReconcileQualifier_Mandatory, \
        LwSciBufIpcRoute_AffinityNone, \
        LwSciBufIpcRoute_AffinityNone) \
    LW_SCI_BUF_ATTR_KEY_DEF( \
        LwSciBufPyramidAttrKey_Scale, \
        LW_SCI_BUF_PYRKEYTYPE_SCALE, \
        1, \
        LwSciBufKeyAccess_InOut, \
        LwSciBuf_MatchPolicy, \
        LwSciBufValidatePyramidScale, \
        LwSciBufValidatePyramidScale, \
        LwSciBufKeyImportQualifier_Mandatory, \
        LwSciBufKeyReconcileQualifier_Mandatory, \
        LwSciBufIpcRoute_AffinityNone, \
        LwSciBufIpcRoute_AffinityNone) \
    LW_SCI_BUF_ATTR_KEY_DEF( \
        LwSciBufPyramidAttrKey_LevelOffset, \
        LW_SCI_BUF_PYRKEYTYPE_LEVELOFFSET, \
        LW_SCI_BUF_PYRAMID_MAX_LEVELS, \
        LwSciBufKeyAccess_Output, \
        LwSciBuf_IlwalidPolicy, \
        NULL, \
        NULL, \
        LwSciBufKeyImportQualifier_Mandatory, \
        LwSciBufKeyReconcileQualifier_Max, \
        LwSciBufIpcRoute_AffinityNone, \
        LwSciBufIpcRoute_AffinityNone) \
    LW_SCI_BUF_ATTR_KEY_DEF( \
        LwSciBufPyramidAttrKey_LevelSize, \
        LW_SCI_BUF_PYRKEYTYPE_LEVELSIZE, \
        LW_SCI_BUF_PYRAMID_MAX_LEVELS, \
        LwSciBufKeyAccess_Output, \
        LwSciBuf_IlwalidPolicy, \
        LwSciBufValidateBufferSize, \
        LwSciBufValidateBufferSize, \
        LwSciBufKeyImportQualifier_Mandatory, \
        LwSciBufKeyReconcileQualifier_Max, \
        LwSciBufIpcRoute_AffinityNone, \
        LwSciBufIpcRoute_AffinityNone) \
    LW_SCI_BUF_ATTR_KEY_DEF( \
        LwSciBufPyramidAttrKey_Alignment, \
        LW_SCI_BUF_PYRKEYTYPE_ALIGNMENT, \
        1, \
        LwSciBufKeyAccess_Output, \
        LwSciBuf_IlwalidPolicy, \
        LwSciBufValidateAlignmentU64, \
        LwSciBufValidateAlignmentU64, \
        LwSciBufKeyImportQualifier_Mandatory, \
        LwSciBufKeyReconcileQualifier_Max, \
        LwSciBufIpcRoute_AffinityNone, \
        LwSciBufIpcRoute_AffinityNone)

/**
 * @brief LW_SCI_BUF_INT_GENERAL_ATTR describes internal general attributes
 */
#define LW_SCI_BUF_INT_GENERAL_ATTR \
    LW_SCI_BUF_ATTR_KEY_DEF( \
        LwSciBufInternalGeneralAttrKey_EngineArray, \
        LwSciBufHwEngine, \
        LW_SCI_BUF_HW_ENGINE_MAX_NUMBER, \
        LwSciBufKeyAccess_InOut, \
        LwSciBuf_ArrayUnionPolicy, \
        LwSciBufValidateHwEngine, \
        LwSciBufValidateHwEngine, \
        LwSciBufKeyImportQualifier_Optional, \
        LwSciBufKeyReconcileQualifier_Optional, \
        LwSciBufIpcRoute_AffinityNone, \
        LwSciBufIpcRoute_SocAffinity) \
    LW_SCI_BUF_ATTR_KEY_DEF( \
        LwSciBufInternalGeneralAttrKey_MemDomainArray, \
        LwSciBufMemDomain, \
        LwSciBufMemDomain_UpperBound, \
        LwSciBufKeyAccess_InOut, \
        LwSciBuf_ArrayIntersectionPolicy, \
        LwSciBufValidateMemDomain, \
        LwSciBufValidateMemDomain, \
        LwSciBufKeyImportQualifier_Mandatory, \
        LwSciBufKeyReconcileQualifier_Optional, \
        LwSciBufIpcRoute_SocAffinity, \
        LwSciBufIpcRoute_SocAffinity)

/**
 * @brief LW_SCI_BUF_INT_IMG_ATTR describes internal image attributes
 */
#define LW_SCI_BUF_INT_IMAGE_ATTR \
    LW_SCI_BUF_ATTR_KEY_DEF( \
        LwSciBufInternalImageAttrKey_PlaneGobSize, \
        uint32_t, \
        LW_SCI_BUF_IMAGE_MAX_PLANES, \
        LwSciBufKeyAccess_Output, \
        LwSciBuf_IlwalidPolicy, \
        NULL, \
        NULL, \
        LwSciBufKeyImportQualifier_Mandatory, \
        LwSciBufKeyReconcileQualifier_Max, \
        LwSciBufIpcRoute_AffinityNone, \
        LwSciBufIpcRoute_AffinityNone) \
    LW_SCI_BUF_ATTR_KEY_DEF( \
        LwSciBufInternalImageAttrKey_PlaneLog2GobsPerBlockX, \
        uint32_t, \
        LW_SCI_BUF_IMAGE_MAX_PLANES, \
        LwSciBufKeyAccess_Output, \
        LwSciBuf_IlwalidPolicy, \
        NULL, \
        NULL, \
        LwSciBufKeyImportQualifier_Mandatory, \
        LwSciBufKeyReconcileQualifier_Max, \
        LwSciBufIpcRoute_AffinityNone, \
        LwSciBufIpcRoute_AffinityNone) \
    LW_SCI_BUF_ATTR_KEY_DEF( \
        LwSciBufInternalImageAttrKey_PlaneLog2GobsPerBlockY, \
        uint32_t, \
        LW_SCI_BUF_IMAGE_MAX_PLANES, \
        LwSciBufKeyAccess_Output, \
        LwSciBuf_IlwalidPolicy, \
        NULL, \
        NULL, \
        LwSciBufKeyImportQualifier_Mandatory, \
        LwSciBufKeyReconcileQualifier_Max, \
        LwSciBufIpcRoute_AffinityNone, \
        LwSciBufIpcRoute_AffinityNone) \
    LW_SCI_BUF_ATTR_KEY_DEF( \
        LwSciBufInternalImageAttrKey_PlaneLog2GobsPerBlockZ, \
        uint32_t, \
        LW_SCI_BUF_IMAGE_MAX_PLANES, \
        LwSciBufKeyAccess_Output, \
        LwSciBuf_IlwalidPolicy, \
        NULL, \
        NULL, \
        LwSciBufKeyImportQualifier_Mandatory, \
        LwSciBufKeyReconcileQualifier_Max, \
        LwSciBufIpcRoute_AffinityNone, \
        LwSciBufIpcRoute_AffinityNone)

/**
 * @brief LW_SCI_BUF_APP_PRIVATE describes app private keys
 */
#define LW_SCI_BUF_INT_UMD_ATTR \
    LW_SCI_BUF_ATTR_KEY_DEF( \
        LwSciBufInternalAttrKey_LwMediaPrivateFirst, \
        LWListRec, \
        1, \
        LwSciBufKeyAccess_InOut, \
        LwSciBuf_ListUnionPolicy, \
        NULL, \
        NULL, \
        LwSciBufKeyImportQualifier_Optional, \
        LwSciBufKeyReconcileQualifier_Optional, \
        LwSciBufIpcRoute_AffinityNone, \
        LwSciBufIpcRoute_AffinityNone)

/**
 * @brief describes private keys used by lwscibuf
 */
#define LW_SCI_BUF_PRIVATE_ATTR \
    LW_SCI_BUF_ATTR_KEY_DEF( \
        LwSciBufPrivateAttrKey_Size, \
        uint64_t, \
        1, \
        LwSciBufKeyAccess_Output, \
        LwSciBuf_IlwalidPolicy, \
        LwSciBufValidateBufferSize, \
        LwSciBufValidateBufferSize, \
        LwSciBufKeyImportQualifier_Mandatory, \
        LwSciBufKeyReconcileQualifier_Optional, \
        LwSciBufIpcRoute_AffinityNone, \
        LwSciBufIpcRoute_AffinityNone) \
    LW_SCI_BUF_ATTR_KEY_DEF( \
        LwSciBufPrivateAttrKey_Alignment, \
        uint64_t, \
        1, \
        LwSciBufKeyAccess_Output, \
        LwSciBuf_IlwalidPolicy, \
        LwSciBufValidateAlignmentU64, \
        LwSciBufValidateAlignmentU64, \
        LwSciBufKeyImportQualifier_Mandatory, \
        LwSciBufKeyReconcileQualifier_Optional, \
        LwSciBufIpcRoute_AffinityNone, \
        LwSciBufIpcRoute_AffinityNone) \
    LW_SCI_BUF_ATTR_KEY_DEF( \
        LwSciBufPrivateAttrKey_HeapType, \
        LwSciBufHeapType, \
        1, \
        LwSciBufKeyAccess_Output, \
        LwSciBuf_IlwalidPolicy, \
        LwSciBufValidateHeapType, \
        LwSciBufValidateHeapType, \
        LwSciBufKeyImportQualifier_Mandatory, \
        LwSciBufKeyReconcileQualifier_Optional, \
        LwSciBufIpcRoute_SocAffinity, \
        LwSciBufIpcRoute_SocAffinity) \
    LW_SCI_BUF_ATTR_KEY_DEF( \
        LwSciBufPrivateAttrKey_MemDomain, \
        LwSciBufMemDomain, \
        1, \
        LwSciBufKeyAccess_Output, \
        LwSciBuf_IlwalidPolicy, \
        LwSciBufValidateMemDomain, \
        LwSciBufValidateMemDomain, \
        LwSciBufKeyImportQualifier_Mandatory, \
        LwSciBufKeyReconcileQualifier_Optional, \
        LwSciBufIpcRoute_AffinityNone, \
        LwSciBufIpcRoute_AffinityNone) \
    LW_SCI_BUF_ATTR_KEY_DEF( \
        LwSciBufPrivateAttrKey_OriginTopoId, \
        LwSciBufAttrValTopoId, \
        1, \
        LwSciBufKeyAccess_Output, \
        LwSciBuf_IlwalidPolicy, \
        NULL, \
        NULL, \
        LwSciBufKeyImportQualifier_Optional, \
        LwSciBufKeyReconcileQualifier_Optional, \
        LwSciBufIpcRoute_AffinityNone, \
        LwSciBufIpcRoute_AffinityNone) \
    LW_SCI_BUF_ATTR_KEY_DEF( \
        LwSciBufPrivateAttrKey_OtherTopoIdArray, \
        LwSciBufAttrValTopoId, \
        1, \
        LwSciBufKeyAccess_Output, \
        LwSciBuf_IlwalidPolicy, \
        NULL, \
        NULL, \
        LwSciBufKeyImportQualifier_Optional, \
        LwSciBufKeyReconcileQualifier_Optional, \
        LwSciBufIpcRoute_AffinityNone, \
        LwSciBufIpcRoute_AffinityNone) \
    LW_SCI_BUF_ATTR_KEY_DEF( \
        LwSciBufPrivateAttrKey_SciIpcRoute, \
        LwSciBufIpcRoute*, \
        1, \
        LwSciBufKeyAccess_Output, \
        LwSciBuf_IlwalidPolicy, \
        NULL, \
        NULL, \
        LwSciBufKeyImportQualifier_Mandatory, \
        LwSciBufKeyReconcileQualifier_Optional, \
        LwSciBufIpcRoute_AffinityNone, \
        LwSciBufIpcRoute_AffinityNone) \
    LW_SCI_BUF_ATTR_KEY_DEF( \
        LwSciBufPrivateAttrKey_IPCTable, \
        LwSciBufIpcTable*, \
        1, \
        LwSciBufKeyAccess_Output, \
        LwSciBuf_IlwalidPolicy, \
        NULL, \
        NULL, \
        LwSciBufKeyImportQualifier_Mandatory, \
        LwSciBufKeyReconcileQualifier_Optional, \
        LwSciBufIpcRoute_AffinityNone, \
        LwSciBufIpcRoute_AffinityNone) \
    LW_SCI_BUF_ATTR_KEY_DEF( \
        LwSciBufPrivateAttrKey_ConflictKey, \
        uint32_t, \
        1, \
        LwSciBufKeyAccess_Output, \
        LwSciBuf_IlwalidPolicy, \
        NULL, \
        NULL, \
        LwSciBufKeyImportQualifier_Optional, \
        LwSciBufKeyReconcileQualifier_Max, \
        LwSciBufIpcRoute_AffinityNone, \
        LwSciBufIpcRoute_AffinityNone)
/**
 * @}
 */



/**
 * @defgroup LwSciBufAttrObjPriv
 * Structures defined in this group are static structures for all types of
 * attribute keys which are generated by macro expansion of
 * LW_SCI_BUF_ATTR_KEY_DEF.
 *
 * @{
 */
LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 20_10), "LwSciBuf-ADV-MISRAC2012-021")
LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 20_5), "LwSciBuf-ADV-MISRAC2012-020")
#undef LW_SCI_BUF_ATTR_KEY_DEF
#define LW_SCI_BUF_ATTR_KEY_DEF(_key, _datatype, _dem, _perm, _policy,\
    _externalValidateFn, _internalValidateFn, _importQualifier,\
    _reconcileQualifier, _localPeerIpcAffinity, _remotePeerIpcAffinity) \
                                _datatype _key[_dem]; \
                                uint64_t _key##_Size; \
                                LwSciBufAttrStatus _key##_Status;
LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 20_5))
LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 20_10))

/**
 * @brief Static storage structure for attributes
 */
typedef struct {
    LW_SCI_BUF_PUB_GENERAL_ATTR
    LW_SCI_BUF_INT_GENERAL_ATTR
} LwSciBufGeneralAttrObjPriv;

typedef struct {
    LW_SCI_BUF_PUB_RAWBUFFER_ATTR
} LwSciBufRawBufferAttrObjPriv;

typedef struct {
    LW_SCI_BUF_PUB_IMAGE_ATTR
    LW_SCI_BUF_INT_IMAGE_ATTR
} LwSciBufImageAttrObjPriv;

typedef struct {
    LW_SCI_BUF_PUB_TENSOR_ATTR
} LwSciBufTensorAttrObjPriv;

typedef struct {
    LW_SCI_BUF_PUB_ARRAY_ATTR
} LwSciBufArrayAttrObjPriv;

typedef struct {
    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Directive, 4_6), "LwSciBuf-ADV-MISRAC2012-006")
    LW_SCI_BUF_PUB_PYRAMID_ATTR
} LwSciBufPyramidAttrObjPriv;

typedef struct {
    LW_SCI_BUF_INT_UMD_ATTR
} LwSciBufUmdAttrObjPriv;

typedef struct {
    LW_SCI_BUF_PRIVATE_ATTR
} LwSciBufPrivateAttrObjPriv;


/**
 * Callback function typedef for freeing any attribute.
 */
typedef void (*LwSciBufAttrFreeCb)(
    void* attrValPtr);

/**
 * Callback function typedef for cloning any attribute.
 */
typedef LwSciError (*LwSciBufAttrCloneCb)(
    const void* srcValPtr,
    void* dstValPtr);

typedef struct {
    /** Claeanup Callback function for freeing this key */
    LwSciBufAttrFreeCb freeCallback;
    /** Clone Callback function for cloning the value of this key */
    LwSciBufAttrCloneCb cloneCallback;
} LwSciBufAttrKeyCallbackDesc;

/**
 * Qualifiers denoting whether an Attribute Key requires additional checking
 * when an LwSciBufAttrList is imported.
 *
 * @implements{19731960}
 */
typedef enum {
    /** The key must always be set */
    LwSciBufKeyImportQualifier_Mandatory,
    /** The key can be skipped */
    LwSciBufKeyImportQualifier_Optional,
    /** The key may be mandatory in some situations, depending on some other condition */
    LwSciBufKeyImportQualifier_Conditional,
    /** Sentinel value */
    LwSciBufKeyImportQualifier_Max,
} LwSciBufKeyImportQualifier;

/**
 * Qualifiers denoting whether an Attribute Key requires additional checking
 * when an LwSciBufAttrList is reconciled.
 *
 * @implements{19731963}
 */
typedef enum {
    /** The key must always be set */
    LwSciBufKeyReconcileQualifier_Mandatory,
    /** The key can be skipped */
    LwSciBufKeyReconcileQualifier_Optional,
    /** The key may be mandatory in some situations, depending on some other condition */
    LwSciBufKeyReconcileQualifier_Conditional,
    /** Sentinel value */
    LwSciBufKeyReconcileQualifier_Max,
} LwSciBufKeyReconcileQualifier;

/**
 * @brief Attribute List descriptor structure, this structure is bookkeeping
 * for all necessary information to Set/Get Apis
 */
typedef struct {
    /** Key name for debug */
    const char *name;
    /** Key index for debug */
    uint32_t dataIndex;
    /** Size of each element in key-value */
    size_t dataSize;
    /** Total number of elements in key-value */
    uint32_t dataMaxInstance;
    /** offset of this key in bytes from starting address of
     * LwSciBufAttrObjPriv structure
     */
    uintptr_t dataOffset;
    /** offset of this size of key in bytes from starting address of
     * LwSciBufAttrObjPriv structure
     */
    uintptr_t sizeOffset;
    /** offset of this status of key in bytes from starting address of
     * LwSciBufAttrObjPriv structure
     */
    uintptr_t statusOffset;
    /** enum to indicate the accessibility of this key */
    LwSciBufKeyAccess keyAccess;
    /** Policy that will be applied to this key */
    LwSciBuf_ReconcilePolicy recpolicy;
    /** Validation function for the key where set operation is performed by the
     * user of LwSciBuf
     */
    LwSciBufValidateAttrFn externalValidateFn;
    /** Validation function for the key where set operation is performed by the
     * LwSciBuf
     */
    LwSciBufValidateAttrFn internalValidateFn;
    /** Denotes the qualified type on the Reconciled Attribute List being
     * imported.
     *
     * This is to handle the case where certain keys are mandatory in certain
     * kinds of reconciliation, but optional in other cases. */
    LwSciBufKeyImportQualifier importQualifier;

    /** The qualified type on the Attribute List being reconciled.
     *
     * This is to handle the cases where keys may be mandatory during
     * reconciliation.
     */
    LwSciBufKeyReconcileQualifier reconcileQualifier;

    /**
     * IPC route affinity of the attribute for local peer.
     */
    LwSciBufIpcRouteAffinity localPeerIpcAffinity;

    /**
     * IPC route affinity of the attribute for remote peer.
     */
    LwSciBufIpcRouteAffinity remotePeerIpcAffinity;
} LwSciBufAttrKeyDescPriv;

LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 20_5), "LwSciBuf-ADV-MISRAC2012-020")
#undef LW_SCI_BUF_ATTR_KEY_DEF

#endif /* INCLUDED_LWSCIBUF_ATTR_DESC_H */
