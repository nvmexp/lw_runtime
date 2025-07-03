/*
 *  Copyright 2018-2020 LWPU Corporation.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 *
 */
#pragma once

#define WILLOW_NUM_LWLINKS_PER_SWITCH            18

#define WILLOW_INGRESS_REQUEST_TABLE_SIZE        8192
#define WILLOW_INGRESS_RESPONSETABLE_SIZE        8192

#define WILLOW_REMAP_TABLE_SIZE                  0
#define WILLOW_RLAN_TABLE_SIZE                   0
#define WILLOW_RID_TABLE_SIZE                    0
#define WILLOW_GANGED_LINK_TABLE_SIZE            256

#define WILLOW_REMAP_EXT_RANGE_A_TABLE_SIZE      0
#define WILLOW_REMAP_EXT_RANGE_B_TABLE_SIZE      0
#define WILLOW_MULTICAST_REMAP_TABLE_SIZE        0

#define WILLOW_NUM_INGRESS_REQUEST_ENTRIES_PER_GPU  4
#define WILLOW_NUM_INGRESS_RESPONSE_ENTRIES_PER_GPU 6

#define WILLOW_FIRST_FLA_REMAP_SLOT                 0
#define WILLOW_NUM_FLA_REMAP_ENTRIES_PER_GPU        0
#define WILLOW_FLA_TO_TARGET_ID(fla)                0
#define WILLOW_TARGET_ID_TO_FLA_INDEX(targetId)     0
#define WILLOW_TARGET_ID_TO_FLA(targetId)           0

#define WILLOW_FIRST_GPA_REMAP_SLOT                 0
#define WILLOW_NUM_GPA_REMAP_ENTRIES_PER_GPU        0
#define WILLOW_GPA_TO_TARGET_ID(gpa)                (gpa/VOLTA_FABRIC_ADDRESS_RANGE)
#define WILLOW_TARGET_ID_TO_GPA_INDEX(targetId)     0
#define WILLOW_TARGET_ID_TO_GPA(targetId)           (WILLOW_TARGET_ID_TO_GPA_INDEX(targetId) * VOLTA_FABRIC_ADDRESS_RANGE)

#define WILLOW_FIRST_SPA_REMAP_SLOT                 0
#define WILLOW_NUM_SPA_REMAP_ENTRIES_PER_GPU        0

#define NUM_LWLINKS_PER_VOLTA         6
#define VOLTA_FABRIC_ADDRESS_RANGE   (1LL << 36) //64G
#define VOLTA_EGM_ADDRESS_RANGE       0
