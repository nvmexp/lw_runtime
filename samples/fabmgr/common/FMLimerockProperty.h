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

#define LIMEROCK_NUM_LWLINKS_PER_SWITCH            36

#define LIMEROCK_INGRESS_REQUEST_TABLE_SIZE        0
#define LIMEROCK_INGRESS_RESPONSETABLE_SIZE        0

#define LIMEROCK_REMAP_TABLE_SIZE                  2048
#define LIMEROCK_RLAN_TABLE_SIZE                   512
#define LIMEROCK_RID_TABLE_SIZE                    512
#define LIMEROCK_GANGED_LINK_TABLE_SIZE            256

#define LIMEROCK_REMAP_EXT_RANGE_A_TABLE_SIZE      0
#define LIMEROCK_REMAP_EXT_RANGE_B_TABLE_SIZE      0
#define LIMEROCK_MULTICAST_REMAP_TABLE_SIZE        0

#define LIMEROCK_NUM_INGRESS_REQUEST_ENTRIES_PER_GPU  0
#define LIMEROCK_NUM_INGRESS_RESPONSE_ENTRIES_PER_GPU 0
#define LIMEROCK_ADDRESS_RANGE_PER_REMAP_SLOT         (1LL << 36)  // 64G

#define LIMEROCK_FIRST_FLA_REMAP_SLOT                 32
#define LIMEROCK_NUM_FLA_REMAP_ENTRIES_PER_GPU        2 // each entry cover 64G memory, 2 entries are needed to support 128G
#define LIMEROCK_FLA_TO_TARGET_ID(fla)                ((fla/LIMEROCK_ADDRESS_RANGE_PER_REMAP_SLOT - LIMEROCK_FIRST_FLA_REMAP_SLOT) / LIMEROCK_NUM_FLA_REMAP_ENTRIES_PER_GPU)
#define LIMEROCK_TARGET_ID_TO_FLA_INDEX(targetId)     (targetId * LIMEROCK_NUM_FLA_REMAP_ENTRIES_PER_GPU + LIMEROCK_FIRST_FLA_REMAP_SLOT)
#define LIMEROCK_TARGET_ID_TO_FLA(targetId)           (LIMEROCK_TARGET_ID_TO_FLA_INDEX(targetId) * LIMEROCK_ADDRESS_RANGE_PER_REMAP_SLOT)

#define LIMEROCK_FIRST_GPA_REMAP_SLOT                 0
#define LIMEROCK_NUM_GPA_REMAP_ENTRIES_PER_GPU        2 // each entry cover 64G memory, 2 entries are needed to support 128G
#define LIMEROCK_GPA_TO_TARGET_ID(gpa)                ((gpa/LIMEROCK_ADDRESS_RANGE_PER_REMAP_SLOT - LIMEROCK_FIRST_GPA_REMAP_SLOT) / LIMEROCK_NUM_GPA_REMAP_ENTRIES_PER_GPU)
#define LIMEROCK_TARGET_ID_TO_GPA_INDEX(targetId)     (targetId * LIMEROCK_NUM_GPA_REMAP_ENTRIES_PER_GPU + LIMEROCK_FIRST_GPA_REMAP_SLOT)
#define LIMEROCK_TARGET_ID_TO_GPA(targetId)           (LIMEROCK_TARGET_ID_TO_GPA_INDEX(targetId) * AMPERE_FABRIC_ADDRESS_RANGE)

#define LIMEROCK_FIRST_SPA_REMAP_SLOT                 0
#define LIMEROCK_NUM_SPA_REMAP_ENTRIES_PER_GPU        0

#define NUM_LWLINKS_PER_AMPERE          12
#define AMPERE_FABRIC_ADDRESS_RANGE     (1LL << 37)  // 128G
#define AMPERE_EGM_ADDRESS_RANGE        0
