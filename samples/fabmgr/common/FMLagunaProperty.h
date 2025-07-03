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

#define LAGUNA_NUM_LWLINKS_PER_SWITCH            64

#define LAGUNA_INGRESS_REQUEST_TABLE_SIZE        0
#define LAGUNA_INGRESS_RESPONSETABLE_SIZE        0

#define LAGUNA_REMAP_TABLE_SIZE                  4096
#define LAGUNA_RLAN_TABLE_SIZE                   2048
#define LAGUNA_RID_TABLE_SIZE                    2048
#define LAGUNA_GANGED_LINK_TABLE_SIZE            256

#define LAGUNA_REMAP_EXT_RANGE_A_TABLE_SIZE      64
#define LAGUNA_REMAP_EXT_RANGE_B_TABLE_SIZE      64
#define LAGUNA_MULTICAST_REMAP_TABLE_SIZE        128

#define LAGUNA_NUM_INGRESS_REQUEST_ENTRIES_PER_GPU  0
#define LAGUNA_NUM_INGRESS_RESPONSE_ENTRIES_PER_GPU 0

#define LAGUNA_FIRST_FLA_REMAP_SLOT                 64 // FLA starts at index 64 in normal range remap table, due to bug 3142507
#define LAGUNA_NUM_FLA_REMAP_ENTRIES_PER_GPU        2  // One for GPU memory, one for EGM
#define LAGUNA_FLA_ADDRESS_BASE                     0

#define LAGUNA_FLA_TO_TARGET_ID(fla)                ((fla/HOPPER_FABRIC_ADDRESS_RANGE - LAGUNA_FIRST_FLA_REMAP_SLOT) / LAGUNA_NUM_FLA_REMAP_ENTRIES_PER_GPU)
#define LAGUNA_TARGET_ID_TO_FLA_INDEX(targetId)     (targetId * LAGUNA_NUM_FLA_REMAP_ENTRIES_PER_GPU + LAGUNA_FIRST_FLA_REMAP_SLOT)
#define LAGUNA_TARGET_ID_TO_FLA(targetId)           ((LAGUNA_TARGET_ID_TO_FLA_INDEX(targetId) * HOPPER_FABRIC_ADDRESS_RANGE) | LAGUNA_FLA_ADDRESS_BASE)
#define LAGUNA_TARGET_ID_TO_FLA_EGM(targetId)       (((LAGUNA_TARGET_ID_TO_FLA_INDEX(targetId) + 1) * HOPPER_FABRIC_ADDRESS_RANGE) | LAGUNA_FLA_ADDRESS_BASE)
#define LAGUNA_TARGET_ID_TO_FLA_EGM_INDEX(targetId) (LAGUNA_TARGET_ID_TO_FLA_INDEX(targetId) + 1)

#define LAGUNA_FIRST_GPA_REMAP_SLOT                 0  // GPA starts at index 0 in extended range B remap table
#define LAGUNA_NUM_GPA_REMAP_ENTRIES_PER_GPU        2  // One for GPU memory, one for EGM
#define LAGUNA_GPA_ADDRESS_BASE                     (0b1000001000000LL << 39) // Extended Range B, Index = Addr[44:39], Addr[49:45] = 5’b00001

#define LAGUNA_GPA_TO_TARGET_ID(gpa)                ((gpa/HOPPER_FABRIC_ADDRESS_RANGE - LAGUNA_FIRST_GPA_REMAP_SLOT) / LAGUNA_NUM_GPA_REMAP_ENTRIES_PER_GPU)
#define LAGUNA_TARGET_ID_TO_GPA_INDEX(targetId)     (targetId * LAGUNA_NUM_GPA_REMAP_ENTRIES_PER_GPU + LAGUNA_FIRST_GPA_REMAP_SLOT)
#define LAGUNA_TARGET_ID_TO_GPA(targetId)           ((LAGUNA_TARGET_ID_TO_GPA_INDEX(targetId) * HOPPER_FABRIC_ADDRESS_RANGE) | LAGUNA_GPA_ADDRESS_BASE)
#define LAGUNA_TARGET_ID_TO_GPA_EGM(targetId)       (((LAGUNA_TARGET_ID_TO_GPA_INDEX(targetId) + 1) * HOPPER_FABRIC_ADDRESS_RANGE) | LAGUNA_GPA_ADDRESS_BASE)

#define LAGUNA_FIRST_SPA_REMAP_SLOT                 0 // GPA starts at index 0 in normal range remap table, due to bug 3142507
#define LAGUNA_NUM_SPA_REMAP_ENTRIES_PER_GPU        1
#define LAGUNA_SPA_ADDRESS_BASE                     0 // Normal Range

#define LAGUNA_SPA_TO_SPA_REMAP_INDEX(spa)          (spa/HOPPER_FABRIC_ADDRESS_RANGE + LAGUNA_FIRST_SPA_REMAP_SLOT)
#define LAGUNA_SPA_TO_TARGET_ID(spa)                ((spa/HOPPER_FABRIC_ADDRESS_RANGE - LAGUNA_FIRST_SPA_REMAP_SLOT) / LAGUNA_NUM_SPA_REMAP_ENTRIES_PER_GPU)
#define LAGUNA_TARGET_ID_TO_SPA_INDEX(targetId)     (targetId * LAGUNA_NUM_SPA_REMAP_ENTRIES_PER_GPU + LAGUNA_FIRST_SPA_REMAP_SLOT)

#define NUM_LWLINKS_PER_HOPPER          18
#define HOPPER_FABRIC_ADDRESS_RANGE     (1LL << 39)  // 512G
#define HOPPER_EGM_ADDRESS_RANGE        (HOPPER_FABRIC_ADDRESS_RANGE * 2)

// Addr[51:50] = 2’b11: Multicast
#define LAGUNA_MULTICAST_FABRIC_ADDR_VALUE  0x3
#define LAGUNA_MULTICAST_FABRIC_ADDR_SHIFT  50
// groupId = Addr[45:39]
#define LAGUNA_MULTICAST_GROUP_ID_SHIFT     39
