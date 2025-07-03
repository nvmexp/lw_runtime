/*
 *  Copyright 2018-2019 LWPU Corporation.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 *
 */

#ifndef FMINTERNAL_API_MSG_H
#define FMINTERNAL_API_MSG_H

typedef struct
{
    unsigned int version;
    char gpuUuid[FM_UUID_BUFFER_SIZE];
} fm_msg_prepare_gpu_for_reset_t;

#define fm_msg_prepare_gpu_for_reset_version1 MAKE_FM_PARAM_VERSION(fm_msg_prepare_gpu_for_reset_t,1)
#define fm_msg_prepare_gpu_for_reset_version fm_msg_prepare_gpu_for_reset_version1

typedef struct
{
    unsigned int version;
    char gpuUuid[FM_UUID_BUFFER_SIZE];
} fm_msg_shutdown_gpu_lwlink_t;

#define fm_msg_shutdown_gpu_lwlink_version1 MAKE_FM_PARAM_VERSION(fm_msg_shutdown_gpu_lwlink_t,1)
#define fm_msg_shutdown_gpu_lwlink_version fm_msg_shutdown_gpu_lwlink_version1

typedef struct
{
    unsigned int version;
    char gpuUuid[FM_UUID_BUFFER_SIZE];
} fm_msg_reset_gpu_lwlink_t;

#define fm_msg_reset_gpu_lwlink_version1 MAKE_FM_PARAM_VERSION(fm_msg_reset_gpu_lwlink_t,1)
#define fm_msg_reset_gpu_lwlink_version fm_msg_reset_gpu_lwlink_version1


typedef struct
{
    unsigned int version;
    char gpuUuid[FM_UUID_BUFFER_SIZE];
} fm_msg_complete_gpu_reset_t;

#define fm_msg_complete_gpu_reset_version1 MAKE_FM_PARAM_VERSION(fm_msg_complete_gpu_reset_t,1)
#define fm_msg_complete_gpu_reset_version fm_msg_complete_gpu_reset_version1

#endif /* FMINTERNAL_API_MSG_H */

