/*
 * Copyright (c) 2019-2020, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 */

#ifndef INCLUDED_LWSCIIPC_OS_QNX_H
#define INCLUDED_LWSCIIPC_OS_QNX_H

#ifdef __QNX__
#include <stdio.h>
#include <sys/slog.h>
#include <sys/procmgr.h>

#include <lwos_s3_tegra_log.h>
#if defined(LW_IS_TRACER_ENABLED)
#include <tracer.h>
#endif

/* define LWSCIIPC_DEBUG to enable debug log on LwOsDebugPrint */
/* define CONSOLE_DEBUG to enable log on console instead of LwOsDebugPrint */
/* or you can define these debug flags on each source code */

struct LwSciIpcConfigBlob {
    uint32_t blobSize;
    uint32_t vmid;  /* current vmid */
    uint32_t entryCount;
    uint32_t abilityId;
    /* The LwSciIpcConfigEntry follow the above fields
     * The actual length of this array is entryCount.
     * accessing the LwSciIpcConfigEntry must be done via inline function
     * LwSciIpcConfigEntry_addr()
     */
};

#ifdef CONFIGBLOB_V2
struct LwSciIpcConfigBlob2 {
    uint32_t blobSize;
    uint32_t socid;  /* current socid */
    uint32_t vmid;  /* current vmid */
    uint32_t entryCount;
    uint32_t abilityId;
    uint32_t reserved[10];
    uint32_t checksum;
    /* The LwSciIpcConfigEntry follow the above fields
     * The actual length of this array is entryCount.
     * accessing the LwSciIpcConfigEntry must be done via inline function
     * LwSciIpcConfigEntry_addr()
     */
};
#endif  /* CONFIGBLOB_V2 */

/* config blob shared memory name */
#define CONFIG_SHM "/LwSciIpcConfig"
/* intra-VM channel data shared memory name */
#if !defined(CHANNEL_SHM)
#define CHANNEL_SHM "/dev/shmem/LwSciIpcChannel"
#endif

/* lwsciipc custom ability can be created or looked up w/ this ID */
#ifndef LWSCIIPC_ABILITY_ID
#define LWSCIIPC_ABILITY_ID "LwSciIpcEndpoint"
#endif /* LWSCIIPC_ABILITY_ID */
/* pulse event */
#define SIGEV_PULSE_PRIO_INTR 0x15
#define LW_SCI_IPC_PULSE_DATA 0x4e4f5449

#define PRIV_EVENT_PULSE_CODE   1
#define PRIV_TIMEOUT_PULSE_CODE 16

/*
 * QNX OS specific APIs
 */
struct LwSciIpcConfigEntry;
#if (LW_IS_SAFETY == 0)
uint64_t lwsciipc_qnx_get_us(uint64_t time);
#endif /* (LW_IS_SAFETY == 0) */

/* fill outstr with head string and id since sprintf() is prohibited
 *
 * get "/dev/shmem/LwSciIpcConfig##" SHM node name
 * get "/dev/ivc##" IVC dev node name
 */
LwSciError lwsciipc_os_get_node_name(const char *head, uint32_t id, size_t len,
    char *outstr);
#endif /* __QNX__ */

#endif /* INCLUDED_LWSCIIPC_OS_QNX_H */


