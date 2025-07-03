/*******************************************************************************
    Copyright (c) 2016-2020 LWpu Corporation

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to
    deal in the Software without restriction, including without limitation the
    rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
    sell copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

        The above copyright notice and this permission notice shall be
        included in all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
    THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
    DEALINGS IN THE SOFTWARE.

*******************************************************************************/

/*!
 * @file   lwswitch_sunos.c
 * @brief  LWSwitch driver kernel interface.
 *         TODO: Implement stubs. 
 */

#include "export_lwswitch.h"
#include <sys/sunddi.h>

#define LWSWITCH_OS_ASSERT(_cond)                                               \
    lwswitch_os_assert_log((_cond), "LWSwitch: Assertion failed in OS layer \n")

LwU64
lwswitch_os_get_platform_time
(
    void
)
{
    return 0ULL;
}

void
lwswitch_os_print
(
    const int  log_level,
    const char *fmt,
    ...
)
{
    return;
}

LwlStatus
lwswitch_os_read_registry_dword
(
    void *os_handle,
    const char *name,
    LwU32 *data
)
{
    return -1;
}

void
lwswitch_os_override_platform
(
    void *os_handle,
    LwBool *rtlsim
)
{
    // Never run on RTL
    *rtlsim = LW_FALSE;
}

LwU32
lwswitch_os_get_device_count
(
    void
)
{
    return 0;
}

LwBool
lwswitch_os_is_uuid_in_blacklist
(
    LwUuid *uuid
)
{
    return LW_FALSE;
}

LwlStatus
lwswitch_os_alloc_contig_memory
(
    void *os_handle,
    void **virt_addr,
    LwU32 size,
    LwBool force_dma32
)
{
    return -LWL_ERR_NOT_IMPLEMENTED;
}

void
lwswitch_os_free_contig_memory
(
    void *os_handle,
    void *virt_addr,
    LwU32 size
)
{
    return;
}

LwlStatus
lwswitch_os_map_dma_region
(
    void *os_handle,
    void *cpu_addr,
    LwU64 *dma_handle,
    LwU32 size,
    LwU32 direction
)
{
    return -LWL_ERR_NOT_IMPLEMENTED;
}


LwlStatus
lwswitch_os_unmap_dma_region
(
    void *os_handle,
    void *cpu_addr,
    LwU64 dma_handle,
    LwU32 size,
    LwU32 direction
)
{
    return -LWL_ERR_NOT_IMPLEMENTED;
}

LwlStatus
lwswitch_os_set_dma_mask
(
    void *os_handle,
    LwU32 dma_addr_width
)
{
    return -LWL_ERR_NOT_IMPLEMENTED;
}

LwlStatus
lwswitch_os_read_registery_binary
(
    void *os_handle,
    const char *name,
    LwU8 *data,
    LwU32 length
)
{
    return -LWL_ERR_NOT_IMPLEMENTED;
}

LwlStatus
lwswitch_os_sync_dma_region_for_cpu
(
    void *os_handle,
    LwU64 dma_handle,
    LwU32 size,
    LwU32 direction
)
{
    return -LWL_ERR_NOT_IMPLEMENTED;
}

LwlStatus
lwswitch_os_sync_dma_region_for_device
(
    void *os_handle,
    LwU64 dma_handle,
    LwU32 size,
    LwU32 direction
)
{
    return -LWL_ERR_NOT_IMPLEMENTED;
}

void *
lwswitch_os_malloc_trace
(
    LwLength size,
    const char *file,
    LwU32 line
)
{
    return NULL;
}

void
lwswitch_os_free
(
    void *pMem
)
{
    return;
}

LwLength
lwswitch_os_strlen
(
    const char *str
)
{
    return strlen(str);
}

char*
lwswitch_os_strncpy
(
    char *dest,
    const char *src,
    LwLength length
)
{
    return strncpy(dest, src, length);
}

int
lwswitch_os_strncmp
(
    const char *s1,
    const char *s2,
    LwLength length
)
{
    return strncmp(s1, s2, length);
}

void *
lwswitch_os_memset
(
    void *pDest,
    int value,
    LwLength size
)
{
    return NULL;
}

void *
lwswitch_os_memcpy
(
    void *pDest,
    const void *pSrc,
    LwLength size
)
{
    return NULL;
}

int
lwswitch_os_memcmp
(
    const void *s1,
    const void *s2,
    LwLength size
)
{
    return memcmp(s1, s2, size);
}

LwU32
lwswitch_os_mem_read32
(
    const volatile void * pAddress
)
{
    return 0;
}

void
lwswitch_os_mem_write32
(
    volatile void *pAddress,
    LwU32 data
)
{
}

LwU64
lwswitch_os_mem_read64
(
    const volatile void *pAddress
)
{
    return 0;
}

void
lwswitch_os_mem_write64
(
    volatile void *pAddress,
    LwU64 data
)
{
}

int
lwswitch_os_snprintf
(
    char *pString,
    LwLength size,
    const char *pFormat,
    ...
)
{
    return 0;
}

int
lwswitch_os_vsnprintf
(
    char *buf,
    LwLength size,
    const char *fmt,
    va_list arglist
)
{
    return vsnprintf(buf, size, fmt, arglist);
}

void
lwswitch_os_assert_log
(
    int cond,
    const char *pFormat,
    ...
)
{
}

/*
 * Sleep for specified milliseconds. Yields the CPU to scheduler.
 */
void
lwswitch_os_sleep
(
    unsigned int ms
)
{
    return;
}

LwlStatus
lwswitch_os_acquire_fabric_mgmt_cap
(
    void *osPrivate,
    LwU64 capDescriptor
)
{
    return -LWL_ERR_NOT_IMPLEMENTED;
}

int
lwswitch_os_is_fabric_manager
(
    void *osPrivate
)
{
    return 0;
}

int
lwswitch_os_is_admin
(
    void
)
{
    return 0;
}

LwlStatus
lwswitch_os_get_os_version
(
    LwU32 *pMajorVer,
    LwU32 *pMinorVer,
    LwU32 *pBuildNum
)
{
    return -LWL_ERR_NOT_IMPLEMENTED;
}

/*!
 * @brief: OS specific handling to add an event.
 */
LwlStatus
lwswitch_os_add_client_event
(
    void            *osHandle,
    void            *osPrivate,
    LwU32           eventId
)
{
    return -LWL_ERR_NOT_IMPLEMENTED;
}

/*!
 * @brief: OS specific handling to remove all events corresponding to osPrivate.
 */
LwlStatus
lwswitch_os_remove_client_event
(
    void            *osHandle,
    void            *osPrivate
)
{
    return -LWL_ERR_NOT_IMPLEMENTED;
}

/*!
 * @brief: OS specific handling to notify an event.
 */
LwlStatus
lwswitch_os_notify_client_event
(
    void *osHandle,
    void *osPrivate,
    LwU32 eventId
)
{
    return -LWL_ERR_NOT_IMPLEMENTED;
}

/*!
 * @brief: Gets OS specific support for the REGISTER_EVENTS ioctl
 */
LwlStatus
lwswitch_os_get_supported_register_events_params
(
    LwBool *many_events,
    LwBool *os_descriptor
)
{
    return -LWL_ERR_NOT_IMPLEMENTED;
}
