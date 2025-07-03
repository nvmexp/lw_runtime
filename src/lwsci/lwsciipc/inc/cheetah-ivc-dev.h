/*
 * Copyright (c) 2018-2021, LWPU CORPORATION.  All rights reserved.
 *
 * This header is BSD licensed so anyone can use the definitions to implement
 * compatible drivers/servers.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 3. Neither the name of LWPU CORPORATION nor the names of its contributors
 *    may be used to endorse or promote products derived from this software
 *    without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL LWPU CORPORATION OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROLWREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef INCLUDED_TEGRA_IVC_DEV_H
#define INCLUDED_TEGRA_IVC_DEV_H

#include <stdint.h>
#if defined(LW_IS_TRACER_ENABLED)
#include <tracer.h>
#endif

#ifdef LINUX
#define NOTI_VA_SIZE 4U
#endif /* LINUX */

struct lwsciipc_ivc_info
{
    uint32_t nframes;
    uint32_t frame_size;
    uint32_t queue_offset;
    uint32_t queue_size;
    uint32_t area_size;
    /*
     * ivc has single shared area between 2 VMs out of which it creates
     * 2 fifo: one for receive and another for transmit. rx_first denotes
     * whether receive fifo will start first. If it is true, then receive
     * fifo will start first, otherwise transmit fifo will start first.
     */
    bool     rx_first;
#ifdef LINUX
    uint64_t noti_ipa;
#endif /* LINUX */
    uint16_t noti_irq;
};

#ifdef LINUX
/*  IOCTL magic number */
#define LW_SCI_IPC_IVC_IOCTL_MAGIC 0xAA

/* query ivc info */
#define LW_SCI_IPC_IVC_IOCTL_GET_INFO \
        _IOWR(LW_SCI_IPC_IVC_IOCTL_MAGIC, 1, struct lwsciipc_ivc_info)

/* notify remote */
#define LW_SCI_IPC_IVC_IOCTL_NOTIFY_REMOTE \
        _IO(LW_SCI_IPC_IVC_IOCTL_MAGIC, 2)

#define LW_SCI_IPC_IVC_IOCTL_NUMBER_MAX 2
#endif /* LINUX */

#endif /* INCLUDED_TEGRA_IVC_DEV_H */
