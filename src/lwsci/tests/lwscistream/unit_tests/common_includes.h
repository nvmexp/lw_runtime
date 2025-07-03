//! \file
//! \brief LwSciStream unit testing common includes header.
//!
//! \copyright
//! Copyright (c) 2020-2021 LWPU Corporation. All rights reserved.
//!
//! LWPU Corporation and its licensors retain all intellectual property and
//! proprietary rights in and to this software, related documentation and any
//! modifications thereto. Any use, reproduction, disclosure or distribution
//! of this software and related documentation without an express license
//! agreement from LWPU Corporation is strictly prohibited.

#ifndef COMMON_INCLUDES_H
#define COMMON_INCLUDES_H

#include "lwscistream.h"

constexpr LwSciIpcEndpoint IPCSRC_ENDPOINT = 12345;
constexpr LwSciIpcEndpoint IPCDST_ENDPOINT = 54321;

// Define end point info for ipc channel
typedef struct {
    // channel name
    char chname[21];
    // LwIPC handle
    LwSciIpcEndpoint endpoint = 0U;
    // channel info
    struct LwSciIpcEndpointInfo info;
#ifdef __linux__
    // LwIPC event handle
    int32_t eventHandle = 0;
#else
    int32_t coid = 0;
    int16_t pulsePriority = 0;
    int16_t pulseCode = 0;
#endif
} Endpoint;

// Ipc channel Endpoints extern declarations
extern Endpoint ipcSrc, ipcDst;

#endif // !COMMON_INCLUDES_H
