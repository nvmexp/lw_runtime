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

#pragma once

/* This file defines all the internal function return values */

typedef enum {
    FM_INT_ST_OK                        =  0,
    FM_INT_ST_BADPARAM                  = -1,
    FM_INT_ST_GENERIC_ERROR             = -2,
    
    FM_INT_ST_CONNECTION_NOT_VALID      = -3,
    FM_INT_ST_NOT_SUPPORTED             = -4,
    FM_INT_ST_NOT_CONFIGURED            = -5,
    FM_INT_ST_UNINITIALIZED             = -6,
    FM_INT_ST_IN_USE                    = -7,
    FM_INT_ST_RESOURCE_UNAVAILABLE      = -8,

    // file open/read related errors
    FM_INT_ST_FILE_ILWALID              = -10,
    FM_INT_ST_FILE_OPEN_ERR             = -11,
    FM_INT_ST_FILE_PARSING_ERR          = -12,

    // objects related errors
    FM_INT_ST_ILWALID_NODE              = -20,
    FM_INT_ST_ILWALID_GPU               = -21,
    FM_INT_ST_ILWALID_LWSWITCH          = -22 ,
    FM_INT_ST_ILWALID_PORT              = -23,
    FM_INT_ST_ILWALID_TABLE_ENTRY       = -24,

    // config related errors
    FM_INT_ST_ILWALID_NODE_CFG              = -40,
    FM_INT_ST_ILWALID_GPU_CFG               = -41,
    FM_INT_ST_ILWALID_WILLOW_CFG            = -42,
    FM_INT_ST_ILWALID_PORT_CFG              = -43,
    FM_INT_ST_ILWALID_INGR_REQ_ENTRY_CFG    = -44,
    FM_INT_ST_ILWALID_INGR_RESP_ENTRY_CFG   = -45,
    FM_INT_ST_ILWALID_GANGED_LINK_ENTRY_CFG = -46,

    // invalid connection
    FM_INT_ST_ILWALID_GLOBAL_CONTROL_CONN_TO_LFM = -61,
    FM_INT_ST_ILWALID_LOCAL_CONTROL_CONN_TO_GFM  = -62,

    // interface error
    FM_INT_ST_IOCTL_ERR    = -71,
    FM_INT_ST_MSG_SEND_ERR = -72,

    // config timeout and error
    FM_INT_ST_CFG_TIMEOUT   = -81,
    FM_INT_ST_CFG_ERROR     = -82,

    //to break
    FM_INT_ST_BREAK         = -91,

    //fmrequest status
    FM_INT_ST_PENDING       = -101,
    FM_INT_ST_TIMEOUT       = -102,

    // LWLink error
    FM_INT_ST_LWLINK_ERROR     = -201,
    // version mismatch
    FM_INT_ST_VERSION_MISMATCH = -202
} FMIntReturn_t;
