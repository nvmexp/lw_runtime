/*
 * SPDX-FileCopyrightText: Copyright (c) 2004-2015 LWPU CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#pragma once

#include <lwtypes.h>
#if defined(_MSC_VER)
#pragma warning(disable:4324)
#endif

//
// This file was generated with FINN, an LWPU coding tool.
// Source file: ctrl/ctrl0080/ctrl0080cipher.finn
//
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
//
// Please edit the original FINN IDL file to create desired edits in this header
// See https://confluence.lwpu.com/display/CORERM/FINN for more info on how to
// edit FINN.
//
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)



#include "ctrl/ctrl0080/ctrl0080base.h"

/* LW01_DEVICE_XX/LW03_DEVICE cipher control commands and parameters */

/*
 * LW0080_CTRL_CMD_CIPHER_AES_ENCRYPT
 *
 * The commands runs 128-bit AES block cipher ("Rijndael") over the
 * 128-bit input message in the pt[] array using the session key as
 * the encryption key.  The encrypted result is returned in the
 * 128-bit ct[] array.
 *
 * Possible status values returned are:
 *   LW_ERR_ILWALID_DEVICE
 *     Device is not yet initialized or DH key echange not complete;
 *     try again later.
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_GENERIC
 */
#define LW0080_CTRL_CMD_CIPHER_AES_ENCRYPT (0x801201) /* finn: Evaluated from "(FINN_LW01_DEVICE_0_CIPHER_INTERFACE_ID << 8) | LW0080_CTRL_CMD_CIPHER_AES_ENCRYPT_PARAMS_MESSAGE_ID" */


#define LW0080_CTRL_CMD_CIPHER_AES_ENCRYPT_PARAMS_MESSAGE_ID (0x1U)

typedef struct LW0080_CTRL_CMD_CIPHER_AES_ENCRYPT_PARAMS {
    LwU8 pt[16];         // 128bit message in
    LwU8 ct[16];         // 128bit encrypted message out
} LW0080_CTRL_CMD_CIPHER_AES_ENCRYPT_PARAMS;


/*
 * LW0080_CTRL_CMD_CIPHER_SESSION_KEY
 *
 * The command returns the 128bit session key used for content encoding.
 * Command can fail if DH key exelwtion has not completed, 
 *
 * Possible stus values returened are:
 *   LW_ERR_ILWALID_DEVICE
 *     Device is not yet initialized or DH key echange not complete; try
 *     again later.
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_GENERIC
 *
 */
#define LW0080_CTRL_CMD_CIPHER_SESSION_KEY (0x801202) /* finn: Evaluated from "(FINN_LW01_DEVICE_0_CIPHER_INTERFACE_ID << 8) | LW0080_CTRL_CMD_CIPHER_SESSION_KEY_PARAMS_MESSAGE_ID" */

#define LW0080_CTRL_CMD_CIPHER_SESSION_KEY_PARAMS_MESSAGE_ID (0x2U)

typedef struct LW0080_CTRL_CMD_CIPHER_SESSION_KEY_PARAMS {
    LwU8 sKey[16];       // 128bit session key out
} LW0080_CTRL_CMD_CIPHER_SESSION_KEY_PARAMS;





/*
 * LW0080_CTRL_CMD_CIPHER_GET_DH_RESPONSE
 *
 * Fetch GPU's response to DH Key initiation
 * 
 * The intended use of this function is that ESTABLISH_SESSION_KEY will be used
 * once, then this function will be used to poll for a result.  It is necessary
 * to fetch the result from an exchange before initiating another one.  
 * 
 * If the exchange is still in progress, this function will return LW_OK,
 * and responseReady will be set to FALSE.  If there is no exchange in
 * progress, this function will return LW_ERR_GENERIC and the parameters will not be
 * modified.  If the exchange has completed and there is a result waiting, this
 * function will copy the response into Yprime, set responseReady to TRUE, and
 * modify RM's internal state so it is ready for another exchange.  
 *
 * Y' is the GPU's response in the DH exchange, and SK128 is the 'opaque
 * session key'.  
 */
#define LW0080_CTRL_CMD_CIPHER_GET_DH_RESPONSE (0x801204) /* finn: Evaluated from "(FINN_LW01_DEVICE_0_CIPHER_INTERFACE_ID << 8) | LW0080_CTRL_CMD_CIPHER_DH_RESPONSE_PARAMS_MESSAGE_ID" */

#define LW0080_CTRL_CMD_CIPHER_DH_RESPONSE_PARAMS_MESSAGE_ID (0x4U)

typedef struct LW0080_CTRL_CMD_CIPHER_DH_RESPONSE_PARAMS {
    LwBool responseReady;
    LwU8   Yprime[256];
    LwU8   SK128[16];
} LW0080_CTRL_CMD_CIPHER_DH_RESPONSE_PARAMS;

/*
 * LW0080_CTRL_CMD_CIPHER_SESSION_KEY_STATUS
 *
 * This command retrieves the current session key status.
 *
 * Possible status values returned are:
 *   LW_OK
 *     The current DH session key exchange status is retrieved successfully
 *   LW_ERR_ILWALID_ARGUMENT
 *     This function will return LWOS_STATUS_ERROR_ILWALID_ARGUEMENT if
 *     either the clientId or deviceId is not valid
 */
#define LW0080_CTRL_CMD_CIPHER_SESSION_KEY_STATUS                      (0x801205) /* finn: Evaluated from "(FINN_LW01_DEVICE_0_CIPHER_INTERFACE_ID << 8) | LW0080_CTRL_CMD_CIPHER_SESSION_KEY_STATUS_PARAMS_MESSAGE_ID" */

/* legal status values */
#define LW0080_CTRL_CMD_CIPHER_SESSION_KEY_STATUS_IN_PROGRESS          0:0
#define LW0080_CTRL_CMD_CIPHER_SESSION_KEY_STATUS_IN_PROGRESS_TRUE     (0x00000001)
#define LW0080_CTRL_CMD_CIPHER_SESSION_KEY_STATUS_IN_PROGRESS_FALSE    (0x00000000)
#define LW0080_CTRL_CMD_CIPHER_SESSION_KEY_STATUS_ESTABLISHED          1:1
#define LW0080_CTRL_CMD_CIPHER_SESSION_KEY_STATUS_ESTABLISHED_TRUE     (0x00000001)
#define LW0080_CTRL_CMD_CIPHER_SESSION_KEY_STATUS_ESTABLISHED_FALSE    (0x00000000)
#define LW0080_CTRL_CMD_CIPHER_SESSION_KEY_STATUS_ABORTED              2:2
#define LW0080_CTRL_CMD_CIPHER_SESSION_KEY_STATUS_ABORTED_TRUE         (0x00000001)
#define LW0080_CTRL_CMD_CIPHER_SESSION_KEY_STATUS_ABORTED_FALSE        (0x00000000)
#define LW0080_CTRL_CMD_CIPHER_SESSION_KEY_STATUS_RESPONSE_READY       3:3
#define LW0080_CTRL_CMD_CIPHER_SESSION_KEY_STATUS_RESPONSE_READY_FALSE (0x00000000)
#define LW0080_CTRL_CMD_CIPHER_SESSION_KEY_STATUS_RESPONSE_READY_TRUE  (0x00000001)

#define LW0080_CTRL_CMD_CIPHER_SESSION_KEY_STATUS_PARAMS_MESSAGE_ID (0x5U)

typedef struct LW0080_CTRL_CMD_CIPHER_SESSION_KEY_STATUS_PARAMS {
    LwU32 status;
} LW0080_CTRL_CMD_CIPHER_SESSION_KEY_STATUS_PARAMS;


/*
 * LW0080_CTRL_CMD_CIPHER_GET_DH_CHALLENGE
 *
 * This command retrieves new DHKE challenge from RM.
 *
 * Possible status values returned are:
 *   LW_OK
 *     The current DH challenge buffer is retrieved successfully
 *   LW_ERR_ILWALID_ARGUMENT
 *     This function will return LWOS_STATUS_ERROR_ILWALID_ARGUEMENT if
 *     either the clientId or deviceId is not valid
 */
#define LW0080_CTRL_CMD_CIPHER_GET_DH_CHALLENGE (0x801207) /* finn: Evaluated from "(FINN_LW01_DEVICE_0_CIPHER_INTERFACE_ID << 8) | LW0080_CTRL_CMD_CIPHER_GET_DH_CHALLENGE_PARAMS_MESSAGE_ID" */

#define LW0080_CTRL_CMD_CIPHER_GET_DH_CHALLENGE_PARAMS_MESSAGE_ID (0x7U)

typedef struct LW0080_CTRL_CMD_CIPHER_GET_DH_CHALLENGE_PARAMS {
    LwU8 pX[256];                       // 256 bytes, X = (g ^ M) mod p
    LwU8 pInitializatiolwector[16];     // 16 bytes, random number filled by RM
} LW0080_CTRL_CMD_CIPHER_GET_DH_CHALLENGE_PARAMS;

/*
 * LW0080_CTRL_CMD_CIPHER_SET_DH_RESPONSE
 *
 * This command sets new DHKE response from UMD to RM.
 *
 * Possible status values returned are:
 *   LW_OK
 *     The current DH response buffer is processed successfully
 *   LW_ERR_ILWALID_ARGUMENT
 *     This function will return LWOS_STATUS_ERROR_ILWALID_ARGUEMENT if
 *     either the clientId or deviceId is not valid
 */
#define LW0080_CTRL_CMD_CIPHER_SET_DH_RESPONSE (0x801208) /* finn: Evaluated from "(FINN_LW01_DEVICE_0_CIPHER_INTERFACE_ID << 8) | LW0080_CTRL_CMD_CIPHER_SET_DH_RESPONSE_PARAMS_MESSAGE_ID" */

#define LW0080_CTRL_CMD_CIPHER_SET_DH_RESPONSE_PARAMS_MESSAGE_ID (0x8U)

typedef struct LW0080_CTRL_CMD_CIPHER_SET_DH_RESPONSE_PARAMS {
    LwU8 pY[256];                       // 256 bytes, Y = (g ^ N * X ^ S) mod p
    LwU8 pWrappedSessionKey[16];        // 16 bytes, session key = DM_HASH(X ^ N)
} LW0080_CTRL_CMD_CIPHER_SET_DH_RESPONSE_PARAMS;


/* _ctrl0080cipher_h_ */
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

