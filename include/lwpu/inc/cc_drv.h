/*
* _LWRM_COPYRIGHT_BEGIN_
*
* Copyright 2021 by LWPU Corporation.  All rights reserved.  All
* information contained herein is proprietary and confidential to LWPU
* Corporation.  Any use, reproduction, or disclosure without the written
* permission of LWPU Corporation is prohibited.
*
* _LWRM_COPYRIGHT_END_
*/

#pragma once

#include <lwtypes.h>
#if defined(_MSC_VER)
#pragma warning(disable:4324)
#endif

//
// This file was generated with FINN, an LWPU coding tool.
// Source file: cc_drv.finn
//
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
//
// Please edit the original FINN IDL file to create desired edits in this header
// See https://confluence.lwpu.com/display/CORERM/FINN for more info on how to
// edit FINN.
//
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)



#include "lwtypes.h"

// CLASS LW_CONF_COMPUTE
#define CC_AES_256_GCM_IV_SIZE_BYTES   (0xc) /* finn: Evaluated from "(96 / 8)" */
#define CC_AES_256_GCM_IV_SIZE_DWORD   (0x3) /* finn: Evaluated from "(CC_AES_256_GCM_IV_SIZE_BYTES / 4)" */
#define CC_AES_256_GCM_KEY_SIZE_BYTES  (0x20) /* finn: Evaluated from "(256 / 8)" */
#define CC_AES_256_GCM_KEY_SIZE_DWORD  (0x8) /* finn: Evaluated from "(CC_AES_256_GCM_KEY_SIZE_BYTES / 4)" */

#define CC_HMAC_NONCE_SIZE_BYTES       (0x20) /* finn: Evaluated from "(256 / 8)" */
#define CC_HMAC_NONCE_SIZE_DWORD       (0x8) /* finn: Evaluated from "(CC_HMAC_NONCE_SIZE_BYTES / 4)" */
#define CC_HMAC_KEY_SIZE_BYTES         (0x20) /* finn: Evaluated from "(256 / 8)" */
#define CC_HMAC_KEY_SIZE_DWORD         (0x8) /* finn: Evaluated from "(CC_HMAC_KEY_SIZE_BYTES / 4)" */

#define APM_AES_128_CTR_IV_SIZE_BYTES  (0xc) /* finn: Evaluated from "(96 / 8)" */
#define APM_AES_128_CTR_IV_SIZE_DWORD  (0x3) /* finn: Evaluated from "(APM_AES_128_CTR_IV_SIZE_BYTES / 4)" */
#define APM_AES_128_CTR_KEY_SIZE_BYTES (0x10) /* finn: Evaluated from "(128 / 8)" */
#define APM_AES_128_CTR_KEY_SIZE_DWORD (0x4) /* finn: Evaluated from "(APM_AES_128_CTR_KEY_SIZE_BYTES / 4)" */

// Type is shared between CC control calls and RMKeyStore
typedef enum ROTATE_IV_TYPE {
    ROTATE_IV_ENCRYPT = 0,  // Rotate the IV for encryptBundle
    ROTATE_IV_DECRYPT = 1,  // Rotate the IV for decryptBundle
    ROTATE_IV_HMAC = 2,     // Rotate the IV for hmacBundle
    ROTATE_IV_ALL_VALID = 3, // Rotate the IV for all valid bundles in the KMB
} ROTATE_IV_TYPE;

typedef struct CC_AES_CRYPTOBUNDLE {
    LwU32 iv[CC_AES_256_GCM_IV_SIZE_DWORD];
    LwU32 key[CC_AES_256_GCM_KEY_SIZE_DWORD];
    LwU32 ivMask[CC_AES_256_GCM_IV_SIZE_DWORD];
} CC_AES_CRYPTOBUNDLE;
typedef struct CC_AES_CRYPTOBUNDLE *PCC_AES_CRYPTOBUNDLE;

typedef struct CC_HMAC_CRYPTOBUNDLE {
    LwU32 nonce[CC_HMAC_NONCE_SIZE_DWORD];
    LwU32 key[CC_HMAC_KEY_SIZE_DWORD];
} CC_HMAC_CRYPTOBUNDLE;
typedef struct CC_HMAC_CRYPTOBUNDLE *PCC_HMAC_CRYPTOBUNDLE;

typedef struct CC_KMB {
    CC_AES_CRYPTOBUNDLE encryptBundle;           // Bundle of encyption material

    union {
        CC_HMAC_CRYPTOBUNDLE hmacBundle;  // HMAC bundle used for method stream authenticity
        CC_AES_CRYPTOBUNDLE  decryptBundle;   // Bundle of decryption material
    };
    LwBool bIsWorkLaunch;                        // False if decryption parameters are valid
} CC_KMB;
typedef struct CC_KMB *PCC_KMB;
