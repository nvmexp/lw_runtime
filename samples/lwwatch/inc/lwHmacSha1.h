/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2015 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#ifndef __LW_HMAC_SHA1_H__
#define __LW_HMAC_SHA1_H__

#include "lwSha1.h"

/*!
 * @brief Pointer to a memory accessor function (allocate a specific size of
 * buffer) for use by the SHA-1 hash function.
 *
 * SHA1 library can be used by many different clients, so we need to provide the
 * memory accessor functions which can work in client's environment.
 *
 * @param[in]  size   The requested size in bytes.
 *
 * @return the address of the allocated buffer or NULL when allocation failed.
 */

typedef void * Sha1AllocFunc(LwU32 size);

/*!
 * @brief Pointer to a memory accessor function (Free a specific buffer) for use
 * by the SHA-1 hash function.
 *
 * SHA1 library can be used by many different clients, so we need to provide the
 * memory accessor functions which can work in client's environment.
 *
 * @param[in]  pAddress   The address of the buffer to be freed.
 */

typedef void Sha1FreeFunc(void *pAddress);

/*!
 * @brief   Generates the HMAC-SHA-1 hash value on the data provided.
 *
 * The function does not manipulate the source data directly, as it may not
 * have direct access to it. Therefore, it relies upon the copy function to
 * copy segments of the data into a local buffer before any manipulation takes
 * place.
 *
 * @param[out]  pHash
 *          Pointer to store the hash array. The buffer must be 20 bytes in
 *          length, and the result is stored in big endian format.
 *
 * @param[in]   pMessage
 *          The Message data array to transform. The actual values and make-up
 *          of this parameter are dependent on the copy function.
 *
 * @param[in]   nMessageBytes
 *          The size, in bytes, of the Message data.
 *
 * @param[in]   pKey
 *          The Key data array to transform. The actual values and make-up of
 *          this parameter are dependent on the copy function.
 *
 * @param[in]   nKeyBytes
 *          The size, in bytes, of the Key data.
 *
 * @param[in]   copyFunc
 *          The function responsible for copying data from the source for use by
 *          the hmac-sha1 function.
 *
 * @param[in]   allocFunc
 *          The function responsible for allocating a buffer for use by the
 *          hmac-sha1 function.
 *
 * @param[in]   freeFunc
 *          The function responsible for freeing a buffer which is used by the
 *          hmac-sha1 function.
 *
 * @return TRUE when the generation succeeds otherwise return FALSE.
 */
static LwBool
hmacSha1Generate
(
    LwU8          pHash[LW_SHA1_DIGEST_LENGTH],
    void         *pMessage,
    LwU32         nMessageBytes,
    void         *pKey,
    LwU32         nKeyBytes,
    Sha1CopyFunc  copyFunc,
    Sha1AllocFunc allocFunc,
    Sha1FreeFunc  freeFunc
)
{
    LwU8 K0[LW_SHA1_BLOCK_LENGTH];
    LwU8 ipad[LW_SHA1_BLOCK_LENGTH];
    LwU8 opad[LW_SHA1_BLOCK_LENGTH];
    LwU8 tmp_hash[LW_SHA1_DIGEST_LENGTH];
    LwU8 *tmp_msg1;
    LwU8 tmp_msg2[LW_SHA1_BLOCK_LENGTH + LW_SHA1_DIGEST_LENGTH];
    LwU32 i;

    //
    // FIPS Publication 198 Section 5: HMAC Specification
    // Step 1-3: Determine K0
    //
    _sha1MemZero(K0, LW_SHA1_BLOCK_LENGTH);
    if (nKeyBytes <= LW_SHA1_BLOCK_LENGTH)
    {
        copyFunc(K0, 0, nKeyBytes, pKey);
    }
    else
    {
        sha1Generate(K0, pKey, nKeyBytes, copyFunc);
    }

    //
    // Step 4: K0 ^ ipad
    // Step 7: K0 ^ opad
    //
    for (i = 0; i < LW_SHA1_BLOCK_LENGTH; i++)
    {
        ipad[i] = K0[i] ^ 0x36;
        opad[i] = K0[i] ^ 0x5c;
    }

    // Step 5: Append the stream data to the result of Step 4
    tmp_msg1 = allocFunc(LW_SHA1_BLOCK_LENGTH + nMessageBytes);
    if (tmp_msg1 == NULL)
    {
        return LW_FALSE;
    }

    copyFunc(tmp_msg1, 0, LW_SHA1_BLOCK_LENGTH, ipad);
    copyFunc(tmp_msg1 + LW_SHA1_BLOCK_LENGTH, 0, nMessageBytes, pMessage);

    // Step 6: Apply SHA-1 hash to the stream generated in Step 5
    sha1Generate(tmp_hash, tmp_msg1,
                 LW_SHA1_BLOCK_LENGTH + nMessageBytes, copyFunc);
    freeFunc(tmp_msg1);

    // Step 8: Append the result 
    copyFunc(tmp_msg2, 0, LW_SHA1_BLOCK_LENGTH, opad);
    copyFunc(tmp_msg2 + LW_SHA1_BLOCK_LENGTH, 0,
             LW_SHA1_DIGEST_LENGTH, tmp_hash);

    // Step 9: Apply SHA-1 hash to the result from Step 8
    sha1Generate(pHash, tmp_msg2,
                 LW_SHA1_BLOCK_LENGTH + LW_SHA1_DIGEST_LENGTH, copyFunc);

    return LW_TRUE;
}


#endif /* __LW_HMAC_SHA1_H__ */
