/*
 * Copyright (c) 2020, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property and proprietary
 * rights in and to this software, related documentation and any modifications thereto.
 * Any use, reproduction, disclosure or distribution of this software and related
 * documentation without an express license agreement from LWPU Corporation is strictly
 * prohibited.
 *
 * TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED *AS IS*
 * AND LWPU AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS OR IMPLIED,
 * INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
 * PARTICULAR PURPOSE.  IN NO EVENT SHALL LWPU OR ITS SUPPLIERS BE LIABLE FOR ANY
 * SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES WHATSOEVER (INCLUDING, WITHOUT
 * LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF
 * BUSINESS INFORMATION, OR ANY OTHER PELWNIARY LOSS) ARISING OUT OF THE USE OF OR
 * INABILITY TO USE THIS SOFTWARE, EVEN IF LWPU HAS BEEN ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGES
 */

#pragma once

#define SALT_LENGTH 32
#define PUBLIC_KEY_LENGTH 32

namespace secrets {

// Defines keys and salts used by the ptxEncryptionBuildTool to encrypt internal PTX during the build
// Also used by the internalEncryptionManager in DeviceContext to decrypt internal PTX for built-in modules
//
// These were all generated using optix::detail::generateSalt()
static const unsigned char optixSalt[SALT_LENGTH]             = {40, 192, 54,  125, 170, 156, 38,  211, 97,  49, 90,
                                                     43, 82,  149, 182, 108, 190, 21,  38,  210, 13, 225,
                                                     56, 104, 58,  149, 69,  56,  174, 85,  193, 179};
static const unsigned char vendorSalt[SALT_LENGTH]            = {156, 175, 218, 159, 173, 220, 16,  250, 180, 144, 124,
                                                      170, 131, 253, 47,  134, 222, 206, 90,  201, 208, 32,
                                                      167, 218, 185, 110, 159, 58,  116, 111, 163, 4};
static const unsigned char vendorPublicKey[PUBLIC_KEY_LENGTH] = {73,  39,  228, 50,  147, 127, 29,  164, 94,  202, 210,
                                                                 45,  193, 226, 233, 95,  172, 252, 84,  195, 67,  55,
                                                                 249, 176, 250, 99,  131, 183, 142, 11,  220, 243};

}  // namespace secrets

