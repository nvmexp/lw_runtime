//
//  Copyright (c) 2021 LWPU Corporation.  All rights reserved.
//
//  LWPU Corporation and its licensors retain all intellectual property and proprietary
//  rights in and to this software, related documentation and any modifications thereto.
//  Any use, reproduction, disclosure or distribution of this software and related
//  documentation without an express license agreement from LWPU Corporation is strictly
//  prohibited.
//
//  TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED *AS IS*
//  AND LWPU AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS OR IMPLIED,
//  INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
//  PARTICULAR PURPOSE.  IN NO EVENT SHALL LWPU OR ITS SUPPLIERS BE LIABLE FOR ANY
//  SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES WHATSOEVER (INCLUDING, WITHOUT
//  LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF
//  BUSINESS INFORMATION, OR ANY OTHER PELWNIARY LOSS) ARISING OUT OF THE USE OF OR
//  INABILITY TO USE THIS SOFTWARE, EVEN IF LWPU HAS BEEN ADVISED OF THE POSSIBILITY OF
//  SUCH DAMAGES
//

#pragma once

// Texture info for footprint intrinsic.  This must agree with the
// corresponding struct in the SDK, TextureSampler::Description
struct TextureInfo
{
    unsigned int isInitialized : 1;
    unsigned int reserved1 : 2;
    unsigned int numMipLevels : 5;
    unsigned int logTileWidth : 4;
    unsigned int logTileHeight : 4;
    unsigned int reserved2 : 4;
    unsigned int isUdimBaseTexture : 1;
    unsigned int hasBaseColor : 1;
    unsigned int wrapMode0 : 2;
    unsigned int wrapMode1 : 2;
    unsigned int mipmapFilterMode : 1;
    unsigned int maxAnisotropy : 5;
};

// Texture2DFootprint is binary compatible with the uint4 returned by the texture footprint instructions.
struct Texture2DFootprint
{
    unsigned long long mask;
    unsigned int       tileY : 12;
    unsigned int       reserved1 : 4;  // not used
    unsigned int       dx : 3;
    unsigned int       dy : 3;
    unsigned int       reserved2 : 2;  // not used
    unsigned int       granularity : 4;
    unsigned int       reserved3 : 4;  // not used
    unsigned int       tileX : 12;
    unsigned int       level : 4;
    unsigned int       reserved4 : 16;  // not used
};
