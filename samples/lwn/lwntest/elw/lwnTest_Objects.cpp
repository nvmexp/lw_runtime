/*
* Copyright (c) 2015, Lwpu Corporation.  All rights reserved.
*
* THE INFORMATION CONTAINED HEREIN IS PROPRIETARY AND CONFIDENTIAL TO
* LWPU, CORPORATION.  USE, REPRODUCTION OR DISCLOSURE TO ANY THIRD PARTY
* IS SUBJECT TO WRITTEN PRE-APPROVAL BY LWPU, CORPORATION.
*
*
*/

#include "lwntest_c.h"
#include "lwnTest/lwnTest_Objects.h"

LWNuint lwnTextureGetRegisteredTextureID(const LWNtexture *texture)
{
    const __LWNtextureInternal *alloc = reinterpret_cast<const __LWNtextureInternal *>(texture);
    return alloc->lastRegisteredTextureID;
}

LWNuint lwnSamplerGetRegisteredID(const LWNsampler *sampler)
{
    const __LWNsamplerInternal *alloc = reinterpret_cast<const __LWNsamplerInternal *>(sampler);
    return alloc->lastRegisteredID;
}
