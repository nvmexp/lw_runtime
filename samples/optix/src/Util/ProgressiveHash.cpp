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

#include <Util/ProgressiveHash.h>
// NOTE: Do not include xxh3 in a header file. It redefines assert to ((void)0).
//       Also include this last to avoid clobbering assert in includes.
#include <corelib/misc/xxh3.h>

namespace optix {

ProgressiveHash::ProgressiveHash()
{
    m_state = XXH3_64bits_createState();
    XXH3_64bits_reset( (XXH3_state_t*)m_state );
}

ProgressiveHash::~ProgressiveHash()
{
    XXH3_64bits_freeState( (XXH3_state_t*)m_state );
}

void ProgressiveHash::update( const void* data, size_t len )
{
    XXH3_64bits_update( (XXH3_state_t*)m_state, data, len );
}

size_t ProgressiveHash::digest() const
{
    return (size_t)XXH3_64bits_digest( (XXH3_state_t*)m_state );
}

}
