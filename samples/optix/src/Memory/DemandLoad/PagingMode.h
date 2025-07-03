//
//  Copyright (c) 2020 LWPU Corporation.  All rights reserved.
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

#include <iostream>
#include <string>

namespace optix {

enum class PagingMode
{
    UNKNOWN = 0,
    WHOLE_MIPLEVEL,
    SOFTWARE_SPARSE,
    LWDA_SPARSE_HYBRID,
    LWDA_SPARSE_HARDWARE
};

inline std::string toString( PagingMode mode )
{
    switch( mode )
    {
        case PagingMode::UNKNOWN:
            return "UNKNOWN";
        case PagingMode::WHOLE_MIPLEVEL:
            return "WHOLE_MIPLEVEL";
        case PagingMode::SOFTWARE_SPARSE:
            return "SOFTWARE_SPARSE";
        case PagingMode::LWDA_SPARSE_HYBRID:
            return "LWDA_SPARSE_HYBRID";
        case PagingMode::LWDA_SPARSE_HARDWARE:
            return "LWDA_SPARSE_HARDWARE";
    }
    return std::to_string( static_cast<int>( mode ) );
}

inline std::ostream& operator<<( std::ostream& str, PagingMode mode )
{
    return str << toString( mode );
}

}  // namespace optix
