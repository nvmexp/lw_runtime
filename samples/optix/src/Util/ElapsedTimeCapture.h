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

#include <chrono>

namespace optix {

/// Captures the milliseconds elapsed between it's construction and destruction
/// from the high resolution clock.  Useful for capturing elapsed time across
/// a block of code, regardless of how you exit the block.  Shortlwt returns and
/// exceptions are no problem!
class ElapsedTimeCapture
{
  public:
    explicit ElapsedTimeCapture( int& elapsed )
        : m_elapsed( elapsed )
        , m_start( std::chrono::high_resolution_clock::now() )
    {
    }
    ~ElapsedTimeCapture()
    {
        const auto stop = std::chrono::high_resolution_clock::now();
        m_elapsed       = std::chrono::duration_cast<std::chrono::milliseconds>( stop - m_start ).count();
    }

  private:
    int&                                                        m_elapsed;
    std::chrono::time_point<std::chrono::high_resolution_clock> m_start;
};

}  // namespace optix
