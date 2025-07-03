//
//  Copyright (c) 2019 LWPU Corporation.  All rights reserved.
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

#include <Memory/DemandLoad/PagingServiceLogging.h>

#include <prodlib/system/Knobs.h>
#include <prodlib/system/Logger.h>

namespace optix {
namespace demandLoad {

namespace {

// clang-format off
Knob<int> k_pmll( RT_DSTRING( "rtx.demandPaging.startingLogLevel"), 50, RT_DSTRING( "The log level at which demand paging will start normal logging.  Medium verbose logging and verbose logging occur at the next two higher log levels." ) );
// clang-format on
}

std::mutex g_demandLoadLogMutex;

bool isLogActive()
{
    return prodlib::log::active( k_pmll.get() );
}

std::ostream& logStream()
{
    return llog_stream( k_pmll.get() );
}

bool isLogMediumVerboseActive()
{
    return prodlib::log::active( k_pmll.get() + 1 );
}

std::ostream& logMediumVerboseStream()
{
    return llog_stream( k_pmll.get() + 1 );
}

bool isLogVerboseActive()
{
    return prodlib::log::active( k_pmll.get() + 2 );
}

std::ostream& logVerboseStream()
{
    return llog_stream( k_pmll.get() + 2 );
}

}  // namespace demandLoad
}  // namespace optix
