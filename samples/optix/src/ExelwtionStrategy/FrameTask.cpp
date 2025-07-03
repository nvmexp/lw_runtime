// Copyright (c) 2017, LWPU CORPORATION.
// TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED
// *AS IS* AND LWPU AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS
// OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY
// AND FITNESS FOR A PARTICULAR PURPOSE.  IN NO EVENT SHALL LWPU OR ITS SUPPLIERS
// BE LIABLE FOR ANY SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES
// WHATSOEVER (INCLUDING, WITHOUT LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS,
// BUSINESS INTERRUPTION, LOSS OF BUSINESS INFORMATION, OR ANY OTHER PELWNIARY LOSS)
// ARISING OUT OF THE USE OF OR INABILITY TO USE THIS SOFTWARE, EVEN IF LWPU HAS
// BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES

#include <Context/Context.h>
#include <ExelwtionStrategy/FrameTask.h>
#include <prodlib/system/Knobs.h>

#include <sstream>

using namespace optix;

namespace {
// clang-format off
Knob<std::string> k_limitActiveLaunchIndices( RT_DSTRING( "launch.limitActiveIndices" ), "", RT_DSTRING( "When specified limit which launch indices are active. Syntax: [minX, maxX], [minY, maxY]" ) );
}
// clang-format on

FrameTask::FrameTask( Context* context )
    : m_context( context )
{
    if( !k_limitActiveLaunchIndices.isDefault() )
    {
        minMaxLaunchIndex[0] = 0;
        minMaxLaunchIndex[1] = 0xFFFFFFFF;
        minMaxLaunchIndex[2] = 0;
        minMaxLaunchIndex[3] = 0xFFFFFFFF;
        minMaxLaunchIndex[4] = 0;
        minMaxLaunchIndex[5] = 0xFFFFFFFF;
        // parse the knob and set these values
        // format is: [minX, maxX], [minY, maxY]
        std::istringstream is( k_limitActiveLaunchIndices.get() );
        char               st;
        is >> st >> minMaxLaunchIndex[0] >> st >> minMaxLaunchIndex[1] >> st;
        is >> st;  // ','
        if( is.good() )
        {
            RT_ASSERT_MSG( st == ',', "Error parsing knob: " + k_limitActiveLaunchIndices.get() );
            is >> st >> minMaxLaunchIndex[2] >> st >> minMaxLaunchIndex[3] >> st;
            is >> st;
            if( is.good() )
            {
                RT_ASSERT_MSG( st == ',', "Error parsing knob: " + k_limitActiveLaunchIndices.get() );
                is >> st >> minMaxLaunchIndex[4] >> st >> minMaxLaunchIndex[5] >> st;
                RT_ASSERT_MSG( st == ']', "Error parsing knob: " + k_limitActiveLaunchIndices.get() );
            }
        }
        lprint << "minMaxLaunchIndex set to "
               << "[" << minMaxLaunchIndex[0] << ", " << minMaxLaunchIndex[1] << "]"
               << ", "
               << "[" << minMaxLaunchIndex[2] << ", " << minMaxLaunchIndex[3] << "]"
               << ", "
               << "[" << minMaxLaunchIndex[4] << ", " << minMaxLaunchIndex[5] << "]"
               << "\n";
    }
}

FrameTask::~FrameTask()
{
}

std::shared_ptr<ProfileMapping> FrameTask::getProfileMapping() const
{
    return m_profileMapping;
}

void FrameTask::setProfileMapping( const std::shared_ptr<ProfileMapping>& newMapping )
{
    m_profileMapping = newMapping;
}
