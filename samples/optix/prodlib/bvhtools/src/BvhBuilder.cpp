// Copyright LWPU Corporation 2013
// TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED
// *AS IS* AND LWPU AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS
// OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY
// AND FITNESS FOR A PARTICULAR PURPOSE.  IN NO EVENT SHALL LWPU OR ITS SUPPLIERS
// BE LIABLE FOR ANY SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES
// WHATSOEVER (INCLUDING, WITHOUT LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS,
// BUSINESS INTERRUPTION, LOSS OF BUSINESS INFORMATION, OR ANY OTHER PELWNIARY LOSS)
// ARISING OUT OF THE USE OF OR INABILITY TO USE THIS SOFTWARE, EVEN IF LWPU HAS
// BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES

#include "../include/BvhBuilder.hpp"

#include <g_lwconfig.h>

#include "BVH8Builder.hpp"
#include "ChunkedTrbvhBuilder.hpp"
#include <prodlib/system/Knobs.h>
#include <corelib/misc/String.h>

using namespace prodlib::bvhtools;

namespace {
  Knob<std::string> k_builderSpecAppend( RT_DSTRING("bvhtools.builderSpecAppend"), "", RT_DSTRING("Append entries to the builderSpec, overriding existing entries."));
}

BvhBuilder::~BvhBuilder( void )
{
}

BvhBuilder* BvhBuilder::create( const char* builderSpec )
{
  ParameterList p( builderSpec );

  const char* builder       = p.get( "type", "TRBVH" );
  bool        lwdaAvailable = !p.get( "disableLWDA", false );

  if( std::string( "BVH8" ) == builder )
    return new BVH8Builder( lwdaAvailable );
  else
    return new ChunkedTrbvhBuilder( lwdaAvailable );
}

void BvhBuilder::setBuilderSpec(const char* builderSpec)
{
  m_builderSpec = (builderSpec) ? builderSpec : "";

  // Append knob value
  const std::string builderSpecAppend = k_builderSpecAppend.get();
  if ( !builderSpecAppend.empty() )
  {
    if (builderSpec)
      m_builderSpec += std::string(", ");
    m_builderSpec += builderSpecAppend;
  }

}

void BvhBuilder::build(const std::vector<TriangleMesh>& meshes, bool meshInDeviceMem)
{
  RT_ASSERT_MSG(0, "Not implemented");
}

void BvhBuilder::build(const Group& group, bool groupInDeviceMem)
{
  build( group.numInstances, group.aabbs, groupInDeviceMem );
}

void BvhBuilder::build(int numAabbs, int motionSteps, const float* aabbs, bool aabbsInDeviceMem)
{
  RT_ASSERT_MSG(0, "Not implemented");
}

void BvhBuilder::build(int numInstances, const InstanceDesc* instances, bool inDeviceMem)
{
  RT_ASSERT_MSG(0, "Not implemented");
}

void BvhBuilder::computeMemUsage(const char* builderSpec, bool buildGpu, int numPrims, int motionSteps, InputType type, MemoryUsage* memUsage)
{
  // Default implementation does not handle motion steps
  RT_ASSERT_MSG( motionSteps == 1, "default computeMemUsage called with motionSteps > 1, not implemented" );

  if( type == IT_INSTANCE || type == IT_AABB ) // FIXME: Remove this once we have migrated all code to use this interface
    RT_ASSERT_MSG(0, "Not implemented");
  else
  {
    int numPrimAabbs = (type == IT_PRIMAABB) ? numPrims : 0;
    int numTriangles = (type == IT_TRI) ? numPrims : 0;
    computeMemUsage(builderSpec, buildGpu, numPrimAabbs, numTriangles, memUsage);
  }
}

void prodlib::bvhtools::BvhBuilder::copyHeader(void* dst, size_t dstSize)
{
  RT_ASSERT_MSG(0, "Not implemented");
}
