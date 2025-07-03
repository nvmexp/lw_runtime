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

#pragma once

#include <AS/BuildTypes.h>
#include <Objects/Buffer.h>

#include <memory>
#include <string>
#include <vector>

namespace optix {

class Acceleration;
class Device;
class Program;
struct BvhNode;

class Builder
{
  public:
    Builder( Acceleration* accel, bool isGeometryGroup );
    virtual ~Builder() NOEXCEPT_FALSE;

    // Return the name of the builder.
    virtual const char* getName() const = 0;

    // Return the builder version. This should change at least every time
    // serialized data becomes incompatible.
    virtual unsigned int getVersion() const = 0;

    // Set builder properties.
    virtual void setProperty( const std::string& name, const std::string& value ) = 0;

    // Return the visit program
    virtual Program* getVisitProgram( bool hasMotionAabbs, bool bakePrimitiveEntities ) = 0;

    // Return the group bounding box program
    virtual Program* getBoundingBoxProgram( bool hasMotionAabbs ) = 0;

    // Build for AABB primitives (GeometryGroup)
    virtual BuildSetupRequest setupForBuild( const BuildParameters&                   params,
                                             unsigned int                             totalPrims,
                                             const std::vector<GeometryInstanceData>& gidata ) = 0;
    virtual void build( const BuildParameters& params, const BuildSetup& setup, const std::vector<GeometryInstanceData>& gidata ) = 0;
    virtual void finalizeAfterBuildPrimitives( const BuildParameters& ) = 0;

    // Build for triangle primitives. Forwards to AABB path if not implemented.
    virtual BuildSetupRequest setupForBuild( const BuildParameters&                   params,
                                             unsigned int                             totalPrims,
                                             const std::vector<GeometryInstanceData>& gidata,
                                             const std::vector<TriangleData>&         tridata );
    virtual void build( const BuildParameters&                   params,
                        const BuildSetup&                        setup,
                        const std::vector<GeometryInstanceData>& gidata,
                        const std::vector<TriangleData>&         tridata );
    virtual void finalizeAfterBuildTriangles( const BuildParameters& );

    // Build for Groups
    virtual BuildSetupRequest setupForBuild( const BuildParameters& params, const GroupData& groupdata ) = 0;
    virtual void build( const BuildParameters& params, const BuildSetup& setup, const GroupData& groupdata ) = 0;
    virtual void finalizeAfterBuildGroups( const BuildParameters& ) = 0;

    // Return the traversable handle if it is valid, otherwise return 0
    virtual RtcTraversableHandle getTraversableHandle( unsigned int allDeviceIndex ) const;

    // Used by testing to manipulate the Bvh.
    virtual void overrideBvh_forTesting( const std::vector<BvhNode>& nodes, const std::vector<int>& indices );

  protected:
    Acceleration* m_acceleration    = nullptr;  // acceleration object the builder belongs to
    bool          m_isGeometryGroup = false;
};

}  // namespace optix
