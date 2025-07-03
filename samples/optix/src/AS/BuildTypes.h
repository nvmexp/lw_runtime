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

#include <Memory/MBuffer.h>
#include <Objects/GeometryTriangles.h>
#include <rtcore/interface/rtcore.h>

namespace cort {
struct Aabb;
}

namespace optix {
class LWDADevice;

// Data used for configuring and building with geometry instances (one
// GeometryInstanceData per GeometryInstance).
struct GeometryInstanceData
{
    int primCount  = 0;  // Total number primitives
    int primOffset = 0;  // Offset added to each primitive index
    int primStart  = 0;  // Start index of primitives in array shared by all geometry instances
                         // under a geometry group (i.e. prefix sum)

    unsigned int giOffsetOrChildIndex = 0;  // Object record offset for geometry instance (baked) or the child index (not baked)
    unsigned int giOffset             = 0;  // Alway the object record offset, regardless of baking

    // Motion blur data, required for AABB resampling before BVH building, not needed anymore at time of building
    unsigned int motionSteps = 1;
    float        timeBegin   = 0;
    float        timeEnd     = 0;

    RTgeometryflags flags = RT_GEOMETRY_FLAG_NONE;  // Special flags used by rtcore bvh builds

    const Geometry* g = nullptr;

    const GeometryTriangles* getTriangles() const { return dynamic_cast<const GeometryTriangles*>( g ); }

    unsigned int getMaterialCount() const
    {
        const auto gt = getTriangles();
        // only geometry triangles can have multiple materials
        return gt ? gt->getMaterialCount() : 1;
    }
};

// Data used for configuring and building with triangles. One
// TriangleData per GeometryInstance in addition to GeometryInstance.
struct TriangleData
{
    MBufferHandle indices;                                   // Triangle indices (NULL for a flat triangle list)
    MBufferHandle vertices;                                  // Triangle vertices
    int           triIndicesStride   = 3 * sizeof( int );    // Stride in bytes between elements in indices buffer
    int           vertexStride       = 3 * sizeof( float );  // Stride in bytes between elements in vertices buffer
    RTformat      triIndicesFormat   = RT_FORMAT_UNSIGNED_INT3;  // rtcore supports: RT_FORMAT_UNSIGNED_INT3, RT_FORMAT_UNSIGNED_SHORT3; MK only supports RT_FORMAT_UNSIGNED_INT3
    RTformat      positionFormat     = RT_FORMAT_FLOAT3;  // rtcore supports: RT_FORMAT_FLOAT3, RT_FORMAT_HALF3, RT_FORMAT_FLOAT2, RT_FORMAT_HALF2; MK only supports RT_FORMAT_FLOAT3
    unsigned int  indexBufferOffset  = 0;                 // Offset in bytes
    unsigned int  vertexBufferOffset = 0;                 // Offset in bytes
};

// Data used for configuring and building a group. One per group.
struct GroupData
{
    int              childCount = 0;              // number of children in the group
    MBufferHandle    children;                    // Buffer containing object record addresses of children
    bool             bakeChildAddresses = false;  // Should we bake the child addresses into the primitive entities
    RTvisibilitymask mask               = RT_VISIBILITY_ALL;
    RTinstanceflags  flags              = RT_INSTANCE_FLAG_NONE;
};

struct BuildParameters
{
    // Parameters common to different builders
    bool buildGpu             = true;
    bool refitEnabled         = false;
    bool shouldRefitThisFrame = false;
    int  framesSinceLastBuild = 0;

    DeviceSet buildDevices;

    // Motion input info for builder
    unsigned motionSteps = 0;  // 1 means no motion bvh
    // The following data is not required for the builder, but
    //  used to prepare the build (AABB resampling) and needs to be passed
    //  to traversal in the RTX case
    float motionTimeBegin            = 0.0f;  // Motion time range
    float motionTimeEnd              = 0.0f;
    bool  motionBorderModeClampBegin = true;
    bool  motionBorderModeClampEnd   = true;
};

struct BuildSetupRequest
{
    bool         willRefit      = false;
    unsigned int totalPrims     = 0;
    unsigned int motionSteps    = 0;
    unsigned int motionStride   = 0;
    size_t       tempSize       = 0;
    bool         needAabbs      = false;
    bool         needAabbsOnCpu = false;

    BuildSetupRequest() {}

    BuildSetupRequest( bool willRefit, unsigned int totalPrims, unsigned int motionSteps, unsigned int motionStride, size_t tempSize, bool needAabbs, bool needAabbsOnCpu )
        : willRefit( willRefit )
        , totalPrims( totalPrims )
        , motionSteps( motionSteps )
        , motionStride( motionStride )
        , tempSize( tempSize )
        , needAabbs( needAabbs )
        , needAabbsOnCpu( needAabbsOnCpu )
    {
    }
};

struct BuildSetup
{
    struct PerDevice
    {
        size_t tempSize = 0;

        LWDADevice* buildDevice   = nullptr;
        char*       deviceTempPtr = nullptr;
        cort::Aabb* deviceAabbPtr = nullptr;

        Device*     cpuDevice  = nullptr;
        cort::Aabb* cpuAabbPtr = nullptr;

        PerDevice( size_t tempSize, LWDADevice* buildDevice, char* deviceTempPtr, cort::Aabb* deviceAabbPtr, Device* cpuDevice, cort::Aabb* cpuAabbPtr )
            : tempSize( tempSize )
            , buildDevice( buildDevice )
            , deviceTempPtr( deviceTempPtr )
            , deviceAabbPtr( deviceAabbPtr )
            , cpuDevice( cpuDevice )
            , cpuAabbPtr( cpuAabbPtr )
        {
        }
    };

    bool                   willRefit  = false;
    unsigned int           totalPrims = 0;
    std::vector<PerDevice> perDevice;

    BuildSetup( bool willRefit, unsigned int totalPrims, std::vector<PerDevice>& perDevice )
        : willRefit( willRefit )
        , totalPrims( totalPrims )
        , perDevice( perDevice )
    {
    }
};


}  // namespace optix
