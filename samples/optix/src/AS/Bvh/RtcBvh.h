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

#include <AS/Builder.h>
#include <LWCA/Event.h>
#include <Memory/MBuffer.h>
#include <Memory/MBufferListener.h>
#include <rtcore/interface/types.h>

namespace optix {

enum RtcBvhAccelType
{
    RTC_BVH_ACCEL_TYPE_NOACCEL,
    RTC_BVH_ACCEL_TYPE_BVH2,
    RTC_BVH_ACCEL_TYPE_BVH8,
    RTC_BVH_ACCEL_TYPE_TTU
};

class RtcBvh : public Builder, public MBufferListener
{
  public:
    RtcBvh( Acceleration* accel, bool isGeometryGroup, RtcBvhAccelType type );
    ~RtcBvh();

    void setRtcAccelType( RtcBvhAccelType type );

    // Override Builder methods
    const char*  getName() const override { return "RtcBvh"; }
    unsigned int getVersion() const override { return 0; }

    void setProperty( const std::string& name, const std::string& value ) override;
    Program* getVisitProgram( bool hasMotionAabbs, bool bakePrimitiveEntities ) override;
    Program* getBoundingBoxProgram( bool hasMotionAabbs ) override;

    BuildSetupRequest setupForBuild( const BuildParameters&                   params,
                                     unsigned int                             totalPrims,
                                     const std::vector<GeometryInstanceData>& gidata ) override;
    void build( const BuildParameters& params, const BuildSetup& setup, const std::vector<GeometryInstanceData>& gidata ) override;
    void finalizeAfterBuildPrimitives( const BuildParameters& params );

    BuildSetupRequest setupForBuild( const BuildParameters&                   params,
                                     unsigned int                             totalPrims,
                                     const std::vector<GeometryInstanceData>& gidata,
                                     const std::vector<TriangleData>&         tridata ) override;
    void build( const BuildParameters&                   params,
                const BuildSetup&                        setup,
                const std::vector<GeometryInstanceData>& gidata,
                const std::vector<TriangleData>&         tridata ) override;
    void finalizeAfterBuildTriangles( const BuildParameters& params );

    BuildSetupRequest setupForBuild( const BuildParameters& params, const GroupData& groupdata ) override;
    void build( const BuildParameters& params, const BuildSetup& setup, const GroupData& td ) override;
    void finalizeAfterBuildGroups( const BuildParameters& params );

    RtcTraversableHandle getTraversableHandle( unsigned int allDeviceIndex ) const override;

    // From MBufferListener
    void eventMBufferMAccessDidChange( const MBuffer* buffer, const Device* device, const MAccess& oldMBA, const MAccess& newMBA ) override;
    // builders are assumed to be all RtcBvh
    static void copyMotionAabbs( const std::vector<Builder*>& builders, char* deviceOutputBuffer, LWDADevice* lwdaDevice );

  private:
    size_t setupBuffers( const BuildParameters&                            params,
                         const RtcAccelOptions&                            options,
                         const std::vector<RtcBuildInput>&                 buildInputs,
                         const std::vector<const RtcBuildInputOverrides*>* overrides = nullptr );
    void setAccelOptions( const BuildParameters& params, RtcAccelOptions& options, bool bakeTriangles );
    RtcTraversableHandle computeTraversableHandle( const MAccess& access ) const;
    void compactBuffers();
    void copyCompactedSize( RtcAccelBuffers& buffers, LWDADevice* lwdaDevice, bool& compactedSizeCopied );
    bool shouldCompact( const BuildParameters& params, const RtcAccelOptions& options );

    RtcTraversableType m_traversableType;
    RtcBvhAccelType    m_accelType;
    MBufferHandle      m_accelBuffer;
    size_t             m_bvhSize;
    unsigned int       m_motionSteps                 = 1;
    bool               m_allowCompaction             = true;
    size_t             m_compactedSize               = 0;
    bool               m_lastBuildSupportsRefit      = false;
    bool               m_lastBuildHasUniversalFormat = false;
    lwca::Event        m_event;
};

}  // namespace optix
