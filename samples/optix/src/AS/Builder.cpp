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

#include <AS/Builder.h>

#include <Context/Context.h>
#include <Objects/Acceleration.h>


using namespace optix;


Builder::Builder( Acceleration* accel, bool isGeometryGroup )
    : m_acceleration( accel )
    , m_isGeometryGroup( isGeometryGroup )
{
}

Builder::~Builder() NOEXCEPT_FALSE
{
}

RtcTraversableHandle Builder::getTraversableHandle( unsigned int allDeviceIndex ) const
{
    RT_ASSERT_MSG( !m_acceleration->getContext()->useRtxDataModel(), "This method should only be required with RTX" );
    return 0;
}

BuildSetupRequest Builder::setupForBuild( const BuildParameters&                   params,
                                          unsigned int                             totalPrims,
                                          const std::vector<GeometryInstanceData>& gidata,
                                          const std::vector<TriangleData>&         tridata )
{
    return setupForBuild( params, totalPrims, gidata );
}

void Builder::build( const BuildParameters&                   params,
                     const BuildSetup&                        setup,
                     const std::vector<GeometryInstanceData>& gidata,
                     const std::vector<TriangleData>&         td )
{
    build( params, setup, gidata );
}

void Builder::finalizeAfterBuildTriangles( const BuildParameters& )
{
    RT_ASSERT_FAIL_MSG( "finalizeBuildAfterTriangles should not get ilwoked in this configuration" );
}

void Builder::overrideBvh_forTesting( const std::vector<BvhNode>& nodes, const std::vector<int>& indices )
{
    RT_ASSERT_FAIL_MSG( "Unsupported builder for overrideBvh" );
}
