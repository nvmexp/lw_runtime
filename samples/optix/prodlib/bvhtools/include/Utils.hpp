// Copyright LWPU Corporation 2014
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

//struct float3;
//struct float4;

namespace prodlib
{
namespace bvhtools
{

struct ApexPointMap;
struct PrimitiveAABB;
struct TriangleMesh;

void computeInstanceAabbsAndIlwMatrices( 
    bool            useLwda,
    int             numInstances, 
    int             numModels, 
    int             matrixStride,
    int*            instance_ids, 
    float4*         transforms, 
    ApexPointMap**  hostApexPointMaps,
    PrimitiveAABB*  outWorldSpaceAabbs, 
    float4*         outIlwMatrices );

// Pointers must be on the device.
void createAccelInstanceData(
  int                  numInstances,
  const InstanceDesc*  inInstanceDescs,
  BvhInstanceData*     outInstanceData );

void computeAabb(                                                                
    bool                useLwda,                                                     
    const TriangleMesh& mesh,                                            
    float               result[6] );

// computes the ApexPointMap for the given AABB data (min.x,y,z, max.x,y,z)
// If the resultData isn't large enough for some reason, an error is triggered. 
void computeApexPointMap(
    float aabb[6],
    int resultData[8] );


} // namespace bvhtools
} // namespace prodlib

