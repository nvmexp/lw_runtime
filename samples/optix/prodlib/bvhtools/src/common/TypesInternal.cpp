// Copyright LWPU Corporation 2017
// TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED
// *AS IS* AND LWPU AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS
// OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY
// AND FITNESS FOR A PARTICULAR PURPOSE.  IN NO EVENT SHALL LWPU OR ITS SUPPLIERS
// BE LIABLE FOR ANY SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES
// WHATSOEVER (INCLUDING, WITHOUT LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS,
// BUSINESS INTERRUPTION, LOSS OF BUSINESS INFORMATION, OR ANY OTHER PELWNIARY LOSS)
// ARISING OUT OF THE USE OF OR INABILITY TO USE THIS SOFTWARE, EVEN IF LWPU HAS
// BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES

#include "TypesInternal.hpp"
#include <prodlib/exceptions/IlwalidValue.h>

#include <algorithm>

namespace prodlib
{
namespace bvhtools
{

#ifndef __LWDACC__

#define DUMMY_BUFFER_POINTER    0x0BADBAD0

InputBuffers::InputBuffers(const std::vector<TriangleMesh>& meshes, MemorySpace memSpace)
{
    RT_ASSERT_MSG( !meshes.empty(), "Need at least one mesh when computing InputBuffers" );
    clear();
    inputType                = IT_TRI;
    numPrimitives            = 0;
    maxPrimsInSingleGeometry = 0;

    numInitialArrays = static_cast<int>( meshes.size() );
    trianglesDescArray.setNumElems( numInitialArrays ).materialize().writeDiscardHost();
    trianglesPtrArray.setNumElems( numInitialArrays ).materialize().writeDiscardHost();
    numPrimsArray.setNumElems( numInitialArrays ).materialize().writeDiscardHost();
    geometryIndexArray.setNumElems( numInitialArrays ).materialize().writeDiscardHost();

    numArrays = 0;
    for( const TriangleMesh& mesh : meshes )
    {
        numPrimitives += mesh.numTriangles;
        maxPrimsInSingleGeometry = std::max( static_cast<int>( mesh.numTriangles ), maxPrimsInSingleGeometry );
        trianglesDescArray[numArrays] =
            InputTrianglesDesc( (const char*)mesh.vertices, mesh.numVertices, mesh.indices == NULL ? 0 : numPrimitives * 3,
                                mesh.vertexStride, 3, sizeof( float ), mesh.indexStride,
                                mesh.indices == NULL ? 0 : sizeof( int ), InputTrianglesDesc::Topology_List, 0 );
        InputTrianglesPointers& inputTrisPtrs = trianglesPtrArray[numArrays];
        inputTrisPtrs.vertices                = (const char*)mesh.vertices;
        inputTrisPtrs.indices                 = (const char*)mesh.indices;
        inputTrisPtrs.transform               = NULL;

        geometryIndexArray[numArrays] = 0;
        numPrimsArray[numArrays]      = mesh.numTriangles;

        if( numArrays > 1 )
            if( mesh.numTriangles && ( mesh.triUserIds || mesh.triAlphaFlags ) )
                throw AssertionFailure( RT_EXCEPTION_INFO, "Cannot support primUserIds or triAlphaFlags with multi-mesh input" ); 

        numArrays++;
    }

    if( meshes.size() == 1 )
    {
        const TriangleMesh& mesh = meshes.back();
        if( mesh.numTriangles && mesh.triUserIds )
        {
            primUserIdStride      = mesh.triUserIdStride;
            size_t triUserIdBytes = ( mesh.numTriangles - 1 ) * (size_t)mesh.triUserIdStride + mesh.triUseIdSizeInBytes;
            primUserIds.assignExternal( (unsigned char*)mesh.triUserIds, triUserIdBytes, memSpace );
        }

        primAlphaDefaultFlags = mesh.triAlphaDefaultFlags;
        if( mesh.numTriangles && mesh.triAlphaFlags )
        {
            primFlagStride           = mesh.triAlphaFlagStride;
            size_t triAlphaFlagBytes = ( mesh.numTriangles - 1 ) * (size_t)mesh.triAlphaFlagStride + sizeof( unsigned char );
            primFlags.assignExternal( mesh.triAlphaFlags, triAlphaFlagBytes / sizeof( unsigned char ), memSpace );
            primFlagShift   = mesh.triAlphaFlagShift;
            primFlagIlwerse = mesh.triAlphaFlagIlwerse;
        }
    }
}

InputBuffers::InputBuffers(int numAABBs, const PrimitiveAABB* aabbPtr, MemorySpace memSpace)
{
    inputType = IT_PRIMAABB;
    clear();
    numPrimitives = numAABBs;
    primAabbs.assignExternal(aabbPtr, numAABBs, memSpace);
}

InputBuffers::InputBuffers(int numAABBs, int aabbMotionSteps, const AABB* aabbPtr, MemorySpace memSpace)
{
    inputType = IT_AABB;
    clear();
    numPrimitives = numAABBs;
    motionSteps = aabbMotionSteps;
    maxPrimsInSingleGeometry = numAABBs;
    numInitialArrays = 1;
    numArrays = 1;
    aabbsArray.setNumElems(1).materialize().writeDiscardHost();
    numPrimsArray.setNumElems(1).materialize().writeDiscardHost();
    geometryIndexArray.setNumElems(1).materialize().writeDiscardHost();

    maxPrimsInSingleGeometry = 0;
    InputAABBs &inputAabbs = aabbsArray[0];
    numPrimsArray[0] = numAABBs;
    inputAabbs.aabbs = (const float*)aabbPtr;
    inputAabbs.strideInBytes = sizeof(AABB);
    inputAabbs.flags = 0;
}

InputBuffers::InputBuffers(int numInstances, const InstanceDesc* instanceDescPtr, MemorySpace memSpace)
{
    inputType = IT_INSTANCE;
    clear();
    numPrimitives = numInstances;
    instanceDescs.assignExternal(instanceDescPtr, numInstances, memSpace);
}

// 'canUseDummyPointers' is true if this struct is part of a configuration run to determine the memory requirements for the AS.
// Dummy buffers may then be used as placeholders for real data buffers.
InputBuffers::InputBuffers(int numInputs, const RtcBuildInput* buildInputs, MemorySpace memSpace, bool canUseDummyPointers)
{
    if( buildInputs->type == RTC_BUILD_INPUT_TYPE_TRIANGLE_ARRAY || buildInputs->type == RTC_BUILD_INPUT_TYPE_AABB_ARRAY )
    {
        unsigned int size = (numInputs + 31) / 32;
        unsigned int* flags = opaqueFlagsArray.setNumElems(size).materialize().writeDiscardHost();
        memset(flags, 0, size * sizeof(unsigned int));
    }

    // The entire array must be of the same input type. TODO: Verify or do the check somewhere else?
    if( buildInputs->type == RTC_BUILD_INPUT_TYPE_TRIANGLE_ARRAY )
    {
        inputType = IT_TRI;
        clear();
        numPrimitives = 0;
        numInitialArrays = numInputs;
        trianglesDescArray.setNumElems(numInputs).materialize().writeDiscardHost();
        trianglesPtrArray.setNumElems(numInputs).materialize().writeDiscardHost();
        numPrimsArray.setNumElems(numInputs).materialize().writeDiscardHost();
        geometryIndexArray.setNumElems(numInputs).materialize().writeDiscardHost();

        maxPrimsInSingleGeometry = 0;
        numArrays = 0;

        for (int i = 0; i < numInputs; i++)
        {
            RT_ASSERT( buildInputs[i].type == RTC_BUILD_INPUT_TYPE_TRIANGLE_ARRAY );
            const RtcBuildInputTriangleArray& triArray = buildInputs[i].triangleArray;

            // Check if this geometry is empty
            if ( (triArray.indexSizeInBytes == 0 && triArray.numVertices == 0) || (triArray.indexSizeInBytes != 0 && triArray.numIndices == 0) )
                continue;

            // Setup pointers to buffer resources
            InputTrianglesPointers &inputTrisPtrs = trianglesPtrArray[numArrays];
            if (canUseDummyPointers)
            {
                inputTrisPtrs.vertices = (const char *) DUMMY_BUFFER_POINTER;
                inputTrisPtrs.indices = triArray.indexSizeInBytes != 0 ? (const char *) DUMMY_BUFFER_POINTER : NULL;
            }
            else
            {
                inputTrisPtrs.vertices = (const char *) triArray.vertexBuffer;
                inputTrisPtrs.indices = (const char *) triArray.indexBuffer;
            }
            inputTrisPtrs.transform = (const float *) triArray.transform;

            // Setup triangle descriptor
            trianglesDescArray[numArrays] = InputTrianglesDesc(
                inputTrisPtrs.vertices,
                triArray.numVertices,
                triArray.numIndices,
                triArray.vertexStrideInBytes,
                triArray.vertexFormat == RTC_VERTEX_FORMAT_FLOAT3 || triArray.vertexFormat == RTC_VERTEX_FORMAT_HALF3  ? 3 : 2,
                triArray.vertexFormat == RTC_VERTEX_FORMAT_FLOAT3 || triArray.vertexFormat == RTC_VERTEX_FORMAT_FLOAT2 ? 4 : 2,
                triArray.indexStrideInBytes,
                triArray.indexSizeInBytes,
                InputTrianglesDesc::Topology_List,
                (triArray.flags & RTC_GEOMETRY_FLAG_OPAQUE) != 0
            );

            unsigned int numTris = 0;
            numTris = triArray.indexSizeInBytes != 0 ? triArray.numIndices / 3 : triArray.numVertices / 3;
            numPrimsArray[numArrays] = numTris;
            numPrimitives += numTris;

            maxPrimsInSingleGeometry = max( maxPrimsInSingleGeometry, numTris );

            opaqueFlagsArray[numArrays / 32] |=  ( ( triArray.flags & RTC_GEOMETRY_FLAG_OPAQUE ) ? 1 : 0 ) << ( numArrays & 31 );

            geometryIndexArray[numArrays] = i;
            numArrays++;
        }

        // Handle the case where there are no non-empty inputs

        if (numArrays == 0)
        {
            InputTrianglesPointers &inputTrisPtrs = trianglesPtrArray[0];
            inputTrisPtrs.vertices = (const char *) DUMMY_BUFFER_POINTER;
            inputTrisPtrs.indices = NULL;
            inputTrisPtrs.transform = NULL;
            trianglesDescArray[numArrays] = InputTrianglesDesc(inputTrisPtrs.vertices, 0, 0, sizeof(float3), 3, 4, 0, 0, InputTrianglesDesc::Topology_List, 0);
            numPrimsArray[numArrays] = 0;
            maxPrimsInSingleGeometry = 0;
            geometryIndexArray[0] = 0;
            numArrays = 1;
        }

    }
    else if( buildInputs->type == RTC_BUILD_INPUT_TYPE_AABB_ARRAY )
    {
        inputType = IT_AABB;
        clear();
        numPrimitives = 0;
        numInitialArrays = numInputs;
        aabbsArray.setNumElems(numInputs).materialize().writeDiscardHost();
        numPrimsArray.setNumElems(numInputs).materialize().writeDiscardHost();
        geometryIndexArray.setNumElems(numInputs).materialize().writeDiscardHost();

        maxPrimsInSingleGeometry = 0;
        numArrays = 0;

        for (int i = 0; i < numInputs; i++)
        {
            RT_ASSERT( buildInputs[i].type == RTC_BUILD_INPUT_TYPE_AABB_ARRAY );
            const RtcBuildInputAabbArray& aabbItem = buildInputs[i].aabbArray;

            if (aabbItem.numAabbs == 0)
                continue;

            InputAABBs &inputAabbs = aabbsArray[numArrays];
            numPrimsArray[numArrays] = aabbItem.numAabbs;
            numPrimitives += aabbItem.numAabbs;
            if (canUseDummyPointers)
                inputAabbs.aabbs = (const float*) DUMMY_BUFFER_POINTER;
            else
                inputAabbs.aabbs = (const float*) aabbItem.aabbBuffer;
            inputAabbs.strideInBytes = aabbItem.strideInBytes;
            inputAabbs.flags = 0;
            inputAabbs.flags |= (aabbItem.flags & RTC_GEOMETRY_FLAG_OPAQUE) ? (uint32_t) InputAABBs::Flags_Opaque : 0;

            maxPrimsInSingleGeometry = max( maxPrimsInSingleGeometry, aabbItem.numAabbs );

            geometryIndexArray[numArrays] = i;
            numArrays++;
        }

    }
    else if( buildInputs->type == RTC_BUILD_INPUT_TYPE_INSTANCE_ARRAY )
    {
        RT_ASSERT(numInputs == 1);
        const RtcBuildInputInstanceArray& instItem = buildInputs->instanceArray;
        inputType = IT_INSTANCE;
        clear();
        numPrimitives = instItem.numInstances;
        numInitialArrays = 1;
        if( canUseDummyPointers )
            instanceDescs.assignExternal( (const prodlib::bvhtools::InstanceDesc*) DUMMY_BUFFER_POINTER, numPrimitives, memSpace );
        else
            instanceDescs.assignExternal( (const prodlib::bvhtools::InstanceDesc*) instItem.instanceDescs, numPrimitives, memSpace );
    }
    else
        RT_ASSERT_MSG(0, "Unsupported input");
}

bool InputBuffers::needsGeometryIndexRemap() const
{
    return numInitialArrays != numArrays;
}

bool InputBuffers::isAABBs(void) const
{
    return inputType == IT_AABB || inputType == IT_PRIMAABB || inputType == IT_INSTANCE;
}

void InputBuffers::clear(void)
{
    numInitialArrays = 0;
    numArrays = 0;
    numPrimitives = 0;
    motionSteps = 1;

    primAabbs = EmptyBuf;

    instanceDescs = EmptyBuf;

    primUserIds = EmptyBuf;
    primFlags = EmptyBuf;
    instMatrices = EmptyBuf;
    instRoots = EmptyBuf;

    trianglesDescArray = EmptyBuf;
    trianglesPtrArray = EmptyBuf;
    aabbsArray = EmptyBuf;
    numPrimsArray = EmptyBuf;
}

void InputBuffers::materialize(LwdaUtils* lwdaUtils) const
{
    primAabbs.materialize(lwdaUtils);
    instanceDescs.materialize(lwdaUtils);
    primUserIds.materialize(lwdaUtils);
    primFlags.materialize(lwdaUtils);
    instMatrices.materialize(lwdaUtils);
    instRoots.materialize(lwdaUtils);
}

void InputBuffers::freeMem(void) const
{
    primAabbs.freeMem();
    instanceDescs.freeMem();
    primUserIds.freeMem();
    primFlags.freeMem();
    instMatrices.freeMem();
    instRoots.freeMem();
}

#endif // __LWDACC__

} // namespace bvhtools
} // namespace prodlib
