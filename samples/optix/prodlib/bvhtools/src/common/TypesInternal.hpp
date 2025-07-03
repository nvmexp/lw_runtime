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

#pragma once

#include "../../include/Types.hpp"
#include "RemapListLenEncoding.hpp"
#include "Intrinsics.hpp"
#include <vector>

#ifndef __LWDACC__
#   include "BufferRef.hpp"
#   include <string.h>
#   include <rtcore/interface/rtcore.h>
#endif

#ifdef __LWDACC__
#   include <lwda_fp16.h>
#endif

// TODO: Experiment with different block sizes. Make this templated?
// Must be 2^n >= 32
#define INPUT_ARRAY_INDEXER_BLOCK_SIZE   128

namespace prodlib
{
namespace bvhtools
{
//------------------------------------------------------------------------
// Generic axis-aligned bounding box type.

struct AABB
{
    float3                  lo;                 // Min corner.
    float3                  hi;                 // Max corner.

    INLINE                  AABB                (void)                                      {}
    INLINE                  AABB                (float3 lo, float3 hi)                      : lo(lo), hi(hi) {}
    INLINE                  AABB                (float lo, float hi)                        : lo(make_float3(lo, lo, lo)), hi(make_float3(hi, hi, hi)) {}
    INLINE                  AABB                (PrimitiveAABB v)                           : lo(make_float3(v.lox, v.loy, v.loz)), hi(make_float3(v.hix, v.hiy, v.hiz)) {}
    INLINE                  AABB                (AABB a, AABB b)                            { lo.x = fminf(a.lo.x, b.lo.x); lo.y = fminf(a.lo.y, b.lo.y); lo.z = fminf(a.lo.z, b.lo.z); hi.x = fmaxf(a.hi.x, b.hi.x); hi.y = fmaxf(a.hi.y, b.hi.y); hi.z = fmaxf(a.hi.z, b.hi.z); }
    INLINE                  AABB                (float3 v0, float3 v1, float3 v2)           { lo.x = fminf(fminf(v0.x, v1.x), v2.x); lo.y = fminf(fminf(v0.y, v1.y), v2.y); lo.z = fminf(fminf(v0.z, v1.z), v2.z); hi.x = fmaxf(fmaxf(v0.x, v1.x), v2.x); hi.y = fmaxf(fmaxf(v0.y, v1.y), v2.y); hi.z = fmaxf(fmaxf(v0.z, v1.z), v2.z); }

    INLINE void             include             (AABB a)                                    { *this = AABB(*this, a); }
    INLINE bool             isValid             (void) const                                { return lo.x <= hi.x && lo.y <= hi.y && lo.z <= hi.z; }
    INLINE bool             contains            (const AABB& a) const                       { return lo.x <= a.lo.x && lo.y <= a.lo.y && lo.z <= a.lo.z && hi.x >= a.hi.x && hi.y >= a.hi.y && hi.z >= a.hi.z; }

    INLINE float3           getCenter           (void) const                                { return make_float3((lo.x + hi.x) * 0.5f, (lo.y + hi.y) * 0.5f, (lo.z + hi.z) * 0.5f); }
    INLINE float3           getSize             (void) const                                { return make_float3(hi.x - lo.x, hi.y - lo.y, hi.z - lo.z); }
    INLINE float            getSizeMaxRcp       (void) const                                { float3 s = getSize(); return 1.0f / fmaxf(fmaxf(s.x, s.y), s.z); }
    INLINE float            getHalfArea         (void) const                                { float3 s = getSize(); return s.x * s.y + s.y * s.z + s.z * s.x; }
    INLINE float            getArea             (void) const                                { return getHalfArea() * 2.0f; }

    // Transform the input isotropically so that every point within this AABB fits in the range [0,1].

    static INLINE float3    transformRelative   (float3 v, float3 center, float sizeMaxRcp) { return make_float3((v.x - center.x) * sizeMaxRcp + 0.5f, (v.y - center.y) * sizeMaxRcp + 0.5f, (v.z - center.z) * sizeMaxRcp + 0.5f); }
    INLINE float3           transformRelative   (float3 v) const                            { return transformRelative(v, getCenter(), getSizeMaxRcp()); }
    INLINE AABB             transformRelative   (AABB v) const                              { float3 center = getCenter(); float sizeMaxRcp = getSizeMaxRcp(); return AABB(transformRelative(v.lo, center, sizeMaxRcp), transformRelative(v.hi, center, sizeMaxRcp)); }
};

//------------------------------------------------------------------------

struct Range
{
#ifndef __LWDACC__
  Range() 
  {
    memset( this, 0, sizeof(Range) );
  }
#endif
  __host__ __device__
  Range( int s, int e )
  {
    start = s;
      end = e;
  }
  
  int start;
  int end;

  __host__ __device__
  int span() const { return end - start; }
};


//------------------------------------------------------------------------
// Input model for the BVH builder.

// 20 B
// Compact representation of the format and layout of a triangle geometry.
// This may not change between refits calls and can therefore be retained on device.
struct InputTrianglesDesc
{
public:
    // The type and dimensionality of the input vertices. 
    enum VertexFormat
    {
        VertexFormat_AlignedFloat4 = 0,  // 16-byte stride and alignment
        VertexFormat_Float3,
        VertexFormat_Float2,
        VertexFormat_Half3,
        VertexFormat_Half2
    };

    // Bytes per index. May be 2 (ushort) or 4 (uint), if using indices. _Must_ be 0 otherwise.
    enum IndexByteSize
    {
        IndexByteSize_0 = 0, // No index buffer
        IndexByteSize_2,
        IndexByteSize_4
    };

    // Topology: either triangle list or triangle strips.
    enum Topology
    {
        Topology_List = 0,
        Topology_Strip
    };

    // Flags.
    // Force opaque: This overrides the alpha/opaque flag of individual primitives.
    enum Flags
    {
        Flags_Opaque = 1
    };

private:
    uint32_t  m_numVertices;
    uint32_t  m_numIndices; // TODO: This is only accessible through prime. Would be nice to get rid of it, or limit to 16 bits, and have a 16-byte aligned struct
    uint32_t  m_vertexStride;
    uint32_t  m_indexStride;
    uint32_t  m_packedVertexFormat    : 3;
    uint32_t  m_packedIndexStride     : 1;
    uint32_t  m_packedIndexByteSize   : 2;
    uint32_t  m_packedTopology        : 1;
    uint32_t  m_packedFlags           : 1;

public:

#ifndef __LWDACC__
    InputTrianglesDesc() {}
    InputTrianglesDesc(
        const char* vertexBuffer,
        uint32_t    numVertices,
        uint32_t    numIndices,
        uint32_t    vertexStride,
        uint32_t    vertexDim,
        uint32_t    vertexByteSize,
        uint32_t    indexStrideInBytes,
        uint32_t    indexByteSize,
        uint32_t    topology,
        bool        isOpaque)
    {
        m_numVertices = numVertices;
        m_numIndices = numIndices;
        m_vertexStride = vertexStride;

        m_indexStride = indexStrideInBytes;
        if (indexStrideInBytes == 0 && indexByteSize != 0)
            m_indexStride = indexByteSize == 2 ? sizeof(short) * 3 : sizeof(int) * 3;

        if (vertexDim == 3)
        {
            if (vertexByteSize == 4)
            {
                if (vertexStride == 4*sizeof(float) && ((uintptr_t)vertexBuffer & 15) == 0) // float4 with proper alignment - fast loads possible
                    m_packedVertexFormat = (uint32_t) VertexFormat_AlignedFloat4;
                else
                    m_packedVertexFormat = (uint32_t) VertexFormat_Float3;
            }
            else if (vertexByteSize == 2)
                m_packedVertexFormat = (uint32_t) VertexFormat_Half3;
            else
                RT_ASSERT_MSG(false, "Unsupported vertex byte size. Must be either 2 (half) or 4 (float)");
        }
        else if (vertexDim == 2)
        {
            if (vertexByteSize == 4)
              m_packedVertexFormat = (uint32_t) VertexFormat_Float2;
            else if (vertexByteSize == 2)
              m_packedVertexFormat = (uint32_t) VertexFormat_Half2;
            else
                RT_ASSERT_MSG(false, "Unsupported vertex byte size. Must be either 2 (half) or 4 (float)");
        }
        else
          RT_ASSERT_MSG(false, "Unsupported vertex dimensionality. Must be either 2 or 3");

        if( indexByteSize == 0)
          m_packedIndexByteSize = (uint32_t) IndexByteSize_0;
        else if( indexByteSize == 2)
          m_packedIndexByteSize = (uint32_t) IndexByteSize_2;
        else if( indexByteSize == 4)
          m_packedIndexByteSize = (uint32_t) IndexByteSize_4;
        else
          RT_ASSERT_MSG(false, "Unsupported index byte size. Must be 0, 2 or 4 bytes");

        m_packedTopology = topology;
        m_packedFlags = 0;
        m_packedFlags |= isOpaque ? (uint32_t) Flags_Opaque : 0;
    }
#endif

    INLINE __device__ __host__ uint32_t getVertexStride(void) const { return m_vertexStride; }
    INLINE __device__ __host__ uint32_t getIndexStride(void) const  { return m_indexStride; }
    INLINE __device__ __host__ uint32_t getNumVertices(void) const  { return m_numVertices; }
    INLINE __device__ __host__ uint32_t getNumIndices(void) const   { return m_numIndices; }
    
    INLINE __device__ __host__ VertexFormat getVertexFormat(void) const
    {
      return (VertexFormat) m_packedVertexFormat;
    }
    INLINE __device__ __host__ uint32_t getVertexDim(void) const
    {
        if (m_packedVertexFormat == (uint32_t) VertexFormat_Float2 || m_packedVertexFormat == (uint32_t) VertexFormat_Half2)
            return 2;
        return 3;
    }
    INLINE __device__ __host__ uint32_t getVertexByteSize(void) const
    {
        if (m_packedVertexFormat == (uint32_t) VertexFormat_Half2 || m_packedVertexFormat == (uint32_t) VertexFormat_Half3)
            return 2;
        return 4;
    }

    // Only valid for 4-byte indices
    INLINE __device__ __host__ uint32_t getIndexByteSize(void) const
    {
        if (m_packedIndexByteSize == (uint32_t) IndexByteSize_0)
          return 0;
        else if (m_packedIndexByteSize == (uint32_t) IndexByteSize_2)
          return 2;
        // indexStride == (uint32_t) IndexByteSize_4
        return 4;
    }

    INLINE __device__ __host__ bool isOpaque(void) const
    {
      return (m_packedFlags & (uint32_t) Flags_Opaque) != 0;
    }

    INLINE __device__ __host__ uint32_t hasIndexBuffer(void) const
    {
      return m_packedIndexByteSize != (uint32_t) IndexByteSize_0;
    }

};

// 24 B
// Triangle geometry pointers to data buffers.
// Uploaded to device whenever a build or refit is performed.
struct InputTrianglesPointers
{
    const char*   vertices;
    const char*   indices;        // Optional. Contents may not change for refits. TODO: Put in a separate buffer?
    const float*  transform;      // Optional. TODO: Put in a separate buffer?
};

// 16 B
// Uploaded on build/refit.
// Minor note: stride and flags could remain on device.
struct InputAABBs
{
    // Flags.
    // Force opaque: This overrides the alpha/opaque flag of individual primitives.
    enum Flags
    {
        Flags_Opaque = 1
    };

    const float*    aabbs;
    uint32_t        strideInBytes;
    uint32_t        flags : 1; // TODO: Steal a bit from strideInBytes?

    __device__ __host__ bool isOpaque() const
    {
      return (flags & (uint32_t) Flags_Opaque) != 0;
    }
};


#ifndef __LWDACC__

// TODO: This structure is very similar to ModelBuffers and perhaps could be merged with it.
struct InputBuffers
{
  InputType                       inputType;
  int                             numPrimitives;      // Total number of primitives
  int                             motionSteps=1;      // Motion steps per AABB (1 means static)
  
  BufferRef<const PrimitiveAABB>  primAabbs;          // Abstract primitive AABBs + primitive index. 

  BufferRef<const InstanceDesc>   instanceDescs;      // Instance descriptors


  BufferRef<>                         primUserIds;
  int                                 primUserIdStride = sizeof( int );
  BufferRef<const unsigned char>      primFlags;
  int                                 primFlagStride = sizeof( unsigned char );
  int                                 primFlagShift = 0;
  int                                 primFlagIlwerse = 0;
  unsigned char                       primAlphaDefaultFlags = 0;
  BufferRef<const float>              instMatrices;
  int                                 instMatrixStride = sizeof( float );
  bool                                instMatrixWorldToObject = true;
  BufferRef<const unsigned long long> instRoots;
  int                                 instRootStride = sizeof( unsigned long long );
  int                                 instRootOffset = 0;

  // Array inputs

  int                                 numInitialArrays = 0;           // Incoming input count (incl. empty inputs)
  int                                 numArrays = 0;                  // Number of arrays after empty inputs are removed
  int                                 maxPrimsInSingleGeometry = 0;   // Maximum number of primitives in any one geometry

  BufferRef<InputAABBs>               aabbsArray;           // AABB geometry
  BufferRef<InputTrianglesDesc>       trianglesDescArray;   // Triangle geometry format and layout
  BufferRef<InputTrianglesPointers>   trianglesPtrArray;    // Triangle geometry pointers
  BufferRef<unsigned int>             numPrimsArray;        // Number of primitives for each geometry/instance array
  BufferRef<unsigned int>             opaqueFlagsArray;     // Opaque flag for tris and AABBs. 1 bit per item

  BufferRef<int>                      geometryIndexArray;   // Maps index in aabbs/trianglesDesc/trianglesPtrArray to a geometryIndex.
                                                            // Non-null only if there are empty inputs.

  InputBuffers(const std::vector<TriangleMesh>& meshes, MemorySpace memSpace);
  InputBuffers(int numAABBs, const PrimitiveAABB* aabbPtr, MemorySpace memSpace);
  InputBuffers(int numAABBs, int motionSteps, const AABB* aabbPtr, MemorySpace memSpace);
  InputBuffers(int numInstances, const InstanceDesc* instanceDescPtr, MemorySpace memSpace);

  // 'canUseDummyPointers' is true if this struct is part of a configuration run to determine the memory requirements for the AS.
  // Dummy buffers may then be used as placeholders for real data buffers.
  InputBuffers(int numInputs, const RtcBuildInput* buildInputs, MemorySpace memSpace, bool canUseDummyPointers = false);

  bool needsGeometryIndexRemap() const;
  bool isAABBs(void) const;
  void clear(void);
  void materialize(LwdaUtils* lwdaUtils) const;
  void freeMem(void) const;
};

//------------------------------------------------------------------------
// Buffers mapping global primitive indices to input array/primitive indices

struct InputArrayIndexBuffers
{
    BufferRef<unsigned int>         arrayBaseGlobalIndex; // Global start index of each input array
    BufferRef<unsigned int>         arrayTransitionBits;  // Treating the buffer as an array of bits, a bit is set for each entry in arrayBaseGlobalIndex
    BufferRef<int>                  blockStartArrayIndex; // Input array index at start of each 128-bit block
    BufferRef<int>                  geometryIndexArray;   // Only needed if there were empty input geometries
};

struct ModelBuffers
{
    InputType                       inputType;

    int                             numPrimitives;        // Total number of primitives (AABBs or triangles).
    int                             indexStride;          // Bytes between conselwtive items in 'indices'.
    int                             vertexStride;         // Bytes between conselwtive items in 'vertices'.

    BufferRef<const PrimitiveAABB>  aabbs;                // Abstract primitive AABBs. EmptyBuf => primitives are triangles.
    BufferRef<const char>           indices;              // Vertex index triplets. EmptyBuf => each triangle is defined by 3 conselwtive vertices.
    BufferRef<const float>          vertices;             // Vertex position triplets.
    BufferRef<const PrimitiveAABB>  splitAABBs;           // Additional primitive AABBs produced by triangle splitting.

    InputArrayIndexBuffers          inputArrayIndexing;   // Global start index of each input array

    ModelBuffers(void)
    {
        clear();
    }

    ModelBuffers(const TriangleMesh& mesh, MemorySpace memSpace)
    {
        size_t indexBytes = (mesh.numTriangles && mesh.indices) ? (mesh.numTriangles - 1) * (size_t)mesh.indexStride + sizeof(int3) : 0;
        size_t vertexBytes = (mesh.numTriangles && mesh.numVertices && mesh.vertices) ? (mesh.numVertices - 1) * (size_t)mesh.vertexStride + sizeof(float3) : 0;
        clear();
        numPrimitives   = mesh.numTriangles;
        indexStride     = mesh.indexStride;
        vertexStride    = mesh.vertexStride;
        indices         .assignExternal((const char*) mesh.indices, indexBytes, memSpace);
        vertices        .assignExternal((const float*) mesh.vertices, vertexBytes / sizeof(float), memSpace);
    }

    ModelBuffers(const Group& group, MemorySpace memSpace)
    {
        clear();
        numPrimitives = group.numInstances;
        aabbs.assignExternal(group.aabbs, group.numInstances, memSpace);
    }

    ModelBuffers(int numAABBs, const PrimitiveAABB* aabbPtr, MemorySpace memSpace)
    {
        clear();
        numPrimitives = numAABBs;
        aabbs.assignExternal(aabbPtr, numAABBs, memSpace);
    }

    ModelBuffers(BufferRef<const PrimitiveAABB> aabbBuf)
    {
        clear();
        numPrimitives = (int)aabbBuf.getNumElems();
        aabbs = aabbBuf;
    }

    ModelBuffers(const ModelBuffers& other, BufferRef<const PrimitiveAABB> newSplitAABBs)
    {
        *this = other;
        splitAABBs = newSplitAABBs;
    }

    bool isAABBs(void) const
    {
        return inputType == IT_AABB || inputType == IT_PRIMAABB || inputType == IT_INSTANCE;
    }

    bool isValid(void) const
    {
        // Check constants.

        if (numPrimitives   < 0 ||
            indexStride     < 0 || indexStride  % sizeof(int)   != 0 ||
            vertexStride    < 0 || vertexStride % sizeof(float) != 0)
        {
            return false;
        }

        // No primitives => all buffers must be empty.

        if (numPrimitives == 0)
            return (aabbs.getNumElems() == 0 && indices.getNumBytes() == 0 && vertices.getNumBytes() == 0);

        // AABBs => must have one AABB per primitive and no indices/vertices.

        if (aabbs.getNumElems() != 0)
            return (aabbs.getNumElems() >= (size_t)numPrimitives && indices.getNumBytes() == 0 && vertices.getNumBytes() == 0);

        // Must have at least one vertex.

        if ((size_t)vertexStride < sizeof(float3) || vertices.getNumBytes() < sizeof(float3))
            return false;

        // Indexed triangles => must have one index triplet per primitive.

        if (indices.getNumElems() != 0)
            return ((size_t)indexStride >= sizeof(int3) &&  indices.getNumBytes() >= (numPrimitives - 1) * (size_t)indexStride + sizeof(int3));

        // Non-indexed triangles => must have 3 vertex triplets per primitive.

        return (vertices.getNumBytes() >= ((size_t)numPrimitives * 3 - 1) * (size_t)vertexStride + sizeof(float3));
    }

    void clear(void)
    {
        numPrimitives   = 0;
        indexStride     = 0;
        vertexStride    = 0;
        aabbs           = EmptyBuf;
        indices         = EmptyBuf;
        vertices        = EmptyBuf;
        splitAABBs      = EmptyBuf;
    }

    void materialize(LwdaUtils* lwdaUtils) const
    {
        aabbs       .materialize(lwdaUtils);
        indices     .materialize(lwdaUtils);
        vertices    .materialize(lwdaUtils);
        splitAABBs  .materialize(lwdaUtils);
    }

    void freeMem(void) const
    {
        aabbs       .freeMem();
        indices     .freeMem();
        vertices    .freeMem();
        splitAABBs  .freeMem();
    }
};

#endif // __LWDACC__

//------------------------------------------------------------------------
// Buffers mapping global primitive indices to input array/primitive indices, represented using CPU/GPU pointers.
//

struct InputArrayIndexPointers
{
    const unsigned int*             arrayBaseGlobalIndex;         // Global start index of each input array
    const unsigned int*             arrayTransitionBits;          // Treating the buffer as an array of bits, a bit is set for each entry in arrayBaseGlobalIndex
    const int*                      blockStartArrayIndex;         // Input array index at start of each 128-bit block
    const int*                      geometryIndexArray;           // Only needed if there were empty input geometries. Otherwise arrayIndex == geometryIndex

    INLINE InputArrayIndexPointers(void)
    {
        arrayBaseGlobalIndex          = NULL;
        arrayTransitionBits           = NULL;
        blockStartArrayIndex          = NULL;
        geometryIndexArray            = NULL;
    }

#ifndef __LWDACC__
    INLINE InputArrayIndexPointers(const InputArrayIndexBuffers& buf, MemorySpace memSpace)
    {
        arrayBaseGlobalIndex          = buf.arrayBaseGlobalIndex.read(memSpace);
        arrayTransitionBits           = buf.arrayTransitionBits.read(memSpace);
        blockStartArrayIndex          = buf.blockStartArrayIndex.read(memSpace);
        geometryIndexArray            = buf.geometryIndexArray.read(memSpace);
    }
#endif

    INLINE int getArrayIndex(int globalPrimitiveIndex)
    {
        int block = globalPrimitiveIndex / INPUT_ARRAY_INDEXER_BLOCK_SIZE;
        const unsigned int *bits = &arrayTransitionBits[block * (INPUT_ARRAY_INDEXER_BLOCK_SIZE / 32)];

        // Count bits set from position 0 to bitPos
        int bitPos = globalPrimitiveIndex & (INPUT_ARRAY_INDEXER_BLOCK_SIZE - 1);

        // Process 32 bit chunks at a time
        int transitionCount = 0;
#ifdef __LWDACC__
        #pragma unroll
        for (int i = 0; i < INPUT_ARRAY_INDEXER_BLOCK_SIZE / 32; i++)
#else
        for (int i = 0; i < INPUT_ARRAY_INDEXER_BLOCK_SIZE / 32; i++)
#endif
        {
            int chunkHiBitPos = (i + 1)*32;
            int shift = chunkHiBitPos - bitPos;
#ifndef __LWDACC__
            // Don't shift by more than 31 if running CPU path
            if (shift >= 32)
                continue;
#endif
            // Right shift clamps to [0, 32] in LWCA, so no need to check upper bound
            unsigned int mask = ~0U >> max(0, shift);
            transitionCount += __popc(bits[i] & mask);
        }

        return blockStartArrayIndex[block] + transitionCount;
    }
    INLINE int getGeometryIndex(int arrayIndex)
    {
        return geometryIndexArray ? geometryIndexArray[arrayIndex] : arrayIndex;
    }
    INLINE int getLocalPrimitiveIndex(int globalPrimitiveIndex, int arrayIndex)
    {
        return globalPrimitiveIndex - arrayBaseGlobalIndex[arrayIndex];
    }

};

//------------------------------------------------------------------------
// Input model for the BVH builder, represented using CPU/GPU pointers.

struct ModelPointers
{
    int                             numPrimitives;        // Total number of primitives (AABBs or triangles).
    int                             indexStride;          // Bytes between conselwtive items in 'indices'.
    int                             vertexStride;         // Bytes between conselwtive items in 'vertices'.

    const PrimitiveAABB*            aabbs;                // Abstract primitive AABBs. NULL => primitives are triangles.
    const char*                     indices;              // Vertex index triplets. NULL => each triangle is defined by 3 conselwtive vertices.
    const float*                    vertices;             // Vertex position triplets.
    const PrimitiveAABB*            splitAABBs;           // Additional primitive AABBs produced by triangle splitting.

    InputArrayIndexPointers         inputArrayIndexing;

    const char*                     inputAABBPtr=NULL;         // Backdoor to use input AABBs (no primitive index)
    int                             inputAABBStrideInBytes=0;  // Stride including motion steps

    INLINE ModelPointers(void)
    {
        numPrimitives   = 0;
        indexStride     = 0;
        vertexStride    = 0;
        aabbs           = NULL;
        indices         = NULL;
        vertices        = NULL;
        splitAABBs      = NULL;
    }

#ifndef __LWDACC__
    INLINE ModelPointers(const ModelBuffers& buf, MemorySpace memSpace)
    {
        numPrimitives         = buf.numPrimitives;
        indexStride           = buf.indexStride;
        vertexStride          = buf.vertexStride;
        aabbs                 = buf.aabbs.read(memSpace);

        indices               = buf.indices.read(memSpace);
        vertices              = buf.vertices.read(memSpace);
        splitAABBs            = buf.splitAABBs.read(memSpace);

        inputArrayIndexing    = InputArrayIndexPointers(buf.inputArrayIndexing, memSpace);
    }

    INLINE ModelPointers( const char* aabbPtr, int numAabbs, int aabbStride, MemorySpace memSpace )
    {
        numPrimitives          = numAabbs;
        indexStride            = 0;
        vertexStride           = 0; 
        aabbs                  = NULL; 
        indices                = NULL; 
        vertices               = NULL; 
        splitAABBs             = NULL; 

        inputAABBPtr           = aabbPtr;
        inputAABBStrideInBytes = aabbStride;
    }
#endif

    INLINE bool isAABBs(void) const
    {
        return (aabbs != NULL || vertexStride == 0);
    }

    INLINE int3 loadVertexIds(int triangleIdx) const // only allowed if isAABBs() == false
    {
        return (indices) ?
            fastUnalignedLoadInt3((char*)indices + (size_t)triangleIdx * indexStride) :
            make_int3(triangleIdx * 3 + 0, triangleIdx * 3 + 1, triangleIdx * 3 + 2);
    }

    INLINE void loadVertexPositions(float3& v0, float3& v1, float3& v2, int3 vertexIds) const // only allowed if isAABBs() == false
    {
        if (vertexStride == 4*sizeof(float) && ((uintptr_t)vertices & 15) == 0) // float4 with proper alignment
        {
            v0 = make_float3(LDG_OR_GLOBAL((float4*)vertices + vertexIds.x));
            v1 = make_float3(LDG_OR_GLOBAL((float4*)vertices + vertexIds.y));
            v2 = make_float3(LDG_OR_GLOBAL((float4*)vertices + vertexIds.z));
        }
        else
        {
            v0 = fastUnalignedLoadFloat3((char*)vertices + vertexIds.x * (size_t)vertexStride);
            v1 = fastUnalignedLoadFloat3((char*)vertices + vertexIds.y * (size_t)vertexStride);
            v2 = fastUnalignedLoadFloat3((char*)vertices + vertexIds.z * (size_t)vertexStride);
        }
    }

    INLINE void loadVertexPositions(float3& v0, float3& v1, float3& v2, int triangleIdx) const // only allowed if isAABBs() == false
    {
        loadVertexPositions(v0, v1, v2, loadVertexIds(triangleIdx));
    }

    // Load the AABB of either an original input primitive, or a primitive produced by triangle splitting.
    // Original input primitives are indexed with integers 0 .. 2^30-1 (two highest bits are zero).
    // Primitives produced by triangle splitting are indexed with 2^30 .. 2^31-1 (highest bit is zero, second highest bit is one).
    // The highest bit is reserved by several building blocks for distinguishing between leaf nodes and internal nodes.
    //
    INLINE PrimitiveAABB loadPrimitiveAABB(int primitiveIdx) const // always allowed
    {
        // Backdoor for motion refit from input AABBs
        if ( inputAABBPtr )
        {
            const AABB &b = *(AABB *) ( inputAABBPtr + (primitiveIdx * inputAABBStrideInBytes) );
            PrimitiveAABB aabb  = {};
            aabb.lox = b.lo.x;
            aabb.loy = b.lo.y;
            aabb.loz = b.lo.z;
            aabb.hix = b.hi.x;
            aabb.hiy = b.hi.y;
            aabb.hiz = b.hi.z;
            aabb.primitiveIdx = primitiveIdx;
            return aabb;
        }
        if (primitiveIdx >= (1 << 30))
        {
            return loadCachedAlign16(&splitAABBs[primitiveIdx - (1 << 30)]);
        }
        else if (isAABBs())
        {
            return loadCachedAlign16(&aabbs[primitiveIdx]);
        }
        else
        {
            float3 v0, v1, v2;
            loadVertexPositions(v0, v1, v2, primitiveIdx);

            PrimitiveAABB aabb  = {};
            aabb.lox            = fminf(fminf(v0.x, v1.x), v2.x);
            aabb.loy            = fminf(fminf(v0.y, v1.y), v2.y);
            aabb.loz            = fminf(fminf(v0.z, v1.z), v2.z);
            aabb.hix            = fmaxf(fmaxf(v0.x, v1.x), v2.x);
            aabb.hiy            = fmaxf(fmaxf(v0.y, v1.y), v2.y);
            aabb.hiz            = fmaxf(fmaxf(v0.z, v1.z), v2.z);
            aabb.primitiveIdx   = primitiveIdx;
            return aabb;
        }
    }

    INLINE int getArrayIndex(int globalPrimitiveIndex)
    {
        return inputArrayIndexing.getArrayIndex(globalPrimitiveIndex);
    }
    INLINE int getGeometryIndex(int arrayIndex)
    {
        return inputArrayIndexing.getGeometryIndex(arrayIndex);
    }
    INLINE int getLocalPrimitiveIndex(int globalPrimitiveIndex, int arrayIndex)
    {
        return inputArrayIndexing.getLocalPrimitiveIndex(globalPrimitiveIndex, arrayIndex);
    }

};

enum PrimBitsFormat
{
    PRIMBITS_NONE              = 0, // No primBits
    PRIMBITS_DIRECT_32         = 1, // Primitive index, geometry index, mode and opaque flag all fit within 32 bits
    PRIMBITS_INDIRECT_32_TO_64 = 2, // 32-bit primitive index and 32-bit geometry index are accessed via an indirection
    PRIMBITS_DIRECT_64         = 4, // 32-bit primitive and geometry indices are stored directly
    PRIMBITS_LEGACY_DIRECT_32  = 8, // "Old" format which fails if opaque flag and primitive and geometry indices don't fit together in 32 bits. TODO: This should go away once BVH2 and BVH8 supports the other formats
};


//------------------------------------------------------------------------

} // namespace bvhtools
} // namespace prodlib
