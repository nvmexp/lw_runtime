// Copyright LWPU Corporation 2016
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
#include "../include/BvhBuilder.hpp"

#include "bounds/ApexPointMapConstructor.hpp"
#include "trbvh/MortonTriangleSplitter.hpp"
#include "misc/MortonSorter.hpp"
#include "bvh8/AacBuilder.hpp"
#include "bvh8/WideBvhPlanner.hpp"
#include "bvh8/BVH8Constructor.hpp"
#include "bvh8/BVH8Fitter.hpp"
#include "misc/GatherPrimBits.hpp"
#include "misc/InputAdapter.hpp"

namespace prodlib
{
namespace bvhtools
{
//------------------------------------------------------------------------
// Dedicated builder for 8-wide BVHs.
// For further details, please refer to:
// https://p4viewer.lwpu.com/get///research/research/hylitie/docs/Efficient-RT%202016-07-09.pptx 
//------------------------------------------------------------------------

class BVH8Builder : public BvhBuilder
{
public:
                                BVH8Builder             (bool lwdaAvailable = true);
    virtual                     ~BVH8Builder            (void);

    virtual void                setDisableLwda          (bool disableLwda)          { m_disableLwda = disableLwda; }
    virtual void                setLwdaStream           (lwdaStream_t stream);

    virtual void                build                   (const std::vector<TriangleMesh>& meshes, bool meshInDeviceMem) override;
    virtual void                build                   (int numAabbs, const PrimitiveAABB* aabbs, bool aabbsInDeviceMem) override;
    virtual void                build                   (int numAabbs, int motionSteps, const float* aabbs, bool aabbsInDeviceMem) override;
    virtual void                build                   (int numInstances, const InstanceDesc* instances, bool inDeviceMem) override;
    virtual void                build                   (int numInputs, const RtcBuildInput* buildInputs, bool inDeviceMem) override;

    virtual void                copyHeader              (void* dst, size_t dstSize);

    static void                 computeMemUsageStatic   ( const char* builderSpec, bool buildGpu, const InputBuffers& input, MemoryUsage* memUsage );
    virtual void                computeMemUsage         (const char* builderSpec, bool buildGpu, int numPrims, int motionSteps, InputType type, MemoryUsage* memUsage) override;
    virtual void                computeMemUsage         (const char* builderSpec, bool buildGpu, int numAabbs, int numTriangles, MemoryUsage* memUsage) override;
    static void                 computeMemUsageStatic   (const char* builderSpec, bool buildGpu, int numPrims, InputType type, MemoryUsage* memUsage);

    virtual void                computeMemUsage         (const char* builderSpec, const std::vector<TriangleMesh>& meshes, bool meshInDeviceMem, bool buildGpu, MemoryUsage* memUsage) override;
    virtual void                computeMemUsage         (const char* builderSpec, bool buildGpu, int numInputs, const RtcBuildInput* buildInputs, MemoryUsage* memUsage) override;
    static void                 computeMemUsageStatic   (const char* builderSpec, bool buildGpu, int numInputs, const RtcBuildInput* buildInputs, MemoryUsage* memUsage);

    virtual void                freeData                (void);             // Frees all memory.
    virtual void                setDataLocation         (bool inDeviceMem); // No duplicate buffers remain after the call.

    int                         getNumBVH8Triangles     (void) const { return (int)m_buf.triangles.getNumElems(); }
    int                         getNumRemaps            (void) const { return (int)m_buf.remap.getNumElems(); }

    virtual const BvhNode*      getNodeBufferPtr        (bool inDeviceMem = false) { return (const BvhNode*)getBVH8NodeBufferPtr(inDeviceMem); }
    virtual const WoopTriangle* getWoopBufferPtr        (bool inDeviceMem = false) { return NULL; }
    virtual const int*          getRemapBufferPtr       (bool inDeviceMem = false) { return m_buf.remap.read((inDeviceMem) ? MemorySpace_LWDA : MemorySpace_Host); }
    virtual const ApexPointMap* getApexPointMapPtr      (bool inDeviceMem = false) { return m_buf.apexPointMap.read((inDeviceMem) ? MemorySpace_LWDA : MemorySpace_Host); }
    virtual const unsigned char* getOutputBufferPtr     (bool inDeviceMem = false) { return m_outputBuffer.read((inDeviceMem) ? MemorySpace_LWDA : MemorySpace_Host); }
    const BVH8Node*             getBVH8NodeBufferPtr    (bool inDeviceMem = false) { return m_buf.nodes.read((inDeviceMem) ? MemorySpace_LWDA : MemorySpace_Host); }
    const BVH8Triangle*         getBVH8TriangleBufferPtr(bool inDeviceMem = false) { return m_buf.triangles.read((inDeviceMem) ? MemorySpace_LWDA : MemorySpace_Host); }

    virtual size_t              getNodeBufferSize       (void) const  { return m_buf.nodes.getNumBytes(); }
    virtual size_t              getRemapBufferSize      (void) const  { return getNumRemaps() * sizeof(int); }
    virtual size_t              getWoopBufferSize       (void) const  { return 0; }
    virtual size_t              getOutputBufferSize     (void) const  { return m_outputBuffer.getNumBytes(); }
    size_t                      getBVH8TriangleBufferSize(void) const { return getNumBVH8Triangles() * sizeof(BVH8Triangle); }
    virtual size_t              getApexPointMapSize     (void) const { return m_buf.apexPointMap.getNumBytes(); }

    virtual void                setExternalBuffers      (unsigned char* outputPtr, size_t outputNumBytes, 
                                                         unsigned char* tempPtr, size_t tempNumBytes, 
                                                         unsigned char* readbackPtr, size_t readbackNumBytes, bool buildGpu);

    virtual void                setExternalOutputBuffer (unsigned char* outputPtr, size_t outputNumBytes, bool buildGpu); // BL: Why is this needed?

private:
    void                        buildFromInputBuffers   (const InputBuffers& input);
    void                        configureBuild          (const InputBuffers& input);
    void                        exelwteBuild            (void);
    void                        exelwteBuildWithTiming  (void);

private:
                                BVH8Builder             (const BVH8Builder&); // forbidden
    BVH8Builder&                operator=               (const BVH8Builder&); // forbidden

private:
    // Initialized by the constructor.

    LwdaUtils*                  m_lwdaUtils;            // NULL if LWCA is not available.
    LwdaUtils                   m_lwdaUtilsImpl;        // Note: This must be the first non-pointer member, so that its destructor gets called last.

    InputAdapter                m_inputAdapter;
    ApexPointMapConstructor     m_apmConstructor;
    MortonTriangleSplitter      m_splitter;
    MortonSorter                m_mortonSorter;
    AacBuilder                  m_aacBuilder;
    WideBvhPlanner              m_planner;
    BVH8Constructor             m_constructor;
    BVH8Fitter                  m_fitter;
    GatherPrimBits              m_gatherPrimBits;

    // Initialized by the public setter methods.

    bool                        m_disableLwda;

    // Initialized by configureBuild().

    bool                        m_useLwda;
    InputType                   m_inputType;
    bool                        m_supportRefit;
    bool                        m_refitOnly;
    bool                        m_freeTempsAfterBuild;
    int                         m_lwdaTimingRepeats;

    int                         m_numInputPrims;
    int                         m_maxSplitPrims;
    bool                        m_enableSplits;

    int                         m_buildCount = 0;

    bool                        m_usePrimBits;

    // Top-level CPU/GPU memory buffers allocated by the builder.

    BufferRef<>                 m_outputBuffer;         // Output of the builder.
    BufferRef<int>              m_readbackBuffer;       // Outputs that are read by the CPU.
    BufferRef<>                 m_tempBuffer;           // Temporary data during build.

    // Individual data arrays used by the building blocks.
    // These aggregated/overlaid within the top-level buffers by configureBuild().

    struct BuildingBlockBuffers
    {
        // Builder outputs.

        BufferRef<BvhHeader>              header;               // <tiny> user configurable
        BufferRef<BVH8Node>               nodes;                // ~ 80 bytes * numNodes
        BufferRef<BVH8NodeAux>            nodeAux;              // ~ 16 bytes * numNodes
        BufferRef<int>                    remap;                // ~ 4  bytes * maxSplitPrims
        BufferRef<BvhInstanceData>        bvhInstanceData;      // ~ 112 bytes * numInputPrims, IF m_inputType == IT_INSTANCE
        BufferRef<BVH8Triangle>           triangles;            // ~ 48 bytes * maxSplitPrims
        BufferRef<ApexPointMap>           apexPointMap;         // <tiny>
        BufferRef<>                       primBits;             // ~ 4 or 8 bytes * maxSplitPrims
        BufferRef<unsigned int>           arrayBaseGlobalIndex; // ~ 4  bytes * numInputs
        BufferRef<unsigned int>           arrayTransitionBits;  // ~ 4  bytes * ceil(numPrims/32)
        BufferRef<int>                    blockStartArrayIndex; // ~ 4  bytes * ceil(numPrims/128)
        BufferRef<int>                    geometryIndexArray;   // ~ 4  bytes * numInputs

        // Temp arrays.

        BufferRef<PrimitiveAABB>          splitAABBs;           // ~ 15 bytes * maxSplitPrims, IF m_enableSplits == true
        BufferRef<int>                    primOrder;            // ~ 4  bytes * maxSplitPrims
        BufferRef<>                       mortonCodes;          // ~ 8  bytes * maxSplitPrims
        BufferRef<PlannerBinaryNode>      binaryNodes;          // ~ 32 bytes * maxSplitPrims
        BufferRef<float3>                 coalescedVertices;    // ~ 12 bytes * numPrims

        BufferRef<>                       tmpPrimBits;          // ~ 4 or 8 bytes * maxSplitPrims
        BufferRef<InputAABBs>             inputAabbArray;       // ~ 16 bytes * numAabbInputs, IF m_inputType == IT_AABB
        BufferRef<InputTrianglesDesc>     inputTriangleDescs;   // ~ 16 bytes * numInputs, IF m_inputType == IT_TRI
        BufferRef<InputTrianglesPointers> inputTrianglePtrs;    // ~ 24 bytes * numInputs, IF m_inputType == IT_TRI

        // Temp scalars.

        BufferRef<int>                    numNodes;             // <tiny>
        BufferRef<Range>                  splitPrimRange;       // <tiny>
        BufferRef<Range>                  sortPrimRange;        // <tiny>
    
        // Building block internal temps.

        BufferRef<>                       apmTemp;              // <tiny>
        BufferRef<>                       splitterTemp;         // ~ 4  bytes * maxInputPrims, IF m_enableSplits == true
        BufferRef<>                       mortonSorterTemp;     // ~ 24 bytes * maxSplitPrims
        BufferRef<>                       aacBuilderTemp;       // ~ 8  bytes * maxSplitPrims
        BufferRef<>                       plannerTempA;         // ~ 8  bytes * maxSplitPrims
        BufferRef<>                       plannerTempB;         // ~ 8  bytes * maxSplitPrims
        BufferRef<>                       constructorTemp;      // <tiny>
        BufferRef<>                       fitterTemp;           // <tiny>
    };
    BuildingBlockBuffers        m_buf;

    BvhHeader m_header;
};

//------------------------------------------------------------------------
} // namespace bvhtools
} // namespace prodlib
