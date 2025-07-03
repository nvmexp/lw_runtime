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
#include "../include/BvhBuilder.hpp"
#include "common/ParameterList.hpp"

#include "bounds/ApexPointMapConstructor.hpp"
#include "misc/MortonSorter.hpp"
#include "misc/MotionRefitter.hpp"
#include "chunking/Chunker.hpp"
#include "trbvh/MortonTriangleSplitter.hpp"
#include "trbvh/TrBvhBuilder.hpp"
#include "chunking/TreeTopTrimmer.hpp"
#include "chunking/TopTreeConnector.hpp"
#include "misc/OptixColwerter.hpp"
#include "misc/TriangleWooper.hpp"
#include "misc/GatherPrimBits.hpp"
#include "misc/InputAdapter.hpp"

namespace prodlib
{
namespace bvhtools
{
//------------------------------------------------------------------------
// BVH class
//
// builderSpec is a string of the form "builder=TRBVH, splitBeta=0.30"
// that allows specifying custom values for different the builder
// parameters. An empty string instructs the builder to use the default
// value for each parameter, which is typically the right thing to do.
//
// For a list of supported parameters, please search for "p.get("
// in ChunkedTrbvhBuilder.cpp. For a detailed description of any
// individual parameter, please see the header of the corresponding
// building block.
//
// TODO: May choose between host/device memory using an enum
// instead of a bool.
//
// TODO [tkarras]: Write down the full list of builder parameters in
// optix_prime_private.h.

class ChunkedTrbvhBuilder : public BvhBuilder
{
public:
                                ChunkedTrbvhBuilder     (bool lwdaAvailable = true);
    virtual                     ~ChunkedTrbvhBuilder    (void);

    virtual void                setDisableLwda          (bool disableLwda)          { m_disableLwda = disableLwda; }
    virtual void                setLwdaStream           (lwdaStream_t stream);

    virtual void                build                   (const std::vector<TriangleMesh>& meshes, bool meshInDeviceMem) override;
    virtual void                build                   (int numAabbs, const PrimitiveAABB* aabbs, bool aabbsInDeviceMem) override;
    virtual void                build                   (int numAabbs, int motionSteps, const float* aabbs, bool aabbsInDeviceMem) override;
    virtual void                build                   (int numInstances, const InstanceDesc* instances, bool inDeviceMem) override;
    virtual void                build                   (int numInputs, const RtcBuildInput* buildInputs, bool inDeviceMem) override;

    virtual void                copyHeader              (void* dst, size_t dstSize);

    virtual void                computeMemUsage         (const char* builderSpec, bool buildGpu, int numPrims, int motionSteps, InputType type, MemoryUsage* memUsage);
    virtual void                computeMemUsage         (const char* builderSpec, bool buildGpu, int numPrimAabbs, int numTriangles, MemoryUsage* memUsage);

    virtual void                computeMemUsage         (const char* builderSpec, const std::vector<TriangleMesh>& meshes, bool meshInDeviceMem, bool buildGpu, MemoryUsage* memUsage) override;
    virtual void                computeMemUsage         (const char* builderSpec, bool buildGpu, int numInputs, const RtcBuildInput *buildInputs, MemoryUsage* memUsage) override;

    static void                 computeMemUsageStatic   ( const char* builderSpec, bool buildGpu, const InputBuffers& input, MemoryUsage* memUsage );
    static void                 computeMemUsageStatic   (const char* builderSpec, bool buildGpu, int numPrims, int motionSteps, InputType type, MemoryUsage* memUsage);
    static void                 computeMemUsageStatic   (const char* builderSpec, bool buildGpu, int numInputs, const RtcBuildInput *buildInputs, MemoryUsage* memUsage);

    virtual void                freeData                (void);             // Frees all memory.
    virtual void                setDataLocation         (bool inDeviceMem); // No duplicate buffers remain after the call.

    virtual const unsigned char* getOutputBufferPtr     (bool inDeviceMem = false) { return (unsigned char*)m_outputBuffer.read((inDeviceMem) ? MemorySpace_LWDA : MemorySpace_Host); }
    virtual const BvhNode*      getNodeBufferPtr        (bool inDeviceMem = false) { return m_buf.nodes       .read((inDeviceMem) ? MemorySpace_LWDA : MemorySpace_Host); }
    virtual const int*          getRemapBufferPtr       (bool inDeviceMem = false) { return m_buf.remap       .read((inDeviceMem) ? MemorySpace_LWDA : MemorySpace_Host); }
    virtual const WoopTriangle* getWoopBufferPtr        (bool inDeviceMem = false) { return m_buf.woop        .read((inDeviceMem) ? MemorySpace_LWDA : MemorySpace_Host); }
    virtual const ApexPointMap* getApexPointMapPtr      (bool inDeviceMem = false) { return m_buf.apexPointMap.read((inDeviceMem) ? MemorySpace_LWDA : MemorySpace_Host); }

    virtual size_t              getOutputBufferSize     (void) const { return m_outputBuffer.getNumBytes(); }
    virtual size_t              getNodeBufferSize       (void) const { return m_buf.nodes.getNumBytes(); }
    virtual size_t              getRemapBufferSize      (void) const { return m_buf.remap.getNumBytes(); }
    virtual size_t              getWoopBufferSize       (void) const { return m_buf.woop.getNumBytes(); }
    virtual size_t              getApexPointMapSize     (void) const { return m_buf.apexPointMap.getNumBytes(); }

    virtual void                setExternalBuffers      (unsigned char* outputPtr, size_t outputNumBytes, 
                                                         unsigned char* tempPtr, size_t tempNumBytes, 
                                                         unsigned char* readbackPtr, size_t readbackNumBytes, bool buildGpu);

    virtual void                setExternalOutputBuffer (unsigned char* outputPtr, size_t outputNumBytes, bool buildGpu);

private:
    void                        buildFromInputBuffers   (const InputBuffers& input);
    void                        configureBuild          (const InputBuffers& input);
    void                        exelwteBuild            (void);
    void                        exelwteBuildWithTiming  (void);
    
    void                        noAccelHackExelwte      (void);

    void                        chooseChunkSize         (/*out*/ int&           maxSplitPrimsPerChunk,
                                                         /*out*/ int&           preferredTopTreePrims,
                                                         int                    numInputPrims,
                                                         int                    maxTotalSplitPrims,
                                                         float                  splitBeta,
                                                         const ParameterList&   p);

    void                        checkpoint              (const char* name);
    void                        printBufferSizes        ();

private:
                                ChunkedTrbvhBuilder     (const ChunkedTrbvhBuilder&); // forbidden
    ChunkedTrbvhBuilder&        operator=               (const ChunkedTrbvhBuilder&); // forbidden

private:

    // Initialized by the constructor.

    LwdaUtils*              m_lwdaUtils;            // NULL if LWCA is not available.
    LwdaUtils               m_lwdaUtilsImpl;        // Note: This must be the first non-pointer member, so that its destructor gets called last.

    InputAdapter            m_inputAdapter;
    ApexPointMapConstructor m_apmConstructor;
    MortonSorter            m_chunkSorter;
    Chunker                 m_chunker;
    MortonTriangleSplitter  m_splitter;
    MortonSorter            m_bottomSorter;
    TrbvhBuilder            m_bottomTrbvh;
    TreeTopTrimmer          m_trimmer;
    MortonSorter            m_topSorter;
    TrbvhBuilder            m_topTrbvh;
    TopTreeConnector        m_connector;
    OptixColwerter          m_colwerter;
    TriangleWooper          m_wooper;
    GatherPrimBits          m_gatherPrimBits;
    MotionRefitter          m_motionRefitter;

    // Initialized by the public setter methods.

    bool                    m_disableLwda;

    // Initialized by configureBuild().

    bool                    m_useLwda;
    InputType               m_inputType;
    bool                    m_enableChunking;
    bool                    m_outputOptixNodes;
    bool                    m_outputWoopTriangles;
    bool                    m_freeTempsAfterBuild;
    int                     m_numInputPrims;
    int                     m_motionSteps = 1;
    bool                    m_enableSplits;
    int                     m_lwdaTimingRepeats;    // Non-zero and m_useLwda is true => enable timing printouts in exelwteBuildWithTiming().
    bool                    m_noAccelHack;
    bool                    m_supportRefit;
    bool                    m_refitOnly;

    bool                    m_usePrimBits;

    // TEMP
    ModelBuffers            m_refitModel;

    // Top-level CPU/GPU memory buffers allocated by the builder.

    BufferRef<>             m_outputBuffer;         // Output of the builder.
    BufferRef<int>          m_readbackBuffer;       // Outputs that are read by the CPU.
    BufferRef<>             m_tempBuffer;           // Temporary data during build.

    // Individual data arrays used by the building blocks.
    // These aggregated/overlaid within the top-level buffers by configureBuild().

    struct BuildingBlockBuffers
    {
        // Builder outputs.
        BufferRef<BvhHeader>              header;               // <tiny> user configurable
        BufferRef<BvhNode>                nodes;                // ~ 64 bytes * (maxTotalSplitPrims + preferredTopTreePrims).  Subrange of allNodes.
        BufferRef<BvhNode>                allNodes;             // ~ 64 bytes * (maxTotalSplitPrims + preferredTopTreePrims) * (motionSteps)
        BufferRef<int>                    remap;                // ~ 4  bytes * (maxTotalSplitPrims + preferredTopTreePrims)
        BufferRef<BvhInstanceData>        bvhInstanceData;      // ~ 112 bytes * numInputPrims, IF m_inputType == IT_INSTANCE
        BufferRef<WoopTriangle>           woop;                 // ~ 48 bytes * maxTotalSplitPrims, IF m_outputWoopTriangles = true
        BufferRef<ApexPointMap>           apexPointMap;         // <tiny>
        BufferRef<int>                    primBits;             // ~ 4  bytes * maxSplitPrims
        BufferRef<unsigned int>           arrayBaseGlobalIndex; // ~ 4  bytes * numInputs
        BufferRef<unsigned int>           arrayTransitionBits;  // ~ 4  bytes * ceil(numPrims/32)
        BufferRef<int>                    blockStartArrayIndex; // ~ 4  bytes * ceil(numPrims/128)
        BufferRef<int>                    geometryIndexArray;   // ~ 4  bytes * numInputs


        // Temp arrays.

        BufferRef<PrimitiveAABB>          tmpAabbs;             // ~ 15 bytes * maxSplitPrimsPerChunk, IF m_enableSplits = true. Temp storage for split tri refs or instance AABBs
        BufferRef<unsigned long long>     chunkMortonCodes;     // ~ 8  bytes * numInputPrims, IF m_enableChunking = true
        BufferRef<int>                    chunkPrimOrder;       // ~ 4  bytes * numInputPrims, IF m_enableChunking = true
        BufferRef<Range>                  chunkPrimRanges;      // ~ 8  bytes * maxChunks
        BufferRef<unsigned long long>     bottomMortonCodes;    // ~ 8  bytes * maxSplitPrimsPerChunk
        BufferRef<int>                    bottomPrimOrder;      // ~ 4  bytes * maxSplitPrimsPerChunk
        BufferRef<int>                    bottomNodeParents;    // ~ 4  bytes * maxSplitPrimsPerChunk
        BufferRef<unsigned long long>     topMortonCodes;       // ~ 8  bytes * preferredTopTreePrims, IF m_enableChunking = true
        BufferRef<int>                    topPrimOrder;         // ~ 4  bytes * preferredTopTreePrims, IF m_enableChunking = true
        BufferRef<int>                    topNodeParents;       // ~ 4  bytes * preferredTopTreePrims, IF m_enableChunking = true
        BufferRef<PrimitiveAABB>          trimmedAabbs;         // ~ 32 bytes * preferredTopTreePrims, IF m_enableChunking = true
        BufferRef<int>                    tmpRemap;             // ~ 4  bytes * (maxTotalSplitPrims + preferredTopTreePrims), IF m_outputOptixNodes = true

        BufferRef<float3>                 coalescedVertices;    // ~ 12 bytes * numPrims
        BufferRef<>                       tmpPrimBits;          // ~ 4 - 8  bytes * numPrims
        BufferRef<InputAABBs>             inputAabbArray;       // ~ 16 bytes * numInputs, IF m_inputType == IT_AABB
        BufferRef<InputTrianglesDesc>     inputTriangleDescs;   // ~ 16 bytes * numInputs, IF m_inputType == IT_TRI
        BufferRef<InputTrianglesPointers> inputTrianglePtrs;    // ~ 24 bytes * numInputs, IF m_inputType == IT_TRI

        // Temp scalars.

        BufferRef<int>                    numNodes;             // <tiny>
        BufferRef<int>                    numChunks;            // <tiny>
        BufferRef<int>                    numRemaps;            // <tiny>
        BufferRef<int>                    lwtoffLevel;          // <tiny>
        BufferRef<Range>                  splitPrimRange;       // <tiny>
        BufferRef<Range>                  sortPrimRange;        // <tiny>
        BufferRef<Range>                  trbvhNodeRange;       // <tiny>
        BufferRef<Range>                  trimmedAabbRange;     // <tiny>

        // Building block internal temps.

        BufferRef<>                       aabbCalcTemp;         // <tiny>
        BufferRef<>                       chunkSorterTemp;      // ~ 24 bytes * numInputPrims, IF m_enableChunking = true
        BufferRef<>                       splitterTemp;         // ~ 4  bytes * maxInputPrimsPerChunk, IF m_inputAABBs = false
        BufferRef<>                       bottomSorterTemp;     // ~ 24 bytes * maxSplitPrimsPerChunk
        BufferRef<>                       bottomTrbvhTemp;      // <tiny>
        BufferRef<>                       topSorterTemp;        // ~ 24 bytes * preferredTopTreePrims, IF m_enableChunking = true
        BufferRef<>                       topTrbvhTemp;         // <tiny>
        BufferRef<>                       colwerterTemp;        // ~ 8  bytes * (maxTotalSplitPrims + preferredTopTreePrims), IF m_outputOptixNodes = true
    };
    BuildingBlockBuffers    m_buf;

    BvhHeader m_header;
};

//------------------------------------------------------------------------
} // namespace bvhtools
} // namespace prodlib
