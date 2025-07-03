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

#include "BVH8Builder.hpp"
#include "common/ParameterList.hpp"
#include <prodlib/system/Knobs.h>
#include <corelib/misc/String.h>
#include <prodlib/exceptions/Assert.h>
#include <prodlib/exceptions/IlwalidOperation.h>
#include <prodlib/exceptions/IlwalidValue.h>
#include <corelib/math/MathUtil.h>
#include <algorithm>
#include <iostream>

namespace
{
    Knob<bool>  k_memPrintUsage     (RT_DSTRING("bvhtools.memPrintUsage"),      0,  RT_DSTRING("Print max and final memory usage"));
    Knob<int>   k_lwdaTimingRepeats (RT_DSTRING("bvhtools.lwdaTimingRepeats"),  0,  RT_DSTRING("Run BVH builder several times and print detailed timing info"));
}

using namespace prodlib::bvhtools;
using namespace prodlib;

//------------------------------------------------------------------------

BVH8Builder::BVH8Builder(bool lwdaAvailable)
:   m_lwdaUtils             ((lwdaAvailable) ? &m_lwdaUtilsImpl : NULL),

    m_disableLwda           (false),

    m_useLwda               (false),
    m_supportRefit          (false),
    m_refitOnly             (false),
    m_freeTempsAfterBuild   (false),
    m_lwdaTimingRepeats     (0),

    m_numInputPrims         (0),
    m_maxSplitPrims         (0),
    m_enableSplits          (false)
{
}

//------------------------------------------------------------------------

BVH8Builder::~BVH8Builder(void)
{
}

//------------------------------------------------------------------------

void BVH8Builder::setLwdaStream(lwdaStream_t stream)
{
    if (!m_lwdaUtils)
        return;

    // Stream changed => make sure there are no pending async ops on the old one.

    if (stream != m_lwdaUtils->getDefaultStream())
        m_lwdaUtils->streamSynchronize();

    // Set stream.

    m_lwdaUtils->setDefaultStream(stream);
}

//------------------------------------------------------------------------

void BVH8Builder::build(const std::vector<TriangleMesh>& meshes, bool meshInDeviceMem)
{
    // Check for errors.

    for( const TriangleMesh& mesh : meshes )
    {
        if( mesh.indexStride < 0 )
            throw IlwalidValue( RT_EXCEPTION_INFO, "indexStride must be non-negative", mesh.indexStride );

        if( mesh.vertexStride < 0 )
            throw IlwalidValue( RT_EXCEPTION_INFO, "vertexStride must be non-negative", mesh.vertexStride );

        if( mesh.numVertices > 0 && !mesh.vertices )
            throw IlwalidValue( RT_EXCEPTION_INFO, "NULL vertex pointer" );

        if( meshInDeviceMem && !m_lwdaUtils )
            throw IlwalidValue( RT_EXCEPTION_INFO, "LWCA not available" );
    }

    InputBuffers input(meshes, (meshInDeviceMem) ? MemorySpace_LWDA : MemorySpace_Host);
    buildFromInputBuffers(input);
}

//------------------------------------------------------------------------

void BVH8Builder::build(int numAabbs, const PrimitiveAABB* aabbs, bool aabbsInDeviceMem)
{
    // Check for errors.

    if (numAabbs < 0)
        throw IlwalidValue(RT_EXCEPTION_INFO, "numAabbs must be non-negative", numAabbs);

    if (!aabbs && numAabbs > 0)
        throw IlwalidValue(RT_EXCEPTION_INFO, "NULL aabbs pointer");

    if (aabbsInDeviceMem && !m_lwdaUtils)
        throw IlwalidValue(RT_EXCEPTION_INFO, "LWCA not available");

    InputBuffers input(numAabbs, aabbs, (aabbsInDeviceMem) ? MemorySpace_LWDA : MemorySpace_Host);
    buildFromInputBuffers(input);
}

//------------------------------------------------------------------------

void BVH8Builder::build(int numAabbs, int motionSteps, const float* aabbs, bool aabbsInDeviceMem)
{
  // Check for errors.

  if (numAabbs < 0)
      throw IlwalidValue( RT_EXCEPTION_INFO, "numAabbs must be non-negative", numAabbs );

  // REMOVE once we support motion steps
  if (motionSteps > 1)
    throw IlwalidValue( RT_EXCEPTION_INFO, "Motion steps > 1 not supported in Bvh8", motionSteps );

  if (motionSteps < 1)
    throw IlwalidValue( RT_EXCEPTION_INFO, "motionSteps must be positive", motionSteps );

  if (!aabbs && numAabbs > 0)
      throw IlwalidValue( RT_EXCEPTION_INFO, "NULL aabbs pointer" );

  if (aabbsInDeviceMem && !m_lwdaUtils)
      throw IlwalidValue( RT_EXCEPTION_INFO, "LWCA not available" );

  InputBuffers input(numAabbs, motionSteps, (const AABB*)aabbs, (aabbsInDeviceMem) ? MemorySpace_LWDA : MemorySpace_Host);
  buildFromInputBuffers(input);
}

//------------------------------------------------------------------------------

void BVH8Builder::build(int numInstances, const InstanceDesc* instancesPtr, bool inDeviceMem)
{
  // Check for errors.

  if (numInstances < 0)
      throw IlwalidValue( RT_EXCEPTION_INFO, "numInstances must be non-negative", numInstances );

  if (!instancesPtr && numInstances > 0)
      throw IlwalidValue( RT_EXCEPTION_INFO, "NULL instances pointer" );

  if (inDeviceMem && !m_lwdaUtils)
      throw IlwalidValue( RT_EXCEPTION_INFO, "LWCA not available" );

  InputBuffers input(numInstances, instancesPtr, (inDeviceMem) ? MemorySpace_LWDA : MemorySpace_Host);
  buildFromInputBuffers(input);
}

//------------------------------------------------------------------------

void BVH8Builder::build(int numInputs, const RtcBuildInput* buildInputs, bool inDeviceMem)
{
  if (numInputs > 0 && buildInputs[0].type == RTC_BUILD_INPUT_TYPE_INSTANCE_ARRAY)
  {
      const RtcBuildInputInstanceArray& inst = buildInputs[0].instanceArray;
      build(inst.numInstances, (const InstanceDesc*) inst.instanceDescs, inDeviceMem);
      return;
  }

  if( inDeviceMem && !m_lwdaUtils )
    throw IlwalidValue( RT_EXCEPTION_INFO, "LWCA not available" );

  InputBuffers input(numInputs, buildInputs, (inDeviceMem) ? MemorySpace_LWDA : MemorySpace_Host);
  buildFromInputBuffers(input);
}

//------------------------------------------------------------------------

void BVH8Builder::buildFromInputBuffers(const InputBuffers & input)
{
  try
  {
    input.materialize(m_lwdaUtils);
    configureBuild(input);
    exelwteBuildWithTiming();

    if (!m_useLwda && m_outputBuffer.isExternal())
      m_outputBuffer.readLWDA();

    if(m_freeTempsAfterBuild)
      m_tempBuffer.freeMem();
  }
  catch(...)
  {
    input.freeMem();
    freeData();
    throw;
  }
}

//------------------------------------------------------------------------------
void BVH8Builder::computeMemUsageStatic( const char* builderSpec, bool buildGpu, const InputBuffers& input, MemoryUsage* memUsage )
{
    // Configure build.

    BVH8Builder tmp( buildGpu );
    tmp.setBuilderSpec( builderSpec );
    tmp.configureBuild( input );

    // Collect allocation sizes

    size_t outputSize   = tmp.m_outputBuffer.getAllocSize();
    size_t tempSize     = tmp.m_tempBuffer.getAllocSize();
    size_t readbackSize = tmp.m_readbackBuffer.getAllocSize();

    memUsage->header     = tmp.m_buf.header.getNumBytes();
    memUsage->nodes      = tmp.m_buf.nodes.getNumBytes();
    memUsage->remap      = tmp.m_buf.remap.getNumBytes();
    memUsage->output     = outputSize;
    memUsage->temp       = tempSize;
    memUsage->readback   = readbackSize;
    memUsage->totalMax   = outputSize + readbackSize + tempSize;
    memUsage->totalFinal = outputSize + readbackSize + ( ( tmp.m_freeTempsAfterBuild ) ? 0 : tempSize );
}

//------------------------------------------------------------------------------
void BVH8Builder::computeMemUsageStatic(const char* builderSpec, bool buildGpu, int numPrims, InputType type, MemoryUsage* memUsage )
{
  if (memUsage == NULL)
    return;

  // Setup fake model buffers.

  char* dummyBuffer = (char*)0x10; // dummy 16 byte aligned buffer
  std::unique_ptr<InputBuffers> input;
  MemorySpace memSpace = MemorySpace_LWDA;
  if (type == IT_TRI)
  {
    TriangleMesh mesh; 
    mesh.numTriangles = numPrims;
    mesh.numVertices = 3 * numPrims;
    mesh.vertices = (float*)dummyBuffer;
    mesh.vertexStride = sizeof(float3);
    input.reset(new InputBuffers( {mesh}, memSpace));
  }
  else if (type == IT_AABB)
  {
    input.reset(new InputBuffers(numPrims, /*motionSteps*/ 1, (const AABB*)dummyBuffer, memSpace));
  }
  else if (type == IT_PRIMAABB)
  {
    input.reset(new InputBuffers(numPrims, (PrimitiveAABB*)dummyBuffer, memSpace));
  }
  else if (type == IT_INSTANCE)
  {
    input.reset(new InputBuffers(numPrims, (InstanceDesc*)dummyBuffer, memSpace));
  }
  else
  {
    RT_ASSERT_MSG(0, "Unhandled input type.");
  }

  computeMemUsageStatic( builderSpec, buildGpu, *input, memUsage );
}

//------------------------------------------------------------------------

void BVH8Builder::computeMemUsage( const char*                      builderSpec,
                                   const std::vector<TriangleMesh>& meshes,
                                   bool                             meshInDeviceMem,
                                   bool                             buildGpu,
                                   MemoryUsage*                     memUsage )
{
    // TODO: This is very similar between the three builders. Refactor.
    if( memUsage == NULL )
        return;

    // Setup fake model buffers.
    InputBuffers input( meshes, ( meshInDeviceMem ) ? MemorySpace_LWDA : MemorySpace_Host );

    computeMemUsageStatic( builderSpec, buildGpu, input, memUsage );
}

//------------------------------------------------------------------------

void BVH8Builder::computeMemUsageStatic( const char* builderSpec, bool buildGpu, int numInputs, const RtcBuildInput* buildInputs, MemoryUsage* memUsage )
{
    // TODO: This is very similar between the three builders. Refactor.
    if (memUsage == NULL)
      return;

    // Setup fake model buffers.
    MemorySpace memSpace = MemorySpace_LWDA;
    InputBuffers input( numInputs, buildInputs, memSpace, true );

    computeMemUsageStatic( builderSpec, buildGpu, input, memUsage );
}

//------------------------------------------------------------------------

void BVH8Builder::computeMemUsage(const char * builderSpec, bool buildGpu, int numPrims, int motionSteps, InputType type, MemoryUsage * memUsage)
{
  // REMOVE once we support motion steps
  if (motionSteps > 1)
    throw IlwalidValue( RT_EXCEPTION_INFO, "Motion steps > 1 not supported", motionSteps );

  computeMemUsageStatic(builderSpec, buildGpu, numPrims, type, memUsage);
}

//------------------------------------------------------------------------

void BVH8Builder::computeMemUsage(const char* builderSpec, bool buildGpu, int numPrimAabbs, int numTriangles, MemoryUsage* memUsage)
{
  RT_ASSERT(!(numPrimAabbs > 0 && numTriangles > 0));
  int numPrims = 0; 
  InputType type = IT_TRI;
  if (numPrimAabbs > 0)
  {
    numPrims = numPrimAabbs;
    type = IT_PRIMAABB;
  }
  computeMemUsageStatic(builderSpec, buildGpu, numPrims, type, memUsage);
}

//------------------------------------------------------------------------

void BVH8Builder::computeMemUsage(const char* builderSpec, bool buildGpu, int numInputs, const RtcBuildInput* buildInputs, MemoryUsage* memUsage)
{
  computeMemUsageStatic(builderSpec, buildGpu, numInputs, buildInputs, memUsage);
}

//------------------------------------------------------------------------

void BVH8Builder::freeData(void)
{
    m_outputBuffer      .freeMem();
    m_tempBuffer        .freeMem();
    m_readbackBuffer    .freeMem();
    m_supportRefit      = false;
}

//------------------------------------------------------------------------

void BVH8Builder::setDataLocation(bool inDeviceMem)
{
    MemorySpace memSpace = (inDeviceMem) ? MemorySpace_LWDA : MemorySpace_Host;
    m_outputBuffer      .freeMemExcept(memSpace);
    m_tempBuffer        .freeMemExcept(memSpace);
    m_readbackBuffer    .freeMemExcept(memSpace);
}

//------------------------------------------------------------------------

void BVH8Builder::setExternalBuffers(unsigned char* outputPtr, size_t outputNumBytes, unsigned char* tempPtr, size_t tempNumBytes, unsigned char* readbackPtr, size_t readbackNumBytes, bool buildGpu)
{
    MemorySpace memSpace = (buildGpu) ? MemorySpace_LWDA : MemorySpace_Host;
    
    m_outputBuffer.assignExternal(outputPtr, outputNumBytes / m_outputBuffer.getBytesPerElem(), memSpace);
    m_tempBuffer.assignExternal(tempPtr, tempNumBytes / m_tempBuffer.getBytesPerElem(), memSpace);
    m_readbackBuffer.assignExternal((int*)readbackPtr, readbackNumBytes / m_readbackBuffer.getBytesPerElem(), memSpace);
    
    m_outputBuffer.materialize(m_lwdaUtils);
    m_tempBuffer.materialize(m_lwdaUtils);
    m_readbackBuffer.materialize(m_lwdaUtils);
}   

//------------------------------------------------------------------------
// Special case of the above for just the output buffer. Needed for refit if the memory manager moves the allocation.
void BVH8Builder::setExternalOutputBuffer(unsigned char* outputPtr, size_t outputNumBytes, bool buildGpu)
{
    MemorySpace memSpace = (buildGpu) ? MemorySpace_LWDA : MemorySpace_Host;
    
    m_outputBuffer.assignExternal(outputPtr, outputNumBytes / m_outputBuffer.getBytesPerElem(), memSpace);
    m_outputBuffer.materialize(m_lwdaUtils);
}

//------------------------------------------------------------------------

void BVH8Builder::configureBuild(const InputBuffers& input)
{
    bool refitPossible = (m_supportRefit && input.numPrimitives == (int)m_buf.remap.getNumElems());

    // Parse builderSpec.

    ParameterList p(m_builderSpec.c_str());
        
    const char* builder                 = p.get("builder",                  "BVH8");
    bool        disableLWDA             = p.get("disableLWDA",              false);
    float       splitBeta               = p.get("splitBeta",                0.3f); // 0.f for fast build
    bool        colwertTriangles        = p.get("colwertTriangles",         true);
    const char* squeezeModeStr          = p.get("squeezeMode",              "default");
    float       allocExtra              = p.get("allocExtra",               0.0f);
    int         maxBranchingFactor      = p.get("maxBranchingFactor",       8);
    size_t      headerSize              = p.get("headerSize",               corelib::roundUpPow2(sizeof(BvhHeader),sizeof(BvhNode)));
    int         maxLeafSize             = p.get("maxLeafSize",              3);
    bool        bakePrimBitsToTriangles = p.get("bakePrimBitsToTriangles",  false);

    m_useLwda               = (m_lwdaUtils && !m_disableLwda && !disableLWDA);
    m_inputType             = input.inputType;
    m_supportRefit          = p.get("supportRefit",         false);
    m_refitOnly             = p.get("refitOnly",            false);
    m_freeTempsAfterBuild   = p.get("freeTemps",            !m_refitOnly); // Free temps by default unless refitting.
    m_lwdaTimingRepeats     = p.get("lwdaTimingRepeats",    k_lwdaTimingRepeats.get());

    // This will likely go away, and primBits will become mandatory
    m_usePrimBits           = p.get( "usePrimBits", false );

    // Check for errors.

    if (strcmp(builder, "BVH8") != 0)
        throw IlwalidValue(RT_EXCEPTION_INFO, "Builder must be 'BVH8'");

    if (!(splitBeta >= 0.0f))
        throw IlwalidValue(RT_EXCEPTION_INFO, "splitBeta must be non-negative", splitBeta);

    if (m_refitOnly && !m_supportRefit)
        throw IlwalidOperation(RT_EXCEPTION_INFO, "Tried to set refitOnly=1 without supportRefit=1");

    if (!(allocExtra >= 0.0f))
        throw IlwalidValue(RT_EXCEPTION_INFO, "allocExtra must be non-negative", allocExtra);

    if (m_lwdaTimingRepeats < 0)
        throw IlwalidValue(RT_EXCEPTION_INFO, "lwdaTimingRepeats must be non-negative", m_lwdaTimingRepeats);

    // TODO: We need some validation of input buffers
    //if (!model.isValid())
    //    throw IlwalidOperation(RT_EXCEPTION_INFO, "Invalid model buffers specified");

    if (headerSize > 0 && headerSize < sizeof(BvhHeader))
      throw IlwalidValue( RT_EXCEPTION_INFO, "specified headerSize too small" );
    
    // Parse squeezeMode.

    WideBvhPlanner::SqueezeMode squeezeMode;
    if (strcmp(squeezeModeStr, "default") == 0)         squeezeMode = WideBvhPlanner::SqueezeMode_Default;
    else if (strcmp(squeezeModeStr, "maxsqueeze") == 0) squeezeMode = WideBvhPlanner::SqueezeMode_MaxSqueeze;
    else if (strcmp(squeezeModeStr, "nosqueeze") == 0)  squeezeMode = WideBvhPlanner::SqueezeMode_NoSqueeze;
    else throw IlwalidValue(RT_EXCEPTION_INFO, "Invalid squeezeMode specified", squeezeModeStr);

    // Sanitize parameters.

    if (!refitPossible)
        m_refitOnly = false;

    if (input.isAABBs())
    {
        splitBeta = 0.0f;
        colwertTriangles = false;
    }

    if (m_supportRefit)
        splitBeta = 0.0f;

    m_numInputPrims   = input.numPrimitives;
    uint64_t maxSplitPrims64 = m_numInputPrims + (uint64_t)std::max((float)m_numInputPrims * splitBeta, 0.0f);
    m_maxSplitPrims   = (int)maxSplitPrims64;
    m_enableSplits    = (m_maxSplitPrims > m_numInputPrims);

    if (maxSplitPrims64 >= (1 << 30)) // See ModelPointers::loadPrimitiveAABB() for rationale.
        throw IlwalidValue(RT_EXCEPTION_INFO, "Number of primitives exceeds internal limits after splitting");

    if( m_inputType != IT_TRI )
        bakePrimBitsToTriangles = false;

    // Reset buffers, retaining outputs that we need from the previous round.
    {
        BuildingBlockBuffers old = m_buf;
        m_buf = BuildingBlockBuffers();

        // Retain outputs only if we are refitting.

        if (!m_refitOnly)
        {
            m_outputBuffer.unmaterialize().detachChildren();
            m_buf.header.setNumBytes(headerSize);
            m_buf.numNodes.setNumElems(1);
        }
        else
        {
            m_buf.header                = old.header;
            m_buf.nodes                 = old.nodes;
            m_buf.nodeAux               = old.nodeAux;
            m_buf.triangles             = old.triangles;
            m_buf.remap                 = old.remap;
            m_buf.apexPointMap          = old.apexPointMap;
            m_buf.bvhInstanceData       = old.bvhInstanceData;
            m_buf.numNodes              = old.numNodes;
            m_buf.primBits              = old.primBits;
            m_buf.arrayBaseGlobalIndex  = old.arrayBaseGlobalIndex;
            m_buf.arrayTransitionBits   = old.arrayTransitionBits;
            m_buf.blockStartArrayIndex  = old.blockStartArrayIndex;
            m_buf.geometryIndexArray    = old.geometryIndexArray;
            m_buf.inputTriangleDescs    = old.inputTriangleDescs;
        }

        // Never retain temps.

        m_tempBuffer.unmaterialize().detachChildren();
    }

    ModelBuffers model;
    {
        InputAdapter::Config c;

        c.useLwda                 = m_useLwda;
        c.lwdaUtils               = m_lwdaUtils;

        c.inBuffers               = &input;

        c.outArrayTransitionBits  = m_buf.arrayTransitionBits;
        c.outBlockStartArrayIndex = m_buf.blockStartArrayIndex;
        c.outArrayBaseGlobalIndex = m_buf.arrayBaseGlobalIndex;
        c.outGeometryIndexArray   = m_buf.geometryIndexArray;

        c.usePrimBits             = m_usePrimBits;
        c.refitOnly               = m_refitOnly;

        c.allowedPrimBitsFormats  = PRIMBITS_LEGACY_DIRECT_32;

        c.outAabbs                = m_buf.splitAABBs;
        c.outTmpPrimBits          = m_buf.tmpPrimBits;
        c.outBvhInstanceData      = m_buf.bvhInstanceData;
        c.outAabbArray            = m_buf.inputAabbArray;
        c.outTriangleDescArray    = m_buf.inputTriangleDescs;
        c.outTrianglePtrArray     = m_buf.inputTrianglePtrs;
        c.outCoalescedVertices    = m_buf.coalescedVertices;

        m_inputAdapter.configure(c);

        model                     = m_inputAdapter.getModel();
    }

    // Configure ApexPointMapConstructor.
    {
        ApexPointMapConstructor::Config c;
        c.lwca                  = (m_useLwda) ? m_lwdaUtils : NULL;
        //c.apmResolution = 5; // fast trace
        //c.apmResolution = 1; // fast build
        p.get(c.apmResolution,  "apmResolution");

        c.outApexPointMap       = m_buf.apexPointMap;
        c.tempBuffer            = m_buf.apmTemp;
        c.inModel               = model;

        size_t oldApmSize = m_buf.apexPointMap.getNumBytes();
        m_apmConstructor.configure(c);

        if (m_refitOnly && m_buf.apexPointMap.getNumBytes() != oldApmSize)
            throw IlwalidOperation(RT_EXCEPTION_INFO, "apmResolution must remain the same when refitting");
    }

    // Configure MortonTriangleSplitter.

    if (m_enableSplits)
    {
        MortonTriangleSplitter::Config c;
        c.lwca                              = (m_useLwda) ? m_lwdaUtils : NULL;
        c.maxInputTris                      = m_numInputPrims;
        c.maxOutputPrims                    = m_maxSplitPrims;

        c.splitBeta                         = splitBeta;
        p.get(c.splitTuningRounds,          "splitTuningRounds");
        p.get(c.splitPriorityX,             "splitPriorityX");
        p.get(c.splitPriorityY,             "splitPriorityY");
        p.get(c.splitMaxAABBsPerTriangle,   "splitMaxAABBsPerTriangle");
        p.get(c.splitEpsilon,               "splitEpsilon");

        c.outSplitAABBs                     = m_buf.splitAABBs;
        c.outPrimIndices                    = m_buf.primOrder;
        c.outPrimRange                      = m_buf.splitPrimRange;
        c.tempBuffer                        = m_buf.splitterTemp;

        c.inTriOrder                        = EmptyBuf;
        c.inModel                           = model;
        c.inApexPointMap                    = m_buf.apexPointMap;

        m_splitter.configure(c);
    }

    // Configure MortonSorter.

    if (!m_refitOnly)
    {
        MortonSorter::Config c;
        c.lwca                      = (m_useLwda) ? m_lwdaUtils : NULL;
        c.maxPrims                  = m_maxSplitPrims;
        p.get(c.bytesPerMortonCode, "bytesPerMortonCode");

        c.outMortonCodes            = m_buf.mortonCodes;
        c.outPrimOrder              = m_buf.primOrder;
        c.outPrimRange              = m_buf.sortPrimRange;
        c.tempBuffer                = m_buf.mortonSorterTemp;

        c.inPrimOrder               = (m_enableSplits) ? m_buf.primOrder : EmptyBuf;
        c.inModel                   = ModelBuffers(model, m_buf.splitAABBs);
        c.inApexPointMap            = m_buf.apexPointMap;

        m_mortonSorter.configure(c);
    }

    // Configure AacBuilder.

    if (!m_refitOnly)
    {
        AacBuilder::Config c;
        c.lwca                      = (m_useLwda) ? m_lwdaUtils : NULL;
        c.maxPrims                  = m_maxSplitPrims;

        p.get(c.bytesPerMortonCode, "bytesPerMortonCode");
        //c.aacMaxClusterSize = 4; // fast build
        p.get(c.aacMaxClusterSize,  "aacMaxClusterSize");
        p.get(c.aacPenaltyTermCoef, "aacPenaltyTermCoef");

        c.outNodes                  = m_buf.binaryNodes;
        c.tempBuffer                = m_buf.aacBuilderTemp;

        c.inPrimOrder               = m_buf.primOrder;
        c.inMortonCodes             = m_buf.mortonCodes;
        c.inPrimRange               = m_buf.sortPrimRange;
        c.inModel                   = ModelBuffers(model, m_buf.splitAABBs);
        c.inApexPointMap            = m_buf.apexPointMap;

        m_aacBuilder.configure(c);
    }

    // Configure WideBvhPlanner.

    if (!m_refitOnly)
    {
        WideBvhPlanner::Config c;
        c.lwca                      = (m_useLwda) ? m_lwdaUtils : NULL;
        c.maxBranchingFactor        = maxBranchingFactor;
        c.maxLeafSize               = maxLeafSize;
        c.squeezeMode               = squeezeMode;

        c.sahNodeCost               = p.get("sahNodeCost", 1.0f);
        c.sahPrimCost               = p.get("sahPrimCost", input.inputType == IT_INSTANCE ? 2.0f : 1.0f);
        c.sahRootCost               = 0.0f; // don't care

        c.allowMultipleRoots        = false;
        c.minAreaPerRoot            = 0.0f; // don't care
        c.sahTopLevelCost           = 0.0f; // don't care

        c.outNumWideNodes           = m_buf.numNodes;
        c.outNumRoots               = EmptyBuf; // don't care
        c.outSubtreeRoots           = EmptyBuf; // don't care
        c.tempBufferA               = m_buf.plannerTempA;
        c.tempBufferB               = m_buf.plannerTempB;

        c.ioBinaryNodes             = m_buf.binaryNodes;
        c.inPrimRange               = m_buf.sortPrimRange;
        c.inApexPointMap            = m_buf.apexPointMap;

        m_planner.configure(c);
    }

    // Configure BVH8Constructor.

    if (!m_refitOnly)
    {
        BVH8Constructor::Config c;
        c.lwca                  = (m_useLwda) ? m_lwdaUtils : NULL;
        c.maxPrims              = m_maxSplitPrims;
        c.maxNodes              = m_planner.getMaxWideNodes();
        c.maxBranchingFactor    = maxBranchingFactor;
        c.maxLeafSize           = maxLeafSize;

        c.outNodes              = m_buf.nodes;
        c.outNodeAux            = m_buf.nodeAux;
        c.outRemap              = m_buf.remap;
        c.tempBuffer            = m_buf.constructorTemp;

        c.inNumNodes            = m_buf.numNodes;
        c.inBinaryNodes         = m_buf.binaryNodes;

        m_constructor.configure(c);
    }

    // Configure BVH8Fitter.
    {
        BVH8Fitter::Config c;
        c.lwca                      = (m_useLwda) ? m_lwdaUtils : NULL;
        c.maxPrims                  = m_maxSplitPrims;
        c.maxLeafSize               = maxLeafSize;
        p.get(c.numReorderRounds,   "numReorderRounds");

        c.translateIndices          = m_enableSplits;
        c.removeDuplicates          = (m_enableSplits && maxLeafSize > 1);
        c.reorderChildSlots         = p.get("reorderChildSlots", !m_refitOnly); // Default: enabled for rebuild, disabled for refit.
        c.colwertTriangles          = colwertTriangles;
        c.supportRefit              = m_supportRefit;

        c.ioNodes                   = m_buf.nodes;
        c.ioNodeAux                 = m_buf.nodeAux;
        c.ioRemap                   = m_buf.remap;
        c.outTriangles              = m_buf.triangles;
        c.tempBuffer                = m_buf.fitterTemp;

        c.inNumNodes                = m_buf.numNodes;
        c.inModel                   = ModelBuffers(model, m_buf.splitAABBs);

        m_fitter.configure(c);
    }

    // Configure GatherPrimBits

    if( m_usePrimBits && m_inputType != IT_INSTANCE )
    {
        GatherPrimBits::Config c;

        c.lwca                      = (m_useLwda) ? m_lwdaUtils : NULL;
        c.flags                     = m_inputAdapter.getPrimBitsFlags();
        c.numRemaps                 = m_maxSplitPrims;
        c.inPrimBitsRaw             = m_buf.tmpPrimBits.reinterpretRaw();
        c.inRemap                   = m_buf.remap;
        c.numPrims                  = m_numInputPrims;
        c.primBitsFormat            = m_inputAdapter.getPrimBitsFormat();

        if( bakePrimBitsToTriangles )
        {
            c.useBufferOverlay      = true;
            c.outPrimBitsOffset     = offsetof(BVH8Triangle, primBits);
            c.outPrimBitsStride     = sizeof(BVH8Triangle);
            c.outPrimBitsRaw        = m_buf.triangles.reinterpretRaw();
        }
        else
        {
            c.outPrimBitsRaw        = m_buf.primBits.reinterpretRaw();
        }

        m_gatherPrimBits.configure(c);
    }

    // Group the temp buffers into manageable pieces.

    BufferRef<> bnodeTemps        = overlay(m_buf.binaryNodes, m_buf.mortonSorterTemp);                   // ~ 32 bytes * maxSplitPrims
    BufferRef<> mcTemps           = overlay(m_buf.mortonCodes, m_buf.plannerTempA, m_buf.splitterTemp);   // ~ 8  bytes * maxSplitPrims
    BufferRef<> aacTemps          = overlay(m_buf.aacBuilderTemp, m_buf.plannerTempB);                    // ~ 8  bytes * maxSplitPrims
    BufferRef<> aabbTemps         = aggregate(m_buf.splitAABBs);                                          // 0-15 bytes * maxSplitPrims
    BufferRef<> primOrder         = aggregate(m_buf.primOrder);                                           // ~ 4  bytes * maxSplitPrims
    BufferRef<> nodeAuxTemp;      // placeholder for nodeAux, see below                                   // 0-3  bytes * maxSplitPrims
    BufferRef<> rangeTemps        = aggregate(m_buf.splitPrimRange, m_buf.sortPrimRange);                 // <tiny>
    BufferRef<> tinyTemps         = overlay(m_buf.apmTemp, m_buf.constructorTemp, m_buf.fitterTemp);      // <tiny>

    // Further group the temps based on whether we can overlay them with specific output buffers.

    BufferRef<> nodeOverlays      = aggregate(mcTemps, aacTemps);                     //  ~ 16 bytes * maxSplitPrims
    BufferRef<> triOverlays       = aggregate(bnodeTemps, aabbTemps, nodeAuxTemp);    // 32-50 bytes * maxSplitPrims
    BufferRef<> remapOverlays     = primOrder;                                        //  ~ 4  bytes * maxSplitPrims
    BufferRef<> separateTemps     = overlay(rangeTemps, tinyTemps);                   // <tiny>

    // Layout top-level buffers.

    m_tempBuffer.aggregate(separateTemps);

    if (!m_refitOnly)
    {
        m_outputBuffer
            .aggregate(m_buf.header)            
            .aggregate(overlay(m_buf.nodes, nodeOverlays))
            .aggregate(overlay(m_buf.remap, remapOverlays))
            .aggregate(m_buf.bvhInstanceData)
            .aggregate(m_buf.apexPointMap)
            .aggregate(m_buf.numNodes);

        if (m_supportRefit)
            m_outputBuffer.aggregate(m_buf.nodeAux);
        else
            nodeAuxTemp.aggregate(m_buf.nodeAux);
    }

    if (colwertTriangles && !m_refitOnly)
        m_outputBuffer.aggregate(overlay(m_buf.triangles, triOverlays));
    else
        m_tempBuffer.aggregate(triOverlays);

    m_tempBuffer
        .aggregate(m_buf.inputAabbArray)
        .aggregate(m_buf.inputTrianglePtrs)
        .aggregate(m_buf.coalescedVertices)
        .aggregate(m_buf.tmpPrimBits);

    if (!m_refitOnly)
    {
        m_outputBuffer
            .aggregate(m_buf.primBits);

        BufferRef<> &refitReusable = m_supportRefit ? m_outputBuffer : m_tempBuffer;
        refitReusable
            .aggregate(m_buf.arrayTransitionBits)
            .aggregate(m_buf.blockStartArrayIndex)
            .aggregate(m_buf.arrayBaseGlobalIndex)
            .aggregate(m_buf.geometryIndexArray)
            .aggregate(m_buf.inputTriangleDescs);
    }

    // The builder lwrrently relies on a previous configure in order to do 
    // a refit. If the previous configure did not happen, then this would be it,
    // so call again.
    if(!m_refitOnly && p.get("refitOnly", false)) 
      configureBuild( input );


    // Materialize top-level buffers.

    m_outputBuffer.setAllocExtra(allocExtra).materialize(m_lwdaUtils);
    m_tempBuffer.setAllocExtra(allocExtra).materialize(m_lwdaUtils);
    m_readbackBuffer.setAllocExtra(0.0f).materialize(m_lwdaUtils);
}

//------------------------------------------------------------------------------

void BVH8Builder::exelwteBuild(void)
{
    // Allocate buffers up front to make sure we have enough memory.

    MemorySpace memSpace = (m_useLwda) ? MemorySpace_LWDA : MemorySpace_Host;
    if( m_buildCount == 0 )
    {
        m_outputBuffer  .access(AccessType_Allocate, memSpace);
        m_readbackBuffer.access(AccessType_Allocate, memSpace);
        m_readbackBuffer.access(AccessType_Allocate, MemorySpace_Host);
        m_tempBuffer    .access(AccessType_Allocate, memSpace);
    }

    // Buffers are initially uninitialized (except outputBuffer if refitting)

    if(m_refitOnly)
      m_outputBuffer.readWriteLWDA();
    else
      m_outputBuffer.markAsUninitialized();
    m_readbackBuffer.markAsUninitialized();
    m_tempBuffer.markAsUninitialized();

    // Refit => fast path.

    if (m_refitOnly)
    {
        if( m_buildCount == 0 )
        {
          // The following buffers are in overlays, so the earlier initialization
          // doesn't propagate to them. Call writeLWDA() to manually mark them as 
          // initialized.
          m_buf.nodes.writeLWDA();
          m_buf.remap.writeLWDA();
        }
        m_inputAdapter.execute();
        m_apmConstructor.execute(); // Need to update the apex point map when refitting, even though the builder itself does not need it.
        m_fitter.execute();
        m_tempBuffer.markAsUninitialized();
        return;
    }


    // Init header

    if( m_buf.header.getNumElems() > 0 )
    {
      m_header.flags                  = HF_TYPE_BVH8;
      m_header.numEntities            = m_numInputPrims;
      m_header.numNodes               = (unsigned)m_buf.nodes.getNumElems();
      m_header.nodesOffset            = m_buf.nodes.getOffsetInTopmost();
      m_header.nodeParentsOffset      = 0;
      m_header.numRemaps              = (unsigned)m_buf.remap.getNumElems();
      m_header.remapOffset            = m_buf.remap.getOffsetInTopmost();
      m_header.apmOffset              = m_buf.apexPointMap.getOffsetInTopmost();
      m_header.numTriangles           = (unsigned)m_buf.triangles.getNumElems();   
      m_header.trianglesOffset        = m_buf.triangles.getOffsetInTopmost();    
      m_header.numInstances           = (unsigned)m_buf.bvhInstanceData.getNumElems();
      m_header.instanceDataOffset     = m_buf.bvhInstanceData.getOffsetInTopmost();
      m_header.size                   = m_outputBuffer.getNumBytes();
      m_header.primBitsOffset         = m_buf.primBits.getOffsetInTopmost();
      m_header.numPrimitiveIndexBits  = m_inputAdapter.getNumPrimitiveIndexBits();

      if( m_useLwda )
        m_lwdaUtils->memcpyHtoDAsync( m_buf.header.writeDiscardLWDA(), &m_header, sizeof(m_header) );
      else
        memcpy( m_buf.header.writeDiscardHost(), &m_header, sizeof(m_header) );
    }

    // Execute stages.

    m_inputAdapter.execute();

    m_apmConstructor.execute();

    BufferRef<Range> primRange = EmptyBuf; // (0, maxSplitPrims)
    if (m_enableSplits)
    {
        m_splitter.execute(EmptyBuf);
        primRange = m_buf.splitPrimRange;
    }

    m_mortonSorter.execute(primRange);
    m_aacBuilder.execute();
    m_planner.execute();
    m_constructor.execute();
    m_fitter.execute();

    if( m_usePrimBits && m_inputType != IT_INSTANCE )
      m_gatherPrimBits.execute();

    // Ilwalidate the temp buffer to prevent accidental reads.

    m_tempBuffer.markAsUninitialized();
}

//------------------------------------------------------------------------------

void BVH8Builder::exelwteBuildWithTiming(void)
{
    // Timing disabled => just call exelwteBuild() directly.
    // Several repeats => do the first one without timing to make sure we get a clean report.

    if (!m_useLwda || !m_lwdaTimingRepeats)
        exelwteBuild();
    else
    {
        if (m_lwdaTimingRepeats > 1)
            exelwteBuild();

        // Enable timing.

        m_lwdaUtils->resetTiming(true);

        // Execute the build multiple times and take the minimum, for accurate results.

        for (int repeat = 0; repeat < std::max(m_lwdaTimingRepeats - 1, 1); repeat++)
        {
            m_lwdaUtils->repeatTiming();
            m_lwdaUtils->beginTimer("BVH8Builder");
            exelwteBuild();
            m_lwdaUtils->endTimer();
        }

        // Print results.

        printf("\n");
        m_lwdaUtils->printTiming(m_numInputPrims);
        printf("\n");

        // Disable timing.

        m_lwdaUtils->resetTiming(false);
    }

    // Print memory usage if requested.

    if (k_memPrintUsage.get())
    {
        size_t totalMin = m_outputBuffer.getAllocSize() + m_readbackBuffer.getAllocSize();
        size_t totalMax = totalMin + m_tempBuffer.getAllocSize();
        std::cerr << "MemMax:" << totalMax << " MemMin:" << totalMin << "\n";
        std::cerr << "Output/Tri:" << (double)totalMin / (double)m_numInputPrims << " Temp/Tri:" << (double)(totalMax - totalMin) / (double)m_numInputPrims << "\n";
    }
}

//------------------------------------------------------------------------------

void BVH8Builder::copyHeader(void* dst, size_t dstSize)
{
  RT_ASSERT(dstSize >= sizeof(m_header));
  memcpy(dst, &m_header, sizeof(m_header));
}

//------------------------------------------------------------------------
