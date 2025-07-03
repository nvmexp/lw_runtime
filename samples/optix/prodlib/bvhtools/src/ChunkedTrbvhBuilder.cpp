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

#include "ChunkedTrbvhBuilder.hpp"
#include <prodlib/bvhtools/src/common/Utils.hpp>
#include <prodlib/exceptions/IlwalidValue.h>
#include <prodlib/exceptions/IlwalidOperation.h>
#include <prodlib/system/Knobs.h>
#include <corelib/math/MathUtil.h>
#include <corelib/misc/String.h>
#include <stdint.h>
#include <fstream>
#include <algorithm>

using namespace prodlib::bvhtools;
using namespace prodlib;

namespace {
  Knob<bool>  k_memPrintUsage( RT_DSTRING("bvhtools.memPrintUsage"), 0, RT_DSTRING("Print max and final memory usage"));
  Knob<int>   k_maxPrimsPerChunk( RT_DSTRING("bvhtools.maxPrimsPerChunk"), 0, RT_DSTRING("Override the maximum number of prims per chunk."));
  Knob<int>   k_lwdaTimingRepeats( RT_DSTRING("bvhtools.lwdaTimingRepeats"), 0, RT_DSTRING("Run BVH builder several times and print detailed timing info"));
  
  Knob<std::string> k_dumpPath( RT_DSTRING("bvhtools.dumpPath"), "", RT_DSTRING("Path for dump files") );
  Knob<std::string> k_checkpoints( RT_DSTRING("bvhtools.checkpoints"), "", RT_DSTRING("[read|write][_all|_<name>]"));
}


static int fileCounter = 0;

//------------------------------------------------------------------------

static std::string pathJoin( const std::string& base, const std::string& name )
{
  if( base.empty() )
    return name;
  else 
    return base + "/" + name;
}

//------------------------------------------------------------------------

static bool parseCheckpoints( const std::string& checkpoints, std::string& action, std::string& name)
{
  if( checkpoints.empty() )
    return false;

  size_t pos = checkpoints.find( '_' );
  if( pos == std::string::npos )
  {
    action = checkpoints;
    name = "all";
  }
  else
  {
    action = checkpoints.substr(0,pos);
    name = checkpoints.substr(pos+1);
  }

  return action == "write" || action == "read";
}

//------------------------------------------------------------------------

template <class T>
static std::ostream& operator<<(std::ostream& out, const prodlib::bvhtools::BufferRef<T>& buffer)
{
  size_t size = buffer.getNumBytes();
  out.write( (const char*)&size, sizeof(size) ); 
  if( size )
  {
    buffer.access(AccessType_Allocate, MemorySpace_Host);
    out.write((const char*)buffer.readHost(), size);
  }
  return out;
}

//------------------------------------------------------------------------

template <class T>
static std::istream& operator>>(std::istream& in, const prodlib::bvhtools::BufferRef<T>& buffer)
{
  size_t size = 0;
  in.read((char*)&size, sizeof(size));
  if(size != buffer.getNumBytes())
  {
    lwarn << "Sizes in file (" << size << ") does not match size of buffer (" << buffer.getNumBytes() << ")\n";
    in.seekg(size, in.lwr);
  }
  else 
  {
     in.read((char*)buffer.writeDiscardHost(), size);
     buffer.readLWDA(); // Make sure that data gets uploaded.
  }

  return in;
}

//------------------------------------------------------------------------

static void dumpInput(const char* builderSpec, bool disableLWDA, const InputBuffers& input)
{
  // MANTODO: Make this work again
#if 0
  if( !input.primAabbs.isMaterialized() && !input.indices.isMaterialized() && !input.vertices.isMaterialized() )
    return;

  std::string action, cpName;
  if( !parseCheckpoints(k_checkpoints.get(), action, cpName) )
    return;
  if(action == "read")
  {
    fileCounter++;
    return;
  }

  std::string filename = corelib::stringf( "bvhtools-%03d-input.bin", fileCounter++ );
  if(cpName != "all"  && cpName != "input")
    return;

  filename = pathJoin( k_dumpPath.get(), filename );
  std::fstream fs(filename, std::ios::out | std::ios::binary );
  if(fs.fail())
  {
    lwarn << "Could not open file " << filename << "\n";
    return;
  }
  
  fs << builderSpec   << "\n";
  fs << disableLWDA   << "\n";
  fs << input.inputType << "\n";
  fs << input.numPrimitives << "\n";
  fs << input.indexStride   << "\n";
  fs << input.vertexStride  << "\n"; 

  fs << input.primAabbs;
  fs << input.indices;
  fs << input.vertices;
  fs << input.aabbs;
  fs << input.instanceDescs;
#endif
}


//------------------------------------------------------------------------

void ChunkedTrbvhBuilder::checkpoint(const char* name)
{
  std::string action, cpName;
  if(!parseCheckpoints(k_checkpoints.get(), action, cpName))
    return;  

  std::string filename = corelib::stringf( "bvhtools-%03d-%s.bin", fileCounter++, name );
  if(cpName != "all"  && cpName != name)
    return;

  bool write = (action == "write");
  filename = pathJoin( k_dumpPath.get(), filename );
  std::fstream fs(filename, std::ios::binary | (write ? std::ios::out : std::ios::in) );
  if(fs.fail())
  {
    lwarn << "Could not open file " << name << "\n";
    return;
  }
  
  const char* FILE_TYPE = "CHECKPT\n"; // 8 bytes
  if( write )
  {
    fs << FILE_TYPE;
    fs << m_outputBuffer;
    fs << m_tempBuffer;
    fs << m_readbackBuffer;
  }
  else 
  {
    char buf[8];
    fs.read( buf, 8 );
    RT_ASSERT( std::string( buf, 8 ) == FILE_TYPE );

    fs >> m_outputBuffer;
    fs >> m_tempBuffer;
    fs >> m_readbackBuffer;      
  }  
}



//------------------------------------------------------------------------

ChunkedTrbvhBuilder::ChunkedTrbvhBuilder(bool lwdaAvailable)
:   m_lwdaUtils             ((lwdaAvailable) ? &m_lwdaUtilsImpl : NULL),

    m_disableLwda           (false),

    m_useLwda               (false),
    m_inputType             (IT_TRI),
    m_enableChunking        (false),
    m_outputOptixNodes      (false),
    m_outputWoopTriangles   (true),
    m_freeTempsAfterBuild   (false),
    m_numInputPrims         (0),
    m_lwdaTimingRepeats     (0)
{
}

//------------------------------------------------------------------------

ChunkedTrbvhBuilder::~ChunkedTrbvhBuilder(void)
{
}

//------------------------------------------------------------------------

void ChunkedTrbvhBuilder::setLwdaStream(lwdaStream_t stream)
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

void ChunkedTrbvhBuilder::build(const std::vector<TriangleMesh>& meshes, bool meshInDeviceMem)
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

    InputBuffers input( meshes, ( meshInDeviceMem ) ? MemorySpace_LWDA : MemorySpace_Host );
    buildFromInputBuffers( input );
}

//------------------------------------------------------------------------

void ChunkedTrbvhBuilder::build(int numAabbs, const PrimitiveAABB* aabbs, bool aabbsInDeviceMem)
{
    // Check for errors.

    if (numAabbs < 0)
        throw IlwalidValue( RT_EXCEPTION_INFO, "numAabbs must be non-negative", numAabbs );

    if (!aabbs && numAabbs > 0)
        throw IlwalidValue( RT_EXCEPTION_INFO, "NULL aabbs pointer" );

    if (aabbsInDeviceMem && !m_lwdaUtils)
        throw IlwalidValue( RT_EXCEPTION_INFO, "LWCA not available" );

    InputBuffers input(numAabbs, aabbs, (aabbsInDeviceMem) ? MemorySpace_LWDA : MemorySpace_Host);
    buildFromInputBuffers(input);
}

//------------------------------------------------------------------------

void ChunkedTrbvhBuilder::build(int numAabbs, int motionSteps, const float* aabbs, bool aabbsInDeviceMem)
{
  // Check for errors.

    if (numAabbs < 0)
        throw IlwalidValue( RT_EXCEPTION_INFO, "numAabbs must be non-negative", numAabbs );

    if (motionSteps < 1)
      throw IlwalidValue( RT_EXCEPTION_INFO, "motionSteps must be positive", motionSteps );

    if (!aabbs && numAabbs > 0)
        throw IlwalidValue( RT_EXCEPTION_INFO, "NULL aabbs pointer" );

    if (aabbsInDeviceMem && !m_lwdaUtils)
        throw IlwalidValue( RT_EXCEPTION_INFO, "LWCA not available" );

    InputBuffers input(numAabbs, motionSteps, (const AABB*) aabbs, (aabbsInDeviceMem) ? MemorySpace_LWDA : MemorySpace_Host);
    buildFromInputBuffers(input);
}

//------------------------------------------------------------------------------
void ChunkedTrbvhBuilder::build(int numInstances, const InstanceDesc* instancesPtr, bool inDeviceMem)
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

//------------------------------------------------------------------------------

void ChunkedTrbvhBuilder::build(int numInputs, const RtcBuildInput* buildInputs, bool inDeviceMem)
{
  if (numInputs > 0 && buildInputs[0].type == RTC_BUILD_INPUT_TYPE_INSTANCE_ARRAY)
  {
      const RtcBuildInputInstanceArray &inst = buildInputs[0].instanceArray;
      build(inst.numInstances, (const InstanceDesc*) inst.instanceDescs, inDeviceMem);
      return;
  }

  if( inDeviceMem && !m_lwdaUtils )
    throw IlwalidValue( RT_EXCEPTION_INFO, "LWCA not available" );

  InputBuffers input(numInputs, buildInputs, (inDeviceMem) ? MemorySpace_LWDA : MemorySpace_Host);
  buildFromInputBuffers(input);
}

//------------------------------------------------------------------------------
void ChunkedTrbvhBuilder::computeMemUsageStatic( const char* builderSpec, bool buildGpu, const InputBuffers& input, MemoryUsage* memUsage )
{
    // Configure build.

    ChunkedTrbvhBuilder tmp( buildGpu );
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

void ChunkedTrbvhBuilder::computeMemUsageStatic(const char* builderSpec, bool buildGpu, int numPrims, int motionSteps, InputType type, MemoryUsage* memUsage )
{
    // TODO: This is very similar between the three builders. Refactor.
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
      input.reset(new InputBuffers(numPrims, motionSteps, (const AABB*)dummyBuffer, memSpace));
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
    memUsage->nodes *= motionSteps;
}

//------------------------------------------------------------------------

void ChunkedTrbvhBuilder::computeMemUsageStatic( const char* builderSpec, bool buildGpu, int numInputs, const RtcBuildInput* buildInputs, MemoryUsage* memUsage )
{
    // TODO: This is very similar between the three builders. Refactor.
    if (memUsage == NULL)
      return;

    MemorySpace memSpace = MemorySpace_LWDA;
    InputBuffers input( numInputs, buildInputs, memSpace, true );

    computeMemUsageStatic( builderSpec, buildGpu, input, memUsage );
}

//------------------------------------------------------------------------

void ChunkedTrbvhBuilder::computeMemUsage(const char * builderSpec, bool buildGpu, int numPrims, int motionSteps, InputType type, MemoryUsage * memUsage)
{
  computeMemUsageStatic(builderSpec, buildGpu, numPrims, motionSteps, type, memUsage);
}

//------------------------------------------------------------------------

void ChunkedTrbvhBuilder::computeMemUsage(const char* builderSpec, bool buildGpu, int numPrimAabbs, int numTriangles, MemoryUsage* memUsage)
{
  RT_ASSERT(!(numPrimAabbs > 0 && numTriangles > 0));
  int numPrims = 0; 
  InputType type = IT_TRI;
  if (numPrimAabbs > 0)
  {
    numPrims = numPrimAabbs;
    type = IT_PRIMAABB;
  }
  computeMemUsageStatic(builderSpec, buildGpu, numPrims, 1, type, memUsage);
}

//------------------------------------------------------------------------

void ChunkedTrbvhBuilder::computeMemUsage( const char*                      builderSpec,
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

void ChunkedTrbvhBuilder::computeMemUsage(const char* builderSpec, bool buildGpu, int numInputs, const RtcBuildInput* buildInputs, MemoryUsage* memUsage)
{
  computeMemUsageStatic(builderSpec, buildGpu, numInputs, buildInputs, memUsage);
}

//------------------------------------------------------------------------

void ChunkedTrbvhBuilder::setDataLocation(bool inDeviceMem)
{
    MemorySpace memSpace = (inDeviceMem) ? MemorySpace_LWDA : MemorySpace_Host;
    m_outputBuffer.freeMemExcept(memSpace);
    m_tempBuffer.freeMemExcept(memSpace);
    m_readbackBuffer.freeMemExcept(memSpace);
}

//------------------------------------------------------------------------

void ChunkedTrbvhBuilder::freeData( void )
{
    m_outputBuffer.freeMem();
    m_tempBuffer.freeMem();
    m_readbackBuffer.freeMem();
}

//------------------------------------------------------------------------

void ChunkedTrbvhBuilder::buildFromInputBuffers(const InputBuffers & input)
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


//------------------------------------------------------------------------

void ChunkedTrbvhBuilder::configureBuild(const InputBuffers& input)
{
    dumpInput(m_builderSpec.c_str(), m_disableLwda, input);
  
    // Parse builderSpec.

    ParameterList p(m_builderSpec.c_str());

    const char* builder         = p.get("builder",      "TRBVH");
    bool        disableLWDA     = p.get("disableLWDA",  false);
    float       splitBeta       = p.get("splitBeta",    0.3f);
    float       allocExtra      = p.get("allocExtra",   0.0f);
    size_t      headerSize      = p.get("headerSize",   corelib::roundUpPow2(sizeof(BvhHeader),sizeof(BvhNode)));

    m_useLwda               = (m_lwdaUtils && !m_disableLwda && !disableLWDA && m_lwdaUtils->getSMArch() >= 20);
    m_inputType             = input.inputType;
    m_outputOptixNodes      = p.get("optixNodes",   false);
    m_outputWoopTriangles   = p.get("useWoop",      true);
    m_freeTempsAfterBuild   = p.get("freeTemps",    true);
    m_numInputPrims         = input.numPrimitives;
    m_motionSteps           = input.motionSteps;
    m_lwdaTimingRepeats     = p.get("lwdaTimingRepeats", k_lwdaTimingRepeats.get());
    m_noAccelHack           = p.get("noAccelHack",  false);
    m_supportRefit          = p.get("supportRefit", false);
    m_refitOnly             = p.get("refitOnly",    false);

    // This will likely go away, and primBits will become mandatory
    m_usePrimBits           = p.get( "usePrimBits", false );

    // Check for errors.

    if (strcmp(builder, "TRBVH") != 0)
        throw IlwalidValue( RT_EXCEPTION_INFO, "Builder must be 'TRBVH'" );

    if (m_noAccelHack && (splitBeta != 0 || m_outputOptixNodes || m_outputWoopTriangles))
        throw IlwalidValue( RT_EXCEPTION_INFO, "Other options incompatible with noAccelHack");

    if (!(splitBeta >= 0.0f))
        throw IlwalidValue( RT_EXCEPTION_INFO, "splitBeta must be non-negative", splitBeta );

    if (!(allocExtra >= 0.0f))
        throw IlwalidValue( RT_EXCEPTION_INFO, "allocExtra must be non-negative", allocExtra );

    if (m_lwdaTimingRepeats < 0)
        throw IlwalidValue( RT_EXCEPTION_INFO, "lwdaTimingRepeats must be non-negative", m_lwdaTimingRepeats );

    if (m_outputWoopTriangles && m_inputType != IT_TRI)
        throw IlwalidOperation( RT_EXCEPTION_INFO, "Can output woop triangles only for triangle inputs" );

    if (headerSize > 0 && headerSize < sizeof(BvhHeader))
        throw IlwalidValue( RT_EXCEPTION_INFO, "specified headerSize too small" );

    if (m_refitOnly && m_enableChunking)
        throw IlwalidValue( RT_EXCEPTION_INFO, "Chunking not allowed with refit");
    
    if (m_supportRefit && m_inputType == IT_TRI && splitBeta != 0.0f )
        throw IlwalidOperation( RT_EXCEPTION_INFO, "Splitting not allowed with refit" );

    // Sanitize parameters.

    if (m_inputType != IT_TRI)
        splitBeta = 0.0f; // Only do splitting for triangles

    if (m_outputWoopTriangles)
        m_freeTempsAfterBuild = false; // Temp buffer is tiny with Woop triangles => no point in freeing it.

    RemapListLenEncoding listLenEnc = 
        m_outputOptixNodes      ? RLLE_NONE :  
        m_outputWoopTriangles   ? RLLE_COMPLEMENT_LAST :
                                  RLLE_PACK_IN_FIRST;

    if (listLenEnc == RLLE_PACK_IN_FIRST && m_numInputPrims > RLLEPACK_INDEX_MASK+1)
        throw IlwalidValue( RT_EXCEPTION_INFO, "Number of input primitives exceeds internal limits" );

    // Choose chunking parameters.

    uint64_t maxTotalSplitPrims64 = m_numInputPrims + (uint64_t)((float)m_numInputPrims * splitBeta); // MortonTriangleSplitter rounds down for every chunk, so this is a valid upper bound.
    int maxTotalSplitPrims = (int)maxTotalSplitPrims64;
    int maxSplitPrimsPerChunk, preferredTopTreePrims;
    chooseChunkSize(/*out*/maxSplitPrimsPerChunk, /*out*/preferredTopTreePrims, m_numInputPrims, maxTotalSplitPrims, splitBeta, p);
    int maxInputPrimsPerChunk = max((int)((float)maxSplitPrimsPerChunk / (splitBeta + 1.0f)), 1); // round down
    m_enableSplits = (maxTotalSplitPrims > m_numInputPrims && maxSplitPrimsPerChunk > maxInputPrimsPerChunk);

    if (maxTotalSplitPrims64 >= (1 << 30)) // See ModelPointers::loadPrimitiveAABB() for rationale.
        throw IlwalidValue( RT_EXCEPTION_INFO, "Number of primitives exceeds internal limits after splitting" );

    // Decide whether to enable chunking.

    int forceChunking = p.get("forceChunking", -1); // 0 => force-disable, 1 => force-enable, others => let the builder decide.
    bool modelIsSmall = (m_numInputPrims <= maxInputPrimsPerChunk * 3);

    if (m_numInputPrims < 2)        m_enableChunking = false;   // <2 input primitives is a special case throughout the builder => chunking is not supported.
    else if (forceChunking == 0)    m_enableChunking = false;   // Force-disable?
    else if (forceChunking == 1)    m_enableChunking = true;    // Force-enable?
    else if (m_outputWoopTriangles) m_enableChunking = false;   // Temp buffer is tiny with Woop triangles => chunking would be strictly harmful, and is not supported.
    else if (modelIsSmall)          m_enableChunking = false;   // Chunk size is >30% of the input => chunking would increase memory consumption instead of decreasing it.
    else                            m_enableChunking = true;    // Otherwise => enable chunking to conserve memory.

    // Sanitize chunking parameters.

    if (!m_enableChunking)
    {
        maxInputPrimsPerChunk = m_numInputPrims;
        maxSplitPrimsPerChunk = maxTotalSplitPrims;
        preferredTopTreePrims = 0;
    }
    else if (maxTotalSplitPrims > RLLEPACK_INDEX_MASK+1)
        throw IlwalidValue( RT_EXCEPTION_INFO, "Number of primitives exceeds internal limits with chunking." );

    RT_ASSERT(maxInputPrimsPerChunk <= m_numInputPrims);
    RT_ASSERT(maxSplitPrimsPerChunk <= maxTotalSplitPrims);

    // Reset buffers.

    m_buf = BuildingBlockBuffers();
    m_outputBuffer.unmaterialize().detachChildren();
    m_tempBuffer.unmaterialize().detachChildren();
    m_readbackBuffer.unmaterialize().detachChildren();

    // Setup readback buffer. Make sub-buffers subranges so they get read back at same time.

    if (m_enableChunking)
    {
      m_readbackBuffer.setNumElems(1);
      m_buf.numChunks = m_readbackBuffer.getSubrange(0, 1);
    }

    m_buf.numNodes.setNumElems(1); // TODO: is this needed? Repeated below

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
        c.motionSteps             = m_motionSteps;

        c.allowedPrimBitsFormats  = PRIMBITS_LEGACY_DIRECT_32;

        c.outAabbs                = m_buf.tmpAabbs;
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
    // Note: Setting "c.lwca = NULL" for any individual building block will force it to run on the CPU.
    {
        ApexPointMapConstructor::Config c;
        c.lwca                  = (m_useLwda) ? m_lwdaUtils : NULL;
        p.get(c.apmResolution,  "apmResolution");

        c.outApexPointMap       = m_buf.apexPointMap;
        c.tempBuffer            = m_buf.aabbCalcTemp;
        c.inModel               = model;

        m_apmConstructor.configure(c);
    }
    
    if (m_refitOnly)
        m_refitModel = model;

    // Configure MortonSorter for chunker input primitives.

    if (m_enableChunking)
    {
        MortonSorter::Config c;
        c.lwca                  = (m_useLwda) ? m_lwdaUtils : NULL;
        c.maxPrims              = model.numPrimitives;
        c.bytesPerMortonCode    = sizeof(unsigned long long);

        c.outMortonCodes        = m_buf.chunkMortonCodes.reinterpretRaw();
        c.outPrimOrder          = m_buf.chunkPrimOrder;
        c.outPrimRange          = m_buf.sortPrimRange;
        c.tempBuffer            = m_buf.chunkSorterTemp;

        c.inModel               = model;
        c.inApexPointMap        = m_buf.apexPointMap;

        m_chunkSorter.configure(c);
    }

    // Configure Chunker.

    if (m_enableChunking)
    {
        Chunker::Config c;
        c.lwca                  = (m_useLwda) ? m_lwdaUtils : NULL;
        c.numPrims              = m_numInputPrims;
        c.maxChunkPrims         = maxInputPrimsPerChunk;
        c.preferredTopTreePrims = preferredTopTreePrims;

        c.outPrimRanges         = m_buf.chunkPrimRanges;
        c.outNumChunks          = m_buf.numChunks;
        c.outLwtoffLevel        = m_buf.lwtoffLevel;
        c.allocTrimmedAABBs     = m_buf.trimmedAabbs;
        c.inPrimMorton          = m_buf.chunkMortonCodes;

        m_chunker.configure(c);
    }

    // Configure MortonTriangleSplitter.

    if (m_enableSplits)
    {
        MortonTriangleSplitter::Config c;
        c.lwca                              = (m_useLwda) ? m_lwdaUtils : NULL;
        c.maxInputTris                      = maxInputPrimsPerChunk;
        c.maxOutputPrims                    = maxSplitPrimsPerChunk;

        c.splitBeta                         = splitBeta;
        p.get(c.splitTuningRounds,          "splitTuningRounds");
        p.get(c.splitPriorityX,             "splitPriorityX");
        p.get(c.splitPriorityY,             "splitPriorityY");
        p.get(c.splitMaxAABBsPerTriangle,   "splitMaxAABBsPerTriangle");
        p.get(c.splitEpsilon,               "splitEpsilon");

        c.outSplitAABBs                     = m_buf.tmpAabbs;
        c.outPrimIndices                    = m_buf.bottomPrimOrder;
        c.outPrimRange                      = m_buf.splitPrimRange;
        c.tempBuffer                        = m_buf.splitterTemp;

        c.inTriOrder                        = (m_enableChunking) ? m_buf.chunkPrimOrder : EmptyBuf;
        c.inModel                           = model;
        c.inApexPointMap                    = m_buf.apexPointMap;

        m_splitter.configure(c);
    }
    

    // Callwlate maximum number of nodes, remaps, and trimmed AABBs.

    int maxTopTreePrims = (int)m_buf.trimmedAabbs.getNumElems();                             // Callwlated by Chunker::configure().
    uint64_t maxNodes   = max(maxTotalSplitPrims64, 1ull) + max(maxTopTreePrims - 1, 1) + 2; // Bottom + top + root + OptixColwerter-root.
    uint64_t maxRemaps  = max(maxTotalSplitPrims64, 1ull) + max(maxTopTreePrims, 1);         // Bottom + top. TrbvhBuilder always outputs at least one entry.
    if( maxNodes > INT_MAX )
        throw IlwalidValue( RT_EXCEPTION_INFO, "Required number of bvh nodes exceeds internal limits" );
    

    m_buf.header            .setNumBytes(headerSize);
    m_buf.allNodes          .setNumElems(maxNodes * m_motionSteps);
    m_buf.remap             .setNumElems(maxRemaps);
    m_buf.numNodes          .setNumElems(1);
    m_buf.numRemaps         .setNumElems(1);
    m_buf.trimmedAabbRange  .setNumElems(1);

    // We build at the first motion step, using a subrange
    m_buf.nodes            = m_buf.allNodes.getSubrange(0, maxNodes);

    if (m_supportRefit)
        m_buf.bottomNodeParents.setNumElems(maxNodes);

    if (m_outputOptixNodes)
        m_buf.tmpRemap.setNumElems(maxRemaps);

    // Configure bottom-level MortonSorter.
    {
        MortonSorter::Config c;
        c.lwca                  = (m_useLwda) ? m_lwdaUtils : NULL;
        c.maxPrims              = maxSplitPrimsPerChunk;
        c.bytesPerMortonCode    = sizeof(unsigned long long);

        c.outMortonCodes        = m_buf.bottomMortonCodes.reinterpretRaw();
        c.outPrimOrder          = m_buf.bottomPrimOrder;
        c.outPrimRange          = m_buf.sortPrimRange;
        c.tempBuffer            = m_buf.bottomSorterTemp;

        c.inPrimOrder           = (m_enableSplits) ? m_buf.bottomPrimOrder : (m_enableChunking) ? m_buf.chunkPrimOrder : EmptyBuf;
        c.inModel               = ModelBuffers(model, m_buf.tmpAabbs);
        c.inApexPointMap        = m_buf.apexPointMap;

        m_bottomSorter.configure(c);
    }

    // Configure bottom-level TrbvhBuilder.
    {
        TrbvhBuilder::Config c;
        c.lwca                      = (m_useLwda) ? m_lwdaUtils : NULL;
        c.maxPrims                  = maxSplitPrimsPerChunk;
        c.listLenEnc                = listLenEnc;
        p.get(c.sahNodeCost,        "sahNodeCost");
        p.get(c.sahPrimCost,        "sahPrimCost");
        p.get(c.maxLeafSize,        "maxLeafSize");
        p.get(c.optTreeletSize,     "optTreeletSize");
        p.get(c.optRounds,          "optRounds");
        p.get(c.optGamma,           "optGamma");
        p.get(c.optAdaptiveGamma,   "optAdaptiveGamma");

        c.outNodeParents            = m_buf.bottomNodeParents;
        c.outNodeRange              = m_buf.trbvhNodeRange;
        c.tempBuffer                = m_buf.bottomTrbvhTemp;

        c.ioNodes                   = m_buf.nodes;
        c.ioNumNodes                = m_buf.numNodes;
        c.ioRemap                   = (m_outputOptixNodes) ? m_buf.tmpRemap : m_buf.remap;
        c.ioNumRemaps               = m_buf.numRemaps;
        c.ioMortonCodes             = m_buf.bottomMortonCodes;
        c.inPrimOrder               = m_buf.bottomPrimOrder;
        c.inPrimRange               = m_buf.sortPrimRange;
        c.inModel                   = ModelBuffers(model, m_buf.tmpAabbs);

        m_bottomTrbvh.configure(c);
    }

    // Configure TreeTopTrimmer

    if (m_enableChunking)
    {
        TreeTopTrimmer::Config c;
        c.lwca              = (m_useLwda) ? m_lwdaUtils : NULL;
        c.maxNodes          = max(maxSplitPrimsPerChunk - 1, 1);

        c.ioTrimmed         = m_buf.trimmedAabbs;
        c.ioTrimmedRange    = m_buf.trimmedAabbRange;
        c.ioNodes           = m_buf.nodes;
        c.inLwtoffLevel     = m_buf.lwtoffLevel;
        c.inNodeRange       = m_buf.trbvhNodeRange;
        c.inNodeParents     = m_buf.bottomNodeParents;

        m_trimmer.configure(c);
    }

    // Configure top-level MortonSorter.

    if (m_enableChunking)
    {
        MortonSorter::Config c;
        c.lwca                  = (m_useLwda) ? m_lwdaUtils : NULL;
        c.maxPrims              = maxTopTreePrims;
        c.bytesPerMortonCode    = sizeof(unsigned long long);

        c.outMortonCodes        = m_buf.topMortonCodes.reinterpretRaw();
        c.outPrimOrder          = m_buf.topPrimOrder;
        c.outPrimRange          = m_buf.sortPrimRange;
        c.tempBuffer            = m_buf.topSorterTemp;

        c.inPrimOrder           = EmptyBuf;
        c.inModel               = ModelBuffers(m_buf.trimmedAabbs);
        c.inApexPointMap        = m_buf.apexPointMap;

        m_topSorter.configure(c);
    }

    // Configure top-level TrbvhBuilder.

    if (m_enableChunking)
    {
        TrbvhBuilder::Config c;
        c.lwca                      = (m_useLwda) ? m_lwdaUtils : NULL;
        c.maxPrims                  = maxTopTreePrims;
        c.listLenEnc                = RLLE_NONE;

        c.sahNodeCost               = 1.0f;     // makes no difference
        c.sahPrimCost               = 0.0f;     // makes no difference
        c.maxLeafSize               = 1;        // no collapsing of leaves
        p.get(c.optTreeletSize,     "optTreeletSize");
        p.get(c.optRounds,          "optRounds");
        c.optGamma                  = 1;        // force high-quality build
        c.optAdaptiveGamma          = false;    // force high-quality build

        c.outNodeParents            = m_buf.topNodeParents;
        c.outNodeRange              = m_buf.trbvhNodeRange;
        c.tempBuffer                = m_buf.topTrbvhTemp;

        c.ioNodes                   = m_buf.nodes;
        c.ioNumNodes                = m_buf.numNodes;
        c.ioRemap                   = (m_outputOptixNodes) ? m_buf.tmpRemap : m_buf.remap;
        c.ioNumRemaps               = m_buf.numRemaps;
        c.ioMortonCodes             = m_buf.topMortonCodes;
        c.inPrimOrder               = m_buf.topPrimOrder;
        c.inPrimRange               = m_buf.sortPrimRange;
        c.inModel                   = ModelBuffers(m_buf.trimmedAabbs);

        m_topTrbvh.configure(c);
    }

    // Configure TopTreeConnector

    if (m_enableChunking)
    {
        TopTreeConnector::Config c;
        c.lwca              = (m_useLwda) ? m_lwdaUtils : NULL;
        c.maxNodes          = max(maxTopTreePrims - 1, 1);

        c.outRoot           = m_buf.nodes.getSubrange((m_outputOptixNodes) ? 1 : 0, 1);
        c.ioNodes           = m_buf.nodes;
        c.inNodeRange       = m_buf.trbvhNodeRange;
        c.inRemap           = (m_outputOptixNodes) ? m_buf.tmpRemap : m_buf.remap;
        c.inTrimmedAabbs    = m_buf.trimmedAabbs;

        m_connector.configure(c);
    }

    // Configure OptixColwerter.

    if (m_outputOptixNodes)
    {
        OptixColwerter::Config c;
        c.lwca              = (m_useLwda) ? m_lwdaUtils : NULL;
        c.maxNodes          = (int)maxNodes;
        c.bake              = true;
        c.shiftNodes        = false;

        c.outRemap          = m_buf.remap;
        c.tempBuffer        = m_buf.colwerterTemp;

        c.ioNodes           = m_buf.nodes;
        c.inNumNodes        = m_buf.numNodes;
        c.inRemap           = m_buf.tmpRemap;
        c.inApexPointMap    = m_buf.apexPointMap;

        m_colwerter.configure(c);
    }


    // Configure TriangleWooper.

    if (m_outputWoopTriangles)
    {
        TriangleWooper::Config c;
        c.lwca                  = (m_useLwda) ? m_lwdaUtils : NULL;
        c.uncomplementRemaps    = true;

        c.outWoop               = m_buf.woop;

        c.ioRemap               = m_buf.remap;
        //c.inNumRemaps           = m_refitOnly ? EmptyBuf : m_buf.numRemaps;
        c.inNumRemaps           = m_buf.numRemaps;
        c.inModel               = model;

        m_wooper.configure(c);
    }

    // Configure motion refitter

    if( m_motionSteps > 1 )
    {

      RT_ASSERT_MSG(m_inputType == IT_AABB, "motion BVH only supported for AABB input");

      MotionRefitter::Config c;      
      c.lwca         = (m_useLwda) ? m_lwdaUtils : NULL;
      c.numAabbs     = m_numInputPrims;
      c.motionSteps  = m_motionSteps;
      c.maxNodes     = (unsigned)m_buf.nodes.getNumElems();

      c.ioBvh          = m_buf.header;
      c.ioAllNodes     = m_buf.allNodes;
      c.inAabbs        = input.aabbsArray;
      c.ioNodeParents  = m_buf.bottomNodeParents; 

      m_motionRefitter.configure(c);     
        
    }
   
    // Configure GatherPrimBits
    
    if( m_usePrimBits && m_inputType != IT_INSTANCE )
    {
        GatherPrimBits::Config c;

        c.lwca                  = (m_useLwda) ? m_lwdaUtils : NULL;
        c.flags                 = m_inputAdapter.getPrimBitsFlags();
        c.numRemaps             = (int)maxRemaps;
        c.inPrimBitsRaw         = m_buf.tmpPrimBits.reinterpretRaw();
        c.inRemap               = m_buf.remap;
        c.numPrims              = m_numInputPrims;
        c.outPrimBitsRaw        = m_buf.primBits.reinterpretRaw();
        c.primBitsFormat        = m_inputAdapter.getPrimBitsFormat();

        m_gatherPrimBits.configure(c);
    }

    // General notes about buffer layout:
    // - Remember to aggregate/overlay all BuildingBlockBuffers within one of the top-level buffers.
    // - Try to overlay as many buffers as possible to conserve memory.
    // - Experiment freely. If you make a mistake, BufferRef will report an error.
    // - If the size of m_tempBuffer changes significantly, please update the constants in chooseChunkSize().

    // Setup initial buffer layout.

    BufferRef<> nodeOverlay = overlay( m_buf.allNodes );  // Nodes at all motion steps
    BufferRef<> woopOverlay = overlay(m_buf.woop);

    m_outputBuffer
        .aggregate(m_buf.header)
        .aggregate(nodeOverlay)
        .aggregate(m_buf.remap)
        .aggregate(m_buf.numRemaps) // TODO: Moving this here for refit. Try to eliminate the need for it.
        .aggregate(woopOverlay)
        .aggregate(m_buf.bvhInstanceData)
        .aggregate(m_buf.apexPointMap)
        .aggregate(m_buf.primBits)
        .aggregate(m_buf.inputTriangleDescs)
        .aggregate(m_buf.arrayTransitionBits)
        .aggregate(m_buf.blockStartArrayIndex)
        .aggregate(m_buf.arrayBaseGlobalIndex)
        .aggregate(m_buf.geometryIndexArray);

    m_tempBuffer
        .aggregate(m_buf.chunkPrimOrder)
        .aggregate(m_buf.chunkPrimRanges)
        .aggregate(m_buf.trimmedAabbs)
        .aggregate(m_buf.tmpRemap)
        .aggregate(m_buf.numNodes)
        .aggregate(m_buf.lwtoffLevel)
        .aggregate(m_buf.splitPrimRange)
        .aggregate(m_buf.sortPrimRange)
        .aggregate(m_buf.trbvhNodeRange)
        .aggregate(m_buf.trimmedAabbRange)
        .aggregate(m_buf.inputAabbArray)
        .aggregate(m_buf.inputTrianglePtrs)
        .aggregate(m_buf.coalescedVertices)
        .aggregate(m_buf.tmpPrimBits);


    // Group the remaining buffers.

    BufferRef<> preprocessTemps     = aggregate(m_buf.chunkMortonCodes, m_buf.chunkSorterTemp);
    BufferRef<> postprocessTemps    = overlay(m_buf.colwerterTemp);
    BufferRef<> trbvhSortTemps      = overlay(m_buf.bottomSorterTemp, m_buf.topSorterTemp);
    BufferRef<> nodeParents         = overlay(m_buf.bottomNodeParents, m_buf.topNodeParents);
    BufferRef<> trbvhIOBufs;    
    if (m_supportRefit)
    {
      trbvhIOBufs = aggregate(m_buf.tmpAabbs);
      m_outputBuffer.aggregate(nodeParents);
    }
    else
    {
      trbvhIOBufs = aggregate(m_buf.tmpAabbs, nodeParents);
    }

    BufferRef<> otherTemps =
         overlay(m_buf.aabbCalcTemp)
        .overlay(aggregate(m_buf.bottomTrbvhTemp, overlay(m_buf.splitterTemp, m_buf.bottomMortonCodes), m_buf.bottomPrimOrder))
        .overlay(aggregate(m_buf.topTrbvhTemp, m_buf.topMortonCodes, m_buf.topPrimOrder));

    // Sprinkle the groups in appropriate places.

    nodeOverlay.overlay(preprocessTemps);

    if (!m_enableChunking)
        nodeOverlay.overlay(trbvhSortTemps);
    else
        m_tempBuffer.aggregate(trbvhSortTemps);

    BufferRef<> tempBundle = overlay(aggregate(trbvhIOBufs, otherTemps), postprocessTemps);
    if (m_outputWoopTriangles)
        woopOverlay.overlay(tempBundle);
    else
        m_tempBuffer.aggregate(tempBundle);

    // Materialize the top-level buffers.

    m_outputBuffer.setAllocExtra(allocExtra).materialize(m_lwdaUtils);
    m_tempBuffer.setAllocExtra(allocExtra).materialize(m_lwdaUtils);
    m_readbackBuffer.setAllocExtra(0.0f).materialize(m_lwdaUtils);


    if( k_memPrintUsage.get() )
    {
        //printBufferSizes();
        size_t totalMin = m_outputBuffer.getAllocSize() + m_readbackBuffer.getAllocSize();
        size_t totalMax = totalMin + m_tempBuffer.getAllocSize();
        lprint << "NumPrims:" << m_numInputPrims << " MemMax:" << totalMax << " MemMin:" << totalMin << "\n";
    }
}

//------------------------------------------------------------------------------

void ChunkedTrbvhBuilder::setExternalBuffers(unsigned char* outputPtr, size_t outputNumBytes, 
                                             unsigned char* tempPtr, size_t tempNumBytes, 
                                             unsigned char* readbackPtr, size_t readbackNumBytes, bool buildGpu)
{
    MemorySpace memSpace = (buildGpu) ? MemorySpace_LWDA : MemorySpace_Host;

    m_outputBuffer.assignExternal(outputPtr, outputNumBytes / m_outputBuffer.getBytesPerElem(), memSpace);
    m_tempBuffer.assignExternal(tempPtr, tempNumBytes / m_tempBuffer.getBytesPerElem(), memSpace);
    m_readbackBuffer.assignExternal((int*)readbackPtr, readbackNumBytes / m_readbackBuffer.getBytesPerElem(), memSpace);

    m_outputBuffer.materialize(m_lwdaUtils);
    m_tempBuffer.materialize(m_lwdaUtils);
    m_readbackBuffer.materialize(m_lwdaUtils);
}

//------------------------------------------------------------------------------

// Special case of the above for just the output buffer. Needed for refit if the memory manager moves the allocation.
void ChunkedTrbvhBuilder::setExternalOutputBuffer(unsigned char* outputPtr, size_t outputNumBytes, bool buildGpu)
{
    MemorySpace memSpace = (buildGpu) ? MemorySpace_LWDA : MemorySpace_Host;

    m_outputBuffer.assignExternal(outputPtr, outputNumBytes / m_outputBuffer.getBytesPerElem(), memSpace);
    m_outputBuffer.materialize(m_lwdaUtils);
}

//------------------------------------------------------------------------------


void ChunkedTrbvhBuilder::exelwteBuild(void)
{
    MemorySpace memSpace = (m_useLwda) ? MemorySpace_LWDA : MemorySpace_Host;

    // Allocate all buffers up front to make sure we have enough memory.

    m_outputBuffer.access(AccessType_Allocate, memSpace);
    m_tempBuffer.access(AccessType_Allocate, memSpace);
    m_readbackBuffer.access(AccessType_Allocate, memSpace);
        
    // All buffers are initially uninitialized.

    if (m_refitOnly)
        m_outputBuffer.readWriteLWDA();   
    else
        m_outputBuffer.markAsUninitialized();
    m_readbackBuffer.markAsUninitialized();
    m_tempBuffer.markAsUninitialized();
        
    // Init header


    if (!m_refitOnly)
    {
        if( m_buf.header.getNumElems() > 0 )
        {
          m_header.flags                = HF_TYPE_BVH2;
          if (m_outputOptixNodes)
            m_header.flags              |= HF_OPTIX_NODES;
          else if(!m_outputWoopTriangles)
            m_header.flags              |= HF_RLLE_PACK_IN_FIRST;
          m_header.numEntities          = m_numInputPrims;          
          m_header.numNodes             = (unsigned)m_buf.nodes.getNumElems();
          m_header.nodesOffset          = m_buf.nodes.getOffsetInTopmost();
          if (m_supportRefit) 
          {      
            RT_ASSERT(m_buf.bottomNodeParents.getNumElems() >= m_header.numNodes);
            m_header.nodeParentsOffset  = m_buf.bottomNodeParents.getOffsetInTopmost();
          }
          else
            m_header.nodeParentsOffset    = 0;
          m_header.numRemaps              = (unsigned)m_buf.remap.getNumElems();
          m_header.remapOffset            = m_buf.remap.getOffsetInTopmost();
          m_header.apmOffset              = m_buf.apexPointMap.getOffsetInTopmost();
          m_header.numTriangles           = (unsigned)m_buf.woop.getNumElems();   
          m_header.trianglesOffset        = m_buf.woop.getOffsetInTopmost();    
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

        // Init counters.

        m_buf.numRemaps.clear(memSpace, 0);
        if (m_enableChunking)
            m_buf.trimmedAabbRange.clear(memSpace, 0);

        int numNodesInit = 0;
        if (m_enableChunking)
            numNodesInit++; // Top-level BVH root.
        if (m_outputOptixNodes)
            numNodesInit++; // OptixColwerter root.

        if (numNodesInit == 0)
            m_buf.numNodes.clear(memSpace, 0);
        else if (m_useLwda)
            m_lwdaUtils->setInt(m_buf.numNodes.writeDiscardLWDA(), numNodesInit);
        else
            *m_buf.numNodes.writeDiscardHost() = numNodesInit;
    }

    m_inputAdapter.execute();

    // Execute ApexPointMapConstructor.

    m_apmConstructor.execute();

    if (m_refitOnly)
    {
      ModelPointers model(m_refitModel, MemorySpace_LWDA);
      refitBvh2(m_outputBuffer.readWriteLWDA(), (int)m_buf.nodes.getNumElems(), 0, &model );
    }
    else
    {
        // Execute chunker

        int numChunks = 1;
        if( m_noAccelHack )
        {
          noAccelHackExelwte();
          numChunks = 0;
        }
        if (m_enableChunking)
        {
            m_chunkSorter.execute(EmptyBuf);
            m_chunker.execute();                                    checkpoint("chunker");
            numChunks = *m_buf.numChunks.readHost();
            RT_ASSERT( numChunks > 0 );
        }

        // Build bottom-level BVH for each chunk.

        if (m_lwdaUtils && m_enableChunking)
            m_lwdaUtils->beginTimer("Bottom-level");

        for (int chunkIdx = 0; chunkIdx < numChunks; chunkIdx++)
        {
            BufferRef<Range> primRange = (m_enableChunking)
                ? m_buf.chunkPrimRanges.getSubrange(chunkIdx, 1)    // Reordered input primitives for one chunk.
                : EmptyBuf;                                         // Full range of input primitives.

            if (m_enableSplits)
            {
                m_splitter.execute(primRange);                      checkpoint("splitter");
                primRange = m_buf.splitPrimRange;
            }

            m_bottomSorter.execute(primRange);
            m_bottomTrbvh.execute();                                checkpoint("trbvhBuilder");

            if (m_enableChunking)
            {
                m_trimmer.execute();                                checkpoint("trimmer");
            }
        }

        if (m_lwdaUtils && m_enableChunking)
            m_lwdaUtils->endTimer();

        // Build top-level BVH to connect the chunks.

        if (m_enableChunking)
        {
            if (m_lwdaUtils)
                m_lwdaUtils->beginTimer("Top-level");

            m_topSorter.execute(m_buf.trimmedAabbRange);
            m_topTrbvh.execute();                                   checkpoint("topTrbvhBuilder");
            m_connector.execute();

            if (m_lwdaUtils)
                m_lwdaUtils->endTimer();
        }

        // Execute OptixColwerter and TriangleWooper.

        if (m_outputOptixNodes)
        {
            m_colwerter.execute();
            if (m_supportRefit)
            {
              RT_ASSERT( m_useLwda );
              updateNodeParentsBvh2( m_outputBuffer.readWriteLWDA(), (int)m_buf.nodes.getNumElems() );
            }
            else if (m_motionSteps > 1)
            {
              RT_ASSERT( m_useLwda );
              // Update tmp buffer for node parents
              m_buf.bottomNodeParents.writeLWDA();  // make overlay valid
              updateNodeParentsBvh2( m_outputBuffer.readWriteLWDA(), (int)m_buf.nodes.getNumElems(), m_buf.bottomNodeParents.readWriteLWDA() );
            }
        }
    }

    if (m_outputWoopTriangles)
        m_wooper.execute();

    if ( m_motionSteps > 1 )
        m_motionRefitter.execute();

    if( m_usePrimBits && !m_refitOnly && m_inputType != IT_INSTANCE )
      m_gatherPrimBits.execute();
                                                                checkpoint("output");
    // Ilwalidate the temp buffer to prevent accidental reads.

    m_tempBuffer.markAsUninitialized();
}

//------------------------------------------------------------------------------

void ChunkedTrbvhBuilder::exelwteBuildWithTiming(void)
{
    // Timing disabled => just call exelwteBuild() directly.

    if (!m_useLwda || !m_lwdaTimingRepeats)
    {
        exelwteBuild();
        return;
    }

    // Several repeats => do the first one without timing to make sure we get a clean report.

    if (m_lwdaTimingRepeats > 1)
        exelwteBuild();

    // Enable timing.

    m_lwdaUtils->resetTiming(true);

    // Execute the build multiple times and take the minimum, for accurate results.

    for (int repeat = 0; repeat < std::max(m_lwdaTimingRepeats - 1, 1); repeat++)
    {
        m_lwdaUtils->repeatTiming();
        m_lwdaUtils->beginTimer("ChunkedTrbvhBuilder");
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

//-------------------------------------------------------------------------

void ChunkedTrbvhBuilder::chooseChunkSize(
    /*out*/ int&            maxSplitPrimsPerChunk,
    /*out*/ int&            preferredTopTreePrims,
    int                     numInputPrims,
    int                     maxTotalSplitPrims,
    float                   splitBeta,
    const ParameterList&    p)
{
    // Query user-specified soft upper bound for the size of m_tempBuffer.

    uint64_t userPreferredTempBytes = p.get("chunkBytes", 0);
    uint64_t preferredTempBytes = userPreferredTempBytes;

    // Unspecified => pick a reasonable default.

    if (preferredTempBytes == 0)
    {
        if (m_useLwda)
        {
            size_t free, total;
            m_lwdaUtils->getMemInfo(free, total);
            preferredTempBytes = total / 10; // 10% of total GPU memory
        }
        else
        {
            preferredTempBytes = 512 << 20; // 512 MB on the host
        }
    }

    // Assume that the size of m_tempBuffer behaves approximately as follows.
    // The assumption was verified experimentally by tkarras on 2015/09/28.
    //
    // tempBytes <=
    //      bytesPerSplitPrim   * max(maxSplitPrimsPerChunk, preferredTopTreePrims) +
    //      bytesPerTopPrim     * preferredTopTreePrims +
    //      bytesPerInputPrim   * numInputPrims +
    //      constantBytes

    float splitAABBsPerSplitPrim = fminf(splitBeta * 2.0f / (splitBeta + 1.0f), 1.0f); // See MortonTriangleSplitter for details.
    int bytesPerSplitPrim   = 40 + (int)ceilf(splitAABBsPerSplitPrim * 32.0f); // nodeParents(4) + trbvhTemp(12) + sortTemp(24) + splitAABBs
    int bytesPerTopPrim     = 32;       // trimmedAabbs(32)
    int bytesPerInputPrim   = 4;        // chunkPrimOrder(4)
    int constantBytes       = 16 << 10; // Measured to be below 16KB in practice.

    // Assume that we will choose preferredTopTreePrims as follows:
    // - No larger than maxSplitPrimsPerChunk to prevent the top-level builder from consuming more temp memory than the bottom-level builder.
    // - No larger than a pre-defined limit to prevent the trimmedAabbs from consuming excessive amounts of memory.
    //
    // preferredTopTreePrims = min(maxSplitPrimsPerChunk, topTreeLimit)

    int topTreeLimit = (32 << 20) / bytesPerTopPrim; // 32MB seems like a reasonable soft upper bound for the top-level tree.

    // Callwlate the largest allowed maxSplitPrimsPerChunk so that tempBytes <= preferredTempBytes based to the above assumptions.

    uint64_t minTempBytes = bytesPerInputPrim * uint64_t(numInputPrims) + constantBytes;
    if( minTempBytes > preferredTempBytes )
    {
      if( userPreferredTempBytes != 0 )
          throw IlwalidValue( RT_EXCEPTION_INFO, "User specified chunk size for this model must be greater than ", minTempBytes + bytesPerSplitPrim + bytesPerTopPrim );

      // preferredTempBytes is a default value so increase it based on model size (which is likely large)
      int defaultChunks = 32;
      preferredTempBytes = minTempBytes + (uint64_t)max(numInputPrims / defaultChunks, 1) * (bytesPerSplitPrim + bytesPerTopPrim);
    }

    int64_t spareBytes              = preferredTempBytes - minTempBytes;
    int64_t splitPrimsIfBelowLimit  = spareBytes / (bytesPerSplitPrim + bytesPerTopPrim);
    int64_t splitPrimsIfAboveLimit  = (spareBytes - bytesPerTopPrim * topTreeLimit) / bytesPerSplitPrim;
    RT_ASSERT( splitPrimsIfBelowLimit <= INT_MAX && splitPrimsIfAboveLimit <= INT_MAX );
    maxSplitPrimsPerChunk           = (int)std::max(splitPrimsIfBelowLimit, splitPrimsIfAboveLimit); 

    // Clamp to a reasonable lower bound.

    maxSplitPrimsPerChunk = max(maxSplitPrimsPerChunk, 32 << 10); // ~2.9MB

    // Override with explicit builderSpec param if specified.

    int maxSplitPrimsPerChunkOverride = k_maxPrimsPerChunk.get();
    if( maxSplitPrimsPerChunkOverride == 0 )
      maxSplitPrimsPerChunkOverride = p.get("maxSplitPrimsPerChunk", 0);
    if (maxSplitPrimsPerChunkOverride > 0)
        maxSplitPrimsPerChunk = maxSplitPrimsPerChunkOverride;

    // Clamp to legal range.

    maxSplitPrimsPerChunk = min(maxSplitPrimsPerChunk, maxTotalSplitPrims);
    maxSplitPrimsPerChunk = max(maxSplitPrimsPerChunk, 1);

    // Callwlate preferredTopTreePrims.

    preferredTopTreePrims = min(maxSplitPrimsPerChunk, topTreeLimit);

    // Override with explicit builderSpec param if specified.

    int preferredTopTreePrimsOverride = p.get("preferredTopTreePrims", 0);
    if (preferredTopTreePrimsOverride > 0)
        preferredTopTreePrims = preferredTopTreePrimsOverride;
}

//------------------------------------------------------------------------------

void ChunkedTrbvhBuilder::printBufferSizes()
{
#define PRINT(x) ltemp << #x ":" << m_buf.x.getNumBytes() << "\n"

  PRINT(header);
  PRINT(nodes);           
  PRINT(remap); 
  PRINT(bvhInstanceData);
  PRINT(woop);            
  PRINT(apexPointMap);    
  PRINT(tmpAabbs);     
  PRINT(chunkMortonCodes);
  PRINT(chunkPrimOrder);  
  PRINT(chunkPrimRanges); 
  PRINT(bottomMortonCodes);
  PRINT(bottomPrimOrder); 
  PRINT(bottomNodeParents);
  PRINT(topMortonCodes);  
  PRINT(topPrimOrder);    
  PRINT(topNodeParents);  
  PRINT(trimmedAabbs);    
  PRINT(tmpRemap);        
  PRINT(numNodes);        
  PRINT(numChunks);       
  PRINT(numRemaps);       
  PRINT(lwtoffLevel);     
  PRINT(splitPrimRange);  
  PRINT(sortPrimRange);   
  PRINT(trbvhNodeRange);  
  PRINT(trimmedAabbRange);
  PRINT(aabbCalcTemp);    
  PRINT(chunkSorterTemp); 
  PRINT(splitterTemp);    
  PRINT(bottomSorterTemp);
  PRINT(bottomTrbvhTemp); 
  PRINT(topSorterTemp);   
  PRINT(topTrbvhTemp);    
  PRINT(colwerterTemp);   

#undef PRINT
}

//------------------------------------------------------------------------------

void ChunkedTrbvhBuilder::copyHeader(void* dst, size_t dstSize)
{
  RT_ASSERT(dstSize >= sizeof(m_header));
  memcpy(dst, &m_header, sizeof(m_header));
}

//------------------------------------------------------------------------------

void ChunkedTrbvhBuilder::noAccelHackExelwte(void)
{
  BvhNode* nodes = m_buf.nodes.writeDiscardHost();
  int*     remap = m_buf.remap.writeDiscardHost();

 // Lots of braces to make clang compile without warning about sub objects
 float big = FLT_MAX;
 BvhNode root = {{{                                               // clang-format off
    -big, -big, -big, big, big, big,  ~0,   4,
       0,    0,    0,   0,   0,   0,   0,   0  }}};

 // A null ray will hit the second child.  If this happens we need to provide a range that works
 // for child0 to child1 [0,m_numInputPrims], but if doing one child at a time, child1's interval 
 // needs to be empty.
 BvhNode leaf = {{{
     -big, -big, -big,  big,  big,  big,                0, m_numInputPrims,
      big,  big,  big, -big, -big, -big,  m_numInputPrims, m_numInputPrims }}}; // clang-format on

 nodes[0] = root;
 nodes[1] = leaf;
 for( int i=0; i < m_numInputPrims; ++i )
   remap[i] = i;

 // Force upload
 m_buf.nodes.readLWDA();
 m_buf.remap.readLWDA();
}

//-------------------------------------------------------------------------
