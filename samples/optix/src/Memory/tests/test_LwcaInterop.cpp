#include <srcTests.h>

#include <corelib/system/LwdaDriver.h>
#include <optixu/optixpp.h>
#include <prodlib/system/Knobs.h>

using namespace optix;
using namespace optixu;
using namespace testing;

using corelib::lwdaDriver;

#define LWDA_CHK( func )                                                                                               \
    do                                                                                                                 \
    {                                                                                                                  \
        LWresult code = func;                                                                                          \
        EXPECT_THAT( code, LWDA_SUCCESS );                                                                             \
    } while( 0 )

class ActiveDevice
{
  public:
    ActiveDevice( int dev )
        : m_dev( dev )
    {
        LWcontext newCtx;
        LWDA_CHK( lwdaDriver().LwDevicePrimaryCtxRetain( &newCtx, m_dev ) );
        EXPECT_TRUE( newCtx != nullptr );
        LWcontext context = nullptr;
        LWDA_CHK( lwdaDriver().LwCtxGetLwrrent( &context ) );
        if( context != newCtx )
        {
            m_prevCtx = context;
            LWDA_CHK( lwdaDriver().LwCtxSetLwrrent( newCtx ) );
        }
    }

    ~ActiveDevice()
    {
        if( m_prevCtx )
            LWDA_CHK( lwdaDriver().LwCtxSetLwrrent( m_prevCtx ) );
        LWDA_CHK( lwdaDriver().LwDevicePrimaryCtxRelease( m_dev ) );
    }

  private:
    int       m_dev     = 0;
    LWcontext m_prevCtx = nullptr;
};

struct LwdaInteropParams
{
    std::vector<int> optixDevices;
    unsigned         inputBufferType  = RT_BUFFER_INPUT;
    unsigned         outputBufferType = RT_BUFFER_OUTPUT;
};

std::string typeStr( unsigned type )
{
    std::string str;
    if( type & RT_BUFFER_INPUT )
        str += "I";
    if( type & RT_BUFFER_OUTPUT )
        str += "O";
    if( type & RT_BUFFER_COPY_ON_DIRTY )
        str += "Cd";
    if( type & RT_BUFFER_GPU_LOCAL )
        str += "Gl";

    return str;
}

std::ostream& operator<<( std::ostream& out, const LwdaInteropParams& params )
{
    out << "{";
    for( size_t i = 0; i < params.optixDevices.size(); ++i )
    {
        if( i > 0 )
            out << ",";
        out << params.optixDevices[i];
    }
    out << "} in:" << typeStr( params.inputBufferType ) << " out:" << typeStr( params.outputBufferType );

    return out;
}

class LwdaInteropBase : public Test
{
  public:
    Context            context;
    Buffer             inBuffer;
    Buffer             outBuffer;
    std::vector<void*> inDevPtrs;
    std::vector<void*> outDevPtrs;
    std::vector<void*> lwdaBuffers;
    std::vector<int>   allLwdaDevices;   // lwca devices ordinal indexed by OptiX ordinal
    std::vector<int>   allOptixDevices;  // all available optix device ordinals
    std::vector<int>   optixDevices;     // optix device ordinals
    std::vector<int>   lwdaDevices;      // lwca device ordinals
    size_t             bufferSize       = 1024;
    unsigned           inputBufferType  = RT_BUFFER_INPUT;
    unsigned           outputBufferType = RT_BUFFER_OUTPUT;
    const int          MULTIPLE         = -1;

    std::unique_ptr<ScopedKnobSetter> loadBalanceKnob;

    virtual void SetUp()
    {
        optixDevices = {0};
    }

    virtual void TearDown()
    {
        for( size_t i = 0; i < lwdaBuffers.size(); ++i )
            freeLwdaBuffer( (int)i );

        if( context )
            context->destroy();
        context = nullptr;
    }

    void init()
    {
        context = Context::create();

        computeAllDevices();
        if( !lwdaDevices.empty() )
            optixDevices = mapToOptix( lwdaDevices );
        else if( optixDevices.empty() )
            optixDevices = allOptixDevices;
        lwdaDevices      = getLwdaDevices( optixDevices );
        context->setDevices( optixDevices.begin(), optixDevices.end() );

        Program rg =
            context->createProgramFromPTXFile( ptxPath( "test_Memory", "interop.lw" ), "copy_input_to_output" );
        context->setEntryPointCount( 1 );
        context->setRayGenerationProgram( 0, rg );
        context->setRayTypeCount( 1 );

        inBuffer = context->createBuffer( inputBufferType, RT_FORMAT_INT, bufferSize );
        context["input"]->set( inBuffer );
        outBuffer = context->createBuffer( outputBufferType, RT_FORMAT_INT, bufferSize );
        context["output"]->set( outBuffer );

        context->validate();
        context->compile();

        inDevPtrs.resize( lwdaDevices.size() );
        outDevPtrs.resize( lwdaDevices.size() );
        lwdaBuffers.resize( lwdaDevices.size() );
    }

    void computeAllDevices()
    {
        int count;
        rtDeviceGetDeviceCount( (unsigned*)&count );
        allLwdaDevices.resize( count );
        allOptixDevices.resize( count );
        for( int i = 0; i < count; ++i )
        {
            rtDeviceGetAttribute( i, RT_DEVICE_ATTRIBUTE_LWDA_DEVICE_ORDINAL, sizeof( int ), &allLwdaDevices[i] );
            allOptixDevices[i] = i;
        }
    }

    std::vector<int> mapToOptix( std::vector<int>& lwdaDevices )
    {
        // Compute mapping
        std::map<int, int> lwdaToOptix;
        int count;
        rtDeviceGetDeviceCount( (unsigned*)&count );
        for( int i = 0; i < count; ++i )
        {
            int lwdaOrdinal;
            rtDeviceGetAttribute( i, RT_DEVICE_ATTRIBUTE_LWDA_DEVICE_ORDINAL, sizeof( int ), &lwdaOrdinal );
            lwdaToOptix[lwdaOrdinal] = i;
        }

        // map lwdaDevices to optixDevices
        std::vector<int> optixDevices;
        for( size_t i = 0; i < lwdaDevices.size(); ++i )
        {
            bool deviceFound = ( lwdaToOptix.find( lwdaDevices[i] ) != lwdaToOptix.end() );
            EXPECT_TRUE( deviceFound );
            if( deviceFound )
                optixDevices.push_back( lwdaToOptix[lwdaDevices[i]] );
        }

        return optixDevices;
    }

    std::vector<int> getLwdaDevices( std::vector<int>& optixDevices )
    {
        std::vector<int> lwdaDevices( optixDevices.size() );
        for( size_t i = 0; i < optixDevices.size(); ++i )
            rtDeviceGetAttribute( optixDevices[i], RT_DEVICE_ATTRIBUTE_LWDA_DEVICE_ORDINAL, sizeof( int ), &lwdaDevices[i] );

        return lwdaDevices;
    }

    int selectUnusedOptixDevice()
    {
        if( optixDevices.size() == allOptixDevices.size() )
            return -1;

        for( int j = 0; j < (int)allOptixDevices.size(); ++j )
        {
            bool used = false;
            for( int i = 0; i < (int)optixDevices.size(); ++i )
            {
                if( optixDevices[i] == allOptixDevices[j] )
                {
                    used = true;
                    break;
                }
            }

            if( !used )
                return j;
        }

        return -1;
    }

    void addDevice( int optixDev )
    {
        optixDevices.push_back( optixDev );
        lwdaDevices.push_back( allLwdaDevices[optixDev] );
        inDevPtrs.resize( optixDevices.size() );
        outDevPtrs.resize( optixDevices.size() );
        lwdaBuffers.resize( optixDevices.size() );

        context->setDevices( optixDevices.begin(), optixDevices.end() );
    }

    // removes the last one
    int removeDevice()
    {
        freeLwdaBuffer( (int)optixDevices.size() - 1 );

        int removedDevice = optixDevices.back();
        optixDevices.pop_back();
        lwdaDevices.resize( optixDevices.size() );
        inDevPtrs.resize( optixDevices.size() );
        outDevPtrs.resize( optixDevices.size() );
        lwdaBuffers.resize( optixDevices.size() );

        context->setDevices( optixDevices.begin(), optixDevices.end() );

        return removedDevice;
    }

    Buffer createBufferForLwda( const std::string& var )
    {
        unsigned bufferType = ( var == "input" ) ? inputBufferType : outputBufferType;
        Buffer   buf        = context->createBufferForLWDA( bufferType, RT_FORMAT_INT, bufferSize );
        context[var]->set( buf );
        return buf;
    }

    void resize( size_t newBufferSize )
    {
        bufferSize = newBufferSize;
        inBuffer->setSize( newBufferSize );
        outBuffer->setSize( newBufferSize );
    }

    std::vector<void*> getDevicePointers( Buffer& buf )
    {
        std::vector<void*> ptrs( optixDevices.size() );
        for( size_t i = 0; i < optixDevices.size(); ++i )
            ptrs[i]   = buf->getDevicePointer( optixDevices[i] );

        return ptrs;
    }

    void setDevicePointers( Buffer& buf, std::vector<void*>& ptrs )
    {
        for( size_t i = 0; i < optixDevices.size(); ++i )
            buf->setDevicePointer( optixDevices[i], ptrs[i] );
    }

    void launchOptixCopy() { context->launch( 0, bufferSize ); }

    void* allocLwdaBuffer( int devIdx )
    {
        //LWDA_CHK( lwdaSetDevice( lwdaDevices[devIdx] ) );
        ActiveDevice ad( lwdaDevices[devIdx] );
        LWDA_CHK( lwdaDriver().LwMemAlloc( (LWdeviceptr*)&lwdaBuffers[devIdx], bufferSize * sizeof( int ) ) );
        return lwdaBuffers[devIdx];
    }

    void freeLwdaBuffer( int devIdx )
    {
        if( lwdaBuffers[devIdx] )
        {
            ActiveDevice ad( lwdaDevices[devIdx] );
            LWDA_CHK( lwdaDriver().LwMemFree( (LWdeviceptr)lwdaBuffers[devIdx] ) );
            lwdaBuffers[devIdx] = nullptr;
        }
    }

    std::vector<void*> allocLwdaBuffers()
    {
        std::vector<void*> ptrs;
        for( size_t i = 0; i < lwdaDevices.size(); ++i )
            ptrs.push_back( allocLwdaBuffer( int( i ) ) );
        return ptrs;
    }

    // Returns the value stored in the buffer if all values are the same, MULTIPLE otherwise
    int readViaOptix( Buffer& buffer )
    {
        int* ptr = (int*)buffer->map();
        int  val = ptr[0];
        for( size_t i = 0; i < bufferSize; ++i )
        {

            if( ptr[i] != ptr[0] )
            {
                val = MULTIPLE;
                break;
            }
        }
        buffer->unmap();

        return val;
    }

    // Returns the value stored in the buffer if all values are the same, MULTIPLE otherwise
    int readViaLwda( int lwdaDev, void* ptr )
    {
        std::vector<int> vals( bufferSize );
        //LWDA_CHK( lwdaSetDevice(lwdaDev) );
        ActiveDevice ad( lwdaDev );
        LWDA_CHK( lwdaDriver().LwMemcpyDtoH( vals.data(), (LWdeviceptr)ptr, vals.size() * sizeof( int ) ) );
        for( size_t i = 0; i < vals.size(); ++i )
            if( vals[i] != vals[0] )
                return MULTIPLE;
        return vals[0];
    }

    int readViaLwda( std::vector<void*>& ptrs )
    {
        int val = MULTIPLE;
        for( size_t i = 0; i < lwdaDevices.size(); ++i )
        {
            int thisVal = readViaLwda( lwdaDevices[i], ptrs[i] );
            if( i > 0 && thisVal != val )
                return MULTIPLE;
            val = thisVal;
        }
        return val;
    }

    // The distribution of work across GPUs is undefined and possible non-deterministic.
    // But the composited results should yield the correct final buffer
    int compositeViaLwda( std::vector<void*>& ptrs )
    {
        std::vector<int> final( bufferSize, 0 );

        std::vector<int> vals( bufferSize );
        for( size_t d = 0; d < lwdaDevices.size(); ++d )
        {
            ActiveDevice ad( lwdaDevices[d] );
            LWDA_CHK( lwdaDriver().LwMemcpyDtoH( vals.data(), (LWdeviceptr)ptrs[d], vals.size() * sizeof( int ) ) );
            for( size_t i = 0; i < vals.size(); ++i )
                if( final[i] == 0 )
                    final[i] = vals[i];
                else if( final[i] != vals[i] && vals[i] != 0 )
                    return MULTIPLE;
        }

        for( size_t i = 0; i < final.size(); ++i )
            if( final[i] != final[0] )
                return MULTIPLE;

        return final[0];
    }

    void writeViaOptix( Buffer& buffer, int val )
    {
        int* ptr = (int*)buffer->map();
        for( size_t i = 0; i < bufferSize; ++i )
            ptr[i]    = val;
        buffer->unmap();
    }

    void writeViaLwda( int lwdaDev, void* ptr, int val )
    {
        std::vector<int> vals( bufferSize, val );
        ActiveDevice     ad( lwdaDev );
        LWDA_CHK( lwdaDriver().LwMemcpyHtoD( (LWdeviceptr)ptr, vals.data(), vals.size() * sizeof( int ) ) );
    }

    void writeViaLwda( std::vector<void*>& ptrs, int val )
    {
        for( size_t i = 0; i < lwdaDevices.size(); ++i )
            writeViaLwda( lwdaDevices[i], ptrs[i], val );
    }
};

class LwdaInterop1Gpu : public LwdaInteropBase
{
};

class LwdaInterop2Gpu : public LwdaInteropBase
{
  public:
    virtual void SetUp() { optixDevices = {0, 1}; }
};

class LwdaInterop3Gpu : public LwdaInteropBase
{
  public:
    virtual void SetUp() { optixDevices = {0, 1, 2}; }
};


TEST_F_DEV( LwdaInterop2Gpu, EachGpuContributesToOutput )
{
    // This test must pass before the other multi-gpu tests can be considered valid.
    inputBufferType = RT_BUFFER_INPUT_OUTPUT | RT_BUFFER_GPU_LOCAL;
    init();
    inDevPtrs = getDevicePointers( inBuffer );
    writeViaLwda( 0, inDevPtrs[0], 42 );
    writeViaLwda( 1, inDevPtrs[1], 43 );
    writeViaOptix( outBuffer, 0 );

    launchOptixCopy();

    EXPECT_THAT( readViaOptix( outBuffer ), Eq( MULTIPLE ) );
}

TEST_F( LwdaInterop1Gpu, GetPointerSucceedsOnUnattachedBuffer )
{
    init();
    Buffer unattached = context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_INT, bufferSize );

    EXPECT_NO_THROW( unattached->getDevicePointer( 0 ) );
}

TEST_F( LwdaInterop1Gpu, CanGetPreviouslySetPointer )
{
    init();
    int di = 0, odev = optixDevices[di];
    inDevPtrs[di] = allocLwdaBuffer( di );
    inBuffer->setDevicePointer( odev, inDevPtrs[di] );

    EXPECT_THAT( inBuffer->getDevicePointer( odev ), Eq( inDevPtrs[di] ) );
}

TEST_F( LwdaInterop2Gpu, CannotMixGetAndSetPointersOnDifferentDevices )
{
    init();
    inDevPtrs[0] = allocLwdaBuffer( 0 );
    inBuffer->setDevicePointer( optixDevices[0], inDevPtrs[0] );

    EXPECT_ANY_THROW( inBuffer->getDevicePointer( optixDevices[1] ) );
}

TEST_F( LwdaInterop1Gpu, SetPointerFailsOnOutputBuffer )
{
    init();
    int di = 0, odev = optixDevices[di];
    outDevPtrs[di] = allocLwdaBuffer( di );

    EXPECT_ANY_THROW( outBuffer->setDevicePointer( odev, outDevPtrs[di] ) );
}

TEST_F( LwdaInterop1Gpu, SetPointerFailsOnInputOutputBuffer )
{
    outputBufferType = RT_BUFFER_INPUT_OUTPUT;
    init();
    int di = 0, odev = optixDevices[di];
    outDevPtrs[di] = allocLwdaBuffer( di );

    EXPECT_ANY_THROW( outBuffer->setDevicePointer( odev, outDevPtrs[di] ) );
}

TEST_F( LwdaInterop1Gpu, SetPointerFailsForNonOptiXDevice )
{
    init();
    int   odev   = selectUnusedOptixDevice();
    void* devPtr = &odev;  // dummy values

    EXPECT_ANY_THROW( inBuffer->setDevicePointer( odev, devPtr ) );
}

TEST_F( LwdaInterop3Gpu, LaunchFailsWithIncompleteSetOfInteropPointers )
{
    init();
    inDevPtrs[0] = inBuffer->getDevicePointer( 0 );
    inDevPtrs[1] = inBuffer->getDevicePointer( 1 );

    EXPECT_ANY_THROW( launchOptixCopy() );
}


TEST_F( LwdaInterop1Gpu, MarkDirtyFailsWithoutCopyOnDirty )
{
    init();

    EXPECT_ANY_THROW( inBuffer->markDirty() );
}

TEST_F( LwdaInterop1Gpu, MarkDirtyFailsWithoutGetOrSetPointer )
{
    inputBufferType = RT_BUFFER_INPUT | RT_BUFFER_COPY_ON_DIRTY;
    init();

    EXPECT_ANY_THROW( inBuffer->markDirty() );
}

TEST_F( LwdaInterop2Gpu, MarkDirtyFailsWithMultiplePointers )
{
    inputBufferType = RT_BUFFER_INPUT | RT_BUFFER_COPY_ON_DIRTY;
    init();
    inDevPtrs = getDevicePointers( inBuffer );

    EXPECT_ANY_THROW( inBuffer->markDirty() );
}

TEST_F( LwdaInterop1Gpu, WorksWithGpuLocalOutput )
{
    outputBufferType = RT_BUFFER_INPUT_OUTPUT | RT_BUFFER_GPU_LOCAL;
    init();
    int di = 0, odev = optixDevices[di], cdev = lwdaDevices[di];
    outDevPtrs[di] = outBuffer->getDevicePointer( odev );
    writeViaOptix( inBuffer, 42 );

    launchOptixCopy();

    // We can't read GPU_LOCAL via OptiX, but we can via LWCA.
    EXPECT_THAT( readViaLwda( cdev, outDevPtrs[di] ), Eq( 42 ) );
}


/////////////////////////////////////
// Resize
/////////////////////////////////////

TEST_F( LwdaInterop1Gpu, ResizePreservesSetPointer )
{
    init();
    int di = 0, odev = optixDevices[di], cdev = lwdaDevices[di];
    inDevPtrs[di] = allocLwdaBuffer( di );
    inBuffer->setDevicePointer( odev, inDevPtrs[di] );
    writeViaLwda( cdev, inDevPtrs[di], 42 );
    writeViaOptix( outBuffer, 0 );
    launchOptixCopy();
    EXPECT_THAT( readViaOptix( outBuffer ), Eq( 42 ) );

    resize( bufferSize / 2 );
    writeViaLwda( cdev, inDevPtrs[di], 43 );  // writing to same pointer should affect output
    launchOptixCopy();

    EXPECT_THAT( readViaOptix( outBuffer ), Eq( 43 ) );
}

TEST_F( LwdaInterop1Gpu, ResizeRevertsGetPointer )
{
    inputBufferType = RT_BUFFER_INPUT | RT_BUFFER_COPY_ON_DIRTY;
    init();
    int di = 0, odev = optixDevices[di];
    inDevPtrs[di] = inBuffer->getDevicePointer( odev );

    resize( bufferSize / 2 );

    EXPECT_ANY_THROW( inBuffer->markDirty() );  // markDirty() only works if there is a single interop pointer
}

//TEST_F( LwdaInterop1Gpu, ResizeRevertsPolicyAfterGetPointer )
//{
//  init();
//  MBufferPolicy origPolicy = getPolicy( inputBuffer );
//  int di = 0, odev = optixDevices[di], cdev = lwdaDevices[di];
//  inDevPtrs[di] = inBuffer->getDevicePointer(odev);
//  EXPECT_THAT( getPolicy( inputBuffer ), Ne(origPolicy) );
//
//  resize( bufferSize / 2 );
//
//  EXPECT_THAT( getPolicy( inputBuffer ), Eq(origPolicy) );
//}


/////////////////////////////////////
// Changing the number of devices
/////////////////////////////////////

TEST_F( LwdaInterop2Gpu, LaunchFailsWithIncompleteGetPointersAfterAddingDevice__3Gpu )
{
    init();
    inDevPtrs = getDevicePointers( inBuffer );
    addDevice( 2 );

    EXPECT_ANY_THROW( launchOptixCopy() );
}

TEST_F( LwdaInterop2Gpu, GetAllPointersWorksAfterAddingDevice__3Gpu )
{
    init();
    inDevPtrs = getDevicePointers( inBuffer );
    addDevice( 2 );
    inDevPtrs[2] = inBuffer->getDevicePointer( optixDevices[2] );
    writeViaLwda( inDevPtrs, 42 );  // still using previously gotten pointers
    writeViaOptix( outBuffer, 0 );

    launchOptixCopy();

    EXPECT_THAT( readViaOptix( outBuffer ), Eq( 42 ) );
}

TEST_F( LwdaInterop2Gpu, GetAllPointersWorksAfterRemovingDevice )
{
    init();
    inDevPtrs = getDevicePointers( inBuffer );
    removeDevice();
    writeViaLwda( inDevPtrs, 42 );  // still using previously gotten pointers
    writeViaOptix( outBuffer, 0 );

    launchOptixCopy();

    EXPECT_THAT( readViaOptix( outBuffer ), Eq( 42 ) );
}

TEST_F( LwdaInterop2Gpu, SetAllPointersWorksAfterAddingDevice__3Gpu )
{
    init();
    inDevPtrs = allocLwdaBuffers();
    setDevicePointers( inBuffer, inDevPtrs );
    addDevice( 2 );
    inDevPtrs[2] = allocLwdaBuffer( 2 );
    inBuffer->setDevicePointer( optixDevices[2], inDevPtrs[2] );
    writeViaLwda( inDevPtrs, 42 );  // still using previously gotten pointers
    writeViaOptix( outBuffer, 0 );

    launchOptixCopy();

    EXPECT_THAT( readViaOptix( outBuffer ), Eq( 42 ) );
}

TEST_F( LwdaInterop2Gpu, SetAllPointersWorksAfterRemovingDevice )
{
    init();
    inDevPtrs = allocLwdaBuffers();
    setDevicePointers( inBuffer, inDevPtrs );
    removeDevice();
    writeViaLwda( inDevPtrs, 42 );  // still using previously gotten pointers
    writeViaOptix( outBuffer, 0 );

    launchOptixCopy();

    EXPECT_THAT( readViaOptix( outBuffer ), Eq( 42 ) );
}

TEST_F( LwdaInterop2Gpu, BufferMapWorksAfterGetting2PointersAndRemovingDevice )
{
    init();
    inDevPtrs = getDevicePointers( inBuffer );

    removeDevice();

    EXPECT_NO_THROW( inBuffer->map() );  // Will fail if we still have multiple pointers
}

TEST_F( LwdaInterop2Gpu, RemovingDeviceWithGotPointerRevertsGetPointer )
{
    inputBufferType = RT_BUFFER_INPUT | RT_BUFFER_COPY_ON_DIRTY;
    init();
    int di = 1, odev = optixDevices[di];
    inDevPtrs[di] = inBuffer->getDevicePointer( odev );

    removeDevice();

    EXPECT_ANY_THROW( inBuffer->markDirty() );  // markDirty() only works if there is a single interop pointer
}


/////////////////////////////////////
// BufferMap
/////////////////////////////////////

TEST_F( LwdaInterop1Gpu, BufferMapOverridesLwdaForInput )
{
    init();
    int di = 0, odev = optixDevices[di], cdev = lwdaDevices[di];
    inDevPtrs[di] = inBuffer->getDevicePointer( odev );
    writeViaOptix( inBuffer, 24 );
    writeViaLwda( cdev, inDevPtrs[di], 42 );
    writeViaOptix( outBuffer, 0 );

    launchOptixCopy();

    EXPECT_THAT( readViaOptix( outBuffer ), Eq( 24 ) );

    writeViaLwda( cdev, inDevPtrs[di], 43 );
    writeViaOptix( inBuffer, 34 );

    launchOptixCopy();

    EXPECT_THAT( readViaOptix( outBuffer ), Eq( 34 ) );
}

TEST_F( LwdaInterop1Gpu, MarkDirtyOverridesBufferMap )
{
    inputBufferType = RT_BUFFER_INPUT | RT_BUFFER_COPY_ON_DIRTY;
    init();
    int di = 0, odev = optixDevices[di], cdev = lwdaDevices[di];
    inDevPtrs[di] = inBuffer->getDevicePointer( odev );
    writeViaOptix( inBuffer, 24 );
    writeViaLwda( cdev, inDevPtrs[di], 42 );
    writeViaOptix( outBuffer, 0 );

    inBuffer->markDirty();
    launchOptixCopy();

    EXPECT_THAT( readViaOptix( outBuffer ), Eq( 42 ) );
}

TEST_F( LwdaInterop1Gpu, DISABLED_BufferMapSyncsFromSinglePointer )  // Needs to be debugged
{
    init();
    int di = 0, odev = optixDevices[di], cdev = lwdaDevices[di];
    inDevPtrs[di] = inBuffer->getDevicePointer( odev );
    writeViaLwda( cdev, inDevPtrs[di], 42 );

    EXPECT_THAT( readViaOptix( inBuffer ), Eq( 42 ) );
}

TEST_F( LwdaInterop2Gpu, DISABLED_BufferMapDoesNotSyncFromMultiplePointers )  // Needs to be debugged
{
    init();
    writeViaOptix( inBuffer, 0 );
    inDevPtrs = getDevicePointers( inBuffer );
    writeViaLwda( lwdaDevices[0], inDevPtrs[0], 42 );
    writeViaLwda( lwdaDevices[1], inDevPtrs[1], 43 );

    EXPECT_THAT( readViaOptix( inBuffer ), Eq( 0 ) );
}

TEST_F( LwdaInterop2Gpu, BufferMapFailsWithMultiplePointers )
{
    init();
    inDevPtrs = getDevicePointers( inBuffer );

    EXPECT_ANY_THROW( inBuffer->map() );
}


/////////////////////////////////////
// Syncing
/////////////////////////////////////

TEST_F( LwdaInterop2Gpu, AutoSyncsOnEveryLaunch )
{
    inputBufferType = RT_BUFFER_INPUT;
    init();
    int di = 0, odev = optixDevices[di], cdev = lwdaDevices[di];
    inDevPtrs[di] = inBuffer->getDevicePointer( odev );
    writeViaLwda( cdev, inDevPtrs[di], 42 );
    writeViaOptix( outBuffer, 0 );

    launchOptixCopy();
    EXPECT_THAT( readViaOptix( outBuffer ), Eq( 42 ) );

    writeViaLwda( cdev, inDevPtrs[di], 43 );
    launchOptixCopy();
    EXPECT_THAT( readViaOptix( outBuffer ), Eq( 43 ) );
}

TEST_F( LwdaInterop2Gpu, SyncsOnlyWhenDirtyWithCopyOnDirty )
{
    inputBufferType = RT_BUFFER_INPUT | RT_BUFFER_COPY_ON_DIRTY;
    bufferSize      = 1 << 20;  // Make the buffer big enough that it gets split over multiple GPUs
    init();
    writeViaOptix( inBuffer, 1 );
    writeViaOptix( outBuffer, 0 );
    launchOptixCopy();  // ensure that the device buffers get initialized to 1

    int di = 0, odev = optixDevices[di], cdev = lwdaDevices[di];
    inDevPtrs[di] = inBuffer->getDevicePointer( odev );  // should not mark the buffer dirty
    writeViaLwda( cdev, inDevPtrs[di], 42 );
    launchOptixCopy();
    EXPECT_THAT( readViaOptix( outBuffer ), Eq( MULTIPLE ) );  // other device buffer didn't get synced

    inBuffer->markDirty();
    launchOptixCopy();
    EXPECT_THAT( readViaOptix( outBuffer ), Eq( 42 ) );
}


//////////////////////////////////////
// Parameterized tests
//////////////////////////////////////

class LwdaInteropP : public LwdaInteropBase, public WithParamInterface<LwdaInteropParams>
{
  public:
    virtual void SetUp()
    {
        const LwdaInteropParams& p = GetParam();
        optixDevices               = p.optixDevices;
        inputBufferType            = p.inputBufferType;
        outputBufferType           = p.outputBufferType;
        init();
    }
};

#define SKIP_ON_SET_POINTER_RESTRICTIONS( type )                                                                       \
    {                                                                                                                  \
        if( ( type & RT_BUFFER_OUTPUT ) && !( type & RT_BUFFER_GPU_LOCAL ) )                                           \
        {                                                                                                              \
            std::cerr << "WARNING:Skipping this test.\n";                                                              \
            return;                                                                                                    \
        }                                                                                                              \
    }

#define SKIP_ON_SINGLE_POINTER_WITH_GPU_LOCAL()                                                                        \
    {                                                                                                                  \
        if( optixDevices.size() && inputBufferType & RT_BUFFER_GPU_LOCAL )                                             \
        {                                                                                                              \
            std::cerr << "WARNING:Skipping this test.\n";                                                              \
            return;                                                                                                    \
        }                                                                                                              \
    }

TEST_P( LwdaInteropP, AccessViaOptixWorks )  // Sanity check
{
    writeViaOptix( inBuffer, 42 );
    writeViaOptix( outBuffer, 0 );

    launchOptixCopy();

    EXPECT_THAT( readViaOptix( outBuffer ), Eq( 42 ) );
}

TEST_P( LwdaInteropP, GetSinglePointerWorksForInput )
{
    SKIP_ON_SINGLE_POINTER_WITH_GPU_LOCAL();
    int di = 0, odev = optixDevices[di], cdev = lwdaDevices[di];
    inDevPtrs[di] = inBuffer->getDevicePointer( odev );
    writeViaLwda( cdev, inDevPtrs[di], 42 );
    writeViaOptix( outBuffer, 0 );

    launchOptixCopy();

    EXPECT_THAT( readViaOptix( outBuffer ), Eq( 42 ) );
}

TEST_P( LwdaInteropP, GetSinglePointerWorksForOutput )
{
    int di = 0, odev = optixDevices[di], cdev = lwdaDevices[di];
    outDevPtrs[di] = outBuffer->getDevicePointer( odev );
    writeViaLwda( cdev, outDevPtrs[di], 0 );
    writeViaOptix( inBuffer, 42 );

    launchOptixCopy();

    EXPECT_THAT( readViaLwda( cdev, outDevPtrs[di] ), Eq( 42 ) );
}

TEST_P( LwdaInteropP, SetSinglePointerWorksForInput )
{
    SKIP_ON_SET_POINTER_RESTRICTIONS( inputBufferType );
    SKIP_ON_SINGLE_POINTER_WITH_GPU_LOCAL();
    int di = 0, odev = optixDevices[di], cdev = lwdaDevices[di];
    inDevPtrs[di] = allocLwdaBuffer( di );
    inBuffer      = createBufferForLwda( "input" );
    inBuffer->setDevicePointer( odev, inDevPtrs[di] );
    writeViaLwda( cdev, inDevPtrs[di], 42 );
    writeViaOptix( outBuffer, 0 );

    launchOptixCopy();

    EXPECT_THAT( readViaOptix( outBuffer ), Eq( 42 ) );
}

TEST_P( LwdaInteropP, SetSinglePointerWorksForOutput )
{
    SKIP_ON_SET_POINTER_RESTRICTIONS( outputBufferType );
    int di = 0, odev = optixDevices[di], cdev = lwdaDevices[di];
    outDevPtrs[di] = allocLwdaBuffer( di );
    outBuffer      = createBufferForLwda( "output" );
    outBuffer->setDevicePointer( odev, outDevPtrs[di] );
    writeViaLwda( cdev, outDevPtrs[di], 0 );
    writeViaOptix( inBuffer, 42 );

    launchOptixCopy();

    EXPECT_THAT( readViaLwda( cdev, outDevPtrs[di] ), Eq( 42 ) );
}


TEST_P( LwdaInteropP, GetMultiplePointersWorksForInput )
{
    inDevPtrs = getDevicePointers( inBuffer );
    writeViaLwda( inDevPtrs, 42 );
    writeViaOptix( outBuffer, 0 );

    launchOptixCopy();

    EXPECT_THAT( readViaOptix( outBuffer ), Eq( 42 ) );
}

TEST_P( LwdaInteropP, GetMultiplePointersWorksForOutput )
{
    outDevPtrs = getDevicePointers( outBuffer );
    writeViaLwda( outDevPtrs, 0 );
    writeViaOptix( inBuffer, 42 );

    launchOptixCopy();

    //EXPECT_THAT( readViaLwda( outDevPtrs ), Eq(42) );
    EXPECT_THAT( compositeViaLwda( outDevPtrs ), Eq( 42 ) );
}

TEST_P( LwdaInteropP, SetMultiplePointersWorksForInput )
{
    SKIP_ON_SET_POINTER_RESTRICTIONS( inputBufferType );
    inDevPtrs = allocLwdaBuffers();
    inBuffer  = createBufferForLwda( "input" );
    setDevicePointers( inBuffer, inDevPtrs );
    writeViaLwda( inDevPtrs, 42 );
    writeViaOptix( outBuffer, 0 );

    launchOptixCopy();

    EXPECT_THAT( readViaOptix( outBuffer ), Eq( 42 ) );
}

TEST_P( LwdaInteropP, SetMultiplePointersWorksForOutput )
{
    SKIP_ON_SET_POINTER_RESTRICTIONS( outputBufferType );
    outDevPtrs = allocLwdaBuffers();
    outBuffer  = createBufferForLwda( "output" );
    setDevicePointers( outBuffer, outDevPtrs );
    writeViaLwda( outDevPtrs, 0 );
    writeViaOptix( inBuffer, 42 );

    launchOptixCopy();

    EXPECT_THAT( compositeViaLwda( outDevPtrs ), Eq( 42 ) );
}


std::vector<LwdaInteropParams> getParams( int numDevices )
{
    std::vector<LwdaInteropParams> params;

    if( numDevices == 1 )
    {
        LwdaInteropParams p1;
        p1.optixDevices = {0};
        params.push_back( p1 );
    }

    if( numDevices == 2 )
    {
        // reversed ordinals
        LwdaInteropParams p2;
        p2.optixDevices = {1, 0};
        params.push_back( p2 );

        // input/output buffers
        LwdaInteropParams p3;
        p3.optixDevices     = {0, 1};
        p3.inputBufferType  = RT_BUFFER_INPUT_OUTPUT;
        p3.outputBufferType = RT_BUFFER_INPUT_OUTPUT;
        params.push_back( p3 );

        // gpu local
        LwdaInteropParams p4;
        p4.optixDevices    = {0, 1};
        p4.inputBufferType = RT_BUFFER_INPUT_OUTPUT | RT_BUFFER_GPU_LOCAL;
        params.push_back( p4 );
    }

    return params;
}

INSTANTIATE_TEST_SUITE_P( With1Gpu, LwdaInteropP, ValuesIn( getParams( 1 ) ) );
INSTANTIATE_TEST_SUITE_P( With2Gpu, LwdaInteropP, ValuesIn( getParams( 2 ) ) );

static unsigned int getNumActiveDevices()
{
    RTcontext context = nullptr;
    rtContextCreate( &context );
    unsigned int numActiveDevices = 0U;
    rtContextGetDeviceCount( context, &numActiveDevices );
    rtContextDestroy( context );
    return numActiveDevices;
}

std::vector<std::string> gpuCountFilters()
{
    std::vector<std::string> filters;

    // Filter tests based on number of GPUs
    const unsigned int numDevices = getNumActiveDevices();
    if( numDevices < 3 )
        filters.push_back( "*3Gpu*" );
    if( numDevices < 2 )
        filters.push_back( "*2Gpu*" );

    if( !filters.empty() )
        std::cerr << "WARNING: filtering tests based on number of GPUs: " << numDevices << "\n\n";

    return filters;
}

SrcTestFilter interop( gpuCountFilters );
