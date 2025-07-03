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

#include <Util/RuntimeStateDumper.h>

#include <LWCA/Memory.h>
#include <Context/ObjectManager.h>
#include <Device/LWDADevice.h>
#include <ExelwtionStrategy/CommonRuntime.h>
#include <Objects/Buffer.h>
#include <corelib/misc/String.h>
#include <prodlib/exceptions/IlwalidValue.h>

using namespace optix;
using namespace corelib;
using prodlib::IlwalidValue;


RuntimeStateDumper::RuntimeStateDumper( int launchCount )
    : m_launchCount( launchCount )
{
}

void RuntimeStateDumper::computeBufferSizes( ObjectManager* om )
{
    const auto& buffers = om->getBuffers();
    m_bufferSizes.resize( buffers.linearArraySize(), 0 );
    for( auto id_buffer = buffers.mapBegin(), end = buffers.mapEnd(); id_buffer != end; ++id_buffer )
    {
        const int     id     = id_buffer->first;
        const Buffer* buffer = id_buffer->second;
        m_bufferSizes[id]    = buffer->getTotalSizeInBytes();
    }
}

void RuntimeStateDumper::dump( LWDADevice* device, cort::Global* global_d )
{
    dump( device, global_d, m_bufferSizes );
}

void RuntimeStateDumper::dump( LWDADevice* device, cort::Global* global_d, std::vector<size_t>& bufferSizes )
{
    device->makeLwrrent();
    m_deviceId = device->allDeviceListIndex();

    cort::Global      global;
    std::vector<char> bufferTable;
    dumpBuffer( "global", global_d, sizeof( global ), &global );
    dumpBuffer( "objectRecord", global.objectRecords, global.objectRecordsSize );
    dumpBuffer( "bufferTable", global.bufferTable, global.numBuffers * sizeof( cort::Buffer ), &bufferTable );
    dumpBuffer( "programTable", global.programTable, global.numPrograms * sizeof( cort::ProgramHeader ) );
    dumpBuffer( "textureTable", global.textureTable, global.numTextures * sizeof( cort::TextureSampler ) );
    //global.statusReturn;
    //global.profileData;

    // Dump buffers from table
    cort::Buffer* buffers = reinterpret_cast<cort::Buffer*>( bufferTable.data() );
    for( unsigned i = 0; i < global.numBuffers; ++i )
    {
        if( buffers[i].dd.texUnit == -3 )
            dumpBuffer( stringf( "buffer_%04u", i ), buffers[i].dd.data, bufferSizes[i] );
    }
}

void RuntimeStateDumper::dumpBuffer( const std::string& name, const void* buffer_d, size_t size, std::vector<char>* out_buffer_h /*= nullptr */ )
{
    std::vector<char>  tmp_buffer_h;
    std::vector<char>& buffer_h = ( out_buffer_h ) ? *out_buffer_h : tmp_buffer_h;
    buffer_h.resize( size );
    dumpBuffer( name, buffer_d, size, buffer_h.data() );
}

void RuntimeStateDumper::dumpBuffer( const std::string& name, const void* buffer_d, size_t size, void* buffer_h )
{
    std::string   filename = "rt_" + name + stringf( "_L%04d_D%d.bin", m_launchCount, m_deviceId );
    std::ofstream out( filename, std::ofstream::binary );
    if( out.bad() )
        throw IlwalidValue( RT_EXCEPTION_INFO, "Cannot open file", filename );

    if( buffer_d )
    {
        lwca::memcpyDtoH( buffer_h, reinterpret_cast<LWdeviceptr>( buffer_d ), size );
        out.write( (char*)buffer_h, size );
    }
}
