/*
* Copyright (c) 2017, LWPU CORPORATION.  All rights reserved.
*
* LWPU Corporation and its licensors retain all intellectual property and proprietary
* rights in and to this software, related documentation and any modifications thereto.
* Any use, reproduction, disclosure or distribution of this software and related
* documentation without an express license agreement from LWPU Corporation is strictly
* prohibited.
*
* TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED *AS IS*
* AND LWPU AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS OR IMPLIED,
* INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
* PARTICULAR PURPOSE.  IN NO EVENT SHALL LWPU OR ITS SUPPLIERS BE LIABLE FOR ANY
* SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES WHATSOEVER (INCLUDING, WITHOUT
* LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF
* BUSINESS INFORMATION, OR ANY OTHER PELWNIARY LOSS) ARISING OUT OF THE USE OF OR
* INABILITY TO USE THIS SOFTWARE, EVEN IF LWPU HAS BEEN ADVISED OF THE POSSIBILITY OF
* SUCH DAMAGES
*/

#include <srcTests.h>

#include <LWCA/Memory.h>
#include <Context/Context.h>
#include <Device/DeviceManager.h>
#include <Memory/DeviceSpecificTable.h>
#include <Memory/MemoryManager.h>

#include <algorithm>

using namespace optix;
using namespace testing;

namespace {

const size_t HEIGHT = 10U;

struct TestDeviceSpecificTableBase : Test
{
    void SetUp() override
    {
        RTcontext ctx_api;
        ASSERT_EQ( RT_SUCCESS, rtContextCreate( &ctx_api ) );
        m_context = reinterpret_cast<Context*>( ctx_api );
        m_context->getDeviceManager()->enableActiveDevices();
    }

    void TearDown() override
    {
        rtContextDestroy( reinterpret_cast<RTcontext>( m_context ) );
        m_context = nullptr;
    }

    Context* m_context = nullptr;
};

template <typename Table>
struct TestDeviceSpecificTableBaseT : TestDeviceSpecificTableBase
{
    void SetUp() override
    {
        TestDeviceSpecificTableBase::SetUp();
        m_dst.reset( new Table( m_context ) );
        m_dst->resize( HEIGHT );
    }

    void TearDown() override
    {
        m_dst.reset();
        TestDeviceSpecificTableBase::TearDown();
    }

    std::unique_ptr<Table> m_dst;
};

struct Interleaved
{
    struct DeviceIndependent
    {
        int food;
    };

    struct DeviceDependent
    {
        int          dead;
        int          beef;
        unsigned int allDevicesIndex;
    };

    DeviceIndependent di;
    DeviceDependent   dd;
};

using Table = DeviceSpecificTable<Interleaved>;

struct TestDeviceSpecificTable : TestDeviceSpecificTableBaseT<Table>
{
    void fillDiTable()
    {
        Table::DiTableLock di( *m_dst );
        std::fill( &di[0], &di[HEIGHT], Interleaved::DeviceIndependent{0xf00d} );
    }

    void fillDdTable()
    {
        for( auto device : m_context->getDeviceManager()->activeDevices() )
        {
            Table::DdTableLock dd( *m_dst, device->allDeviceListIndex() );
            std::fill( &dd[0], &dd[HEIGHT], Interleaved::DeviceDependent{0xdead, 0xbeef, device->allDeviceListIndex()} );
        }
    }
};

}  // namespace

TEST_F( TestDeviceSpecificTable, UpdateEntireTable )
{
    fillDiTable();
    fillDdTable();

    m_context->getMemoryManager()->syncAllMemoryBeforeLaunch();
    m_dst->sync();

    std::vector<Interleaved> deviceDataCopy;
    deviceDataCopy.resize( HEIGHT );
    for( auto device : m_context->getDeviceManager()->activeDevices() )
    {
        const unsigned int allDevicesIndex = device->allDeviceListIndex();
        Interleaved*       deviceDataPtr{m_dst->getInterleavedTableDevicePtr( allDevicesIndex )};
        LWresult           status{LWDA_SUCCESS};
        lwca::memcpyDtoH( &deviceDataCopy[0], reinterpret_cast<LWdeviceptr>( deviceDataPtr ),
                          sizeof( Interleaved ) * HEIGHT, &status );
        ASSERT_EQ( LWDA_SUCCESS, status );
        for( size_t i = 0; i < HEIGHT; ++i )
        {
            ASSERT_EQ( 0xf00d, deviceDataCopy[i].di.food );
            ASSERT_EQ( 0xdead, deviceDataCopy[i].dd.dead );
            ASSERT_EQ( 0xbeef, deviceDataCopy[i].dd.beef );
            ASSERT_EQ( allDevicesIndex, deviceDataCopy[i].dd.allDevicesIndex );
        }
    }
}

TEST_F( TestDeviceSpecificTable, UpdateDiTable )
{
    fillDiTable();

    m_context->getMemoryManager()->syncAllMemoryBeforeLaunch();
    m_dst->sync();

    std::vector<Interleaved> deviceDataCopy;
    deviceDataCopy.resize( HEIGHT );
    for( auto device : m_context->getDeviceManager()->activeDevices() )
    {
        Interleaved* deviceDataPtr{m_dst->getInterleavedTableDevicePtr( device->allDeviceListIndex() )};
        LWresult     status{LWDA_SUCCESS};
        lwca::memcpyDtoH( &deviceDataCopy[0], reinterpret_cast<LWdeviceptr>( deviceDataPtr ),
                          sizeof( Interleaved ) * HEIGHT, &status );
        ASSERT_EQ( LWDA_SUCCESS, status );
        for( size_t i = 0; i < HEIGHT; ++i )
        {
            ASSERT_EQ( 0xf00d, deviceDataCopy[i].di.food );
        }
    }
}

TEST_F( TestDeviceSpecificTable, UpdateDdTable )
{
    fillDdTable();

    m_context->getMemoryManager()->syncAllMemoryBeforeLaunch();
    m_dst->sync();

    std::vector<Interleaved> deviceDataCopy;
    deviceDataCopy.resize( HEIGHT );
    for( auto device : m_context->getDeviceManager()->activeDevices() )
    {
        const unsigned int allDevicesIndex = device->allDeviceListIndex();
        Interleaved*       deviceDataPtr{m_dst->getInterleavedTableDevicePtr( allDevicesIndex )};
        LWresult           status{LWDA_SUCCESS};
        lwca::memcpyDtoH( &deviceDataCopy[0], reinterpret_cast<LWdeviceptr>( deviceDataPtr ),
                          sizeof( Interleaved ) * HEIGHT, &status );
        ASSERT_EQ( LWDA_SUCCESS, status );
        for( size_t i = 0; i < HEIGHT; ++i )
        {
            ASSERT_EQ( 0xdead, deviceDataCopy[i].dd.dead );
            ASSERT_EQ( 0xbeef, deviceDataCopy[i].dd.beef );
            ASSERT_EQ( allDevicesIndex, deviceDataCopy[i].dd.allDevicesIndex );
        }
    }
}

TEST_F( TestDeviceSpecificTable, ResizeReturnsTrueOnRealloc )
{
    ASSERT_TRUE( m_dst->resize( HEIGHT * 2 ) );
    ASSERT_FALSE( m_dst->resize( HEIGHT ) );
}

TEST_F( TestDeviceSpecificTable, ResizePreservesDataOnRealloc )
{
    fillDiTable();

    ASSERT_TRUE( m_dst->resize( HEIGHT * 2 ) );

    Table::DiTableLock di( *m_dst );
    for( size_t i = 0; i < HEIGHT; ++i )
    {
        ASSERT_EQ( 0xf00d, di[i].food );
    }
}

TEST_F( TestDeviceSpecificTable, GetTableSizeInBytes )
{
    ASSERT_EQ( sizeof( Interleaved ) * HEIGHT, m_dst->getTableSizeInBytes() );
}

TEST_F( TestDeviceSpecificTable, FillToCapacity )
{
    ASSERT_TRUE( m_dst->resize( m_dst->capacity() + 1U ) );
    ASSERT_GE( m_dst->capacity(), HEIGHT * 2 );
    {
        Table::DiTableLock di( *m_dst );
        std::fill( &di[0], &di[m_dst->capacity()], Interleaved::DeviceIndependent{0xf00d} );
    }
}

TEST_F( TestDeviceSpecificTable, MappingTableMarksRecordsDirty )
{
    m_dst->mapDeviceIndependentTable( 1, 2 );
    m_dst->mapDeviceIndependentTable( 5, 6 );
    m_dst->mapDeviceIndependentTable( 11, 12 );
    m_dst->mapDeviceIndependentTable( 8, 9 );

    ASSERT_THAT( m_dst->getDeviceIndependentTableDirtyIndicesForTest(),
                 ElementsAre( Pair( 1, 2 ), Pair( 5, 6 ), Pair( 11, 12 ), Pair( 8, 9 ) ) );

    m_dst->mapDeviceIndependentTable( 3, 7 );
    m_dst->mapDeviceIndependentTable( 0 );

    ASSERT_THAT( m_dst->getDeviceIndependentTableDirtyIndicesForTest(),
                 ElementsAre( Pair( 0, 9 ), Pair( 11, 12 ), Pair( -1, -1 ), Pair( -1, -1 ) ) );

    m_dst->mapDeviceIndependentTable( 13, 15 );
    m_dst->mapDeviceIndependentTable( 10, 15 );

    ASSERT_THAT( m_dst->getDeviceIndependentTableDirtyIndicesForTest(),
                 ElementsAre( Pair( 0, 15 ), Pair( -1, -1 ), Pair( -1, -1 ), Pair( -1, -1 ) ) );
}

TEST_F( TestDeviceSpecificTable, SyncCopiesChangedRecords )
{
    fillDiTable();
    fillDdTable();

    m_context->getMemoryManager()->syncAllMemoryBeforeLaunch();
    m_dst->sync();

    *m_dst->mapDeviceIndependentTable( 0 ) = Interleaved::DeviceIndependent{0x0bad};

    for( auto device : m_context->getDeviceManager()->activeDevices() )
    {
        const unsigned int deviceIndex = device->allDeviceListIndex();
        *m_dst->mapDeviceDependentTable( deviceIndex, 1 ) = Interleaved::DeviceDependent{0xface, 0xfeed, deviceIndex};
        *m_dst->mapDeviceDependentTable( deviceIndex, 3 ) = Interleaved::DeviceDependent{0x0dad, 0x0b0d, deviceIndex};
        *m_dst->mapDeviceDependentTable( deviceIndex, 2 ) = Interleaved::DeviceDependent{0x0c0d, 0xcafe, deviceIndex};
    }
    m_dst->sync();

    std::vector<Interleaved> deviceDataCopy;
    deviceDataCopy.resize( HEIGHT );
    for( auto device : m_context->getDeviceManager()->activeDevices() )
    {
        const unsigned int allDevicesIndex = device->allDeviceListIndex();
        Interleaved*       deviceDataPtr{m_dst->getInterleavedTableDevicePtr( allDevicesIndex )};
        LWresult           status{LWDA_SUCCESS};
        lwca::memcpyDtoH( &deviceDataCopy[0], reinterpret_cast<LWdeviceptr>( deviceDataPtr ),
                          sizeof( Interleaved ) * HEIGHT, &status );
        ASSERT_EQ( LWDA_SUCCESS, status );

        ASSERT_EQ( 0x0bad, deviceDataCopy[0].di.food );
        ASSERT_EQ( 0xface, deviceDataCopy[1].dd.dead );
        ASSERT_EQ( 0xfeed, deviceDataCopy[1].dd.beef );
        ASSERT_EQ( 0x0c0d, deviceDataCopy[2].dd.dead );
        ASSERT_EQ( 0xcafe, deviceDataCopy[2].dd.beef );
        ASSERT_EQ( 0x0dad, deviceDataCopy[3].dd.dead );
        ASSERT_EQ( 0x0b0d, deviceDataCopy[3].dd.beef );
    }
}

namespace {

struct SmallDi
{
    char colon;
};

struct SmallDd
{
    unsigned int allDevicesIndex;
    char         space;
};

struct SmallInterleaved
{
    SmallDi di;
    SmallDd dd;
};

using SmallTable =
    DeviceSpecificTable<SmallInterleaved, SmallDi, SmallDd, sizeof( SmallInterleaved ) - sizeof( SmallDi ) - sizeof( SmallDd )>;

struct TestDeviceSpecificTablePadding : TestDeviceSpecificTableBaseT<SmallTable>
{
    void fillDiTable()
    {
        SmallTable::DiTableLock di( *m_dst );
        std::fill( &di[0], &di[HEIGHT], SmallDi{':'} );
    }

    void fillDdTable()
    {
        for( auto device : m_context->getDeviceManager()->activeDevices() )
        {
            SmallTable::DdTableLock dd( *m_dst, device->allDeviceListIndex() );
            std::fill( &dd[0], &dd[HEIGHT], SmallDd{device->allDeviceListIndex(), ' '} );
        }
    }
};

}  // namespace

TEST_F( TestDeviceSpecificTablePadding, TableWithPadding )
{
    fillDiTable();
    fillDdTable();

    m_context->getMemoryManager()->syncAllMemoryBeforeLaunch();
    m_dst->sync();

    std::vector<SmallInterleaved> deviceDataCopy;
    deviceDataCopy.resize( HEIGHT );
    for( auto device : m_context->getDeviceManager()->activeDevices() )
    {
        const unsigned int allDevicesIndex = device->allDeviceListIndex();
        SmallInterleaved*  deviceDataPtr{m_dst->getInterleavedTableDevicePtr( allDevicesIndex )};
        LWresult           status{LWDA_SUCCESS};
        lwca::memcpyDtoH( &deviceDataCopy[0], reinterpret_cast<LWdeviceptr>( deviceDataPtr ),
                          sizeof( SmallInterleaved ) * HEIGHT, &status );
        ASSERT_EQ( LWDA_SUCCESS, status );
        for( size_t i = 0; i < HEIGHT; ++i )
        {
            ASSERT_EQ( ':', deviceDataCopy[i].di.colon );
            ASSERT_EQ( allDevicesIndex, deviceDataCopy[i].dd.allDevicesIndex );
            ASSERT_EQ( ' ', deviceDataCopy[i].dd.space );
        }
    }
}

namespace {

struct SmallInterleavedDdFirst
{
    SmallDd dd;
    SmallDi di;
};

using SmallTableDdFirst =
    DeviceSpecificTable<SmallInterleavedDdFirst, SmallDi, SmallDd, sizeof( SmallInterleavedDdFirst ) - sizeof( SmallDi ) - sizeof( SmallDd ), false>;

struct TestDeviceSpecificTablePaddingDdFirst : TestDeviceSpecificTableBaseT<SmallTableDdFirst>
{
    void fillDiTable()
    {
        SmallTableDdFirst::DiTableLock di( *m_dst );
        std::fill( &di[0], &di[HEIGHT], SmallDi{':'} );
    }

    void fillDdTable()
    {
        for( auto device : m_context->getDeviceManager()->activeDevices() )
        {
            SmallTableDdFirst::DdTableLock dd( *m_dst, device->allDeviceListIndex() );
            std::fill( &dd[0], &dd[HEIGHT], SmallDd{device->allDeviceListIndex(), ' '} );
        }
    }
};

}  // namespace

TEST_F( TestDeviceSpecificTablePaddingDdFirst, TableWithPadding )
{
    fillDiTable();
    fillDdTable();

    m_context->getMemoryManager()->syncAllMemoryBeforeLaunch();
    m_dst->sync();

    std::vector<SmallInterleavedDdFirst> deviceDataCopy;
    deviceDataCopy.resize( HEIGHT );
    for( auto device : m_context->getDeviceManager()->activeDevices() )
    {
        const unsigned int       allDevicesIndex = device->allDeviceListIndex();
        SmallInterleavedDdFirst* deviceDataPtr{m_dst->getInterleavedTableDevicePtr( allDevicesIndex )};
        LWresult                 status{LWDA_SUCCESS};
        lwca::memcpyDtoH( &deviceDataCopy[0], reinterpret_cast<LWdeviceptr>( deviceDataPtr ),
                          sizeof( SmallInterleavedDdFirst ) * HEIGHT, &status );
        ASSERT_EQ( LWDA_SUCCESS, status );
        for( size_t i = 0; i < HEIGHT; ++i )
        {
            ASSERT_EQ( ':', deviceDataCopy[i].di.colon );
            ASSERT_EQ( allDevicesIndex, deviceDataCopy[i].dd.allDevicesIndex );
            ASSERT_EQ( ' ', deviceDataCopy[i].dd.space );
        }
    }
}
