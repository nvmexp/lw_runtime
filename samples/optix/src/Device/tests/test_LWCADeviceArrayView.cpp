//
//  Copyright (c) 2020 LWPU Corporation.  All rights reserved.
//
//  LWPU Corporation and its licensors retain all intellectual property and proprietary
//  rights in and to this software, related documentation and any modifications thereto.
//  Any use, reproduction, disclosure or distribution of this software and related
//  documentation without an express license agreement from LWPU Corporation is strictly
//  prohibited.
//
//  TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED *AS IS*
//  AND LWPU AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS OR IMPLIED,
//  INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
//  PARTICULAR PURPOSE.  IN NO EVENT SHALL LWPU OR ITS SUPPLIERS BE LIABLE FOR ANY
//  SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES WHATSOEVER (INCLUDING, WITHOUT
//  LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF
//  BUSINESS INFORMATION, OR ANY OTHER PELWNIARY LOSS) ARISING OUT OF THE USE OF OR
//  INABILITY TO USE THIS SOFTWARE, EVEN IF LWPU HAS BEEN ADVISED OF THE POSSIBILITY OF
//  SUCH DAMAGES
//

#include <srcTests.h>

#include <Context/Context.h>
#include <Device/CPUDevice.h>
#include <Device/LWDADevice.h>
#include <Device/DeviceManager.h>

#include <algorithm>

using namespace optix;
using namespace testing;

namespace {

struct LWDADeviceArrayViewTest : Test
{
    LWDADeviceArrayViewTest()
    {
        rtContextCreate( &m_contextApi );
        m_context         = reinterpret_cast<Context*>( m_contextApi );
        DeviceManager* dm = m_context->getDeviceManager();
        m_cpuDevice       = dm->cpuDevice();
        m_lwdaDevice      = dm->primaryLWDADevice();
    }
    ~LWDADeviceArrayViewTest() { rtContextDestroy( m_contextApi ); }

    unsigned int countLWDADevices( const LWDADeviceArrayView& view );

    RTcontext m_contextApi = nullptr;
    Context*  m_context    = nullptr;
    Device*   m_cpuDevice  = nullptr;
    Device*   m_lwdaDevice = nullptr;
};

unsigned int LWDADeviceArrayViewTest::countLWDADevices( const LWDADeviceArrayView& view )
{
    unsigned int count = 0;
    for( LWDADevice* device : view )
    {
        ++count;
    }
    return count;
}

}  // namespace

TEST_F( LWDADeviceArrayViewTest, dereference_end_iterator_yields_nullptr )
{
    const DeviceArray devices;
    ASSERT_EQ( nullptr, *std::end( LWDADeviceArrayView( devices ) ) );
}

TEST_F( LWDADeviceArrayViewTest, empty_device_array )
{
    DeviceArray devices{};
    LWDADeviceArrayView view{devices};

    ASSERT_EQ( 0, countLWDADevices( view ) );
}

TEST_F( LWDADeviceArrayViewTest, skips_only_cpu_device )
{
    DeviceArray devices{m_cpuDevice};
    LWDADeviceArrayView view{devices};

    ASSERT_EQ( 0, countLWDADevices( view ) );
}

TEST_F( LWDADeviceArrayViewTest, skips_cpu_device_at_begin )
{
    DeviceArray         devices{m_cpuDevice, m_lwdaDevice};
    LWDADeviceArrayView view{devices};

    ASSERT_EQ( 1, countLWDADevices( view ) );
}

TEST_F( LWDADeviceArrayViewTest, skips_cpu_device_at_end )
{
    DeviceArray         devices{m_lwdaDevice, m_cpuDevice};
    LWDADeviceArrayView view{devices};

    ASSERT_EQ( 1, countLWDADevices( view ) );
}

TEST_F( LWDADeviceArrayViewTest, counts_all_LWDA_devices )
{
    DeviceArray         devices{m_lwdaDevice, m_cpuDevice, m_lwdaDevice};
    LWDADeviceArrayView view{devices};

    ASSERT_EQ( 2, countLWDADevices( view ) );
}

TEST_F( LWDADeviceArrayViewTest, begin_iterators_compare_equal )
{
    DeviceArray devices{};
    LWDADeviceArrayView view{devices};

    ASSERT_EQ( view.begin(), view.begin() );
}

TEST_F( LWDADeviceArrayViewTest, end_iterators_compare_equal )
{
    DeviceArray devices{};
    LWDADeviceArrayView view{devices};

    ASSERT_EQ( view.end(), view.end() );
}

TEST_F( LWDADeviceArrayViewTest, begin_end_iterators_of_empty_container_compare_equal )
{
    DeviceArray devices{};
    LWDADeviceArrayView view{devices};

    ASSERT_EQ( view.begin(), view.end() );
}

TEST_F( LWDADeviceArrayViewTest, begin_end_iterators_of_non_empty_container_compare_different )
{
    DeviceArray devices{m_lwdaDevice};
    LWDADeviceArrayView view{devices};

    ASSERT_NE( view.begin(), view.end() );
}

TEST_F( LWDADeviceArrayViewTest, iterator_for_empty_container_equals_default_constructed )
{
    DeviceArray devices{};
    LWDADeviceArrayView view{devices};
    LWDADeviceArrayView::const_iterator iter;

    ASSERT_EQ( view.begin(), iter );
    ASSERT_EQ( view.end(), iter );
}

TEST_F( LWDADeviceArrayViewTest, iterator_at_end_of_non_empty_container_equals_default_constructed )
{
    DeviceArray devices{m_lwdaDevice};
    LWDADeviceArrayView view{devices};
    LWDADeviceArrayView::const_iterator iter1 = view.begin();
    LWDADeviceArrayView::const_iterator iter2;

    ++iter1;

    ASSERT_EQ( iter1, iter2 );
}
