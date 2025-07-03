#include <srcTests.h>

#include <LWCA/ComputeCapability.h>
#include <Context/Context.h>
#include <Device/CPUDevice.h>
#include <Device/LWDADevice.h>
#include <Device/Device.h>

using namespace optix;
using namespace testing;

namespace {

//------------------------------------------------------------------------------
class DeviceCastTest : public Test
{
  public:
    DeviceCastTest() {}
    virtual ~DeviceCastTest() {}

    void SetUp() override
    {
        m_context = new Context();
        m_context->setEntryPointCount( 1 );
        m_context->setRayTypeCount( 1 );
    }

    void TearDown() override
    {
        m_context->tearDown();
        delete m_context;
    }

    Context* m_context;
};

//------------------------------------------------------------------------------

template <typename T>
DeviceType staticDeviceTypeTester()
{
    return T::m_deviceType;
}

// LWDADevice cast

TEST_F( DeviceCastTest, NullptrToDeviceType )
{
    Device* device = nullptr;

    LWDADevice* lwdaDevice = deviceCast<LWDADevice>( device );

    EXPECT_EQ( lwdaDevice, nullptr );
}

TEST_F( DeviceCastTest, NullptrToDeviceTypeConst )
{
    const Device* device = nullptr;

    const LWDADevice* lwdaDevice = deviceCast<const LWDADevice>( device );

    EXPECT_EQ( lwdaDevice, nullptr );
}

TEST_F( DeviceCastTest, StaticDeviceTypeLWDADevice )
{

    EXPECT_EQ( staticDeviceTypeTester<LWDADevice>(), LWDA_DEVICE );
}

TEST_F( DeviceCastTest, DeviceToLWDADevice )
{
    Device* device = new LWDADevice( m_context, 0, optix::lwca::SM_NONE() );

    LWDADevice* lwdaDevice = deviceCast<LWDADevice>( device );

    EXPECT_TRUE( device->isA( LWDA_DEVICE ) );
    EXPECT_NE( lwdaDevice, nullptr );
}

TEST_F( DeviceCastTest, DeviceToLWDADeviceConst )
{
    const Device* device = new LWDADevice( m_context, 0, optix::lwca::SM_NONE() );

    const LWDADevice* lwdaDevice = deviceCast<const LWDADevice>( device );

    EXPECT_TRUE( device->isA( LWDA_DEVICE ) );
    EXPECT_NE( lwdaDevice, nullptr );
}

// CPUDevice cast

TEST_F( DeviceCastTest, StaticDeviceTypeCPUDevice )
{

    EXPECT_EQ( staticDeviceTypeTester<CPUDevice>(), CPU_DEVICE );
}

TEST_F( DeviceCastTest, DeviceToCPUDevice )
{
    Device* device = new CPUDevice( m_context );

    CPUDevice* cpuDevice = deviceCast<CPUDevice>( device );

    EXPECT_TRUE( device->isA( CPU_DEVICE ) );
    EXPECT_NE( cpuDevice, nullptr );
}

TEST_F( DeviceCastTest, DeviceToCPUDeviceConst )
{
    const Device* device = new CPUDevice( m_context );

    const CPUDevice* cpuDevice = deviceCast<const CPUDevice>( device );

    EXPECT_TRUE( device->isA( CPU_DEVICE ) );
    EXPECT_NE( cpuDevice, nullptr );
}

}  // end namespace
