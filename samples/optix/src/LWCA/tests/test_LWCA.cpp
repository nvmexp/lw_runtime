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

#include <srcTests.h>

#include <LWCA/Context.h>
#include <LWCA/Device.h>
#include <LWCA/Function.h>
#include <LWCA/Module.h>
#include <LWCA/Stream.h>

#include <corelib/system/LwdaDriver.h>

#include <fstream>
#include <sstream>

TEST( LWDA_module, Context )
{
    LWresult res;
    res = corelib::lwdaDriver().LwInit( 0 );
    EXPECT_EQ( LWDA_SUCCESS, res );

    optix::lwca::Device device;
    device = optix::lwca::Device::get( 0, &res );
    EXPECT_EQ( LWDA_SUCCESS, res );

    optix::lwca::Context ctx = optix::lwca::Context::create( LW_CTX_LMEM_RESIZE_TO_MAX | LW_CTX_MAP_HOST, device, &res );
    EXPECT_EQ( LWDA_SUCCESS, res );

    ctx.destroy( &res );
    EXPECT_EQ( LWDA_SUCCESS, res );
}

TEST( LWDA_module, ModuleExelwtion )
{
    LWresult res;
    res = corelib::lwdaDriver().LwInit( 0 );
    EXPECT_EQ( LWDA_SUCCESS, res );

    optix::lwca::Device device = optix::lwca::Device::get( 0, &res );
    EXPECT_EQ( LWDA_SUCCESS, res );

    optix::lwca::Context ctx = optix::lwca::Context::create( LW_CTX_LMEM_RESIZE_TO_MAX | LW_CTX_MAP_HOST, device, &res );
    EXPECT_EQ( LWDA_SUCCESS, res );

    std::string   filename = dataPath() + "/LWCA/simpleKernel.ptx";
    std::ifstream ptxFile( filename.c_str() );
    std::string   ptx( ( std::istreambuf_iterator<char>( ptxFile ) ), std::istreambuf_iterator<char>() );
    ptxFile.close();

    optix::lwca::Module module = optix::lwca::Module::loadData( (const void*)ptx.c_str(), &res );
    EXPECT_EQ( LWDA_SUCCESS, res );
    optix::lwca::Function function = module.getFunction( "test_kernel", &res );
    EXPECT_EQ( LWDA_SUCCESS, res );

    optix::lwca::Stream stream = optix::lwca::Stream::create( LW_STREAM_DEFAULT, &res );
    EXPECT_EQ( LWDA_SUCCESS, res );
    function.launchKernel( 1, 1, 1, 1, 1, 1, 0, stream, nullptr, nullptr, &res );
    EXPECT_EQ( LWDA_SUCCESS, res );

    module.unload( &res );
    EXPECT_EQ( LWDA_SUCCESS, res );

    ctx.destroy( &res );
    EXPECT_EQ( LWDA_SUCCESS, res );
}
