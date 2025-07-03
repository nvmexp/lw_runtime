// Copyright (c) 2018, LWPU CORPORATION.
// TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED
// *AS IS* AND LWPU AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS
// OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY
// AND FITNESS FOR A PARTICULAR PURPOSE.  IN NO EVENT SHALL LWPU OR ITS SUPPLIERS
// BE LIABLE FOR ANY SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES
// WHATSOEVER (INCLUDING, WITHOUT LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS,
// BUSINESS INTERRUPTION, LOSS OF BUSINESS INFORMATION, OR ANY OTHER PELWNIARY LOSS)
// ARISING OUT OF THE USE OF OR INABILITY TO USE THIS SOFTWARE, EVEN IF LWPU HAS
// BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES

#include <LWCA/Context.h>
#include <LWCA/Device.h>
#include <Context/RTCore.h>
#include <Lwcm.h>
#include <Util/RecordCompile.h>
#include <Util/optixUuid.h>

#include <corelib/system/LwdaDriver.h>

#include <fstream>
#include <iostream>
#include <memory>
#include <string.h>

void printUsageAndExit( const std::string& argv0 )
{
    std::cerr << "Usage: " << argv0 << " [options] [compile option json path] [bitcode path]\n";
    std::cerr << "Options:\n";
    std::cerr << "   -k | --knobs     Knobs to pass to RTCore.\n";
    exit( 1 );
}

int main( int argc, char** argv )
{
    if( argc < 2 )
        printUsageAndExit( argv[0] );
    std::string knobs;
    std::string arg;
    int         i;
    for( i = 1; i < argc; ++i )
    {
        arg = std::string( argv[i] );
        if( arg[0] == '-' )
        {
            if( arg == "-k" || arg == "--knobs" )
            {
                if( i + 1 == argc )
                    printUsageAndExit( argv[0] );

                knobs = argv[i + 1];
                ++i;
            }
            else
            {
                printUsageAndExit( argv[0] );
            }
        }
        else
            break;
    }

    if( i + 1 >= argc || argc - i > 2 )
        printUsageAndExit( argv[0] );

    std::string jsonPath( argv[i] );
    std::string binPath( argv[i + 1] );

    // Load the specified replay file.
    RtcCompileOptions options;
    std::string       bitcode;
    std::string       name;
    recordcompile::loadCompileCall( jsonPath, binPath, name, options, bitcode );

    // Initialize lwca
    LWresult result = corelib::lwdaDriver().LwInit( 0 );
    if( result != LWDA_ERROR_NO_DEVICE && result != LWDA_SUCCESS )
    {
        std::cerr << "Unable to initialize LWCA\n";
        exit( 1 );
    }

    optix::lwca::Device  device      = optix::lwca::Device::get( 0 );
    optix::lwca::Context lwdaContext = optix::lwca::Context::create( LW_CTX_LMEM_RESIZE_TO_MAX, device );

    // Initialize rtcore.
    std::unique_ptr<optix::RTCore> rtcore( new optix::RTCore() );

    int major = 0;
    int minor = 0;
    rtcore->getVersion( &major, &minor, nullptr );
    std::cout << "Initializing RTcore " << major << '.' << minor << '\n';

    rtcore->init( 0, nullptr, knobs.c_str() );

    unsigned int arch, impl;
    result = corelib::lwdaDriver().LwDeviceGetArchImpl( device.get(), &arch, &impl );
    if( result != LWDA_SUCCESS )
    {
        std::cerr << "Unable to retrieve LWCA properties\n";
        exit( 1 );
    }
    // TODO: Replace this with the proper lwdaDriver()-query as soon as the required changes are in (See lwbugs 2517574, 2505872, and 2298583)
    int hasTTU = ( arch == LW_CFG_ARCHITECTURE_TU100 )
                 && ( impl != LW_CFG_IMPLEMENTATION_TU116 && impl != LW_CFG_IMPLEMENTATION_TU117 );

    RtcDeviceProperties        deviceProperties = {};
    // Increment version number to force rtcore to generate a new identifier,
    // and prevent backward compatibility (even without having to rebuild rtcore.)
    deviceProperties.productIdentifier  = { 'O', 'X', '7', /* version number: */ 0 };
    deviceProperties.chipArchitecture   = arch;
    deviceProperties.chipImplementation = impl;
    deviceProperties.hasTTU             = hasTTU;
    static const unsigned int* version  = optix::getOptixUUID();
    memcpy( &deviceProperties.productUuid, version, sizeof( unsigned int ) * 4 );

    RtcDeviceContext devctx = nullptr;
    rtcore->deviceContextCreateForLWDA( lwdaContext.get(), &deviceProperties, &devctx );

    // Call rtcCompileModule with the given arguments.
    RtcCompiledModule sass = nullptr;
    rtcore->compileModule( devctx, &options, bitcode.c_str(), bitcode.size(), &sass );

    std::cout << "Compile complete!\n";

    return 0;
}