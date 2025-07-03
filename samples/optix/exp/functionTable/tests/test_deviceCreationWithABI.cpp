//
//  Copyright (c) 2019 LWPU Corporation.  All rights reserved.
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

#include <optix_types.h>

#include <private/optix_7_enum_printers.h>

#include <exp/context/DeviceContext.h>

#include <lwda_runtime.h>

extern "C"
{
    OptixResult optixQueryFunctionTable( int          abiId,
                                         unsigned int numOptions,
                                         OptixQueryFunctionTableOptions* /*optionKeys*/,
                                         const void** /*optiolwalues*/,
                                         void*  functionTable,
                                         size_t sizeOfFunctionTable );
}


using namespace testing;

#define ASSERT_OPTIX_SUCCESS( call_ ) ASSERT_EQ( OPTIX_SUCCESS, call_ )
#define ASSERT_WITH_OPTIX_ERROR( cond, error) if( !cond ) { FAIL() << error; }

namespace {

    const int MIN_ABI_VERSION = 18;
    struct ABIPair { int abi; size_t tableSize; };
    // These need to remain in sorted order by the abi smallest to biggest.
    std::vector<ABIPair > g_abis = { {20, 288}, {22, 288}, {25, 288}, {38, 296},
                                     {43, 304}, {52, 304}, {OPTIX_ABI_VERSION, 344} };

struct CreateWithABIVersion : Test, WithParamInterface<int>
{
    void SetUp() override
    {
        lwdaFree( 0 );
    }

    size_t getTableSize( int abi )
    {
        for( const ABIPair& abiPair : g_abis )
        {
            if( abi <= abiPair.abi )
                return abiPair.tableSize;
        }
        ADD_FAILURE() << "Failed to find ABIPair for abi " << abi;
        return 0;
    }

    // Only use the part of the table we care about.  If this layout should ever change,
    // you will need to use different tables for the different versions of the ABI.
    struct DummyTable {
        const char* ( *optixGetErrorName )( OptixResult result );
        const char* ( *optixGetErrorString )( OptixResult result );
        OptixResult ( *optixDeviceContextCreate )( LWcontext fromContext, const OptixDeviceContextOptions* options, OptixDeviceContext* context );
        OptixResult ( *optixDeviceContextDestroy )( OptixDeviceContext context );
    };

};

} // end anonymous namespace

TEST_P( CreateWithABIVersion, test )
{
    int abi = GetParam();

    size_t tableSize = getTableSize( abi );

    std::vector<char> tableData( tableSize );
    DummyTable* table = reinterpret_cast<DummyTable*>( tableData.data() );

    ASSERT_OPTIX_SUCCESS( optixQueryFunctionTable( abi, 0, 0, 0, table, tableSize ) );

    OptixDeviceContext context = nullptr;
    ASSERT_OPTIX_SUCCESS( table->optixDeviceContextCreate( nullptr, nullptr, &context ) );

    optix_exp::DeviceContext* deviceContext;
    ASSERT_OPTIX_SUCCESS( implCast( context, deviceContext ) );

    ASSERT_EQ( deviceContext->getAbiVersion(), (optix_exp::OptixABI)abi );

    ASSERT_OPTIX_SUCCESS( table->optixDeviceContextDestroy( context ) );
}

INSTANTIATE_TEST_SUITE_P( functionTable, CreateWithABIVersion, Range(MIN_ABI_VERSION, OPTIX_ABI_VERSION+1, 1) );
