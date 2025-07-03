//
// Copyright (c) 2020 LWPU Corporation.  All rights reserved.
//
// LWPU Corporation and its licensors retain all intellectual property and proprietary
// rights in and to this software, related documentation and any modifications thereto.
// Any use, reproduction, disclosure or distribution of this software and related
// documentation without an express license agreement from LWPU Corporation is strictly
// prohibited.
//
// TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED *AS IS*
// AND LWPU AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS OR IMPLIED,
// INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
// PARTICULAR PURPOSE.  IN NO EVENT SHALL LWPU OR ITS SUPPLIERS BE LIABLE FOR ANY
// SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES WHATSOEVER (INCLUDING, WITHOUT
// LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF
// BUSINESS INFORMATION, OR ANY OTHER PELWNIARY LOSS) ARISING OUT OF THE USE OF OR
// INABILITY TO USE THIS SOFTWARE, EVEN IF LWPU HAS BEEN ADVISED OF THE POSSIBILITY OF
// SUCH DAMAGES

#include <o6/optix.h>
#include <private/optix_6_enum_printers.h>

#include <Context/Context.h>

#include <gtest/gtest.h>

using namespace optix;
using namespace testing;

TEST( ContextAbiCompatibilityTest, abi_16_uses_multithreaded_callbacks )
{
    const RTcontext context = reinterpret_cast<RTcontext>( new Context( Context::ABI_16_USE_MULTITHREADED_DEMAND_LOAD_CALLBACKS_BY_DEFAULT ) );
    int value = 0;

    ASSERT_EQ( RT_SUCCESS, rtContextGetAttribute( context, RT_CONTEXT_ATTRIBUTE_DEMAND_LOAD_MULTITHREADED_CALLBACKS_ENABLED, sizeof( int ), &value ) );
    ASSERT_EQ( 1, value );

    ASSERT_EQ( RT_SUCCESS, rtContextDestroy( context ) );
}

TEST( ContextAbiCompatibilityTest, default_abi_uses_single_threaded_callbacks )
{
    const RTcontext context = reinterpret_cast<RTcontext>( new Context );
    int value = 1;

    ASSERT_EQ( RT_SUCCESS, rtContextGetAttribute( context, RT_CONTEXT_ATTRIBUTE_DEMAND_LOAD_MULTITHREADED_CALLBACKS_ENABLED, sizeof( int ), &value ) );
    ASSERT_EQ( 0, value );

    ASSERT_EQ( RT_SUCCESS, rtContextDestroy( context ) );
}
