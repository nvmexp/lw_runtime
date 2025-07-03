// Copyright (c) 2017, LWPU CORPORATION.
// TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED
// *AS IS* AND LWPU AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS
// OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY
// AND FITNESS FOR A PARTICULAR PURPOSE.  IN NO FUNCTION SHALL LWPU OR ITS SUPPLIERS
// BE LIABLE FOR ANY SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES
// WHATSOEVER (INCLUDING, WITHOUT LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS,
// BUSINESS INTERRUPTION, LOSS OF BUSINESS INFORMATION, OR ANY OTHER PELWNIARY LOSS)
// ARISING OUT OF THE USE OF OR INABILITY TO USE THIS SOFTWARE, EVEN IF LWPU HAS
// BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES

#include <LWCA/Function.h>

#include <LWCA/ErrorCheck.h>
#include <LWCA/Stream.h>
#include <LWCA/TexRef.h>
#include <corelib/system/LwdaDriver.h>
#include <prodlib/exceptions/Assert.h>

using namespace optix;
using namespace optix::lwca;
using namespace corelib;

Function::Function()
    : m_function( nullptr )
{
}

Function::Function( LWfunction function )
    : m_function( function )
{
    m_numRegs    = getAttribute( LW_FUNC_ATTRIBUTE_NUM_REGS );
    m_constSize  = getAttribute( LW_FUNC_ATTRIBUTE_CONST_SIZE_BYTES );
    m_sharedSize = getAttribute( LW_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES );
    m_localSize  = getAttribute( LW_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES );
}

LWfunction Function::get()
{
    return m_function;
}

const LWfunction Function::get() const
{
    return m_function;
}

void Function::setCacheConfig( LWfunc_cache config, LWresult* returnResult )
{
    RT_ASSERT( m_function != nullptr );
    CHECK( lwdaDriver().LwFuncSetCacheConfig( m_function, config ) );
}

void Function::setSharedMemConfig( LWsharedconfig config, LWresult* returnResult )
{
    RT_ASSERT( m_function != nullptr );
    CHECK( lwdaDriver().LwFuncSetSharedMemConfig( m_function, config ) );
}

void Function::launchKernel( unsigned int gridDimX,
                             unsigned int gridDimY,
                             unsigned int gridDimZ,
                             unsigned int blockDimX,
                             unsigned int blockDimY,
                             unsigned int blockDimZ,
                             unsigned int sharedMemBytes,
                             Stream       stream,
                             void**       kernelParams,
                             void**       extra,
                             LWresult*    returnResult )
{
    RT_ASSERT( m_function != nullptr );
    CHECK( lwdaDriver().LwLaunchKernel( m_function, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ,
                                        sharedMemBytes, stream.get(), kernelParams, extra, 0, 0, 0 ) );
}

void Function::setTexRef( int texUnit, const TexRef& texref, LWresult* returnResult ) const
{
    RT_ASSERT( m_function != nullptr );
    CHECK( lwdaDriver().LwParamSetTexRef( m_function, texUnit, texref.get() ) );
}

int Function::getAttribute( LWfunction_attribute attrib, LWresult* returnResult ) const
{
    RT_ASSERT( m_function != nullptr );
    int result = 0;
    CHECK( lwdaDriver().LwFuncGetAttribute( &result, attrib, m_function ) );
    return result;
}

int Function::getNumRegisters() const
{
    return m_numRegs;
}

int Function::getConstSize() const
{
    return m_constSize;
}

int Function::getSharedSize() const
{
    return m_sharedSize;
}

int Function::getLocalSize() const
{
    return m_localSize;
}

// Host Function implementation

HostFunction::HostFunction()
    : m_hostFunction( nullptr )
{
}

HostFunction::HostFunction( LWhostFn function )
    : m_hostFunction( function )
{
}

LWhostFn HostFunction::get()
{
    return m_hostFunction;
}

const LWhostFn HostFunction::get() const
{
    return m_hostFunction;
}

void HostFunction::launchHostFunc( Stream stream, void* userData, LWresult* returnResult )
{
    RT_ASSERT( m_hostFunction != nullptr );
    CHECK( lwdaDriver().LwLaunchHostFunc( stream.get(), m_hostFunction, userData ) );
}
