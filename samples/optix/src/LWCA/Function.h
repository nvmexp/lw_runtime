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

#pragma once

#include <lwca.h>

#if LWDA_VERSION >= 10000 && LWDA_VERSION < 10010
// this is needed because the lwca header in lwca-10.0-nightly-24221979 is incomplete.
typedef void( LWDA_CB* LWhostFn )( void* userData );
#endif

namespace optix {
namespace lwca {

class Stream;
class TexRef;

class Function
{
  public:
    Function();

    // Get the low-level function
    LWfunction       get();
    const LWfunction get() const;

    // Sets the preferred cache configuration for a device function.
    void setCacheConfig( LWfunc_cache config, LWresult* returnResult = nullptr );

    // Sets the shared memory configuration for a device function.
    void setSharedMemConfig( LWsharedconfig config, LWresult* returnResult = nullptr );

    // Launches a LWCA function.
    void launchKernel( unsigned int gridDimX,
                       unsigned int gridDimY,
                       unsigned int gridDimZ,
                       unsigned int blockDimX,
                       unsigned int blockDimY,
                       unsigned int blockDimZ,
                       unsigned int sharedMemBytes,
                       Stream       stream,
                       void**       kernelParams,
                       void**       extra,
                       LWresult*    returnResult = nullptr );

    // Adds a texture-reference to the function's argument list.
    void setTexRef( int texUnit, const TexRef& texref, LWresult* returnResult = nullptr ) const;

    // Returns information about a function.
    int getAttribute( LWfunction_attribute attrib, LWresult* returnResult = nullptr ) const;

    // Attribute shortlwts
    int getNumRegisters() const;
    int getConstSize() const;
    int getSharedSize() const;
    int getLocalSize() const;

  private:
    explicit Function( LWfunction function );
    friend class Module;

    LWfunction m_function;

    int m_numRegs    = 0;
    int m_constSize  = 0;
    int m_sharedSize = 0;
    int m_localSize  = 0;
};

// Additional class for host function callbacks
class HostFunction
{
  public:
    HostFunction();

    // Get the low-level host function
    LWhostFn       get();
    const LWhostFn get() const;

    // Add a host function launch to the stream.
    void launchHostFunc( Stream stream, void* userData, LWresult* returnResult = nullptr );

  private:
    explicit HostFunction( LWhostFn function );

    LWhostFn m_hostFunction;
};

}  // namespace lwca
}  // namespace optix
