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

#pragma once

#include <lwca.h>
#include <string>

namespace optix {
namespace lwca {

class Device;

class Context
{
  public:
    Context();

    // Get the low-level context
    LWcontext       get();
    const LWcontext get() const;


    // Create a LWCA context.
    static Context create( unsigned int flags, const Device& dev, LWresult* returnResult = nullptr );

    // Destroy a LWCA context.
    void destroy( LWresult* returnResult = nullptr );

    // Gets the context's API version.
    unsigned int getApiVersion( LWresult* returnResult = nullptr );

    // Returns if this context is valid (!= NULL)
    bool isValid() const;

    // Returns the preferred cache configuration for the current context.
    LWfunc_cache getCacheConfig( LWresult* returnResult = nullptr );

    // Returns the LWCA context bound to the calling CPU thread.
    static Context getLwrrent( LWresult* returnResult = nullptr );

    // Returns the device ID for the current context.
    static Device getDevice( LWresult* returnResult = nullptr );

    // Returns resource limits.
    static size_t getLimit( LWlimit limit, LWresult* returnResult = nullptr );

    // Returns the status of the flag in the current context.
    static bool getFlag( unsigned int flag, LWresult* returnResult = nullptr );

    // Returns the current shared memory configuration for the current context.
    static LWsharedconfig getSharedMemConfig( LWresult* returnResult = nullptr );

    // Returns numerical values that correspond to the least and greatest stream priorities.
    static void getStreamPriorityRange( int* leastPriority, int* greatestPriority, LWresult* returnResult = nullptr );

    // Pops the current LWCA context from the current CPU thread.
    static Context popLwrrent( LWresult* returnResult = nullptr );

    // Pushes a context on the current CPU thread.
    void pushLwrrent( LWresult* returnResult = nullptr );

    // Sets the preferred cache configuration for the current context.
    static void setCacheConfig( LWfunc_cache config, LWresult* returnResult = nullptr );

    // Makes this context current for the calling CPU thread.
    void setLwrrent( LWresult* returnResult = nullptr );

    // Sets the current context to 0 for the calling CPU thread.
    static void unsetLwrrent( LWresult* returnResult = nullptr );

    // Set resource limits.
    static void setLimit( LWlimit limit, size_t value, LWresult* returnResult = nullptr );

    // Sets the shared memory configuration for the current context.
    static void setSharedMemConfig( LWsharedconfig config, LWresult* returnResult = nullptr );

    // Block for a context's tasks to complete.
    static void synchronize( LWresult* returnResult = nullptr );

    // Get the state of the primary context.
    static void devicePrimaryCtxGetState( const Device& dev, unsigned int* flags, int* active, LWresult* returnResult = nullptr );

    // Release the primary context on the GPU.
    static void devicePrimaryCtxRelease( const Device& dev, LWresult* returnResult = nullptr );

    // Destroy all allocations and reset all state on the primary context.
    static void devicePrimaryCtxReset( const Device& dev, LWresult* returnResult = nullptr );

    // Retain the primary context on the GPU.
    static Context devicePrimaryCtxRetain( const Device& dev, LWresult* returnResult = nullptr );

    // Set flags for the primary context.
    static void devicePrimaryCtxSetFlags( const Device& dev, unsigned int flags, LWresult* returnResult = nullptr );

    // Disables direct access to memory allocations in a peer context and unregisters any registered allocations.
    void disablePeerAccess( const Context& peerContext, LWresult* returnResult = nullptr ) const;

    // Enables direct access to memory allocations in a peer context.
    void enablePeerAccess( const Context& peerContext, unsigned int Flags, LWresult* returnResult = nullptr ) const;

  private:
    explicit Context( LWcontext context );

    LWcontext m_context;
};
}  // namespace lwca
}  // namespace optix
