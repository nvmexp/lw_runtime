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

#include <cstddef>
#include <vector_types.h>


namespace optix {

class Context;

class PrintManager
{
  public:
    PrintManager( Context* context );
    ~PrintManager();

    // Settings interface.
    void setPrintEnabled( bool enabled );
    bool getPrintEnabled() const;
    void setPrintBufferSize( size_t bufsize );
    size_t getPrintBufferSize() const;
    void setPrintLaunchIndex( int x, int y, int z );
    int3 getPrintLaunchIndex() const;

  private:
    // The context we're attached to.
    Context* m_context;

    // Settings.
    bool   m_printEnabled    = false;
    size_t m_printBufferSize = 0;  // The default size is 0, i.e. we use the default buffer size used by the LWCA context.
    int3   m_printLaunchIndex;
};
}
