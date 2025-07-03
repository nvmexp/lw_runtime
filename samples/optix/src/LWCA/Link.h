// Copyright (c) 2017, LWPU CORPORATION.
// TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED
// *AS IS* AND LWPU AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS
// OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY
// AND FITNESS FOR A PARTICULAR PURPOSE.  IN NO MODULE SHALL LWPU OR ITS SUPPLIERS
// BE LIABLE FOR ANY SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES
// WHATSOEVER (INCLUDING, WITHOUT LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS,
// BUSINESS INTERRUPTION, LOSS OF BUSINESS INFORMATION, OR ANY OTHER PELWNIARY LOSS)
// ARISING OUT OF THE USE OF OR INABILITY TO USE THIS SOFTWARE, EVEN IF LWPU HAS
// BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES

#pragma once

#include <LWCA/Module.h>

#include <string>

namespace optix {
namespace lwca {

class Link
{
  public:
    Link();

    // Get the low-level linkState object
    LWlinkState       get();
    const LWlinkState get() const;

    // Creates a pending JIT linker invocation.
    static Link create( unsigned int numOptions, LWjit_option* options, void** optiolwalues, LWresult* returnResult = nullptr );

    // Add an input to a pending linker invocation.
    void addData( LWjitInputType     type,
                  void*              data,
                  size_t             size,
                  const std::string& name,
                  unsigned int       numOptions,
                  LWjit_option*      options,
                  void**             optiolwalues,
                  LWresult*          returnResult = nullptr );

    // Add a file input to a pending linker invocation.
    void addFile( LWjitInputType     type,
                  const std::string& path,
                  unsigned int       numOptions,
                  LWjit_option*      options,
                  void**             optiolwalues,
                  LWresult*          returnResult = nullptr );

    // Complete a pending linker invocation.
    void complete( void** lwbinOut, size_t* sizeOut, LWresult* returnResult = nullptr );

    // Complete a pending linker invocation - alternate version that returns a loaded module
    Module complete( LWresult* returnResult = nullptr );

    // Destroys state for a JIT linker invocation.
    void destroy( LWresult* returnResult = nullptr );

  private:
    explicit Link( LWlinkState linkState );

    LWlinkState m_linkState;
};

}  // namespace lwca
}  // namespace optix
