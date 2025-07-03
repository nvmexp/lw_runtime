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

#include <corelib/compiler/LLVMSupportTypes.h>
#include <corelib/misc/Concepts.h>

#include <string>

namespace llvm {
class DataLayout;
}

namespace optix {
// saveSet includes all values that should be stored to reconstruct the values in the restoreSet
// rematSet includes all values that should be stored or rematerialized to reconstruct the values in the restoreSet
class SaveSetOptimizer : private corelib::AbstractInterface
{
  public:
    static std::unique_ptr<SaveSetOptimizer> create( const std::string& kind, const llvm::DataLayout& DL );

    virtual void run( const corelib::InstSetVector& restoreSet, const std::string& idString ) = 0;
    const corelib::InstSetVector& getSaveSet() const;
    const corelib::InstSetVector& getRematSet() const;

    void disallowPrimitiveIndexRemat();

  protected:
    SaveSetOptimizer( const llvm::DataLayout& DL );

    bool                    m_valid;
    corelib::InstSetVector  m_saveSet;
    corelib::InstSetVector  m_rematSet;
    bool                    m_allowPrimitiveIndexRemat;
    const llvm::DataLayout& m_DL;

  private:
};
}
