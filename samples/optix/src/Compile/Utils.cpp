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

#include <Compile/Utils.h>
#include <corelib/misc/String.h>
#include <prodlib/system/Logger.h>

#include <llvm/IR/DataLayout.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/raw_ostream.h>

using namespace optix;
using namespace corelib;
using namespace llvm;

void optix::printSet( Module* module, const InstSetVector& vals, const char* msg )
{
#if defined( OPTIX_ENABLE_LOGGING )
    std::string        outS;
    raw_string_ostream out( outS );
    if( msg )
        out << msg << " --------------------\n ";

    uint64_t totalBytes = 0;
    if( !vals.empty() )
    {
        DataLayout DL( module );
        for( const Instruction* inst : vals )
        {
            uint64_t size = DL.getTypeAllocSize( inst->getType() );
            out << stringf( "%3zub: ", size ) << *inst << '\n';
            totalBytes += size;
        }
    }
    out << "Count:" << vals.size() << "  Bytes:" << totalBytes << "\n\n";
    lprint << out.str();
#endif
}
