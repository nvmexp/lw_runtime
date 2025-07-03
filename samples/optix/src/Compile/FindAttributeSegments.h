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

#include <vector>


namespace llvm {
class CallInst;
class Function;
class Pass;
}

namespace optix {
struct AttributeSegment
{
    llvm::CallInst* rtPI;
    llvm::CallInst* rtRI;
    AttributeSegment( llvm::CallInst* rtPI, llvm::CallInst* rtRI )
        : rtPI( rtPI )
        , rtRI( rtRI )
    {
    }
};

typedef std::vector<AttributeSegment> AttributeSegmentVector;

AttributeSegmentVector findAttributeSegments( llvm::Function* caller, llvm::Function* rtPI, llvm::Function* rtRI );
AttributeSegmentVector findAttributeSegments( llvm::Function*               caller,
                                              std::vector<llvm::CallInst*>& rtPICalls,
                                              std::vector<llvm::CallInst*>& rtRICalls );
}
