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

#include <Compile/FindAttributeSegments.h>

#include <Util/ContainerAlgorithm.h>

#include <corelib/compiler/ControlDependenceGraph.h>
#include <corelib/compiler/LLVMUtil.h>

#include <prodlib/exceptions/Assert.h>
#include <prodlib/exceptions/CompileError.h>

#include <llvm/IR/Dominators.h>
#include <llvm/IR/InstIterator.h>
#include <llvm/IR/Instructions.h>
#include <llvm/Pass.h>

using namespace llvm;
using namespace optix;
using namespace corelib;
using namespace prodlib;

// -----------------------------------------------------------------------------
AttributeSegmentVector optix::findAttributeSegments( Function* caller, Function* rtPI, Function* rtRI )
{
    RT_ASSERT( caller != nullptr );
    RT_ASSERT( rtPI != nullptr );
    RT_ASSERT( rtRI != nullptr );

    std::vector<CallInst*> rtPICalls = getCallsToFunction( rtPI, caller );
    std::vector<CallInst*> rtRICalls = getCallsToFunction( rtRI, caller );

    return findAttributeSegments( caller, rtPICalls, rtRICalls );
}

// -----------------------------------------------------------------------------
// The detection of attribute segments is based on the control dependce between program blocks.
// A block that contains a call to rtRI must be control dependent on a block that contains a call to rtPI.
// Moreover, if rtPI returns true, then we *must* call rtRI.
// This means that programs like this:
// if ( rtPI )
// {
//   if ( condition )
//   {
//     rtRI;
//   }
// }
// are not legal.
// This means that RI must be *directly* control dependent on the block that calls rtPI.
// So the detection works as follows:
// Given a call to rtRI we query the Control Dependence Graph to get the immediate controller of the
// basic block, this *has to* contain a call to rtPI, if not we have found an error.
// For more examples of illegal behaviors look at TestIntersection.
// If you find bugs in the current implementation please add a reproducer to the tests.
AttributeSegmentVector optix::findAttributeSegments( Function* caller, std::vector<CallInst*>& rtPICalls, std::vector<CallInst*>& rtRICalls )
{
    if( rtPICalls.size() != rtRICalls.size() )
    {
        throw CompileError( RT_EXCEPTION_INFO,
                            "All calls to rtPotentialIntersection must be paired with a call to rtReportIntersection" );
    }

    ControlDependenceGraph cdg;
    cdg.runOnFunction( *caller );
    if( !cdg.isSuccessfullyConstructed() )
    {
        // This message assumes that the CDG construction only fails in (some) cases of endless loops.
        throw CompileError( RT_EXCEPTION_INFO, "Endless loop not supported in this context" );
    }


    AttributeSegmentVector attributeSegments;

    for( CallInst* riCall : rtRICalls )
    {
        BasicBlock* riBlock = riCall->getParent();
        auto        parents = cdg.getDirectParents( riBlock );
        if( parents.size() != 1 )
        {
            throw CompileError( RT_EXCEPTION_INFO,
                                "Call to rtReportIntersection is not directly controlled by a single call to "
                                "rtPotentialIntersection" );
        }

        BasicBlock* piBlock = parents.begin()->first;
        auto        piCall =
            algorithm::find_if( rtPICalls, [piBlock]( CallInst* callInst ) { return callInst->getParent() == piBlock; } );
        if( piCall == rtPICalls.end() )
        {
            throw CompileError( RT_EXCEPTION_INFO,
                                "Call to rtReportIntersection is not directly controlled by a single call to "
                                "rtPotentialIntersection" );
        }

        // We have found a new rtPI-rtRI pair.
        attributeSegments.push_back( AttributeSegment( *piCall, riCall ) );
    }

    return attributeSegments;
}
