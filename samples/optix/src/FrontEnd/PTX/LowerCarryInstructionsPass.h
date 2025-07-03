// Copyright (c) 2020, LWPU CORPORATION.
// TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED
// *AS IS* AND LWPU AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS
// OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY
// AND FITNESS FOR A PARTICULAR PURPOSE.  IN NO EVENT SHALL LWPU OR ITS SUPPLIERS
// BE LIABLE FOR ANY SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES
// WHATSOEVER (INCLUDING, WITHOUT LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS,
// BUSINESS INTERRUPTION, LOSS OF BUSINESS INFORMATION, OR ANY OTHER PELWNIARY LOSS)
// ARISING OUT OF THE USE OF OR INABILITY TO USE THIS SOFTWARE, EVEN IF LWPU HAS
// BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES

/*
 * This pass lowers carry arithmetic instructions (which include add.cc, addc,
 * sub.cc, subc, mad.cc, and madc) to their LLVM equivalents. According to the
 * PTX ISA: 
 *
 *     "No other instructions access the condition code, and there is no
 *     support for setting, clearing, or testing the condition code. The
 *     condition code register is not preserved across calls and is mainly
 *     intended for use in straight-line code sequences for computing
 *     extended-precision integer addition, subtraction, and multiplication." 
 *
 * This means that we can simply create an alloca at the beginning of each
 * function that uses these instructions, then load/store the carry bit when
 * necessary. Mem2reg passes can eliminate those loads and stores later.
 */

#pragma once

namespace llvm {
class FunctionPass;
class StringRef;
}

namespace optix {

// Return true if this pass handles the given optix PTX intrinsic
bool lowerCarryInstructionsPassHandlesIntrinsic( llvm::StringRef intrinsicName );

llvm::FunctionPass* createLowerCarryInstructionsPass();

}  // namespace optix
