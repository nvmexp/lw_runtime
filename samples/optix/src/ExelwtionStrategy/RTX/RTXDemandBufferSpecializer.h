//
// Copyright (c) 2019, LWPU CORPORATION.
//
// TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED
// *AS IS* AND LWPU AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS
// OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY
// AND FITNESS FOR A PARTICULAR PURPOSE.  IN NO EVENT SHALL LWPU OR ITS SUPPLIERS
// BE LIABLE FOR ANY SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES
// WHATSOEVER (INCLUDING, WITHOUT LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS,
// BUSINESS INTERRUPTION, LOSS OF BUSINESS INFORMATION, OR ANY OTHER PELWNIARY LOSS)
// ARISING OUT OF THE USE OF OR INABILITY TO USE THIS SOFTWARE, EVEN IF LWPU HAS
// BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES
//

#pragma once

#include <vector>

namespace llvm {
class CallInst;
class Function;
class Module;
}  // namespace llvm

namespace optix {

// Transform
//
//   declare i32 @rtxiLoadOrRequestBufferElement.foo_ptx0xdeadbeef.other_stuff(i32 addrspace(1)*, i32, i32, i64, i64, i64)
//
//   define i32 @parent_fn(i32 addrspace(1)*) {
//   entry:
//     %1 = call i32 @rtxiLoadOrRequestBufferElement.foo_ptx0xdeadbeef.other_stuff(i32 addrspace(1)* null, i32 undef, i32 4, i64 undef, i64 undef, i64 undef)
//     ret i32 %1
//   }
//
// to
//
//   declare i8* @RTX_requestBufferElement2(i32 addrspace(1)*, i32, i32, i64, i64)
//
//   define i32 @parent_fn(i32 addrspace(1)*) {
//   entry:
//     %element = call i8* @RTX_requestBufferElement2(i32 addrspace(1)* null, i32 undef, i32 4, i64 undef, i64 undef)
//     %1 = addrspacecast i8* %element to [4 x i8] addrspace(1)*
//     %2 = icmp ne [4 x i8] addrspace(1)* %1, null
//     br i1 %2, label %3, label %5
//
//   ; <label>:3                                       ; preds = %entry
//     %4 = load [4 x i8] addrspace(1)* %1
//     store [4 x i8] %4, [4 x i8] addrspace(1)* undef
//     br label %6
//
//   ; <label>:5                                       ; preds = %entry
//     store [4 x i8] undef, [4 x i8] addrspace(1)* undef
//     br label %6
//
//   ; <label>:6                                       ; preds = %5, %3
//     %7 = zext i1 %2 to i32
//     ret i32 %7
//   }
//
// for a 2D buffer element access.
//
class RTXDemandBufferSpecializer
{
  public:
    void runOnFunction( llvm::Function* function );

  private:
    void runOnCall( llvm::Module* module, llvm::CallInst* callInst, std::vector<llvm::CallInst*>& toDelete );
};

}  // namespace optix
