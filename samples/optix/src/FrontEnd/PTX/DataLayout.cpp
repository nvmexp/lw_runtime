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

#include <FrontEnd/PTX/DataLayout.h>

#include <llvm/ADT/Twine.h>
#include <llvm/IR/DataLayout.h>

#include <iostream>

/// Return a new LLVM datalayout struct compatible with this
/// platform and LWCA that matches the specified pointer size.
llvm::DataLayout optix::createDataLayoutForPointerSize( unsigned int pointerSizeInBits )
{
    const char* pointerSizes;
    if( pointerSizeInBits == 64 )
    {
        pointerSizes = "-p:64:64:64";
    }
    else
    {
        // All pointers are 32-bit
        pointerSizes = "-p:32:32:32";
    }
    // Bool has 8-bit alignment
    const char* primitiveSizes = "-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-i128:128:128-f32:32:32-f64:64:64";
    // 16, 32, 64, and 128-bit vectors.
    const char* vectorSizes = "-v16:16:16-v32:32:32-v64:64:64-v128:128:128";
    // Native sizes of 16, 32, and 64
    const char* nativeSizes = "-n16:32:64";
    std::string dl          = std::string( "e" ) + pointerSizes + primitiveSizes + vectorSizes + nativeSizes;
    return llvm::DataLayout( dl );
}

/// Return a new LLVM datalayout struct compatible with this
/// platform and LWCA that matches the current process.
llvm::DataLayout optix::createDataLayoutForLwrrentProcess()
{
    return createDataLayoutForPointerSize( sizeof( void* ) * 8 );
}
