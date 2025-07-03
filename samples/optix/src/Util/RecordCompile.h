// Copyright (c) 2018, LWPU CORPORATION.
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

#include <string>

// Tell rapidjson to let us use std::string
#define RAPIDJSON_HAS_STDSTRING 1

struct RtcCompileOptions;

namespace recordcompile {

// Record the specified compile call.
void recordCompileCall( const std::string& name, const RtcCompileOptions& compileOptions, const std::string& bitcode );

// Load the specified compile call.
void loadCompileCall( const std::string& jsonPath,
                      const std::string& binPath,
                      std::string&       moduleName,
                      RtcCompileOptions& compileOptions,
                      std::string&       bitcode );

}  // namespace optix
