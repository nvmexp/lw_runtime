// Copyright (c) 2016 The Khronos Group Inc.
// 
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and/or associated documentation files (the
// "Materials"), to deal in the Materials without restriction, including
// without limitation the rights to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Materials, and to
// permit persons to whom the Materials are furnished to do so, subject to
// the following conditions:
// 
// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Materials.
// 
// MODIFICATIONS TO THIS FILE MAY MEAN IT NO LONGER ACLWRATELY REFLECTS
// KHRONOS STANDARDS. THE UNMODIFIED, NORMATIVE VERSIONS OF KHRONOS
// SPECIFICATIONS AND HEADER INFORMATION ARE LOCATED AT
//    https://www.khronos.org/registry/
// 
// THE MATERIALS ARE PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
// IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
// CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
// TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
// MATERIALS OR THE USE OR OTHER DEALINGS IN THE MATERIALS.

// Use the SPIR-V 1.1 core instruction set, but with 1.0 versions
// of the GLSL and OpenCL extended instruction sets.
#include <spirv/1.0/GLSL.std.450.h>
#include <spirv/1.0/OpenCL.std.h>
#include <spirv/1.1/spirv.hpp>

namespace {

const GLSLstd450 kSin = GLSLstd450Sin;
const OpenCLLIB::Entrypoints kNative_cos = OpenCLLIB::Native_cos;
const spv::Op kNop = spv::OpNop;

// This instruction is new in SPIR-V 1.1.
const spv::Op kNamedBarrierInit = spv::OpNamedBarrierInitialize;

}  // anonymous namespace
