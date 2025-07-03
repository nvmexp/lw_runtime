// Copyright (c) 2016 Google Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef SOURCE_SPIRV_TARGET_ELW_H_
#define SOURCE_SPIRV_TARGET_ELW_H_

#include <string>

#include "spirv-tools/libspirv.h"

// Returns true if |elw| is a VULKAN environment, false otherwise.
bool spvIsVulkanElw(spv_target_elw elw);

// Returns true if |elw| is an OPENCL environment, false otherwise.
bool spvIsOpenCLElw(spv_target_elw elw);

// Returns true if |elw| is an WEBGPU environment, false otherwise.
bool spvIsWebGPUElw(spv_target_elw elw);

// Returns true if |elw| is an OPENGL environment, false otherwise.
bool spvIsOpenGLElw(spv_target_elw elw);

// Returns true if |elw| is a VULKAN or WEBGPU environment, false otherwise.
bool spvIsVulkanOrWebGPUElw(spv_target_elw elw);

// Returns the version number for the given SPIR-V target environment.
uint32_t spvVersionForTargetElw(spv_target_elw elw);

// Returns a string to use in logging messages that indicates the class of
// environment, i.e. "Vulkan", "WebGPU", "OpenCL", etc.
std::string spvLogStringForElw(spv_target_elw elw);

// Returns a formatted list of all SPIR-V target environment names that
// can be parsed by spvParseTargetElw.
// |pad| is the number of space characters that the begining of each line
//       except the first one will be padded with.
// |wrap| is the max length of lines the user desires. Word-wrapping will
//        occur to satisfy this limit.
std::string spvTargetElwList(const int pad, const int wrap);

#endif  // SOURCE_SPIRV_TARGET_ELW_H_
