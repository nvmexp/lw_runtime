//
// Copyright (C) 2014-2015 LunarG, Inc.
// Modifications Copyright (C) 2020 Advanced Micro Devices, Inc. All rights reserved.
//
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//
//    Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//
//    Redistributions in binary form must reproduce the above
//    copyright notice, this list of conditions and the following
//    disclaimer in the documentation and/or other materials provided
//    with the distribution.
//
//    Neither the name of 3Dlabs Inc. Ltd. nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
// FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
// COPYRIGHT HOLDERS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
// INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
// BUT NOT LIMITED TO, PROLWREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
// LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
// ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

//
// 1) Programmatically fill in instruction/operand information.
//    This can be used for disassembly, printing documentation, etc.
//
// 2) Print documentation from this parameterization.
//

#include "doc.h"

#include <cstdio>
#include <cstring>
#include <algorithm>

namespace spv {
    extern "C" {
        // Include C-based headers that don't have a namespace
        #include "GLSL.ext.KHR.h"
        #include "GLSL.ext.EXT.h"
        #include "GLSL.ext.AMD.h"
        #include "GLSL.ext.LW.h"
    }
}

namespace spv {

//
// Whole set of functions that translate enumerants to their text strings for
// the specification (or their sanitized versions for auto-generating the
// spirv headers.
//
// Also, for masks the ceilings are declared next to these, to help keep them in sync.
// Ceilings should be
//  - one more than the maximum value an enumerant takes on, for non-mask enumerants
//    (for non-sparse enums, this is the number of enumerants)
//  - the number of bits consumed by the set of masks
//    (for non-sparse mask enums, this is the number of enumerants)
//

const char* SourceString(int source)
{
    switch (source) {
    case 0:  return "Unknown";
    case 1:  return "ESSL";
    case 2:  return "GLSL";
    case 3:  return "OpenCL_C";
    case 4:  return "OpenCL_CPP";
    case 5:  return "HLSL";

    default: return "Bad";
    }
}

const char* ExelwtionModelString(int model)
{
    switch (model) {
    case 0:  return "Vertex";
    case 1:  return "TessellationControl";
    case 2:  return "TessellationEvaluation";
    case 3:  return "Geometry";
    case 4:  return "Fragment";
    case 5:  return "GLCompute";
    case 6:  return "Kernel";
    case ExelwtionModelTaskLW: return "TaskLW";
    case ExelwtionModelMeshLW: return "MeshLW";

    default: return "Bad";

    case ExelwtionModelRayGenerationKHR: return "RayGenerationKHR";
    case ExelwtionModelIntersectionKHR:  return "IntersectionKHR";
    case ExelwtionModelAnyHitKHR:        return "AnyHitKHR";
    case ExelwtionModelClosestHitKHR:    return "ClosestHitKHR";
    case ExelwtionModelMissKHR:          return "MissKHR";
    case ExelwtionModelCallableKHR:      return "CallableKHR";
    }
}

const char* AddressingString(int addr)
{
    switch (addr) {
    case 0:  return "Logical";
    case 1:  return "Physical32";
    case 2:  return "Physical64";

    case AddressingModelPhysicalStorageBuffer64EXT: return "PhysicalStorageBuffer64EXT";

    default: return "Bad";
    }
}

const char* MemoryString(int mem)
{
    switch (mem) {
    case MemoryModelSimple:     return "Simple";
    case MemoryModelGLSL450:    return "GLSL450";
    case MemoryModelOpenCL:     return "OpenCL";
    case MemoryModelVulkanKHR:  return "VulkanKHR";

    default: return "Bad";
    }
}

const int ExelwtionModeCeiling = 33;

const char* ExelwtionModeString(int mode)
{
    switch (mode) {
    case 0:  return "Ilwocations";
    case 1:  return "SpacingEqual";
    case 2:  return "SpacingFractionalEven";
    case 3:  return "SpacingFractionalOdd";
    case 4:  return "VertexOrderCw";
    case 5:  return "VertexOrderCcw";
    case 6:  return "PixelCenterInteger";
    case 7:  return "OriginUpperLeft";
    case 8:  return "OriginLowerLeft";
    case 9:  return "EarlyFragmentTests";
    case 10: return "PointMode";
    case 11: return "Xfb";
    case 12: return "DepthReplacing";
    case 13: return "Bad";
    case 14: return "DepthGreater";
    case 15: return "DepthLess";
    case 16: return "DepthUnchanged";
    case 17: return "LocalSize";
    case 18: return "LocalSizeHint";
    case 19: return "InputPoints";
    case 20: return "InputLines";
    case 21: return "InputLinesAdjacency";
    case 22: return "Triangles";
    case 23: return "InputTrianglesAdjacency";
    case 24: return "Quads";
    case 25: return "Isolines";
    case 26: return "OutputVertices";
    case 27: return "OutputPoints";
    case 28: return "OutputLineStrip";
    case 29: return "OutputTriangleStrip";
    case 30: return "VecTypeHint";
    case 31: return "ContractionOff";
    case 32: return "Bad";

    case 4446:  return "PostDepthCoverage";

    case ExelwtionModeOutputLinesLW:            return "OutputLinesLW";
    case ExelwtionModeOutputPrimitivesLW:       return "OutputPrimitivesLW";
    case ExelwtionModeOutputTrianglesLW:        return "OutputTrianglesLW";
    case ExelwtionModeDerivativeGroupQuadsLW:   return "DerivativeGroupQuadsLW";
    case ExelwtionModeDerivativeGroupLinearLW:  return "DerivativeGroupLinearLW";

    case ExelwtionModePixelInterlockOrderedEXT:         return "PixelInterlockOrderedEXT";
    case ExelwtionModePixelInterlockUnorderedEXT:       return "PixelInterlockUnorderedEXT";
    case ExelwtionModeSampleInterlockOrderedEXT:        return "SampleInterlockOrderedEXT";
    case ExelwtionModeSampleInterlockUnorderedEXT:      return "SampleInterlockUnorderedEXT";
    case ExelwtionModeShadingRateInterlockOrderedEXT:   return "ShadingRateInterlockOrderedEXT";
    case ExelwtionModeShadingRateInterlockUnorderedEXT: return "ShadingRateInterlockUnorderedEXT";

    case ExelwtionModeCeiling:
    default: return "Bad";
    }
}

const char* StorageClassString(int StorageClass)
{
    switch (StorageClass) {
    case 0:  return "UniformConstant";
    case 1:  return "Input";
    case 2:  return "Uniform";
    case 3:  return "Output";
    case 4:  return "Workgroup";
    case 5:  return "CrossWorkgroup";
    case 6:  return "Private";
    case 7:  return "Function";
    case 8:  return "Generic";
    case 9:  return "PushConstant";
    case 10: return "AtomicCounter";
    case 11: return "Image";
    case 12: return "StorageBuffer";

    case StorageClassRayPayloadKHR:            return "RayPayloadKHR";
    case StorageClassHitAttributeKHR:          return "HitAttributeKHR";
    case StorageClassIncomingRayPayloadKHR:    return "IncomingRayPayloadKHR";
    case StorageClassShaderRecordBufferKHR:    return "ShaderRecordBufferKHR";
    case StorageClassCallableDataKHR:          return "CallableDataKHR";
    case StorageClassIncomingCallableDataKHR:  return "IncomingCallableDataKHR";

    case StorageClassPhysicalStorageBufferEXT: return "PhysicalStorageBufferEXT";

    default: return "Bad";
    }
}

const int DecorationCeiling = 45;

const char* DecorationString(int decoration)
{
    switch (decoration) {
    case 0:  return "RelaxedPrecision";
    case 1:  return "SpecId";
    case 2:  return "Block";
    case 3:  return "BufferBlock";
    case 4:  return "RowMajor";
    case 5:  return "ColMajor";
    case 6:  return "ArrayStride";
    case 7:  return "MatrixStride";
    case 8:  return "GLSLShared";
    case 9:  return "GLSLPacked";
    case 10: return "CPacked";
    case 11: return "BuiltIn";
    case 12: return "Bad";
    case 13: return "NoPerspective";
    case 14: return "Flat";
    case 15: return "Patch";
    case 16: return "Centroid";
    case 17: return "Sample";
    case 18: return "Ilwariant";
    case 19: return "Restrict";
    case 20: return "Aliased";
    case 21: return "Volatile";
    case 22: return "Constant";
    case 23: return "Coherent";
    case 24: return "NonWritable";
    case 25: return "NonReadable";
    case 26: return "Uniform";
    case 27: return "Bad";
    case 28: return "SaturatedColwersion";
    case 29: return "Stream";
    case 30: return "Location";
    case 31: return "Component";
    case 32: return "Index";
    case 33: return "Binding";
    case 34: return "DescriptorSet";
    case 35: return "Offset";
    case 36: return "XfbBuffer";
    case 37: return "XfbStride";
    case 38: return "FuncParamAttr";
    case 39: return "FP Rounding Mode";
    case 40: return "FP Fast Math Mode";
    case 41: return "Linkage Attributes";
    case 42: return "NoContraction";
    case 43: return "InputAttachmentIndex";
    case 44: return "Alignment";

    case DecorationCeiling:
    default:  return "Bad";

    case DecorationExplicitInterpAMD: return "ExplicitInterpAMD";
    case DecorationOverrideCoverageLW:          return "OverrideCoverageLW";
    case DecorationPassthroughLW:               return "PassthroughLW";
    case DecoratiolwiewportRelativeLW:          return "ViewportRelativeLW";
    case DecorationSecondaryViewportRelativeLW: return "SecondaryViewportRelativeLW";
    case DecorationPerPrimitiveLW:              return "PerPrimitiveLW";
    case DecorationPerViewLW:                   return "PerViewLW";
    case DecorationPerTaskLW:                   return "PerTaskLW";
    case DecorationPerVertexLW:                 return "PerVertexLW";

    case DecorationNonUniformEXT:           return "DecorationNonUniformEXT";
    case DecorationHlslCounterBufferGOOGLE: return "DecorationHlslCounterBufferGOOGLE";
    case DecorationHlslSemanticGOOGLE:      return "DecorationHlslSemanticGOOGLE";
    case DecorationRestrictPointerEXT:      return "DecorationRestrictPointerEXT";
    case DecorationAliasedPointerEXT:       return "DecorationAliasedPointerEXT";
    }
}

const char* BuiltInString(int builtIn)
{
    switch (builtIn) {
    case 0:  return "Position";
    case 1:  return "PointSize";
    case 2:  return "Bad";
    case 3:  return "ClipDistance";
    case 4:  return "LwllDistance";
    case 5:  return "VertexId";
    case 6:  return "InstanceId";
    case 7:  return "PrimitiveId";
    case 8:  return "IlwocationId";
    case 9:  return "Layer";
    case 10: return "ViewportIndex";
    case 11: return "TessLevelOuter";
    case 12: return "TessLevelInner";
    case 13: return "TessCoord";
    case 14: return "PatchVertices";
    case 15: return "FragCoord";
    case 16: return "PointCoord";
    case 17: return "FrontFacing";
    case 18: return "SampleId";
    case 19: return "SamplePosition";
    case 20: return "SampleMask";
    case 21: return "Bad";
    case 22: return "FragDepth";
    case 23: return "HelperIlwocation";
    case 24: return "NumWorkgroups";
    case 25: return "WorkgroupSize";
    case 26: return "WorkgroupId";
    case 27: return "LocalIlwocationId";
    case 28: return "GlobalIlwocationId";
    case 29: return "LocalIlwocationIndex";
    case 30: return "WorkDim";
    case 31: return "GlobalSize";
    case 32: return "EnqueuedWorkgroupSize";
    case 33: return "GlobalOffset";
    case 34: return "GlobalLinearId";
    case 35: return "Bad";
    case 36: return "SubgroupSize";
    case 37: return "SubgroupMaxSize";
    case 38: return "NumSubgroups";
    case 39: return "NumEnqueuedSubgroups";
    case 40: return "SubgroupId";
    case 41: return "SubgroupLocalIlwocationId";
    case 42: return "VertexIndex";                 // TBD: put next to VertexId?
    case 43: return "InstanceIndex";               // TBD: put next to InstanceId?

    case 4416: return "SubgroupEqMaskKHR";
    case 4417: return "SubgroupGeMaskKHR";
    case 4418: return "SubgroupGtMaskKHR";
    case 4419: return "SubgroupLeMaskKHR";
    case 4420: return "SubgroupLtMaskKHR";
    case 4438: return "DeviceIndex";
    case 4440: return "ViewIndex";
    case 4424: return "BaseVertex";
    case 4425: return "BaseInstance";
    case 4426: return "DrawIndex";
    case 5014: return "FragStencilRefEXT";

    case 4992: return "BaryCoordNoPerspAMD";
    case 4993: return "BaryCoordNoPerspCentroidAMD";
    case 4994: return "BaryCoordNoPerspSampleAMD";
    case 4995: return "BaryCoordSmoothAMD";
    case 4996: return "BaryCoordSmoothCentroidAMD";
    case 4997: return "BaryCoordSmoothSampleAMD";
    case 4998: return "BaryCoordPullModelAMD";
    case BuiltInLaunchIdKHR:                 return "LaunchIdKHR";
    case BuiltInLaunchSizeKHR:               return "LaunchSizeKHR";
    case BuiltInWorldRayOriginKHR:           return "WorldRayOriginKHR";
    case BuiltInWorldRayDirectionKHR:        return "WorldRayDirectionKHR";
    case BuiltInObjectRayOriginKHR:          return "ObjectRayOriginKHR";
    case BuiltInObjectRayDirectionKHR:       return "ObjectRayDirectionKHR";
    case BuiltInRayTminKHR:                  return "RayTminKHR";
    case BuiltInRayTmaxKHR:                  return "RayTmaxKHR";
    case BuiltInInstanceLwstomIndexKHR:      return "InstanceLwstomIndexKHR";
    case BuiltInRayGeometryIndexKHR:         return "RayGeometryIndexKHR";
    case BuiltInObjectToWorldKHR:            return "ObjectToWorldKHR";
    case BuiltInWorldToObjectKHR:            return "WorldToObjectKHR";
    case BuiltInHitTKHR:                     return "HitTKHR";
    case BuiltInHitKindKHR:                  return "HitKindKHR";
    case BuiltInIncomingRayFlagsKHR:         return "IncomingRayFlagsKHR";
    case BuiltIlwiewportMaskLW:              return "ViewportMaskLW";
    case BuiltInSecondaryPositionLW:         return "SecondaryPositionLW";
    case BuiltInSecondaryViewportMaskLW:     return "SecondaryViewportMaskLW";
    case BuiltInPositionPerViewLW:           return "PositionPerViewLW";
    case BuiltIlwiewportMaskPerViewLW:       return "ViewportMaskPerViewLW";
//    case BuiltInFragmentSizeLW:             return "FragmentSizeLW";        // superseded by BuiltInFragSizeEXT
//    case BuiltInIlwocationsPerPixelLW:      return "IlwocationsPerPixelLW"; // superseded by BuiltInFragIlwocationCountEXT
    case BuiltInBaryCoordLW:                 return "BaryCoordLW";
    case BuiltInBaryCoordNoPerspLW:          return "BaryCoordNoPerspLW";

    case BuiltInFragSizeEXT:                 return "FragSizeEXT";
    case BuiltInFragIlwocationCountEXT:      return "FragIlwocationCountEXT";

    case 5264: return "FullyCoveredEXT";

    case BuiltInTaskCountLW:           return "TaskCountLW";
    case BuiltInPrimitiveCountLW:      return "PrimitiveCountLW";
    case BuiltInPrimitiveIndicesLW:    return "PrimitiveIndicesLW";
    case BuiltInClipDistancePerViewLW: return "ClipDistancePerViewLW";
    case BuiltInLwllDistancePerViewLW: return "LwllDistancePerViewLW";
    case BuiltInLayerPerViewLW:        return "LayerPerViewLW";
    case BuiltInMeshViewCountLW:       return "MeshViewCountLW";
    case BuiltInMeshViewIndicesLW:     return "MeshViewIndicesLW";
    case BuiltInWarpsPerSMLW:           return "WarpsPerSMLW";
    case BuiltInSMCountLW:              return "SMCountLW";
    case BuiltInWarpIDLW:               return "WarpIDLW";
    case BuiltInSMIDLW:                 return "SMIDLW";

    default: return "Bad";
    }
}

const char* DimensionString(int dim)
{
    switch (dim) {
    case 0:  return "1D";
    case 1:  return "2D";
    case 2:  return "3D";
    case 3:  return "Lwbe";
    case 4:  return "Rect";
    case 5:  return "Buffer";
    case 6:  return "SubpassData";

    default: return "Bad";
    }
}

const char* SamplerAddressingModeString(int mode)
{
    switch (mode) {
    case 0:  return "None";
    case 1:  return "ClampToEdge";
    case 2:  return "Clamp";
    case 3:  return "Repeat";
    case 4:  return "RepeatMirrored";

    default: return "Bad";
    }
}

const char* SamplerFilterModeString(int mode)
{
    switch (mode) {
    case 0: return "Nearest";
    case 1: return "Linear";

    default: return "Bad";
    }
}

const char* ImageFormatString(int format)
{
    switch (format) {
    case  0: return "Unknown";

    // ES/Desktop float
    case  1: return "Rgba32f";
    case  2: return "Rgba16f";
    case  3: return "R32f";
    case  4: return "Rgba8";
    case  5: return "Rgba8Snorm";

    // Desktop float
    case  6: return "Rg32f";
    case  7: return "Rg16f";
    case  8: return "R11fG11fB10f";
    case  9: return "R16f";
    case 10: return "Rgba16";
    case 11: return "Rgb10A2";
    case 12: return "Rg16";
    case 13: return "Rg8";
    case 14: return "R16";
    case 15: return "R8";
    case 16: return "Rgba16Snorm";
    case 17: return "Rg16Snorm";
    case 18: return "Rg8Snorm";
    case 19: return "R16Snorm";
    case 20: return "R8Snorm";

    // ES/Desktop int
    case 21: return "Rgba32i";
    case 22: return "Rgba16i";
    case 23: return "Rgba8i";
    case 24: return "R32i";

    // Desktop int
    case 25: return "Rg32i";
    case 26: return "Rg16i";
    case 27: return "Rg8i";
    case 28: return "R16i";
    case 29: return "R8i";

    // ES/Desktop uint
    case 30: return "Rgba32ui";
    case 31: return "Rgba16ui";
    case 32: return "Rgba8ui";
    case 33: return "R32ui";

    // Desktop uint
    case 34: return "Rgb10a2ui";
    case 35: return "Rg32ui";
    case 36: return "Rg16ui";
    case 37: return "Rg8ui";
    case 38: return "R16ui";
    case 39: return "R8ui";

    default:
        return "Bad";
    }
}

const char* ImageChannelOrderString(int format)
{
    switch (format) {
    case 0:  return "R";
    case 1:  return "A";
    case 2:  return "RG";
    case 3:  return "RA";
    case 4:  return "RGB";
    case 5:  return "RGBA";
    case 6:  return "BGRA";
    case 7:  return "ARGB";
    case 8:  return "Intensity";
    case 9:  return "Luminance";
    case 10: return "Rx";
    case 11: return "RGx";
    case 12: return "RGBx";
    case 13: return "Depth";
    case 14: return "DepthStencil";
    case 15: return "sRGB";
    case 16: return "sRGBx";
    case 17: return "sRGBA";
    case 18: return "sBGRA";

    default: 
        return "Bad";
    }
}

const char* ImageChannelDataTypeString(int type)
{
    switch (type)
    {
    case 0: return "SnormInt8";
    case 1: return "SnormInt16";
    case 2: return "UnormInt8";
    case 3: return "UnormInt16";
    case 4: return "UnormShort565";
    case 5: return "UnormShort555";
    case 6: return "UnormInt101010";
    case 7: return "SignedInt8";
    case 8: return "SignedInt16";
    case 9: return "SignedInt32";
    case 10: return "UnsignedInt8";
    case 11: return "UnsignedInt16";
    case 12: return "UnsignedInt32";
    case 13: return "HalfFloat";
    case 14: return "Float";
    case 15: return "UnormInt24";
    case 16: return "UnormInt101010_2";

    default:
        return "Bad";
    }
}

const int ImageOperandsCeiling = 14;

const char* ImageOperandsString(int format)
{
    switch (format) {
    case ImageOperandsBiasShift:                    return "Bias";
    case ImageOperandsLodShift:                     return "Lod";
    case ImageOperandsGradShift:                    return "Grad";
    case ImageOperandsConstOffsetShift:             return "ConstOffset";
    case ImageOperandsOffsetShift:                  return "Offset";
    case ImageOperandsConstOffsetsShift:            return "ConstOffsets";
    case ImageOperandsSampleShift:                  return "Sample";
    case ImageOperandsMinLodShift:                  return "MinLod";
    case ImageOperandsMakeTexelAvailableKHRShift:   return "MakeTexelAvailableKHR";
    case ImageOperandsMakeTexelVisibleKHRShift:     return "MakeTexelVisibleKHR";
    case ImageOperandsNonPrivateTexelKHRShift:      return "NonPrivateTexelKHR";
    case ImageOperandsVolatileTexelKHRShift:        return "VolatileTexelKHR";
    case ImageOperandsSignExtendShift:              return "SignExtend";
    case ImageOperandsZeroExtendShift:              return "ZeroExtend";

    case ImageOperandsCeiling:
    default:
        return "Bad";
    }
}

const char* FPFastMathString(int mode)
{
    switch (mode) {
    case 0: return "NotNaN";
    case 1: return "NotInf";
    case 2: return "NSZ";
    case 3: return "AllowRecip";
    case 4: return "Fast";

    default:     return "Bad";
    }
}

const char* FPRoundingModeString(int mode)
{
    switch (mode) {
    case 0:  return "RTE";
    case 1:  return "RTZ";
    case 2:  return "RTP";
    case 3:  return "RTN";

    default: return "Bad";
    }
}

const char* LinkageTypeString(int type)
{
    switch (type) {
    case 0:  return "Export";
    case 1:  return "Import";

    default: return "Bad";
    }
}

const char* FuncParamAttrString(int attr)
{
    switch (attr) {
    case 0:  return "Zext";
    case 1:  return "Sext";
    case 2:  return "ByVal";
    case 3:  return "Sret";
    case 4:  return "NoAlias";
    case 5:  return "NoCapture";
    case 6:  return "NoWrite";
    case 7:  return "NoReadWrite";

    default: return "Bad";
    }
}

const char* AccessQualifierString(int attr)
{
    switch (attr) {
    case 0:  return "ReadOnly";
    case 1:  return "WriteOnly";
    case 2:  return "ReadWrite";

    default: return "Bad";
    }
}

const int SelectControlCeiling = 2;

const char* SelectControlString(int cont)
{
    switch (cont) {
    case 0:  return "Flatten";
    case 1:  return "DontFlatten";

    case SelectControlCeiling:
    default: return "Bad";
    }
}

const int LoopControlCeiling = LoopControlPartialCountShift + 1;

const char* LoopControlString(int cont)
{
    switch (cont) {
    case LoopControlUnrollShift:             return "Unroll";
    case LoopControlDontUnrollShift:         return "DontUnroll";
    case LoopControlDependencyInfiniteShift: return "DependencyInfinite";
    case LoopControlDependencyLengthShift:   return "DependencyLength";
    case LoopControlMinIterationsShift:      return "MinIterations";
    case LoopControlMaxIterationsShift:      return "MaxIterations";
    case LoopControlIterationMultipleShift:  return "IterationMultiple";
    case LoopControlPeelCountShift:          return "PeelCount";
    case LoopControlPartialCountShift:       return "PartialCount";

    case LoopControlCeiling:
    default: return "Bad";
    }
}

const int FunctionControlCeiling = 4;

const char* FunctionControlString(int cont)
{
    switch (cont) {
    case 0:  return "Inline";
    case 1:  return "DontInline";
    case 2:  return "Pure";
    case 3:  return "Const";

    case FunctionControlCeiling:
    default: return "Bad";
    }
}

const char* MemorySemanticsString(int mem)
{
    // Note: No bits set (None) means "Relaxed"
    switch (mem) {
    case 0: return "Bad"; // Note: this is a placeholder for 'Consume'
    case 1: return "Acquire";
    case 2: return "Release";
    case 3: return "AcquireRelease";
    case 4: return "SequentiallyConsistent";
    case 5: return "Bad"; // Note: reserved for future expansion
    case 6: return "UniformMemory";
    case 7: return "SubgroupMemory";
    case 8: return "WorkgroupMemory";
    case 9: return "CrossWorkgroupMemory";
    case 10: return "AtomicCounterMemory";
    case 11: return "ImageMemory";

    default:     return "Bad";
    }
}

const int MemoryAccessCeiling = 6;

const char* MemoryAccessString(int mem)
{
    switch (mem) {
    case MemoryAccessVolatileShift:                 return "Volatile";
    case MemoryAccessAlignedShift:                  return "Aligned";
    case MemoryAccessNontemporalShift:              return "Nontemporal";
    case MemoryAccessMakePointerAvailableKHRShift:  return "MakePointerAvailableKHR";
    case MemoryAccessMakePointerVisibleKHRShift:    return "MakePointerVisibleKHR";
    case MemoryAccessNonPrivatePointerKHRShift:     return "NonPrivatePointerKHR";

    default: return "Bad";
    }
}

const char* ScopeString(int mem)
{
    switch (mem) {
    case 0:  return "CrossDevice";
    case 1:  return "Device";
    case 2:  return "Workgroup";
    case 3:  return "Subgroup";
    case 4:  return "Invocation";

    default: return "Bad";
    }
}

const char* GroupOperationString(int gop)
{

    switch (gop)
    {
    case GroupOperationReduce:  return "Reduce";
    case GroupOperationInclusiveScan:  return "InclusiveScan";
    case GroupOperationExclusiveScan:  return "ExclusiveScan";
    case GroupOperationClusteredReduce:  return "ClusteredReduce";
    case GroupOperationPartitionedReduceLW:  return "PartitionedReduceLW";
    case GroupOperationPartitionedInclusiveScanLW:  return "PartitionedInclusiveScanLW";
    case GroupOperationPartitionedExclusiveScanLW:  return "PartitionedExclusiveScanLW";

    default: return "Bad";
    }
}

const char* KernelEnqueueFlagsString(int flag)
{
    switch (flag)
    {
    case 0:  return "NoWait";
    case 1:  return "WaitKernel";
    case 2:  return "WaitWorkGroup";

    default: return "Bad";
    }
}

const char* KernelProfilingInfoString(int info)
{
    switch (info)
    {
    case 0:  return "CmdExecTime";

    default: return "Bad";
    }
}

const char* CapabilityString(int info)
{
    switch (info)
    {
    case 0:  return "Matrix";
    case 1:  return "Shader";
    case 2:  return "Geometry";
    case 3:  return "Tessellation";
    case 4:  return "Addresses";
    case 5:  return "Linkage";
    case 6:  return "Kernel";
    case 7:  return "Vector16";
    case 8:  return "Float16Buffer";
    case 9:  return "Float16";
    case 10: return "Float64";
    case 11: return "Int64";
    case 12: return "Int64Atomics";
    case 13: return "ImageBasic";
    case 14: return "ImageReadWrite";
    case 15: return "ImageMipmap";
    case 16: return "Bad";
    case 17: return "Pipes";
    case 18: return "Groups";
    case 19: return "DeviceEnqueue";
    case 20: return "LiteralSampler";
    case 21: return "AtomicStorage";
    case 22: return "Int16";
    case 23: return "TessellationPointSize";
    case 24: return "GeometryPointSize";
    case 25: return "ImageGatherExtended"; 
    case 26: return "Bad";
    case 27: return "StorageImageMultisample";
    case 28: return "UniformBufferArrayDynamicIndexing";
    case 29: return "SampledImageArrayDynamicIndexing";
    case 30: return "StorageBufferArrayDynamicIndexing";
    case 31: return "StorageImageArrayDynamicIndexing";
    case 32: return "ClipDistance";
    case 33: return "LwllDistance";
    case 34: return "ImageLwbeArray";
    case 35: return "SampleRateShading";
    case 36: return "ImageRect";
    case 37: return "SampledRect";
    case 38: return "GenericPointer";
    case 39: return "Int8";
    case 40: return "InputAttachment";
    case 41: return "SparseResidency";
    case 42: return "MinLod";
    case 43: return "Sampled1D";
    case 44: return "Image1D";
    case 45: return "SampledLwbeArray";
    case 46: return "SampledBuffer";
    case 47: return "ImageBuffer";
    case 48: return "ImageMSArray";
    case 49: return "StorageImageExtendedFormats";
    case 50: return "ImageQuery";
    case 51: return "DerivativeControl";
    case 52: return "InterpolationFunction";
    case 53: return "TransformFeedback";
    case 54: return "GeometryStreams";
    case 55: return "StorageImageReadWithoutFormat";
    case 56: return "StorageImageWriteWithoutFormat";
    case 57: return "MultiViewport";
    case 61: return "GroupNonUniform";
    case 62: return "GroupNonUniformVote";
    case 63: return "GroupNonUniformArithmetic";
    case 64: return "GroupNonUniformBallot";
    case 65: return "GroupNonUniformShuffle";
    case 66: return "GroupNonUniformShuffleRelative";
    case 67: return "GroupNonUniformClustered";
    case 68: return "GroupNonUniformQuad";

    case CapabilitySubgroupBallotKHR: return "SubgroupBallotKHR";
    case CapabilityDrawParameters:    return "DrawParameters";
    case CapabilitySubgroupVoteKHR:   return "SubgroupVoteKHR";

    case CapabilityStorageUniformBufferBlock16: return "StorageUniformBufferBlock16";
    case CapabilityStorageUniform16:            return "StorageUniform16";
    case CapabilityStoragePushConstant16:       return "StoragePushConstant16";
    case CapabilityStorageInputOutput16:        return "StorageInputOutput16";

    case CapabilityStorageBuffer8BitAccess:             return "StorageBuffer8BitAccess";
    case CapabilityUniformAndStorageBuffer8BitAccess:   return "UniformAndStorageBuffer8BitAccess";
    case CapabilityStoragePushConstant8:                return "StoragePushConstant8";

    case CapabilityDeviceGroup: return "DeviceGroup";
    case CapabilityMultiView:   return "MultiView";

    case CapabilityStencilExportEXT: return "StencilExportEXT";

    case CapabilityFloat16ImageAMD:       return "Float16ImageAMD";
    case CapabilityImageGatherBiasLodAMD: return "ImageGatherBiasLodAMD";
    case CapabilityFragmentMaskAMD:       return "FragmentMaskAMD";
    case CapabilityImageReadWriteLodAMD:  return "ImageReadWriteLodAMD";

    case CapabilityAtomicStorageOps:             return "AtomicStorageOps";

    case CapabilitySampleMaskPostDepthCoverage:  return "SampleMaskPostDepthCoverage";
    case CapabilityGeometryShaderPassthroughLW:     return "GeometryShaderPassthroughLW";
    case CapabilityShaderViewportIndexLayerLW:      return "ShaderViewportIndexLayerLW";
    case CapabilityShaderViewportMaskLW:            return "ShaderViewportMaskLW";
    case CapabilityShaderStereoViewLW:              return "ShaderStereoViewLW";
    case CapabilityPerViewAttributesLW:             return "PerViewAttributesLW";
    case CapabilityGroupNonUniformPartitionedLW:    return "GroupNonUniformPartitionedLW";
    case CapabilityRayTracingLW:                    return "RayTracingLW";
    case CapabilityRayTracingProvisionalKHR:        return "RayTracingProvisionalKHR";
    case CapabilityRayQueryProvisionalKHR:          return "RayQueryProvisionalKHR";
    case CapabilityRayTraversalPrimitiveLwllingProvisionalKHR: return "RayTraversalPrimitiveLwllingProvisionalKHR";
    case CapabilityComputeDerivativeGroupQuadsLW:   return "ComputeDerivativeGroupQuadsLW";
    case CapabilityComputeDerivativeGroupLinearLW:  return "ComputeDerivativeGroupLinearLW";
    case CapabilityFragmentBarycentricLW:           return "FragmentBarycentricLW";
    case CapabilityMeshShadingLW:                   return "MeshShadingLW";
    case CapabilityImageFootprintLW:                return "ImageFootprintLW";
//    case CapabilityShadingRateLW:                   return "ShadingRateLW";  // superseded by FragmentDensityEXT
    case CapabilitySampleMaskOverrideCoverageLW:    return "SampleMaskOverrideCoverageLW";
    case CapabilityFragmentDensityEXT:              return "FragmentDensityEXT";

    case CapabilityFragmentFullyCoveredEXT: return "FragmentFullyCoveredEXT";

    case CapabilityShaderNonUniformEXT:                          return "ShaderNonUniformEXT";
    case CapabilityRuntimeDescriptorArrayEXT:                    return "RuntimeDescriptorArrayEXT";
    case CapabilityInputAttachmentArrayDynamicIndexingEXT:       return "InputAttachmentArrayDynamicIndexingEXT";
    case CapabilityUniformTexelBufferArrayDynamicIndexingEXT:    return "UniformTexelBufferArrayDynamicIndexingEXT";
    case CapabilityStorageTexelBufferArrayDynamicIndexingEXT:    return "StorageTexelBufferArrayDynamicIndexingEXT";
    case CapabilityUniformBufferArrayNonUniformIndexingEXT:      return "UniformBufferArrayNonUniformIndexingEXT";
    case CapabilitySampledImageArrayNonUniformIndexingEXT:       return "SampledImageArrayNonUniformIndexingEXT";
    case CapabilityStorageBufferArrayNonUniformIndexingEXT:      return "StorageBufferArrayNonUniformIndexingEXT";
    case CapabilityStorageImageArrayNonUniformIndexingEXT:       return "StorageImageArrayNonUniformIndexingEXT";
    case CapabilityInputAttachmentArrayNonUniformIndexingEXT:    return "InputAttachmentArrayNonUniformIndexingEXT";
    case CapabilityUniformTexelBufferArrayNonUniformIndexingEXT: return "UniformTexelBufferArrayNonUniformIndexingEXT";
    case CapabilityStorageTexelBufferArrayNonUniformIndexingEXT: return "StorageTexelBufferArrayNonUniformIndexingEXT";

    case CapabilityVulkanMemoryModelKHR:                return "VulkanMemoryModelKHR";
    case CapabilityVulkanMemoryModelDeviceScopeKHR:     return "VulkanMemoryModelDeviceScopeKHR";

    case CapabilityPhysicalStorageBufferAddressesEXT:   return "PhysicalStorageBufferAddressesEXT";

    case CapabilityVariablePointers:                    return "VariablePointers";

    case CapabilityCooperativeMatrixLW:     return "CooperativeMatrixLW";
    case CapabilityShaderSMBuiltinsLW:      return "ShaderSMBuiltinsLW";

    case CapabilityFragmentShaderSampleInterlockEXT:        return "CapabilityFragmentShaderSampleInterlockEXT";
    case CapabilityFragmentShaderPixelInterlockEXT:         return "CapabilityFragmentShaderPixelInterlockEXT";
    case CapabilityFragmentShaderShadingRateInterlockEXT:   return "CapabilityFragmentShaderShadingRateInterlockEXT";

    case CapabilityDemoteToHelperIlwocationEXT:             return "DemoteToHelperIlwocationEXT";
    case CapabilityShaderClockKHR:                          return "ShaderClockKHR";

    case CapabilityIntegerFunctions2INTEL:              return "CapabilityIntegerFunctions2INTEL";

    default: return "Bad";
    }
}

const char* OpcodeString(int op)
{
    switch (op) {
    case 0:   return "OpNop";
    case 1:   return "OpUndef";
    case 2:   return "OpSourceContinued";
    case 3:   return "OpSource";
    case 4:   return "OpSourceExtension";
    case 5:   return "OpName";
    case 6:   return "OpMemberName";
    case 7:   return "OpString";
    case 8:   return "OpLine";
    case 9:   return "Bad";
    case 10:  return "OpExtension";
    case 11:  return "OpExtInstImport";
    case 12:  return "OpExtInst";
    case 13:  return "Bad";
    case 14:  return "OpMemoryModel";
    case 15:  return "OpEntryPoint";
    case 16:  return "OpExelwtionMode";
    case 17:  return "OpCapability";
    case 18:  return "Bad";
    case 19:  return "OpTypeVoid";
    case 20:  return "OpTypeBool";
    case 21:  return "OpTypeInt";
    case 22:  return "OpTypeFloat";
    case 23:  return "OpTypeVector";
    case 24:  return "OpTypeMatrix";
    case 25:  return "OpTypeImage";
    case 26:  return "OpTypeSampler";
    case 27:  return "OpTypeSampledImage";
    case 28:  return "OpTypeArray";
    case 29:  return "OpTypeRuntimeArray";
    case 30:  return "OpTypeStruct";
    case 31:  return "OpTypeOpaque";
    case 32:  return "OpTypePointer";
    case 33:  return "OpTypeFunction";
    case 34:  return "OpTypeEvent";
    case 35:  return "OpTypeDeviceEvent";
    case 36:  return "OpTypeReserveId";
    case 37:  return "OpTypeQueue";
    case 38:  return "OpTypePipe";
    case 39:  return "OpTypeForwardPointer";
    case 40:  return "Bad";
    case 41:  return "OpConstantTrue";
    case 42:  return "OpConstantFalse";
    case 43:  return "OpConstant";
    case 44:  return "OpConstantComposite";
    case 45:  return "OpConstantSampler";
    case 46:  return "OpConstantNull";
    case 47:  return "Bad";
    case 48:  return "OpSpecConstantTrue";
    case 49:  return "OpSpecConstantFalse";
    case 50:  return "OpSpecConstant";
    case 51:  return "OpSpecConstantComposite";
    case 52:  return "OpSpecConstantOp";
    case 53:  return "Bad";
    case 54:  return "OpFunction";
    case 55:  return "OpFunctionParameter";
    case 56:  return "OpFunctionEnd";
    case 57:  return "OpFunctionCall";
    case 58:  return "Bad";
    case 59:  return "OpVariable";
    case 60:  return "OpImageTexelPointer";
    case 61:  return "OpLoad";
    case 62:  return "OpStore";
    case 63:  return "OpCopyMemory";
    case 64:  return "OpCopyMemorySized";
    case 65:  return "OpAccessChain";
    case 66:  return "OpInBoundsAccessChain";
    case 67:  return "OpPtrAccessChain";
    case 68:  return "OpArrayLength";
    case 69:  return "OpGenericPtrMemSemantics";
    case 70:  return "OpInBoundsPtrAccessChain";
    case 71:  return "OpDecorate";
    case 72:  return "OpMemberDecorate";
    case 73:  return "OpDecorationGroup";
    case 74:  return "OpGroupDecorate";
    case 75:  return "OpGroupMemberDecorate";
    case 76:  return "Bad";
    case 77:  return "OpVectorExtractDynamic";
    case 78:  return "OpVectorInsertDynamic";
    case 79:  return "OpVectorShuffle";
    case 80:  return "OpCompositeConstruct";
    case 81:  return "OpCompositeExtract";
    case 82:  return "OpCompositeInsert";
    case 83:  return "OpCopyObject";
    case 84:  return "OpTranspose";
    case OpCopyLogical: return "OpCopyLogical";
    case 85:  return "Bad";
    case 86:  return "OpSampledImage";
    case 87:  return "OpImageSampleImplicitLod";
    case 88:  return "OpImageSampleExplicitLod";
    case 89:  return "OpImageSampleDrefImplicitLod";
    case 90:  return "OpImageSampleDrefExplicitLod";
    case 91:  return "OpImageSampleProjImplicitLod";
    case 92:  return "OpImageSampleProjExplicitLod";
    case 93:  return "OpImageSampleProjDrefImplicitLod";
    case 94:  return "OpImageSampleProjDrefExplicitLod";
    case 95:  return "OpImageFetch";
    case 96:  return "OpImageGather";
    case 97:  return "OpImageDrefGather";
    case 98:  return "OpImageRead";
    case 99:  return "OpImageWrite";
    case 100: return "OpImage";
    case 101: return "OpImageQueryFormat";
    case 102: return "OpImageQueryOrder";
    case 103: return "OpImageQuerySizeLod";
    case 104: return "OpImageQuerySize";
    case 105: return "OpImageQueryLod";
    case 106: return "OpImageQueryLevels";
    case 107: return "OpImageQuerySamples";
    case 108: return "Bad";
    case 109: return "OpColwertFToU";
    case 110: return "OpColwertFToS";
    case 111: return "OpColwertSToF";
    case 112: return "OpColwertUToF";
    case 113: return "OpUColwert";
    case 114: return "OpSColwert";
    case 115: return "OpFColwert";
    case 116: return "OpQuantizeToF16";
    case 117: return "OpColwertPtrToU";
    case 118: return "OpSatColwertSToU";
    case 119: return "OpSatColwertUToS";
    case 120: return "OpColwertUToPtr";
    case 121: return "OpPtrCastToGeneric";
    case 122: return "OpGenericCastToPtr";
    case 123: return "OpGenericCastToPtrExplicit";
    case 124: return "OpBitcast";
    case 125: return "Bad";
    case 126: return "OpSNegate";
    case 127: return "OpFNegate";
    case 128: return "OpIAdd";
    case 129: return "OpFAdd";
    case 130: return "OpISub";
    case 131: return "OpFSub";
    case 132: return "OpIMul";
    case 133: return "OpFMul";
    case 134: return "OpUDiv";
    case 135: return "OpSDiv";
    case 136: return "OpFDiv";
    case 137: return "OpUMod";
    case 138: return "OpSRem";
    case 139: return "OpSMod";
    case 140: return "OpFRem";
    case 141: return "OpFMod";
    case 142: return "OpVectorTimesScalar";
    case 143: return "OpMatrixTimesScalar";
    case 144: return "OpVectorTimesMatrix";
    case 145: return "OpMatrixTimesVector";
    case 146: return "OpMatrixTimesMatrix";
    case 147: return "OpOuterProduct";
    case 148: return "OpDot";
    case 149: return "OpIAddCarry";
    case 150: return "OpISubBorrow";
    case 151: return "OpUMulExtended";
    case 152: return "OpSMulExtended";
    case 153: return "Bad";
    case 154: return "OpAny";
    case 155: return "OpAll";
    case 156: return "OpIsNan";
    case 157: return "OpIsInf";
    case 158: return "OpIsFinite";
    case 159: return "OpIsNormal";
    case 160: return "OpSignBitSet";
    case 161: return "OpLessOrGreater";
    case 162: return "OpOrdered";
    case 163: return "OpUnordered";
    case 164: return "OpLogicalEqual";
    case 165: return "OpLogicalNotEqual";
    case 166: return "OpLogicalOr";
    case 167: return "OpLogicalAnd";
    case 168: return "OpLogicalNot";
    case 169: return "OpSelect";
    case 170: return "OpIEqual";
    case 171: return "OpINotEqual";
    case 172: return "OpUGreaterThan";
    case 173: return "OpSGreaterThan";
    case 174: return "OpUGreaterThanEqual";
    case 175: return "OpSGreaterThanEqual";
    case 176: return "OpULessThan";
    case 177: return "OpSLessThan";
    case 178: return "OpULessThanEqual";
    case 179: return "OpSLessThanEqual";
    case 180: return "OpFOrdEqual";
    case 181: return "OpFUnordEqual";
    case 182: return "OpFOrdNotEqual";
    case 183: return "OpFUnordNotEqual";
    case 184: return "OpFOrdLessThan";
    case 185: return "OpFUnordLessThan";
    case 186: return "OpFOrdGreaterThan";
    case 187: return "OpFUnordGreaterThan";
    case 188: return "OpFOrdLessThanEqual";
    case 189: return "OpFUnordLessThanEqual";
    case 190: return "OpFOrdGreaterThanEqual";
    case 191: return "OpFUnordGreaterThanEqual";
    case 192: return "Bad";
    case 193: return "Bad";
    case 194: return "OpShiftRightLogical";
    case 195: return "OpShiftRightArithmetic";
    case 196: return "OpShiftLeftLogical";
    case 197: return "OpBitwiseOr";
    case 198: return "OpBitwiseXor";
    case 199: return "OpBitwiseAnd";
    case 200: return "OpNot";
    case 201: return "OpBitFieldInsert";
    case 202: return "OpBitFieldSExtract";
    case 203: return "OpBitFieldUExtract";
    case 204: return "OpBitReverse";
    case 205: return "OpBitCount";
    case 206: return "Bad";
    case 207: return "OpDPdx";
    case 208: return "OpDPdy";
    case 209: return "OpFwidth";
    case 210: return "OpDPdxFine";
    case 211: return "OpDPdyFine";
    case 212: return "OpFwidthFine";
    case 213: return "OpDPdxCoarse";
    case 214: return "OpDPdyCoarse";
    case 215: return "OpFwidthCoarse";
    case 216: return "Bad";
    case 217: return "Bad";
    case 218: return "OpEmitVertex";
    case 219: return "OpEndPrimitive";
    case 220: return "OpEmitStreamVertex";
    case 221: return "OpEndStreamPrimitive";
    case 222: return "Bad";
    case 223: return "Bad";
    case 224: return "OpControlBarrier";
    case 225: return "OpMemoryBarrier";
    case 226: return "Bad";
    case 227: return "OpAtomicLoad";
    case 228: return "OpAtomicStore";
    case 229: return "OpAtomicExchange";
    case 230: return "OpAtomicCompareExchange";
    case 231: return "OpAtomicCompareExchangeWeak";
    case 232: return "OpAtomicIIncrement";
    case 233: return "OpAtomicIDecrement";
    case 234: return "OpAtomicIAdd";
    case 235: return "OpAtomicISub";
    case 236: return "OpAtomicSMin";
    case 237: return "OpAtomilwMin";
    case 238: return "OpAtomicSMax";
    case 239: return "OpAtomilwMax";
    case 240: return "OpAtomicAnd";
    case 241: return "OpAtomicOr";
    case 242: return "OpAtomicXor";
    case 243: return "Bad";
    case 244: return "Bad";
    case 245: return "OpPhi";
    case 246: return "OpLoopMerge";
    case 247: return "OpSelectionMerge";
    case 248: return "OpLabel";
    case 249: return "OpBranch";
    case 250: return "OpBranchConditional";
    case 251: return "OpSwitch";
    case 252: return "OpKill";
    case 253: return "OpReturn";
    case 254: return "OpReturlwalue";
    case 255: return "OpUnreachable";
    case 256: return "OpLifetimeStart";
    case 257: return "OpLifetimeStop";
    case 258: return "Bad";
    case 259: return "OpGroupAsyncCopy";
    case 260: return "OpGroupWaitEvents";
    case 261: return "OpGroupAll";
    case 262: return "OpGroupAny";
    case 263: return "OpGroupBroadcast";
    case 264: return "OpGroupIAdd";
    case 265: return "OpGroupFAdd";
    case 266: return "OpGroupFMin";
    case 267: return "OpGroupUMin";
    case 268: return "OpGroupSMin";
    case 269: return "OpGroupFMax";
    case 270: return "OpGroupUMax";
    case 271: return "OpGroupSMax";
    case 272: return "Bad";
    case 273: return "Bad";
    case 274: return "OpReadPipe";
    case 275: return "OpWritePipe";
    case 276: return "OpReservedReadPipe";
    case 277: return "OpReservedWritePipe";
    case 278: return "OpReserveReadPipePackets";
    case 279: return "OpReserveWritePipePackets";
    case 280: return "OpCommitReadPipe";
    case 281: return "OpCommitWritePipe";
    case 282: return "OpIsValidReserveId";
    case 283: return "OpGetNumPipePackets";
    case 284: return "OpGetMaxPipePackets";
    case 285: return "OpGroupReserveReadPipePackets";
    case 286: return "OpGroupReserveWritePipePackets";
    case 287: return "OpGroupCommitReadPipe";
    case 288: return "OpGroupCommitWritePipe";
    case 289: return "Bad";
    case 290: return "Bad";
    case 291: return "OpEnqueueMarker";
    case 292: return "OpEnqueueKernel";
    case 293: return "OpGetKernelNDrangeSubGroupCount";
    case 294: return "OpGetKernelNDrangeMaxSubGroupSize";
    case 295: return "OpGetKernelWorkGroupSize";
    case 296: return "OpGetKernelPreferredWorkGroupSizeMultiple";
    case 297: return "OpRetainEvent";
    case 298: return "OpReleaseEvent";
    case 299: return "OpCreateUserEvent";
    case 300: return "OpIsValidEvent";
    case 301: return "OpSetUserEventStatus";
    case 302: return "OpCaptureEventProfilingInfo";
    case 303: return "OpGetDefaultQueue";
    case 304: return "OpBuildNDRange";
    case 305: return "OpImageSparseSampleImplicitLod";
    case 306: return "OpImageSparseSampleExplicitLod";
    case 307: return "OpImageSparseSampleDrefImplicitLod";
    case 308: return "OpImageSparseSampleDrefExplicitLod";
    case 309: return "OpImageSparseSampleProjImplicitLod";
    case 310: return "OpImageSparseSampleProjExplicitLod";
    case 311: return "OpImageSparseSampleProjDrefImplicitLod";
    case 312: return "OpImageSparseSampleProjDrefExplicitLod";
    case 313: return "OpImageSparseFetch";
    case 314: return "OpImageSparseGather";
    case 315: return "OpImageSparseDrefGather";
    case 316: return "OpImageSparseTexelsResident";
    case 317: return "OpNoLine";
    case 318: return "OpAtomicFlagTestAndSet";
    case 319: return "OpAtomicFlagClear";
    case 320: return "OpImageSparseRead";

    case OpModuleProcessed: return "OpModuleProcessed";
    case OpDecorateId:      return "OpDecorateId";

    case 333: return "OpGroupNonUniformElect";
    case 334: return "OpGroupNonUniformAll";
    case 335: return "OpGroupNonUniformAny";
    case 336: return "OpGroupNonUniformAllEqual";
    case 337: return "OpGroupNonUniformBroadcast";
    case 338: return "OpGroupNonUniformBroadcastFirst";
    case 339: return "OpGroupNonUniformBallot";
    case 340: return "OpGroupNonUniformIlwerseBallot";
    case 341: return "OpGroupNonUniformBallotBitExtract";
    case 342: return "OpGroupNonUniformBallotBitCount";
    case 343: return "OpGroupNonUniformBallotFindLSB";
    case 344: return "OpGroupNonUniformBallotFindMSB";
    case 345: return "OpGroupNonUniformShuffle";
    case 346: return "OpGroupNonUniformShuffleXor";
    case 347: return "OpGroupNonUniformShuffleUp";
    case 348: return "OpGroupNonUniformShuffleDown";
    case 349: return "OpGroupNonUniformIAdd";
    case 350: return "OpGroupNonUniformFAdd";
    case 351: return "OpGroupNonUniformIMul";
    case 352: return "OpGroupNonUniformFMul";
    case 353: return "OpGroupNonUniformSMin";
    case 354: return "OpGroupNonUniformUMin";
    case 355: return "OpGroupNonUniformFMin";
    case 356: return "OpGroupNonUniformSMax";
    case 357: return "OpGroupNonUniformUMax";
    case 358: return "OpGroupNonUniformFMax";
    case 359: return "OpGroupNonUniformBitwiseAnd";
    case 360: return "OpGroupNonUniformBitwiseOr";
    case 361: return "OpGroupNonUniformBitwiseXor";
    case 362: return "OpGroupNonUniformLogicalAnd";
    case 363: return "OpGroupNonUniformLogicalOr";
    case 364: return "OpGroupNonUniformLogicalXor";
    case 365: return "OpGroupNonUniformQuadBroadcast";
    case 366: return "OpGroupNonUniformQuadSwap";

    case 4421: return "OpSubgroupBallotKHR";
    case 4422: return "OpSubgroupFirstIlwocationKHR";
    case 4428: return "OpSubgroupAllKHR";
    case 4429: return "OpSubgroupAnyKHR";
    case 4430: return "OpSubgroupAllEqualKHR";
    case 4432: return "OpSubgroupReadIlwocationKHR";

    case 5000: return "OpGroupIAddNonUniformAMD";
    case 5001: return "OpGroupFAddNonUniformAMD";
    case 5002: return "OpGroupFMinNonUniformAMD";
    case 5003: return "OpGroupUMinNonUniformAMD";
    case 5004: return "OpGroupSMinNonUniformAMD";
    case 5005: return "OpGroupFMaxNonUniformAMD";
    case 5006: return "OpGroupUMaxNonUniformAMD";
    case 5007: return "OpGroupSMaxNonUniformAMD";

    case 5011: return "OpFragmentMaskFetchAMD";
    case 5012: return "OpFragmentFetchAMD";

    case OpReadClockKHR:               return "OpReadClockKHR";

    case OpDecorateStringGOOGLE:       return "OpDecorateStringGOOGLE";
    case OpMemberDecorateStringGOOGLE: return "OpMemberDecorateStringGOOGLE";

    case OpGroupNonUniformPartitionLW:       return "OpGroupNonUniformPartitionLW";
    case OpReportIntersectionKHR:            return "OpReportIntersectionKHR";
    case OpIgnoreIntersectionKHR:            return "OpIgnoreIntersectionKHR";
    case OpTerminateRayKHR:                  return "OpTerminateRayKHR";
    case OpTraceRayKHR:                      return "OpTraceRayKHR";
    case OpTypeAccelerationStructureKHR:     return "OpTypeAccelerationStructureKHR";
    case OpExelwteCallableKHR:               return "OpExelwteCallableKHR";
    case OpImageSampleFootprintLW:           return "OpImageSampleFootprintLW";
    case OpWritePackedPrimitiveIndices4x8LW: return "OpWritePackedPrimitiveIndices4x8LW";

    case OpTypeRayQueryProvisionalKHR:                                        return "OpTypeRayQueryProvisionalKHR";
    case OpRayQueryInitializeKHR:                                             return "OpRayQueryInitializeKHR";
    case OpRayQueryTerminateKHR:                                              return "OpRayQueryTerminateKHR";
    case OpRayQueryGenerateIntersectionKHR:                                   return "OpRayQueryGenerateIntersectionKHR";
    case OpRayQueryConfirmIntersectionKHR:                                    return "OpRayQueryConfirmIntersectionKHR";
    case OpRayQueryProceedKHR:                                                return "OpRayQueryProceedKHR";
    case OpRayQueryGetIntersectionTypeKHR:                                    return "OpRayQueryGetIntersectionTypeKHR";
    case OpRayQueryGetRayTMinKHR:                                             return "OpRayQueryGetRayTMinKHR";
    case OpRayQueryGetRayFlagsKHR:                                            return "OpRayQueryGetRayFlagsKHR";
    case OpRayQueryGetIntersectionTKHR:                                       return "OpRayQueryGetIntersectionTKHR";
    case OpRayQueryGetIntersectionInstanceLwstomIndexKHR:                     return "OpRayQueryGetIntersectionInstanceLwstomIndexKHR";
    case OpRayQueryGetIntersectionInstanceIdKHR:                              return "OpRayQueryGetIntersectionInstanceIdKHR";
    case OpRayQueryGetIntersectionInstanceShaderBindingTableRecordOffsetKHR:  return "OpRayQueryGetIntersectionInstanceShaderBindingTableRecordOffsetKHR";
    case OpRayQueryGetIntersectionGeometryIndexKHR:                           return "OpRayQueryGetIntersectionGeometryIndexKHR";
    case OpRayQueryGetIntersectionPrimitiveIndexKHR:                          return "OpRayQueryGetIntersectionPrimitiveIndexKHR";
    case OpRayQueryGetIntersectionBarycentricsKHR:                            return "OpRayQueryGetIntersectionBarycentricsKHR";
    case OpRayQueryGetIntersectionFrontFaceKHR:                               return "OpRayQueryGetIntersectionFrontFaceKHR";
    case OpRayQueryGetIntersectionCandidateAABBOpaqueKHR:                     return "OpRayQueryGetIntersectionCandidateAABBOpaqueKHR";
    case OpRayQueryGetIntersectionObjectRayDirectionKHR:                      return "OpRayQueryGetIntersectionObjectRayDirectionKHR";
    case OpRayQueryGetIntersectionObjectRayOriginKHR:                         return "OpRayQueryGetIntersectionObjectRayOriginKHR";
    case OpRayQueryGetWorldRayDirectionKHR:                                   return "OpRayQueryGetWorldRayDirectionKHR";
    case OpRayQueryGetWorldRayOriginKHR:                                      return "OpRayQueryGetWorldRayOriginKHR";
    case OpRayQueryGetIntersectionObjectToWorldKHR:                           return "OpRayQueryGetIntersectionObjectToWorldKHR";
    case OpRayQueryGetIntersectionWorldToObjectKHR:                           return "OpRayQueryGetIntersectionWorldToObjectKHR";

    case OpTypeCooperativeMatrixLW:         return "OpTypeCooperativeMatrixLW";
    case OpCooperativeMatrixLoadLW:         return "OpCooperativeMatrixLoadLW";
    case OpCooperativeMatrixStoreLW:        return "OpCooperativeMatrixStoreLW";
    case OpCooperativeMatrixMulAddLW:       return "OpCooperativeMatrixMulAddLW";
    case OpCooperativeMatrixLengthLW:       return "OpCooperativeMatrixLengthLW";
    case OpDemoteToHelperIlwocationEXT:     return "OpDemoteToHelperIlwocationEXT";
    case OpIsHelperIlwocationEXT:           return "OpIsHelperIlwocationEXT";

    case OpBeginIlwocationInterlockEXT:     return "OpBeginIlwocationInterlockEXT";
    case OpEndIlwocationInterlockEXT:       return "OpEndIlwocationInterlockEXT";

    default:
        return "Bad";
    }
}

// The set of objects that hold all the instruction/operand
// parameterization information.
InstructionParameters InstructionDesc[OpCodeMask + 1];
OperandParameters ExelwtionModeOperands[ExelwtionModeCeiling];
OperandParameters DecorationOperands[DecorationCeiling];

EnumDefinition OperandClassParams[OperandCount];
EnumParameters ExelwtionModeParams[ExelwtionModeCeiling];
EnumParameters ImageOperandsParams[ImageOperandsCeiling];
EnumParameters DecorationParams[DecorationCeiling];
EnumParameters LoopControlParams[FunctionControlCeiling];
EnumParameters SelectionControlParams[SelectControlCeiling];
EnumParameters FunctionControlParams[FunctionControlCeiling];
EnumParameters MemoryAccessParams[MemoryAccessCeiling];

// Set up all the parameterizing descriptions of the opcodes, operands, etc.
void Parameterize()
{
    // only do this once.
    static bool initialized = false;
    if (initialized)
        return;
    initialized = true;

    // Exceptions to having a result <id> and a resulting type <id>.
    // (Everything is initialized to have both).

    InstructionDesc[OpNop].setResultAndType(false, false);
    InstructionDesc[OpSource].setResultAndType(false, false);
    InstructionDesc[OpSourceContinued].setResultAndType(false, false);
    InstructionDesc[OpSourceExtension].setResultAndType(false, false);
    InstructionDesc[OpExtension].setResultAndType(false, false);
    InstructionDesc[OpExtInstImport].setResultAndType(true, false);
    InstructionDesc[OpCapability].setResultAndType(false, false);
    InstructionDesc[OpMemoryModel].setResultAndType(false, false);
    InstructionDesc[OpEntryPoint].setResultAndType(false, false);
    InstructionDesc[OpExelwtionMode].setResultAndType(false, false);
    InstructionDesc[OpTypeVoid].setResultAndType(true, false);
    InstructionDesc[OpTypeBool].setResultAndType(true, false);
    InstructionDesc[OpTypeInt].setResultAndType(true, false);
    InstructionDesc[OpTypeFloat].setResultAndType(true, false);
    InstructionDesc[OpTypeVector].setResultAndType(true, false);
    InstructionDesc[OpTypeMatrix].setResultAndType(true, false);
    InstructionDesc[OpTypeImage].setResultAndType(true, false);
    InstructionDesc[OpTypeSampler].setResultAndType(true, false);
    InstructionDesc[OpTypeSampledImage].setResultAndType(true, false);
    InstructionDesc[OpTypeArray].setResultAndType(true, false);
    InstructionDesc[OpTypeRuntimeArray].setResultAndType(true, false);
    InstructionDesc[OpTypeStruct].setResultAndType(true, false);
    InstructionDesc[OpTypeOpaque].setResultAndType(true, false);
    InstructionDesc[OpTypePointer].setResultAndType(true, false);
    InstructionDesc[OpTypeForwardPointer].setResultAndType(false, false);
    InstructionDesc[OpTypeFunction].setResultAndType(true, false);
    InstructionDesc[OpTypeEvent].setResultAndType(true, false);
    InstructionDesc[OpTypeDeviceEvent].setResultAndType(true, false);
    InstructionDesc[OpTypeReserveId].setResultAndType(true, false);
    InstructionDesc[OpTypeQueue].setResultAndType(true, false);
    InstructionDesc[OpTypePipe].setResultAndType(true, false);
    InstructionDesc[OpFunctionEnd].setResultAndType(false, false);
    InstructionDesc[OpStore].setResultAndType(false, false);
    InstructionDesc[OpImageWrite].setResultAndType(false, false);
    InstructionDesc[OpDecorationGroup].setResultAndType(true, false);
    InstructionDesc[OpDecorate].setResultAndType(false, false);
    InstructionDesc[OpDecorateId].setResultAndType(false, false);
    InstructionDesc[OpDecorateStringGOOGLE].setResultAndType(false, false);
    InstructionDesc[OpMemberDecorate].setResultAndType(false, false);
    InstructionDesc[OpMemberDecorateStringGOOGLE].setResultAndType(false, false);
    InstructionDesc[OpGroupDecorate].setResultAndType(false, false);
    InstructionDesc[OpGroupMemberDecorate].setResultAndType(false, false);
    InstructionDesc[OpName].setResultAndType(false, false);
    InstructionDesc[OpMemberName].setResultAndType(false, false);
    InstructionDesc[OpString].setResultAndType(true, false);
    InstructionDesc[OpLine].setResultAndType(false, false);
    InstructionDesc[OpNoLine].setResultAndType(false, false);
    InstructionDesc[OpCopyMemory].setResultAndType(false, false);
    InstructionDesc[OpCopyMemorySized].setResultAndType(false, false);
    InstructionDesc[OpEmitVertex].setResultAndType(false, false);
    InstructionDesc[OpEndPrimitive].setResultAndType(false, false);
    InstructionDesc[OpEmitStreamVertex].setResultAndType(false, false);
    InstructionDesc[OpEndStreamPrimitive].setResultAndType(false, false);
    InstructionDesc[OpControlBarrier].setResultAndType(false, false);
    InstructionDesc[OpMemoryBarrier].setResultAndType(false, false);
    InstructionDesc[OpAtomicStore].setResultAndType(false, false);
    InstructionDesc[OpLoopMerge].setResultAndType(false, false);
    InstructionDesc[OpSelectionMerge].setResultAndType(false, false);
    InstructionDesc[OpLabel].setResultAndType(true, false);
    InstructionDesc[OpBranch].setResultAndType(false, false);
    InstructionDesc[OpBranchConditional].setResultAndType(false, false);
    InstructionDesc[OpSwitch].setResultAndType(false, false);
    InstructionDesc[OpKill].setResultAndType(false, false);
    InstructionDesc[OpReturn].setResultAndType(false, false);
    InstructionDesc[OpReturlwalue].setResultAndType(false, false);
    InstructionDesc[OpUnreachable].setResultAndType(false, false);
    InstructionDesc[OpLifetimeStart].setResultAndType(false, false);
    InstructionDesc[OpLifetimeStop].setResultAndType(false, false);
    InstructionDesc[OpCommitReadPipe].setResultAndType(false, false);
    InstructionDesc[OpCommitWritePipe].setResultAndType(false, false);
    InstructionDesc[OpGroupCommitWritePipe].setResultAndType(false, false);
    InstructionDesc[OpGroupCommitReadPipe].setResultAndType(false, false);
    InstructionDesc[OpCaptureEventProfilingInfo].setResultAndType(false, false);
    InstructionDesc[OpSetUserEventStatus].setResultAndType(false, false);
    InstructionDesc[OpRetainEvent].setResultAndType(false, false);
    InstructionDesc[OpReleaseEvent].setResultAndType(false, false);
    InstructionDesc[OpGroupWaitEvents].setResultAndType(false, false);
    InstructionDesc[OpAtomicFlagClear].setResultAndType(false, false);
    InstructionDesc[OpModuleProcessed].setResultAndType(false, false);
    InstructionDesc[OpTypeCooperativeMatrixLW].setResultAndType(true, false);
    InstructionDesc[OpCooperativeMatrixStoreLW].setResultAndType(false, false);
    InstructionDesc[OpBeginIlwocationInterlockEXT].setResultAndType(false, false);
    InstructionDesc[OpEndIlwocationInterlockEXT].setResultAndType(false, false);

    // Specific additional context-dependent operands

    ExelwtionModeOperands[ExelwtionModeIlwocations].push(OperandLiteralNumber, "'Number of <<Invocation,ilwocations>>'");

    ExelwtionModeOperands[ExelwtionModeLocalSize].push(OperandLiteralNumber, "'x size'");
    ExelwtionModeOperands[ExelwtionModeLocalSize].push(OperandLiteralNumber, "'y size'");
    ExelwtionModeOperands[ExelwtionModeLocalSize].push(OperandLiteralNumber, "'z size'");

    ExelwtionModeOperands[ExelwtionModeLocalSizeHint].push(OperandLiteralNumber, "'x size'");
    ExelwtionModeOperands[ExelwtionModeLocalSizeHint].push(OperandLiteralNumber, "'y size'");
    ExelwtionModeOperands[ExelwtionModeLocalSizeHint].push(OperandLiteralNumber, "'z size'");

    ExelwtionModeOperands[ExelwtionModeOutputVertices].push(OperandLiteralNumber, "'Vertex count'");
    ExelwtionModeOperands[ExelwtionModeVecTypeHint].push(OperandLiteralNumber, "'Vector type'");

    DecorationOperands[DecorationStream].push(OperandLiteralNumber, "'Stream Number'");
    DecorationOperands[DecorationLocation].push(OperandLiteralNumber, "'Location'");
    DecorationOperands[DecorationComponent].push(OperandLiteralNumber, "'Component'");
    DecorationOperands[DecorationIndex].push(OperandLiteralNumber, "'Index'");
    DecorationOperands[DecorationBinding].push(OperandLiteralNumber, "'Binding Point'");
    DecorationOperands[DecorationDescriptorSet].push(OperandLiteralNumber, "'Descriptor Set'");
    DecorationOperands[DecorationOffset].push(OperandLiteralNumber, "'Byte Offset'");
    DecorationOperands[DecorationXfbBuffer].push(OperandLiteralNumber, "'XFB Buffer Number'");
    DecorationOperands[DecorationXfbStride].push(OperandLiteralNumber, "'XFB Stride'");
    DecorationOperands[DecorationArrayStride].push(OperandLiteralNumber, "'Array Stride'");
    DecorationOperands[DecorationMatrixStride].push(OperandLiteralNumber, "'Matrix Stride'");
    DecorationOperands[DecorationBuiltIn].push(OperandLiteralNumber, "See <<BuiltIn,*BuiltIn*>>");
    DecorationOperands[DecorationFPRoundingMode].push(OperandFPRoundingMode, "'Floating-Point Rounding Mode'");
    DecorationOperands[DecorationFPFastMathMode].push(OperandFPFastMath, "'Fast-Math Mode'");
    DecorationOperands[DecorationLinkageAttributes].push(OperandLiteralString, "'Name'");
    DecorationOperands[DecorationLinkageAttributes].push(OperandLinkageType, "'Linkage Type'");
    DecorationOperands[DecorationFuncParamAttr].push(OperandFuncParamAttr, "'Function Parameter Attribute'");
    DecorationOperands[DecorationSpecId].push(OperandLiteralNumber, "'Specialization Constant ID'");
    DecorationOperands[DecorationInputAttachmentIndex].push(OperandLiteralNumber, "'Attachment Index'");
    DecorationOperands[DecorationAlignment].push(OperandLiteralNumber, "'Alignment'");

    OperandClassParams[OperandSource].set(0, SourceString, 0);
    OperandClassParams[OperandExelwtionModel].set(0, ExelwtionModelString, nullptr);
    OperandClassParams[OperandAddressing].set(0, AddressingString, nullptr);
    OperandClassParams[OperandMemory].set(0, MemoryString, nullptr);
    OperandClassParams[OperandExelwtionMode].set(ExelwtionModeCeiling, ExelwtionModeString, ExelwtionModeParams);
    OperandClassParams[OperandExelwtionMode].setOperands(ExelwtionModeOperands);
    OperandClassParams[OperandStorage].set(0, StorageClassString, nullptr);
    OperandClassParams[OperandDimensionality].set(0, DimensionString, nullptr);
    OperandClassParams[OperandSamplerAddressingMode].set(0, SamplerAddressingModeString, nullptr);
    OperandClassParams[OperandSamplerFilterMode].set(0, SamplerFilterModeString, nullptr);
    OperandClassParams[OperandSamplerImageFormat].set(0, ImageFormatString, nullptr);
    OperandClassParams[OperandImageChannelOrder].set(0, ImageChannelOrderString, nullptr);
    OperandClassParams[OperandImageChannelDataType].set(0, ImageChannelDataTypeString, nullptr);
    OperandClassParams[OperandImageOperands].set(ImageOperandsCeiling, ImageOperandsString, ImageOperandsParams, true);
    OperandClassParams[OperandFPFastMath].set(0, FPFastMathString, nullptr, true);
    OperandClassParams[OperandFPRoundingMode].set(0, FPRoundingModeString, nullptr);
    OperandClassParams[OperandLinkageType].set(0, LinkageTypeString, nullptr);
    OperandClassParams[OperandFuncParamAttr].set(0, FuncParamAttrString, nullptr);
    OperandClassParams[OperandAccessQualifier].set(0, AccessQualifierString, nullptr);
    OperandClassParams[OperandDecoration].set(DecorationCeiling, DecorationString, DecorationParams);
    OperandClassParams[OperandDecoration].setOperands(DecorationOperands);
    OperandClassParams[OperandBuiltIn].set(0, BuiltInString, nullptr);
    OperandClassParams[OperandSelect].set(SelectControlCeiling, SelectControlString, SelectionControlParams, true);
    OperandClassParams[OperandLoop].set(LoopControlCeiling, LoopControlString, LoopControlParams, true);
    OperandClassParams[OperandFunction].set(FunctionControlCeiling, FunctionControlString, FunctionControlParams, true);
    OperandClassParams[OperandMemorySemantics].set(0, MemorySemanticsString, nullptr, true);
    OperandClassParams[OperandMemoryAccess].set(MemoryAccessCeiling, MemoryAccessString, MemoryAccessParams, true);
    OperandClassParams[OperandScope].set(0, ScopeString, nullptr);
    OperandClassParams[OperandGroupOperation].set(0, GroupOperationString, nullptr);
    OperandClassParams[OperandKernelEnqueueFlags].set(0, KernelEnqueueFlagsString, nullptr);
    OperandClassParams[OperandKernelProfilingInfo].set(0, KernelProfilingInfoString, nullptr, true);
    OperandClassParams[OperandCapability].set(0, CapabilityString, nullptr);
    OperandClassParams[OperandOpcode].set(OpCodeMask + 1, OpcodeString, 0);

    // set name of operator, an initial set of <id> style operands, and the description

    InstructionDesc[OpSource].operands.push(OperandSource, "");
    InstructionDesc[OpSource].operands.push(OperandLiteralNumber, "'Version'");
    InstructionDesc[OpSource].operands.push(OperandId, "'File'", true);
    InstructionDesc[OpSource].operands.push(OperandLiteralString, "'Source'", true);

    InstructionDesc[OpSourceContinued].operands.push(OperandLiteralString, "'Continued Source'");

    InstructionDesc[OpSourceExtension].operands.push(OperandLiteralString, "'Extension'");

    InstructionDesc[OpName].operands.push(OperandId, "'Target'");
    InstructionDesc[OpName].operands.push(OperandLiteralString, "'Name'");

    InstructionDesc[OpMemberName].operands.push(OperandId, "'Type'");
    InstructionDesc[OpMemberName].operands.push(OperandLiteralNumber, "'Member'");
    InstructionDesc[OpMemberName].operands.push(OperandLiteralString, "'Name'");

    InstructionDesc[OpString].operands.push(OperandLiteralString, "'String'");

    InstructionDesc[OpLine].operands.push(OperandId, "'File'");
    InstructionDesc[OpLine].operands.push(OperandLiteralNumber, "'Line'");
    InstructionDesc[OpLine].operands.push(OperandLiteralNumber, "'Column'");

    InstructionDesc[OpExtension].operands.push(OperandLiteralString, "'Name'");

    InstructionDesc[OpExtInstImport].operands.push(OperandLiteralString, "'Name'");

    InstructionDesc[OpCapability].operands.push(OperandCapability, "'Capability'");

    InstructionDesc[OpMemoryModel].operands.push(OperandAddressing, "");
    InstructionDesc[OpMemoryModel].operands.push(OperandMemory, "");

    InstructionDesc[OpEntryPoint].operands.push(OperandExelwtionModel, "");
    InstructionDesc[OpEntryPoint].operands.push(OperandId, "'Entry Point'");
    InstructionDesc[OpEntryPoint].operands.push(OperandLiteralString, "'Name'");
    InstructionDesc[OpEntryPoint].operands.push(OperandVariableIds, "'Interface'");

    InstructionDesc[OpExelwtionMode].operands.push(OperandId, "'Entry Point'");
    InstructionDesc[OpExelwtionMode].operands.push(OperandExelwtionMode, "'Mode'");
    InstructionDesc[OpExelwtionMode].operands.push(OperandOptionalLiteral, "See <<Exelwtion_Mode,Exelwtion Mode>>");

    InstructionDesc[OpTypeInt].operands.push(OperandLiteralNumber, "'Width'");
    InstructionDesc[OpTypeInt].operands.push(OperandLiteralNumber, "'Signedness'");

    InstructionDesc[OpTypeFloat].operands.push(OperandLiteralNumber, "'Width'");

    InstructionDesc[OpTypeVector].operands.push(OperandId, "'Component Type'");
    InstructionDesc[OpTypeVector].operands.push(OperandLiteralNumber, "'Component Count'");

    InstructionDesc[OpTypeMatrix].operands.push(OperandId, "'Column Type'");
    InstructionDesc[OpTypeMatrix].operands.push(OperandLiteralNumber, "'Column Count'");

    InstructionDesc[OpTypeImage].operands.push(OperandId, "'Sampled Type'");
    InstructionDesc[OpTypeImage].operands.push(OperandDimensionality, "");
    InstructionDesc[OpTypeImage].operands.push(OperandLiteralNumber, "'Depth'");
    InstructionDesc[OpTypeImage].operands.push(OperandLiteralNumber, "'Arrayed'");
    InstructionDesc[OpTypeImage].operands.push(OperandLiteralNumber, "'MS'");
    InstructionDesc[OpTypeImage].operands.push(OperandLiteralNumber, "'Sampled'");
    InstructionDesc[OpTypeImage].operands.push(OperandSamplerImageFormat, "");
    InstructionDesc[OpTypeImage].operands.push(OperandAccessQualifier, "", true);

    InstructionDesc[OpTypeSampledImage].operands.push(OperandId, "'Image Type'");

    InstructionDesc[OpTypeArray].operands.push(OperandId, "'Element Type'");
    InstructionDesc[OpTypeArray].operands.push(OperandId, "'Length'");

    InstructionDesc[OpTypeRuntimeArray].operands.push(OperandId, "'Element Type'");

    InstructionDesc[OpTypeStruct].operands.push(OperandVariableIds, "'Member 0 type', +\n'member 1 type', +\n...");

    InstructionDesc[OpTypeOpaque].operands.push(OperandLiteralString, "The name of the opaque type.");

    InstructionDesc[OpTypePointer].operands.push(OperandStorage, "");
    InstructionDesc[OpTypePointer].operands.push(OperandId, "'Type'");

    InstructionDesc[OpTypeForwardPointer].operands.push(OperandId, "'Pointer Type'");
    InstructionDesc[OpTypeForwardPointer].operands.push(OperandStorage, "");

    InstructionDesc[OpTypePipe].operands.push(OperandAccessQualifier, "'Qualifier'");

    InstructionDesc[OpTypeFunction].operands.push(OperandId, "'Return Type'");
    InstructionDesc[OpTypeFunction].operands.push(OperandVariableIds, "'Parameter 0 Type', +\n'Parameter 1 Type', +\n...");

    InstructionDesc[OpConstant].operands.push(OperandVariableLiterals, "'Value'");

    InstructionDesc[OpConstantComposite].operands.push(OperandVariableIds, "'Constituents'");

    InstructionDesc[OpConstantSampler].operands.push(OperandSamplerAddressingMode, "");
    InstructionDesc[OpConstantSampler].operands.push(OperandLiteralNumber, "'Param'");
    InstructionDesc[OpConstantSampler].operands.push(OperandSamplerFilterMode, "");

    InstructionDesc[OpSpecConstant].operands.push(OperandVariableLiterals, "'Value'");

    InstructionDesc[OpSpecConstantComposite].operands.push(OperandVariableIds, "'Constituents'");

    InstructionDesc[OpSpecConstantOp].operands.push(OperandLiteralNumber, "'Opcode'");
    InstructionDesc[OpSpecConstantOp].operands.push(OperandVariableIds, "'Operands'");

    InstructionDesc[OpVariable].operands.push(OperandStorage, "");
    InstructionDesc[OpVariable].operands.push(OperandId, "'Initializer'", true);

    InstructionDesc[OpFunction].operands.push(OperandFunction, "");
    InstructionDesc[OpFunction].operands.push(OperandId, "'Function Type'");

    InstructionDesc[OpFunctionCall].operands.push(OperandId, "'Function'");
    InstructionDesc[OpFunctionCall].operands.push(OperandVariableIds, "'Argument 0', +\n'Argument 1', +\n...");

    InstructionDesc[OpExtInst].operands.push(OperandId, "'Set'");
    InstructionDesc[OpExtInst].operands.push(OperandLiteralNumber, "'Instruction'");
    InstructionDesc[OpExtInst].operands.push(OperandVariableIds, "'Operand 1', +\n'Operand 2', +\n...");

    InstructionDesc[OpLoad].operands.push(OperandId, "'Pointer'");
    InstructionDesc[OpLoad].operands.push(OperandMemoryAccess, "", true);
    InstructionDesc[OpLoad].operands.push(OperandLiteralNumber, "", true);
    InstructionDesc[OpLoad].operands.push(OperandId, "", true);

    InstructionDesc[OpStore].operands.push(OperandId, "'Pointer'");
    InstructionDesc[OpStore].operands.push(OperandId, "'Object'");
    InstructionDesc[OpStore].operands.push(OperandMemoryAccess, "", true);
    InstructionDesc[OpStore].operands.push(OperandLiteralNumber, "", true);
    InstructionDesc[OpStore].operands.push(OperandId, "", true);

    InstructionDesc[OpPhi].operands.push(OperandVariableIds, "'Variable, Parent, ...'");

    InstructionDesc[OpDecorate].operands.push(OperandId, "'Target'");
    InstructionDesc[OpDecorate].operands.push(OperandDecoration, "");
    InstructionDesc[OpDecorate].operands.push(OperandVariableLiterals, "See <<Decoration,'Decoration'>>.");

    InstructionDesc[OpDecorateId].operands.push(OperandId, "'Target'");
    InstructionDesc[OpDecorateId].operands.push(OperandDecoration, "");
    InstructionDesc[OpDecorateId].operands.push(OperandVariableIds, "See <<Decoration,'Decoration'>>.");

    InstructionDesc[OpDecorateStringGOOGLE].operands.push(OperandId, "'Target'");
    InstructionDesc[OpDecorateStringGOOGLE].operands.push(OperandDecoration, "");
    InstructionDesc[OpDecorateStringGOOGLE].operands.push(OperandLiteralString, "'Literal String'");

    InstructionDesc[OpMemberDecorate].operands.push(OperandId, "'Structure Type'");
    InstructionDesc[OpMemberDecorate].operands.push(OperandLiteralNumber, "'Member'");
    InstructionDesc[OpMemberDecorate].operands.push(OperandDecoration, "");
    InstructionDesc[OpMemberDecorate].operands.push(OperandVariableLiterals, "See <<Decoration,'Decoration'>>.");

    InstructionDesc[OpMemberDecorateStringGOOGLE].operands.push(OperandId, "'Structure Type'");
    InstructionDesc[OpMemberDecorateStringGOOGLE].operands.push(OperandLiteralNumber, "'Member'");
    InstructionDesc[OpMemberDecorateStringGOOGLE].operands.push(OperandDecoration, "");
    InstructionDesc[OpMemberDecorateStringGOOGLE].operands.push(OperandLiteralString, "'Literal String'");

    InstructionDesc[OpGroupDecorate].operands.push(OperandId, "'Decoration Group'");
    InstructionDesc[OpGroupDecorate].operands.push(OperandVariableIds, "'Targets'");

    InstructionDesc[OpGroupMemberDecorate].operands.push(OperandId, "'Decoration Group'");
    InstructionDesc[OpGroupMemberDecorate].operands.push(OperandVariableIdLiteral, "'Targets'");

    InstructionDesc[OpVectorExtractDynamic].operands.push(OperandId, "'Vector'");
    InstructionDesc[OpVectorExtractDynamic].operands.push(OperandId, "'Index'");

    InstructionDesc[OpVectorInsertDynamic].operands.push(OperandId, "'Vector'");
    InstructionDesc[OpVectorInsertDynamic].operands.push(OperandId, "'Component'");
    InstructionDesc[OpVectorInsertDynamic].operands.push(OperandId, "'Index'");

    InstructionDesc[OpVectorShuffle].operands.push(OperandId, "'Vector 1'");
    InstructionDesc[OpVectorShuffle].operands.push(OperandId, "'Vector 2'");
    InstructionDesc[OpVectorShuffle].operands.push(OperandVariableLiterals, "'Components'");

    InstructionDesc[OpCompositeConstruct].operands.push(OperandVariableIds, "'Constituents'");

    InstructionDesc[OpCompositeExtract].operands.push(OperandId, "'Composite'");
    InstructionDesc[OpCompositeExtract].operands.push(OperandVariableLiterals, "'Indexes'");

    InstructionDesc[OpCompositeInsert].operands.push(OperandId, "'Object'");
    InstructionDesc[OpCompositeInsert].operands.push(OperandId, "'Composite'");
    InstructionDesc[OpCompositeInsert].operands.push(OperandVariableLiterals, "'Indexes'");

    InstructionDesc[OpCopyObject].operands.push(OperandId, "'Operand'");

    InstructionDesc[OpCopyMemory].operands.push(OperandId, "'Target'");
    InstructionDesc[OpCopyMemory].operands.push(OperandId, "'Source'");
    InstructionDesc[OpCopyMemory].operands.push(OperandMemoryAccess, "", true);

    InstructionDesc[OpCopyMemorySized].operands.push(OperandId, "'Target'");
    InstructionDesc[OpCopyMemorySized].operands.push(OperandId, "'Source'");
    InstructionDesc[OpCopyMemorySized].operands.push(OperandId, "'Size'");
    InstructionDesc[OpCopyMemorySized].operands.push(OperandMemoryAccess, "", true);

    InstructionDesc[OpSampledImage].operands.push(OperandId, "'Image'");
    InstructionDesc[OpSampledImage].operands.push(OperandId, "'Sampler'");

    InstructionDesc[OpImage].operands.push(OperandId, "'Sampled Image'");

    InstructionDesc[OpImageRead].operands.push(OperandId, "'Image'");
    InstructionDesc[OpImageRead].operands.push(OperandId, "'Coordinate'");
    InstructionDesc[OpImageRead].operands.push(OperandImageOperands, "", true);
    InstructionDesc[OpImageRead].operands.push(OperandVariableIds, "", true);

    InstructionDesc[OpImageWrite].operands.push(OperandId, "'Image'");
    InstructionDesc[OpImageWrite].operands.push(OperandId, "'Coordinate'");
    InstructionDesc[OpImageWrite].operands.push(OperandId, "'Texel'");
    InstructionDesc[OpImageWrite].operands.push(OperandImageOperands, "", true);
    InstructionDesc[OpImageWrite].operands.push(OperandVariableIds, "", true);

    InstructionDesc[OpImageSampleImplicitLod].operands.push(OperandId, "'Sampled Image'");
    InstructionDesc[OpImageSampleImplicitLod].operands.push(OperandId, "'Coordinate'");
    InstructionDesc[OpImageSampleImplicitLod].operands.push(OperandImageOperands, "", true);
    InstructionDesc[OpImageSampleImplicitLod].operands.push(OperandVariableIds, "", true);

    InstructionDesc[OpImageSampleExplicitLod].operands.push(OperandId, "'Sampled Image'");
    InstructionDesc[OpImageSampleExplicitLod].operands.push(OperandId, "'Coordinate'");
    InstructionDesc[OpImageSampleExplicitLod].operands.push(OperandImageOperands, "", true);
    InstructionDesc[OpImageSampleExplicitLod].operands.push(OperandVariableIds, "", true);

    InstructionDesc[OpImageSampleDrefImplicitLod].operands.push(OperandId, "'Sampled Image'");
    InstructionDesc[OpImageSampleDrefImplicitLod].operands.push(OperandId, "'Coordinate'");
    InstructionDesc[OpImageSampleDrefImplicitLod].operands.push(OperandId, "'D~ref~'");
    InstructionDesc[OpImageSampleDrefImplicitLod].operands.push(OperandImageOperands, "", true);
    InstructionDesc[OpImageSampleDrefImplicitLod].operands.push(OperandVariableIds, "", true);

    InstructionDesc[OpImageSampleDrefExplicitLod].operands.push(OperandId, "'Sampled Image'");
    InstructionDesc[OpImageSampleDrefExplicitLod].operands.push(OperandId, "'Coordinate'");
    InstructionDesc[OpImageSampleDrefExplicitLod].operands.push(OperandId, "'D~ref~'");
    InstructionDesc[OpImageSampleDrefExplicitLod].operands.push(OperandImageOperands, "", true);
    InstructionDesc[OpImageSampleDrefExplicitLod].operands.push(OperandVariableIds, "", true);

    InstructionDesc[OpImageSampleProjImplicitLod].operands.push(OperandId, "'Sampled Image'");
    InstructionDesc[OpImageSampleProjImplicitLod].operands.push(OperandId, "'Coordinate'");
    InstructionDesc[OpImageSampleProjImplicitLod].operands.push(OperandImageOperands, "", true);
    InstructionDesc[OpImageSampleProjImplicitLod].operands.push(OperandVariableIds, "", true);

    InstructionDesc[OpImageSampleProjExplicitLod].operands.push(OperandId, "'Sampled Image'");
    InstructionDesc[OpImageSampleProjExplicitLod].operands.push(OperandId, "'Coordinate'");
    InstructionDesc[OpImageSampleProjExplicitLod].operands.push(OperandImageOperands, "", true);
    InstructionDesc[OpImageSampleProjExplicitLod].operands.push(OperandVariableIds, "", true);

    InstructionDesc[OpImageSampleProjDrefImplicitLod].operands.push(OperandId, "'Sampled Image'");
    InstructionDesc[OpImageSampleProjDrefImplicitLod].operands.push(OperandId, "'Coordinate'");
    InstructionDesc[OpImageSampleProjDrefImplicitLod].operands.push(OperandId, "'D~ref~'");
    InstructionDesc[OpImageSampleProjDrefImplicitLod].operands.push(OperandImageOperands, "", true);
    InstructionDesc[OpImageSampleProjDrefImplicitLod].operands.push(OperandVariableIds, "", true);

    InstructionDesc[OpImageSampleProjDrefExplicitLod].operands.push(OperandId, "'Sampled Image'");
    InstructionDesc[OpImageSampleProjDrefExplicitLod].operands.push(OperandId, "'Coordinate'");
    InstructionDesc[OpImageSampleProjDrefExplicitLod].operands.push(OperandId, "'D~ref~'");
    InstructionDesc[OpImageSampleProjDrefExplicitLod].operands.push(OperandImageOperands, "", true);
    InstructionDesc[OpImageSampleProjDrefExplicitLod].operands.push(OperandVariableIds, "", true);

    InstructionDesc[OpImageFetch].operands.push(OperandId, "'Image'");
    InstructionDesc[OpImageFetch].operands.push(OperandId, "'Coordinate'");
    InstructionDesc[OpImageFetch].operands.push(OperandImageOperands, "", true);
    InstructionDesc[OpImageFetch].operands.push(OperandVariableIds, "", true);

    InstructionDesc[OpImageGather].operands.push(OperandId, "'Sampled Image'");
    InstructionDesc[OpImageGather].operands.push(OperandId, "'Coordinate'");
    InstructionDesc[OpImageGather].operands.push(OperandId, "'Component'");
    InstructionDesc[OpImageGather].operands.push(OperandImageOperands, "", true);
    InstructionDesc[OpImageGather].operands.push(OperandVariableIds, "", true);

    InstructionDesc[OpImageDrefGather].operands.push(OperandId, "'Sampled Image'");
    InstructionDesc[OpImageDrefGather].operands.push(OperandId, "'Coordinate'");
    InstructionDesc[OpImageDrefGather].operands.push(OperandId, "'D~ref~'");
    InstructionDesc[OpImageDrefGather].operands.push(OperandImageOperands, "", true);
    InstructionDesc[OpImageDrefGather].operands.push(OperandVariableIds, "", true);

    InstructionDesc[OpImageSparseSampleImplicitLod].operands.push(OperandId, "'Sampled Image'");
    InstructionDesc[OpImageSparseSampleImplicitLod].operands.push(OperandId, "'Coordinate'");
    InstructionDesc[OpImageSparseSampleImplicitLod].operands.push(OperandImageOperands, "", true);
    InstructionDesc[OpImageSparseSampleImplicitLod].operands.push(OperandVariableIds, "", true);

    InstructionDesc[OpImageSparseSampleExplicitLod].operands.push(OperandId, "'Sampled Image'");
    InstructionDesc[OpImageSparseSampleExplicitLod].operands.push(OperandId, "'Coordinate'");
    InstructionDesc[OpImageSparseSampleExplicitLod].operands.push(OperandImageOperands, "", true);
    InstructionDesc[OpImageSparseSampleExplicitLod].operands.push(OperandVariableIds, "", true);

    InstructionDesc[OpImageSparseSampleDrefImplicitLod].operands.push(OperandId, "'Sampled Image'");
    InstructionDesc[OpImageSparseSampleDrefImplicitLod].operands.push(OperandId, "'Coordinate'");
    InstructionDesc[OpImageSparseSampleDrefImplicitLod].operands.push(OperandId, "'D~ref~'");
    InstructionDesc[OpImageSparseSampleDrefImplicitLod].operands.push(OperandImageOperands, "", true);
    InstructionDesc[OpImageSparseSampleDrefImplicitLod].operands.push(OperandVariableIds, "", true);

    InstructionDesc[OpImageSparseSampleDrefExplicitLod].operands.push(OperandId, "'Sampled Image'");
    InstructionDesc[OpImageSparseSampleDrefExplicitLod].operands.push(OperandId, "'Coordinate'");
    InstructionDesc[OpImageSparseSampleDrefExplicitLod].operands.push(OperandId, "'D~ref~'");
    InstructionDesc[OpImageSparseSampleDrefExplicitLod].operands.push(OperandImageOperands, "", true);
    InstructionDesc[OpImageSparseSampleDrefExplicitLod].operands.push(OperandVariableIds, "", true);

    InstructionDesc[OpImageSparseSampleProjImplicitLod].operands.push(OperandId, "'Sampled Image'");
    InstructionDesc[OpImageSparseSampleProjImplicitLod].operands.push(OperandId, "'Coordinate'");
    InstructionDesc[OpImageSparseSampleProjImplicitLod].operands.push(OperandImageOperands, "", true);
    InstructionDesc[OpImageSparseSampleProjImplicitLod].operands.push(OperandVariableIds, "", true);

    InstructionDesc[OpImageSparseSampleProjExplicitLod].operands.push(OperandId, "'Sampled Image'");
    InstructionDesc[OpImageSparseSampleProjExplicitLod].operands.push(OperandId, "'Coordinate'");
    InstructionDesc[OpImageSparseSampleProjExplicitLod].operands.push(OperandImageOperands, "", true);
    InstructionDesc[OpImageSparseSampleProjExplicitLod].operands.push(OperandVariableIds, "", true);

    InstructionDesc[OpImageSparseSampleProjDrefImplicitLod].operands.push(OperandId, "'Sampled Image'");
    InstructionDesc[OpImageSparseSampleProjDrefImplicitLod].operands.push(OperandId, "'Coordinate'");
    InstructionDesc[OpImageSparseSampleProjDrefImplicitLod].operands.push(OperandId, "'D~ref~'");
    InstructionDesc[OpImageSparseSampleProjDrefImplicitLod].operands.push(OperandImageOperands, "", true);
    InstructionDesc[OpImageSparseSampleProjDrefImplicitLod].operands.push(OperandVariableIds, "", true);

    InstructionDesc[OpImageSparseSampleProjDrefExplicitLod].operands.push(OperandId, "'Sampled Image'");
    InstructionDesc[OpImageSparseSampleProjDrefExplicitLod].operands.push(OperandId, "'Coordinate'");
    InstructionDesc[OpImageSparseSampleProjDrefExplicitLod].operands.push(OperandId, "'D~ref~'");
    InstructionDesc[OpImageSparseSampleProjDrefExplicitLod].operands.push(OperandImageOperands, "", true);
    InstructionDesc[OpImageSparseSampleProjDrefExplicitLod].operands.push(OperandVariableIds, "", true);

    InstructionDesc[OpImageSparseFetch].operands.push(OperandId, "'Image'");
    InstructionDesc[OpImageSparseFetch].operands.push(OperandId, "'Coordinate'");
    InstructionDesc[OpImageSparseFetch].operands.push(OperandImageOperands, "", true);
    InstructionDesc[OpImageSparseFetch].operands.push(OperandVariableIds, "", true);

    InstructionDesc[OpImageSparseGather].operands.push(OperandId, "'Sampled Image'");
    InstructionDesc[OpImageSparseGather].operands.push(OperandId, "'Coordinate'");
    InstructionDesc[OpImageSparseGather].operands.push(OperandId, "'Component'");
    InstructionDesc[OpImageSparseGather].operands.push(OperandImageOperands, "", true);
    InstructionDesc[OpImageSparseGather].operands.push(OperandVariableIds, "", true);

    InstructionDesc[OpImageSparseDrefGather].operands.push(OperandId, "'Sampled Image'");
    InstructionDesc[OpImageSparseDrefGather].operands.push(OperandId, "'Coordinate'");
    InstructionDesc[OpImageSparseDrefGather].operands.push(OperandId, "'D~ref~'");
    InstructionDesc[OpImageSparseDrefGather].operands.push(OperandImageOperands, "", true);
    InstructionDesc[OpImageSparseDrefGather].operands.push(OperandVariableIds, "", true);

    InstructionDesc[OpImageSparseRead].operands.push(OperandId, "'Image'");
    InstructionDesc[OpImageSparseRead].operands.push(OperandId, "'Coordinate'");
    InstructionDesc[OpImageSparseRead].operands.push(OperandImageOperands, "", true);
    InstructionDesc[OpImageSparseRead].operands.push(OperandVariableIds, "", true);

    InstructionDesc[OpImageSparseTexelsResident].operands.push(OperandId, "'Resident Code'");

    InstructionDesc[OpImageQuerySizeLod].operands.push(OperandId, "'Image'");
    InstructionDesc[OpImageQuerySizeLod].operands.push(OperandId, "'Level of Detail'");

    InstructionDesc[OpImageQuerySize].operands.push(OperandId, "'Image'");

    InstructionDesc[OpImageQueryLod].operands.push(OperandId, "'Image'");
    InstructionDesc[OpImageQueryLod].operands.push(OperandId, "'Coordinate'");

    InstructionDesc[OpImageQueryLevels].operands.push(OperandId, "'Image'");

    InstructionDesc[OpImageQuerySamples].operands.push(OperandId, "'Image'");

    InstructionDesc[OpImageQueryFormat].operands.push(OperandId, "'Image'");

    InstructionDesc[OpImageQueryOrder].operands.push(OperandId, "'Image'");

    InstructionDesc[OpAccessChain].operands.push(OperandId, "'Base'");
    InstructionDesc[OpAccessChain].operands.push(OperandVariableIds, "'Indexes'");

    InstructionDesc[OpInBoundsAccessChain].operands.push(OperandId, "'Base'");
    InstructionDesc[OpInBoundsAccessChain].operands.push(OperandVariableIds, "'Indexes'");

    InstructionDesc[OpPtrAccessChain].operands.push(OperandId, "'Base'");
    InstructionDesc[OpPtrAccessChain].operands.push(OperandId, "'Element'");
    InstructionDesc[OpPtrAccessChain].operands.push(OperandVariableIds, "'Indexes'");

    InstructionDesc[OpInBoundsPtrAccessChain].operands.push(OperandId, "'Base'");
    InstructionDesc[OpInBoundsPtrAccessChain].operands.push(OperandId, "'Element'");
    InstructionDesc[OpInBoundsPtrAccessChain].operands.push(OperandVariableIds, "'Indexes'");

    InstructionDesc[OpSNegate].operands.push(OperandId, "'Operand'");

    InstructionDesc[OpFNegate].operands.push(OperandId, "'Operand'");

    InstructionDesc[OpNot].operands.push(OperandId, "'Operand'");

    InstructionDesc[OpAny].operands.push(OperandId, "'Vector'");

    InstructionDesc[OpAll].operands.push(OperandId, "'Vector'");

    InstructionDesc[OpColwertFToU].operands.push(OperandId, "'Float Value'");

    InstructionDesc[OpColwertFToS].operands.push(OperandId, "'Float Value'");

    InstructionDesc[OpColwertSToF].operands.push(OperandId, "'Signed Value'");

    InstructionDesc[OpColwertUToF].operands.push(OperandId, "'Unsigned Value'");

    InstructionDesc[OpUColwert].operands.push(OperandId, "'Unsigned Value'");

    InstructionDesc[OpSColwert].operands.push(OperandId, "'Signed Value'");

    InstructionDesc[OpFColwert].operands.push(OperandId, "'Float Value'");

    InstructionDesc[OpSatColwertSToU].operands.push(OperandId, "'Signed Value'");

    InstructionDesc[OpSatColwertUToS].operands.push(OperandId, "'Unsigned Value'");

    InstructionDesc[OpColwertPtrToU].operands.push(OperandId, "'Pointer'");

    InstructionDesc[OpColwertUToPtr].operands.push(OperandId, "'Integer Value'");

    InstructionDesc[OpPtrCastToGeneric].operands.push(OperandId, "'Pointer'");

    InstructionDesc[OpGenericCastToPtr].operands.push(OperandId, "'Pointer'");

    InstructionDesc[OpGenericCastToPtrExplicit].operands.push(OperandId, "'Pointer'");
    InstructionDesc[OpGenericCastToPtrExplicit].operands.push(OperandStorage, "'Storage'");

    InstructionDesc[OpGenericPtrMemSemantics].operands.push(OperandId, "'Pointer'");

    InstructionDesc[OpBitcast].operands.push(OperandId, "'Operand'");

    InstructionDesc[OpQuantizeToF16].operands.push(OperandId, "'Value'");

    InstructionDesc[OpTranspose].operands.push(OperandId, "'Matrix'");

    InstructionDesc[OpCopyLogical].operands.push(OperandId, "'Operand'");

    InstructionDesc[OpIsNan].operands.push(OperandId, "'x'");

    InstructionDesc[OpIsInf].operands.push(OperandId, "'x'");

    InstructionDesc[OpIsFinite].operands.push(OperandId, "'x'");

    InstructionDesc[OpIsNormal].operands.push(OperandId, "'x'");

    InstructionDesc[OpSignBitSet].operands.push(OperandId, "'x'");

    InstructionDesc[OpLessOrGreater].operands.push(OperandId, "'x'");
    InstructionDesc[OpLessOrGreater].operands.push(OperandId, "'y'");

    InstructionDesc[OpOrdered].operands.push(OperandId, "'x'");
    InstructionDesc[OpOrdered].operands.push(OperandId, "'y'");

    InstructionDesc[OpUnordered].operands.push(OperandId, "'x'");
    InstructionDesc[OpUnordered].operands.push(OperandId, "'y'");

    InstructionDesc[OpArrayLength].operands.push(OperandId, "'Structure'");
    InstructionDesc[OpArrayLength].operands.push(OperandLiteralNumber, "'Array member'");

    InstructionDesc[OpIAdd].operands.push(OperandId, "'Operand 1'");
    InstructionDesc[OpIAdd].operands.push(OperandId, "'Operand 2'");

    InstructionDesc[OpFAdd].operands.push(OperandId, "'Operand 1'");
    InstructionDesc[OpFAdd].operands.push(OperandId, "'Operand 2'");

    InstructionDesc[OpISub].operands.push(OperandId, "'Operand 1'");
    InstructionDesc[OpISub].operands.push(OperandId, "'Operand 2'");

    InstructionDesc[OpFSub].operands.push(OperandId, "'Operand 1'");
    InstructionDesc[OpFSub].operands.push(OperandId, "'Operand 2'");

    InstructionDesc[OpIMul].operands.push(OperandId, "'Operand 1'");
    InstructionDesc[OpIMul].operands.push(OperandId, "'Operand 2'");

    InstructionDesc[OpFMul].operands.push(OperandId, "'Operand 1'");
    InstructionDesc[OpFMul].operands.push(OperandId, "'Operand 2'");

    InstructionDesc[OpUDiv].operands.push(OperandId, "'Operand 1'");
    InstructionDesc[OpUDiv].operands.push(OperandId, "'Operand 2'");

    InstructionDesc[OpSDiv].operands.push(OperandId, "'Operand 1'");
    InstructionDesc[OpSDiv].operands.push(OperandId, "'Operand 2'");

    InstructionDesc[OpFDiv].operands.push(OperandId, "'Operand 1'");
    InstructionDesc[OpFDiv].operands.push(OperandId, "'Operand 2'");

    InstructionDesc[OpUMod].operands.push(OperandId, "'Operand 1'");
    InstructionDesc[OpUMod].operands.push(OperandId, "'Operand 2'");

    InstructionDesc[OpSRem].operands.push(OperandId, "'Operand 1'");
    InstructionDesc[OpSRem].operands.push(OperandId, "'Operand 2'");

    InstructionDesc[OpSMod].operands.push(OperandId, "'Operand 1'");
    InstructionDesc[OpSMod].operands.push(OperandId, "'Operand 2'");

    InstructionDesc[OpFRem].operands.push(OperandId, "'Operand 1'");
    InstructionDesc[OpFRem].operands.push(OperandId, "'Operand 2'");

    InstructionDesc[OpFMod].operands.push(OperandId, "'Operand 1'");
    InstructionDesc[OpFMod].operands.push(OperandId, "'Operand 2'");

    InstructionDesc[OpVectorTimesScalar].operands.push(OperandId, "'Vector'");
    InstructionDesc[OpVectorTimesScalar].operands.push(OperandId, "'Scalar'");

    InstructionDesc[OpMatrixTimesScalar].operands.push(OperandId, "'Matrix'");
    InstructionDesc[OpMatrixTimesScalar].operands.push(OperandId, "'Scalar'");

    InstructionDesc[OpVectorTimesMatrix].operands.push(OperandId, "'Vector'");
    InstructionDesc[OpVectorTimesMatrix].operands.push(OperandId, "'Matrix'");

    InstructionDesc[OpMatrixTimesVector].operands.push(OperandId, "'Matrix'");
    InstructionDesc[OpMatrixTimesVector].operands.push(OperandId, "'Vector'");

    InstructionDesc[OpMatrixTimesMatrix].operands.push(OperandId, "'LeftMatrix'");
    InstructionDesc[OpMatrixTimesMatrix].operands.push(OperandId, "'RightMatrix'");

    InstructionDesc[OpOuterProduct].operands.push(OperandId, "'Vector 1'");
    InstructionDesc[OpOuterProduct].operands.push(OperandId, "'Vector 2'");

    InstructionDesc[OpDot].operands.push(OperandId, "'Vector 1'");
    InstructionDesc[OpDot].operands.push(OperandId, "'Vector 2'");

    InstructionDesc[OpIAddCarry].operands.push(OperandId, "'Operand 1'");
    InstructionDesc[OpIAddCarry].operands.push(OperandId, "'Operand 2'");

    InstructionDesc[OpISubBorrow].operands.push(OperandId, "'Operand 1'");
    InstructionDesc[OpISubBorrow].operands.push(OperandId, "'Operand 2'");

    InstructionDesc[OpUMulExtended].operands.push(OperandId, "'Operand 1'");
    InstructionDesc[OpUMulExtended].operands.push(OperandId, "'Operand 2'");

    InstructionDesc[OpSMulExtended].operands.push(OperandId, "'Operand 1'");
    InstructionDesc[OpSMulExtended].operands.push(OperandId, "'Operand 2'");

    InstructionDesc[OpShiftRightLogical].operands.push(OperandId, "'Base'");
    InstructionDesc[OpShiftRightLogical].operands.push(OperandId, "'Shift'");

    InstructionDesc[OpShiftRightArithmetic].operands.push(OperandId, "'Base'");
    InstructionDesc[OpShiftRightArithmetic].operands.push(OperandId, "'Shift'");

    InstructionDesc[OpShiftLeftLogical].operands.push(OperandId, "'Base'");
    InstructionDesc[OpShiftLeftLogical].operands.push(OperandId, "'Shift'");

    InstructionDesc[OpLogicalOr].operands.push(OperandId, "'Operand 1'");
    InstructionDesc[OpLogicalOr].operands.push(OperandId, "'Operand 2'");

    InstructionDesc[OpLogicalAnd].operands.push(OperandId, "'Operand 1'");
    InstructionDesc[OpLogicalAnd].operands.push(OperandId, "'Operand 2'");

    InstructionDesc[OpLogicalEqual].operands.push(OperandId, "'Operand 1'");
    InstructionDesc[OpLogicalEqual].operands.push(OperandId, "'Operand 2'");

    InstructionDesc[OpLogicalNotEqual].operands.push(OperandId, "'Operand 1'");
    InstructionDesc[OpLogicalNotEqual].operands.push(OperandId, "'Operand 2'");

    InstructionDesc[OpLogicalNot].operands.push(OperandId, "'Operand'");

    InstructionDesc[OpBitwiseOr].operands.push(OperandId, "'Operand 1'");
    InstructionDesc[OpBitwiseOr].operands.push(OperandId, "'Operand 2'");

    InstructionDesc[OpBitwiseXor].operands.push(OperandId, "'Operand 1'");
    InstructionDesc[OpBitwiseXor].operands.push(OperandId, "'Operand 2'");

    InstructionDesc[OpBitwiseAnd].operands.push(OperandId, "'Operand 1'");
    InstructionDesc[OpBitwiseAnd].operands.push(OperandId, "'Operand 2'");

    InstructionDesc[OpBitFieldInsert].operands.push(OperandId, "'Base'");
    InstructionDesc[OpBitFieldInsert].operands.push(OperandId, "'Insert'");
    InstructionDesc[OpBitFieldInsert].operands.push(OperandId, "'Offset'");
    InstructionDesc[OpBitFieldInsert].operands.push(OperandId, "'Count'");

    InstructionDesc[OpBitFieldSExtract].operands.push(OperandId, "'Base'");
    InstructionDesc[OpBitFieldSExtract].operands.push(OperandId, "'Offset'");
    InstructionDesc[OpBitFieldSExtract].operands.push(OperandId, "'Count'");
    
    InstructionDesc[OpBitFieldUExtract].operands.push(OperandId, "'Base'");
    InstructionDesc[OpBitFieldUExtract].operands.push(OperandId, "'Offset'");
    InstructionDesc[OpBitFieldUExtract].operands.push(OperandId, "'Count'");
    
    InstructionDesc[OpBitReverse].operands.push(OperandId, "'Base'");

    InstructionDesc[OpBitCount].operands.push(OperandId, "'Base'");

    InstructionDesc[OpSelect].operands.push(OperandId, "'Condition'");
    InstructionDesc[OpSelect].operands.push(OperandId, "'Object 1'");
    InstructionDesc[OpSelect].operands.push(OperandId, "'Object 2'");

    InstructionDesc[OpIEqual].operands.push(OperandId, "'Operand 1'");
    InstructionDesc[OpIEqual].operands.push(OperandId, "'Operand 2'");

    InstructionDesc[OpFOrdEqual].operands.push(OperandId, "'Operand 1'");
    InstructionDesc[OpFOrdEqual].operands.push(OperandId, "'Operand 2'");

    InstructionDesc[OpFUnordEqual].operands.push(OperandId, "'Operand 1'");
    InstructionDesc[OpFUnordEqual].operands.push(OperandId, "'Operand 2'");

    InstructionDesc[OpINotEqual].operands.push(OperandId, "'Operand 1'");
    InstructionDesc[OpINotEqual].operands.push(OperandId, "'Operand 2'");

    InstructionDesc[OpFOrdNotEqual].operands.push(OperandId, "'Operand 1'");
    InstructionDesc[OpFOrdNotEqual].operands.push(OperandId, "'Operand 2'");

    InstructionDesc[OpFUnordNotEqual].operands.push(OperandId, "'Operand 1'");
    InstructionDesc[OpFUnordNotEqual].operands.push(OperandId, "'Operand 2'");

    InstructionDesc[OpULessThan].operands.push(OperandId, "'Operand 1'");
    InstructionDesc[OpULessThan].operands.push(OperandId, "'Operand 2'");

    InstructionDesc[OpSLessThan].operands.push(OperandId, "'Operand 1'");
    InstructionDesc[OpSLessThan].operands.push(OperandId, "'Operand 2'");

    InstructionDesc[OpFOrdLessThan].operands.push(OperandId, "'Operand 1'");
    InstructionDesc[OpFOrdLessThan].operands.push(OperandId, "'Operand 2'");

    InstructionDesc[OpFUnordLessThan].operands.push(OperandId, "'Operand 1'");
    InstructionDesc[OpFUnordLessThan].operands.push(OperandId, "'Operand 2'");

    InstructionDesc[OpUGreaterThan].operands.push(OperandId, "'Operand 1'");
    InstructionDesc[OpUGreaterThan].operands.push(OperandId, "'Operand 2'");

    InstructionDesc[OpSGreaterThan].operands.push(OperandId, "'Operand 1'");
    InstructionDesc[OpSGreaterThan].operands.push(OperandId, "'Operand 2'");

    InstructionDesc[OpFOrdGreaterThan].operands.push(OperandId, "'Operand 1'");
    InstructionDesc[OpFOrdGreaterThan].operands.push(OperandId, "'Operand 2'");

    InstructionDesc[OpFUnordGreaterThan].operands.push(OperandId, "'Operand 1'");
    InstructionDesc[OpFUnordGreaterThan].operands.push(OperandId, "'Operand 2'");

    InstructionDesc[OpULessThanEqual].operands.push(OperandId, "'Operand 1'");
    InstructionDesc[OpULessThanEqual].operands.push(OperandId, "'Operand 2'");

    InstructionDesc[OpSLessThanEqual].operands.push(OperandId, "'Operand 1'");
    InstructionDesc[OpSLessThanEqual].operands.push(OperandId, "'Operand 2'");

    InstructionDesc[OpFOrdLessThanEqual].operands.push(OperandId, "'Operand 1'");
    InstructionDesc[OpFOrdLessThanEqual].operands.push(OperandId, "'Operand 2'");

    InstructionDesc[OpFUnordLessThanEqual].operands.push(OperandId, "'Operand 1'");
    InstructionDesc[OpFUnordLessThanEqual].operands.push(OperandId, "'Operand 2'");

    InstructionDesc[OpUGreaterThanEqual].operands.push(OperandId, "'Operand 1'");
    InstructionDesc[OpUGreaterThanEqual].operands.push(OperandId, "'Operand 2'");

    InstructionDesc[OpSGreaterThanEqual].operands.push(OperandId, "'Operand 1'");
    InstructionDesc[OpSGreaterThanEqual].operands.push(OperandId, "'Operand 2'");

    InstructionDesc[OpFOrdGreaterThanEqual].operands.push(OperandId, "'Operand 1'");
    InstructionDesc[OpFOrdGreaterThanEqual].operands.push(OperandId, "'Operand 2'");

    InstructionDesc[OpFUnordGreaterThanEqual].operands.push(OperandId, "'Operand 1'");
    InstructionDesc[OpFUnordGreaterThanEqual].operands.push(OperandId, "'Operand 2'");

    InstructionDesc[OpDPdx].operands.push(OperandId, "'P'");

    InstructionDesc[OpDPdy].operands.push(OperandId, "'P'");

    InstructionDesc[OpFwidth].operands.push(OperandId, "'P'");

    InstructionDesc[OpDPdxFine].operands.push(OperandId, "'P'");

    InstructionDesc[OpDPdyFine].operands.push(OperandId, "'P'");

    InstructionDesc[OpFwidthFine].operands.push(OperandId, "'P'");

    InstructionDesc[OpDPdxCoarse].operands.push(OperandId, "'P'");

    InstructionDesc[OpDPdyCoarse].operands.push(OperandId, "'P'");

    InstructionDesc[OpFwidthCoarse].operands.push(OperandId, "'P'");

    InstructionDesc[OpEmitStreamVertex].operands.push(OperandId, "'Stream'");

    InstructionDesc[OpEndStreamPrimitive].operands.push(OperandId, "'Stream'");

    InstructionDesc[OpControlBarrier].operands.push(OperandScope, "'Exelwtion'");
    InstructionDesc[OpControlBarrier].operands.push(OperandScope, "'Memory'");
    InstructionDesc[OpControlBarrier].operands.push(OperandMemorySemantics, "'Semantics'");

    InstructionDesc[OpMemoryBarrier].operands.push(OperandScope, "'Memory'");
    InstructionDesc[OpMemoryBarrier].operands.push(OperandMemorySemantics, "'Semantics'");

    InstructionDesc[OpImageTexelPointer].operands.push(OperandId, "'Image'");
    InstructionDesc[OpImageTexelPointer].operands.push(OperandId, "'Coordinate'");
    InstructionDesc[OpImageTexelPointer].operands.push(OperandId, "'Sample'");

    InstructionDesc[OpAtomicLoad].operands.push(OperandId, "'Pointer'");
    InstructionDesc[OpAtomicLoad].operands.push(OperandScope, "'Scope'");
    InstructionDesc[OpAtomicLoad].operands.push(OperandMemorySemantics, "'Semantics'");

    InstructionDesc[OpAtomicStore].operands.push(OperandId, "'Pointer'");
    InstructionDesc[OpAtomicStore].operands.push(OperandScope, "'Scope'");
    InstructionDesc[OpAtomicStore].operands.push(OperandMemorySemantics, "'Semantics'");
    InstructionDesc[OpAtomicStore].operands.push(OperandId, "'Value'");

    InstructionDesc[OpAtomicExchange].operands.push(OperandId, "'Pointer'");
    InstructionDesc[OpAtomicExchange].operands.push(OperandScope, "'Scope'");
    InstructionDesc[OpAtomicExchange].operands.push(OperandMemorySemantics, "'Semantics'");
    InstructionDesc[OpAtomicExchange].operands.push(OperandId, "'Value'");

    InstructionDesc[OpAtomicCompareExchange].operands.push(OperandId, "'Pointer'");
    InstructionDesc[OpAtomicCompareExchange].operands.push(OperandScope, "'Scope'");
    InstructionDesc[OpAtomicCompareExchange].operands.push(OperandMemorySemantics, "'Equal'");
    InstructionDesc[OpAtomicCompareExchange].operands.push(OperandMemorySemantics, "'Unequal'");
    InstructionDesc[OpAtomicCompareExchange].operands.push(OperandId, "'Value'");
    InstructionDesc[OpAtomicCompareExchange].operands.push(OperandId, "'Comparator'");

    InstructionDesc[OpAtomicCompareExchangeWeak].operands.push(OperandId, "'Pointer'");
    InstructionDesc[OpAtomicCompareExchangeWeak].operands.push(OperandScope, "'Scope'");
    InstructionDesc[OpAtomicCompareExchangeWeak].operands.push(OperandMemorySemantics, "'Equal'");
    InstructionDesc[OpAtomicCompareExchangeWeak].operands.push(OperandMemorySemantics, "'Unequal'");
    InstructionDesc[OpAtomicCompareExchangeWeak].operands.push(OperandId, "'Value'");
    InstructionDesc[OpAtomicCompareExchangeWeak].operands.push(OperandId, "'Comparator'");

    InstructionDesc[OpAtomicIIncrement].operands.push(OperandId, "'Pointer'");
    InstructionDesc[OpAtomicIIncrement].operands.push(OperandScope, "'Scope'");
    InstructionDesc[OpAtomicIIncrement].operands.push(OperandMemorySemantics, "'Semantics'");

    InstructionDesc[OpAtomicIDecrement].operands.push(OperandId, "'Pointer'");
    InstructionDesc[OpAtomicIDecrement].operands.push(OperandScope, "'Scope'");
    InstructionDesc[OpAtomicIDecrement].operands.push(OperandMemorySemantics, "'Semantics'");

    InstructionDesc[OpAtomicIAdd].operands.push(OperandId, "'Pointer'");
    InstructionDesc[OpAtomicIAdd].operands.push(OperandScope, "'Scope'");
    InstructionDesc[OpAtomicIAdd].operands.push(OperandMemorySemantics, "'Semantics'");
    InstructionDesc[OpAtomicIAdd].operands.push(OperandId, "'Value'");

    InstructionDesc[OpAtomicISub].operands.push(OperandId, "'Pointer'");
    InstructionDesc[OpAtomicISub].operands.push(OperandScope, "'Scope'");
    InstructionDesc[OpAtomicISub].operands.push(OperandMemorySemantics, "'Semantics'");
    InstructionDesc[OpAtomicISub].operands.push(OperandId, "'Value'");

    InstructionDesc[OpAtomilwMin].operands.push(OperandId, "'Pointer'");
    InstructionDesc[OpAtomilwMin].operands.push(OperandScope, "'Scope'");
    InstructionDesc[OpAtomilwMin].operands.push(OperandMemorySemantics, "'Semantics'");
    InstructionDesc[OpAtomilwMin].operands.push(OperandId, "'Value'");

    InstructionDesc[OpAtomilwMax].operands.push(OperandId, "'Pointer'");
    InstructionDesc[OpAtomilwMax].operands.push(OperandScope, "'Scope'");
    InstructionDesc[OpAtomilwMax].operands.push(OperandMemorySemantics, "'Semantics'");
    InstructionDesc[OpAtomilwMax].operands.push(OperandId, "'Value'");

    InstructionDesc[OpAtomicSMin].operands.push(OperandId, "'Pointer'");
    InstructionDesc[OpAtomicSMin].operands.push(OperandScope, "'Scope'");
    InstructionDesc[OpAtomicSMin].operands.push(OperandMemorySemantics, "'Semantics'");
    InstructionDesc[OpAtomicSMin].operands.push(OperandId, "'Value'");

    InstructionDesc[OpAtomicSMax].operands.push(OperandId, "'Pointer'");
    InstructionDesc[OpAtomicSMax].operands.push(OperandScope, "'Scope'");
    InstructionDesc[OpAtomicSMax].operands.push(OperandMemorySemantics, "'Semantics'");
    InstructionDesc[OpAtomicSMax].operands.push(OperandId, "'Value'");

    InstructionDesc[OpAtomicAnd].operands.push(OperandId, "'Pointer'");
    InstructionDesc[OpAtomicAnd].operands.push(OperandScope, "'Scope'");
    InstructionDesc[OpAtomicAnd].operands.push(OperandMemorySemantics, "'Semantics'");
    InstructionDesc[OpAtomicAnd].operands.push(OperandId, "'Value'");

    InstructionDesc[OpAtomicOr].operands.push(OperandId, "'Pointer'");
    InstructionDesc[OpAtomicOr].operands.push(OperandScope, "'Scope'");
    InstructionDesc[OpAtomicOr].operands.push(OperandMemorySemantics, "'Semantics'");
    InstructionDesc[OpAtomicOr].operands.push(OperandId, "'Value'");

    InstructionDesc[OpAtomicXor].operands.push(OperandId, "'Pointer'");
    InstructionDesc[OpAtomicXor].operands.push(OperandScope, "'Scope'");
    InstructionDesc[OpAtomicXor].operands.push(OperandMemorySemantics, "'Semantics'");
    InstructionDesc[OpAtomicXor].operands.push(OperandId, "'Value'");

    InstructionDesc[OpAtomicFlagTestAndSet].operands.push(OperandId, "'Pointer'");
    InstructionDesc[OpAtomicFlagTestAndSet].operands.push(OperandScope, "'Scope'");
    InstructionDesc[OpAtomicFlagTestAndSet].operands.push(OperandMemorySemantics, "'Semantics'");

    InstructionDesc[OpAtomicFlagClear].operands.push(OperandId, "'Pointer'");
    InstructionDesc[OpAtomicFlagClear].operands.push(OperandScope, "'Scope'");
    InstructionDesc[OpAtomicFlagClear].operands.push(OperandMemorySemantics, "'Semantics'");

    InstructionDesc[OpLoopMerge].operands.push(OperandId, "'Merge Block'");
    InstructionDesc[OpLoopMerge].operands.push(OperandId, "'Continue Target'");
    InstructionDesc[OpLoopMerge].operands.push(OperandLoop, "");
    InstructionDesc[OpLoopMerge].operands.push(OperandOptionalLiteral, "");

    InstructionDesc[OpSelectionMerge].operands.push(OperandId, "'Merge Block'");
    InstructionDesc[OpSelectionMerge].operands.push(OperandSelect, "");

    InstructionDesc[OpBranch].operands.push(OperandId, "'Target Label'");

    InstructionDesc[OpBranchConditional].operands.push(OperandId, "'Condition'");
    InstructionDesc[OpBranchConditional].operands.push(OperandId, "'True Label'");
    InstructionDesc[OpBranchConditional].operands.push(OperandId, "'False Label'");
    InstructionDesc[OpBranchConditional].operands.push(OperandVariableLiterals, "'Branch weights'");

    InstructionDesc[OpSwitch].operands.push(OperandId, "'Selector'");
    InstructionDesc[OpSwitch].operands.push(OperandId, "'Default'");
    InstructionDesc[OpSwitch].operands.push(OperandVariableLiteralId, "'Target'");


    InstructionDesc[OpReturlwalue].operands.push(OperandId, "'Value'");

    InstructionDesc[OpLifetimeStart].operands.push(OperandId, "'Pointer'");
    InstructionDesc[OpLifetimeStart].operands.push(OperandLiteralNumber, "'Size'");

    InstructionDesc[OpLifetimeStop].operands.push(OperandId, "'Pointer'");
    InstructionDesc[OpLifetimeStop].operands.push(OperandLiteralNumber, "'Size'");

    InstructionDesc[OpGroupAsyncCopy].operands.push(OperandScope, "'Exelwtion'");
    InstructionDesc[OpGroupAsyncCopy].operands.push(OperandId, "'Destination'");
    InstructionDesc[OpGroupAsyncCopy].operands.push(OperandId, "'Source'");
    InstructionDesc[OpGroupAsyncCopy].operands.push(OperandId, "'Num Elements'");
    InstructionDesc[OpGroupAsyncCopy].operands.push(OperandId, "'Stride'");
    InstructionDesc[OpGroupAsyncCopy].operands.push(OperandId, "'Event'");

    InstructionDesc[OpGroupWaitEvents].operands.push(OperandScope, "'Exelwtion'");
    InstructionDesc[OpGroupWaitEvents].operands.push(OperandId, "'Num Events'");
    InstructionDesc[OpGroupWaitEvents].operands.push(OperandId, "'Events List'");

    InstructionDesc[OpGroupAll].operands.push(OperandScope, "'Exelwtion'");
    InstructionDesc[OpGroupAll].operands.push(OperandId, "'Predicate'");

    InstructionDesc[OpGroupAny].operands.push(OperandScope, "'Exelwtion'");
    InstructionDesc[OpGroupAny].operands.push(OperandId, "'Predicate'");

    InstructionDesc[OpGroupBroadcast].operands.push(OperandScope, "'Exelwtion'");
    InstructionDesc[OpGroupBroadcast].operands.push(OperandId, "'Value'");
    InstructionDesc[OpGroupBroadcast].operands.push(OperandId, "'LocalId'");

    InstructionDesc[OpGroupIAdd].operands.push(OperandScope, "'Exelwtion'");
    InstructionDesc[OpGroupIAdd].operands.push(OperandGroupOperation, "'Operation'");
    InstructionDesc[OpGroupIAdd].operands.push(OperandId, "'X'");

    InstructionDesc[OpGroupFAdd].operands.push(OperandScope, "'Exelwtion'");
    InstructionDesc[OpGroupFAdd].operands.push(OperandGroupOperation, "'Operation'");
    InstructionDesc[OpGroupFAdd].operands.push(OperandId, "'X'");

    InstructionDesc[OpGroupUMin].operands.push(OperandScope, "'Exelwtion'");
    InstructionDesc[OpGroupUMin].operands.push(OperandGroupOperation, "'Operation'");
    InstructionDesc[OpGroupUMin].operands.push(OperandId, "'X'");

    InstructionDesc[OpGroupSMin].operands.push(OperandScope, "'Exelwtion'");
    InstructionDesc[OpGroupSMin].operands.push(OperandGroupOperation, "'Operation'");
    InstructionDesc[OpGroupSMin].operands.push(OperandId, "X");

    InstructionDesc[OpGroupFMin].operands.push(OperandScope, "'Exelwtion'");
    InstructionDesc[OpGroupFMin].operands.push(OperandGroupOperation, "'Operation'");
    InstructionDesc[OpGroupFMin].operands.push(OperandId, "X");

    InstructionDesc[OpGroupUMax].operands.push(OperandScope, "'Exelwtion'");
    InstructionDesc[OpGroupUMax].operands.push(OperandGroupOperation, "'Operation'");
    InstructionDesc[OpGroupUMax].operands.push(OperandId, "X");

    InstructionDesc[OpGroupSMax].operands.push(OperandScope, "'Exelwtion'");
    InstructionDesc[OpGroupSMax].operands.push(OperandGroupOperation, "'Operation'");
    InstructionDesc[OpGroupSMax].operands.push(OperandId, "X");

    InstructionDesc[OpGroupFMax].operands.push(OperandScope, "'Exelwtion'");
    InstructionDesc[OpGroupFMax].operands.push(OperandGroupOperation, "'Operation'");
    InstructionDesc[OpGroupFMax].operands.push(OperandId, "X");

    InstructionDesc[OpReadPipe].operands.push(OperandId, "'Pipe'");
    InstructionDesc[OpReadPipe].operands.push(OperandId, "'Pointer'");
    InstructionDesc[OpReadPipe].operands.push(OperandId, "'Packet Size'");
    InstructionDesc[OpReadPipe].operands.push(OperandId, "'Packet Alignment'");

    InstructionDesc[OpWritePipe].operands.push(OperandId, "'Pipe'");
    InstructionDesc[OpWritePipe].operands.push(OperandId, "'Pointer'");
    InstructionDesc[OpWritePipe].operands.push(OperandId, "'Packet Size'");
    InstructionDesc[OpWritePipe].operands.push(OperandId, "'Packet Alignment'");

    InstructionDesc[OpReservedReadPipe].operands.push(OperandId, "'Pipe'");
    InstructionDesc[OpReservedReadPipe].operands.push(OperandId, "'Reserve Id'");
    InstructionDesc[OpReservedReadPipe].operands.push(OperandId, "'Index'");
    InstructionDesc[OpReservedReadPipe].operands.push(OperandId, "'Pointer'");
    InstructionDesc[OpReservedReadPipe].operands.push(OperandId, "'Packet Size'");
    InstructionDesc[OpReservedReadPipe].operands.push(OperandId, "'Packet Alignment'");

    InstructionDesc[OpReservedWritePipe].operands.push(OperandId, "'Pipe'");
    InstructionDesc[OpReservedWritePipe].operands.push(OperandId, "'Reserve Id'");
    InstructionDesc[OpReservedWritePipe].operands.push(OperandId, "'Index'");
    InstructionDesc[OpReservedWritePipe].operands.push(OperandId, "'Pointer'");
    InstructionDesc[OpReservedWritePipe].operands.push(OperandId, "'Packet Size'");
    InstructionDesc[OpReservedWritePipe].operands.push(OperandId, "'Packet Alignment'");

    InstructionDesc[OpReserveReadPipePackets].operands.push(OperandId, "'Pipe'");
    InstructionDesc[OpReserveReadPipePackets].operands.push(OperandId, "'Num Packets'");
    InstructionDesc[OpReserveReadPipePackets].operands.push(OperandId, "'Packet Size'");
    InstructionDesc[OpReserveReadPipePackets].operands.push(OperandId, "'Packet Alignment'");

    InstructionDesc[OpReserveWritePipePackets].operands.push(OperandId, "'Pipe'");
    InstructionDesc[OpReserveWritePipePackets].operands.push(OperandId, "'Num Packets'");
    InstructionDesc[OpReserveWritePipePackets].operands.push(OperandId, "'Packet Size'");
    InstructionDesc[OpReserveWritePipePackets].operands.push(OperandId, "'Packet Alignment'");

    InstructionDesc[OpCommitReadPipe].operands.push(OperandId, "'Pipe'");
    InstructionDesc[OpCommitReadPipe].operands.push(OperandId, "'Reserve Id'");
    InstructionDesc[OpCommitReadPipe].operands.push(OperandId, "'Packet Size'");
    InstructionDesc[OpCommitReadPipe].operands.push(OperandId, "'Packet Alignment'");

    InstructionDesc[OpCommitWritePipe].operands.push(OperandId, "'Pipe'");
    InstructionDesc[OpCommitWritePipe].operands.push(OperandId, "'Reserve Id'");
    InstructionDesc[OpCommitWritePipe].operands.push(OperandId, "'Packet Size'");
    InstructionDesc[OpCommitWritePipe].operands.push(OperandId, "'Packet Alignment'");

    InstructionDesc[OpIsValidReserveId].operands.push(OperandId, "'Reserve Id'");

    InstructionDesc[OpGetNumPipePackets].operands.push(OperandId, "'Pipe'");
    InstructionDesc[OpGetNumPipePackets].operands.push(OperandId, "'Packet Size'");
    InstructionDesc[OpGetNumPipePackets].operands.push(OperandId, "'Packet Alignment'");

    InstructionDesc[OpGetMaxPipePackets].operands.push(OperandId, "'Pipe'");
    InstructionDesc[OpGetMaxPipePackets].operands.push(OperandId, "'Packet Size'");
    InstructionDesc[OpGetMaxPipePackets].operands.push(OperandId, "'Packet Alignment'");

    InstructionDesc[OpGroupReserveReadPipePackets].operands.push(OperandScope, "'Exelwtion'");
    InstructionDesc[OpGroupReserveReadPipePackets].operands.push(OperandId, "'Pipe'");
    InstructionDesc[OpGroupReserveReadPipePackets].operands.push(OperandId, "'Num Packets'");
    InstructionDesc[OpGroupReserveReadPipePackets].operands.push(OperandId, "'Packet Size'");
    InstructionDesc[OpGroupReserveReadPipePackets].operands.push(OperandId, "'Packet Alignment'");

    InstructionDesc[OpGroupReserveWritePipePackets].operands.push(OperandScope, "'Exelwtion'");
    InstructionDesc[OpGroupReserveWritePipePackets].operands.push(OperandId, "'Pipe'");
    InstructionDesc[OpGroupReserveWritePipePackets].operands.push(OperandId, "'Num Packets'");
    InstructionDesc[OpGroupReserveWritePipePackets].operands.push(OperandId, "'Packet Size'");
    InstructionDesc[OpGroupReserveWritePipePackets].operands.push(OperandId, "'Packet Alignment'");

    InstructionDesc[OpGroupCommitReadPipe].operands.push(OperandScope, "'Exelwtion'");
    InstructionDesc[OpGroupCommitReadPipe].operands.push(OperandId, "'Pipe'");
    InstructionDesc[OpGroupCommitReadPipe].operands.push(OperandId, "'Reserve Id'");
    InstructionDesc[OpGroupCommitReadPipe].operands.push(OperandId, "'Packet Size'");
    InstructionDesc[OpGroupCommitReadPipe].operands.push(OperandId, "'Packet Alignment'");

    InstructionDesc[OpGroupCommitWritePipe].operands.push(OperandScope, "'Exelwtion'");
    InstructionDesc[OpGroupCommitWritePipe].operands.push(OperandId, "'Pipe'");
    InstructionDesc[OpGroupCommitWritePipe].operands.push(OperandId, "'Reserve Id'");
    InstructionDesc[OpGroupCommitWritePipe].operands.push(OperandId, "'Packet Size'");
    InstructionDesc[OpGroupCommitWritePipe].operands.push(OperandId, "'Packet Alignment'");

    InstructionDesc[OpBuildNDRange].operands.push(OperandId, "'GlobalWorkSize'");
    InstructionDesc[OpBuildNDRange].operands.push(OperandId, "'LocalWorkSize'");
    InstructionDesc[OpBuildNDRange].operands.push(OperandId, "'GlobalWorkOffset'");

    InstructionDesc[OpCaptureEventProfilingInfo].operands.push(OperandId, "'Event'");
    InstructionDesc[OpCaptureEventProfilingInfo].operands.push(OperandId, "'Profiling Info'");
    InstructionDesc[OpCaptureEventProfilingInfo].operands.push(OperandId, "'Value'");

    InstructionDesc[OpSetUserEventStatus].operands.push(OperandId, "'Event'");
    InstructionDesc[OpSetUserEventStatus].operands.push(OperandId, "'Status'");

    InstructionDesc[OpIsValidEvent].operands.push(OperandId, "'Event'");

    InstructionDesc[OpRetainEvent].operands.push(OperandId, "'Event'");

    InstructionDesc[OpReleaseEvent].operands.push(OperandId, "'Event'");

    InstructionDesc[OpGetKernelWorkGroupSize].operands.push(OperandId, "'Ilwoke'");
    InstructionDesc[OpGetKernelWorkGroupSize].operands.push(OperandId, "'Param'");
    InstructionDesc[OpGetKernelWorkGroupSize].operands.push(OperandId, "'Param Size'");
    InstructionDesc[OpGetKernelWorkGroupSize].operands.push(OperandId, "'Param Align'");

    InstructionDesc[OpGetKernelPreferredWorkGroupSizeMultiple].operands.push(OperandId, "'Ilwoke'");
    InstructionDesc[OpGetKernelPreferredWorkGroupSizeMultiple].operands.push(OperandId, "'Param'");
    InstructionDesc[OpGetKernelPreferredWorkGroupSizeMultiple].operands.push(OperandId, "'Param Size'");
    InstructionDesc[OpGetKernelPreferredWorkGroupSizeMultiple].operands.push(OperandId, "'Param Align'");

    InstructionDesc[OpGetKernelNDrangeSubGroupCount].operands.push(OperandId, "'ND Range'");
    InstructionDesc[OpGetKernelNDrangeSubGroupCount].operands.push(OperandId, "'Ilwoke'");
    InstructionDesc[OpGetKernelNDrangeSubGroupCount].operands.push(OperandId, "'Param'");
    InstructionDesc[OpGetKernelNDrangeSubGroupCount].operands.push(OperandId, "'Param Size'");
    InstructionDesc[OpGetKernelNDrangeSubGroupCount].operands.push(OperandId, "'Param Align'");

    InstructionDesc[OpGetKernelNDrangeMaxSubGroupSize].operands.push(OperandId, "'ND Range'");
    InstructionDesc[OpGetKernelNDrangeMaxSubGroupSize].operands.push(OperandId, "'Ilwoke'");
    InstructionDesc[OpGetKernelNDrangeMaxSubGroupSize].operands.push(OperandId, "'Param'");
    InstructionDesc[OpGetKernelNDrangeMaxSubGroupSize].operands.push(OperandId, "'Param Size'");
    InstructionDesc[OpGetKernelNDrangeMaxSubGroupSize].operands.push(OperandId, "'Param Align'");

    InstructionDesc[OpEnqueueKernel].operands.push(OperandId, "'Queue'");
    InstructionDesc[OpEnqueueKernel].operands.push(OperandId, "'Flags'");
    InstructionDesc[OpEnqueueKernel].operands.push(OperandId, "'ND Range'");
    InstructionDesc[OpEnqueueKernel].operands.push(OperandId, "'Num Events'");
    InstructionDesc[OpEnqueueKernel].operands.push(OperandId, "'Wait Events'");
    InstructionDesc[OpEnqueueKernel].operands.push(OperandId, "'Ret Event'");
    InstructionDesc[OpEnqueueKernel].operands.push(OperandId, "'Ilwoke'");
    InstructionDesc[OpEnqueueKernel].operands.push(OperandId, "'Param'");
    InstructionDesc[OpEnqueueKernel].operands.push(OperandId, "'Param Size'");
    InstructionDesc[OpEnqueueKernel].operands.push(OperandId, "'Param Align'");
    InstructionDesc[OpEnqueueKernel].operands.push(OperandVariableIds, "'Local Size'");

    InstructionDesc[OpEnqueueMarker].operands.push(OperandId, "'Queue'");
    InstructionDesc[OpEnqueueMarker].operands.push(OperandId, "'Num Events'");
    InstructionDesc[OpEnqueueMarker].operands.push(OperandId, "'Wait Events'");
    InstructionDesc[OpEnqueueMarker].operands.push(OperandId, "'Ret Event'");

    InstructionDesc[OpGroupNonUniformElect].operands.push(OperandScope, "'Exelwtion'");

    InstructionDesc[OpGroupNonUniformAll].operands.push(OperandScope, "'Exelwtion'");
    InstructionDesc[OpGroupNonUniformAll].operands.push(OperandId, "X");

    InstructionDesc[OpGroupNonUniformAny].operands.push(OperandScope, "'Exelwtion'");
    InstructionDesc[OpGroupNonUniformAny].operands.push(OperandId, "X");

    InstructionDesc[OpGroupNonUniformAllEqual].operands.push(OperandScope, "'Exelwtion'");
    InstructionDesc[OpGroupNonUniformAllEqual].operands.push(OperandId, "X");

    InstructionDesc[OpGroupNonUniformBroadcast].operands.push(OperandScope, "'Exelwtion'");
    InstructionDesc[OpGroupNonUniformBroadcast].operands.push(OperandId, "X");
    InstructionDesc[OpGroupNonUniformBroadcast].operands.push(OperandId, "ID");

    InstructionDesc[OpGroupNonUniformBroadcastFirst].operands.push(OperandScope, "'Exelwtion'");
    InstructionDesc[OpGroupNonUniformBroadcastFirst].operands.push(OperandId, "X");

    InstructionDesc[OpGroupNonUniformBallot].operands.push(OperandScope, "'Exelwtion'");
    InstructionDesc[OpGroupNonUniformBallot].operands.push(OperandId, "X");

    InstructionDesc[OpGroupNonUniformIlwerseBallot].operands.push(OperandScope, "'Exelwtion'");
    InstructionDesc[OpGroupNonUniformIlwerseBallot].operands.push(OperandId, "X");

    InstructionDesc[OpGroupNonUniformBallotBitExtract].operands.push(OperandScope, "'Exelwtion'");
    InstructionDesc[OpGroupNonUniformBallotBitExtract].operands.push(OperandId, "X");
    InstructionDesc[OpGroupNonUniformBallotBitExtract].operands.push(OperandId, "Bit");

    InstructionDesc[OpGroupNonUniformBallotBitCount].operands.push(OperandScope, "'Exelwtion'");
    InstructionDesc[OpGroupNonUniformBallotBitCount].operands.push(OperandGroupOperation, "'Operation'");
    InstructionDesc[OpGroupNonUniformBallotBitCount].operands.push(OperandId, "X");

    InstructionDesc[OpGroupNonUniformBallotFindLSB].operands.push(OperandScope, "'Exelwtion'");
    InstructionDesc[OpGroupNonUniformBallotFindLSB].operands.push(OperandId, "X");

    InstructionDesc[OpGroupNonUniformBallotFindMSB].operands.push(OperandScope, "'Exelwtion'");
    InstructionDesc[OpGroupNonUniformBallotFindMSB].operands.push(OperandId, "X");

    InstructionDesc[OpGroupNonUniformShuffle].operands.push(OperandScope, "'Exelwtion'");
    InstructionDesc[OpGroupNonUniformShuffle].operands.push(OperandId, "X");
    InstructionDesc[OpGroupNonUniformShuffle].operands.push(OperandId, "'Id'");

    InstructionDesc[OpGroupNonUniformShuffleXor].operands.push(OperandScope, "'Exelwtion'");
    InstructionDesc[OpGroupNonUniformShuffleXor].operands.push(OperandId, "X");
    InstructionDesc[OpGroupNonUniformShuffleXor].operands.push(OperandId, "Mask");

    InstructionDesc[OpGroupNonUniformShuffleUp].operands.push(OperandScope, "'Exelwtion'");
    InstructionDesc[OpGroupNonUniformShuffleUp].operands.push(OperandId, "X");
    InstructionDesc[OpGroupNonUniformShuffleUp].operands.push(OperandId, "Offset");

    InstructionDesc[OpGroupNonUniformShuffleDown].operands.push(OperandScope, "'Exelwtion'");
    InstructionDesc[OpGroupNonUniformShuffleDown].operands.push(OperandId, "X");
    InstructionDesc[OpGroupNonUniformShuffleDown].operands.push(OperandId, "Offset");

    InstructionDesc[OpGroupNonUniformIAdd].operands.push(OperandScope, "'Exelwtion'");
    InstructionDesc[OpGroupNonUniformIAdd].operands.push(OperandGroupOperation, "'Operation'");
    InstructionDesc[OpGroupNonUniformIAdd].operands.push(OperandId, "X");
    InstructionDesc[OpGroupNonUniformIAdd].operands.push(OperandId, "'ClusterSize'", true);

    InstructionDesc[OpGroupNonUniformFAdd].operands.push(OperandScope, "'Exelwtion'");
    InstructionDesc[OpGroupNonUniformFAdd].operands.push(OperandGroupOperation, "'Operation'");
    InstructionDesc[OpGroupNonUniformFAdd].operands.push(OperandId, "X");
    InstructionDesc[OpGroupNonUniformFAdd].operands.push(OperandId, "'ClusterSize'", true);

    InstructionDesc[OpGroupNonUniformIMul].operands.push(OperandScope, "'Exelwtion'");
    InstructionDesc[OpGroupNonUniformIMul].operands.push(OperandGroupOperation, "'Operation'");
    InstructionDesc[OpGroupNonUniformIMul].operands.push(OperandId, "X");
    InstructionDesc[OpGroupNonUniformIMul].operands.push(OperandId, "'ClusterSize'", true);

    InstructionDesc[OpGroupNonUniformFMul].operands.push(OperandScope, "'Exelwtion'");
    InstructionDesc[OpGroupNonUniformFMul].operands.push(OperandGroupOperation, "'Operation'");
    InstructionDesc[OpGroupNonUniformFMul].operands.push(OperandId, "X");
    InstructionDesc[OpGroupNonUniformFMul].operands.push(OperandId, "'ClusterSize'", true);

    InstructionDesc[OpGroupNonUniformSMin].operands.push(OperandScope, "'Exelwtion'");
    InstructionDesc[OpGroupNonUniformSMin].operands.push(OperandGroupOperation, "'Operation'");
    InstructionDesc[OpGroupNonUniformSMin].operands.push(OperandId, "X");
    InstructionDesc[OpGroupNonUniformSMin].operands.push(OperandId, "'ClusterSize'", true);

    InstructionDesc[OpGroupNonUniformUMin].operands.push(OperandScope, "'Exelwtion'");
    InstructionDesc[OpGroupNonUniformUMin].operands.push(OperandGroupOperation, "'Operation'");
    InstructionDesc[OpGroupNonUniformUMin].operands.push(OperandId, "X");
    InstructionDesc[OpGroupNonUniformUMin].operands.push(OperandId, "'ClusterSize'", true);

    InstructionDesc[OpGroupNonUniformFMin].operands.push(OperandScope, "'Exelwtion'");
    InstructionDesc[OpGroupNonUniformFMin].operands.push(OperandGroupOperation, "'Operation'");
    InstructionDesc[OpGroupNonUniformFMin].operands.push(OperandId, "X");
    InstructionDesc[OpGroupNonUniformFMin].operands.push(OperandId, "'ClusterSize'", true);

    InstructionDesc[OpGroupNonUniformSMax].operands.push(OperandScope, "'Exelwtion'");
    InstructionDesc[OpGroupNonUniformSMax].operands.push(OperandGroupOperation, "'Operation'");
    InstructionDesc[OpGroupNonUniformSMax].operands.push(OperandId, "X");
    InstructionDesc[OpGroupNonUniformSMax].operands.push(OperandId, "'ClusterSize'", true);

    InstructionDesc[OpGroupNonUniformUMax].operands.push(OperandScope, "'Exelwtion'");
    InstructionDesc[OpGroupNonUniformUMax].operands.push(OperandGroupOperation, "'Operation'");
    InstructionDesc[OpGroupNonUniformUMax].operands.push(OperandId, "X");
    InstructionDesc[OpGroupNonUniformUMax].operands.push(OperandId, "'ClusterSize'", true);

    InstructionDesc[OpGroupNonUniformFMax].operands.push(OperandScope, "'Exelwtion'");
    InstructionDesc[OpGroupNonUniformFMax].operands.push(OperandGroupOperation, "'Operation'");
    InstructionDesc[OpGroupNonUniformFMax].operands.push(OperandId, "X");
    InstructionDesc[OpGroupNonUniformFMax].operands.push(OperandId, "'ClusterSize'", true);

    InstructionDesc[OpGroupNonUniformBitwiseAnd].operands.push(OperandScope, "'Exelwtion'");
    InstructionDesc[OpGroupNonUniformBitwiseAnd].operands.push(OperandGroupOperation, "'Operation'");
    InstructionDesc[OpGroupNonUniformBitwiseAnd].operands.push(OperandId, "X");
    InstructionDesc[OpGroupNonUniformBitwiseAnd].operands.push(OperandId, "'ClusterSize'", true);

    InstructionDesc[OpGroupNonUniformBitwiseOr].operands.push(OperandScope, "'Exelwtion'");
    InstructionDesc[OpGroupNonUniformBitwiseOr].operands.push(OperandGroupOperation, "'Operation'");
    InstructionDesc[OpGroupNonUniformBitwiseOr].operands.push(OperandId, "X");
    InstructionDesc[OpGroupNonUniformBitwiseOr].operands.push(OperandId, "'ClusterSize'", true);

    InstructionDesc[OpGroupNonUniformBitwiseXor].operands.push(OperandScope, "'Exelwtion'");
    InstructionDesc[OpGroupNonUniformBitwiseXor].operands.push(OperandGroupOperation, "'Operation'");
    InstructionDesc[OpGroupNonUniformBitwiseXor].operands.push(OperandId, "X");
    InstructionDesc[OpGroupNonUniformBitwiseXor].operands.push(OperandId, "'ClusterSize'", true);

    InstructionDesc[OpGroupNonUniformLogicalAnd].operands.push(OperandScope, "'Exelwtion'");
    InstructionDesc[OpGroupNonUniformLogicalAnd].operands.push(OperandGroupOperation, "'Operation'");
    InstructionDesc[OpGroupNonUniformLogicalAnd].operands.push(OperandId, "X");
    InstructionDesc[OpGroupNonUniformLogicalAnd].operands.push(OperandId, "'ClusterSize'", true);

    InstructionDesc[OpGroupNonUniformLogicalOr].operands.push(OperandScope, "'Exelwtion'");
    InstructionDesc[OpGroupNonUniformLogicalOr].operands.push(OperandGroupOperation, "'Operation'");
    InstructionDesc[OpGroupNonUniformLogicalOr].operands.push(OperandId, "X");
    InstructionDesc[OpGroupNonUniformLogicalOr].operands.push(OperandId, "'ClusterSize'", true);

    InstructionDesc[OpGroupNonUniformLogicalXor].operands.push(OperandScope, "'Exelwtion'");
    InstructionDesc[OpGroupNonUniformLogicalXor].operands.push(OperandGroupOperation, "'Operation'");
    InstructionDesc[OpGroupNonUniformLogicalXor].operands.push(OperandId, "X");
    InstructionDesc[OpGroupNonUniformLogicalXor].operands.push(OperandId, "'ClusterSize'", true);

    InstructionDesc[OpGroupNonUniformQuadBroadcast].operands.push(OperandScope, "'Exelwtion'");
    InstructionDesc[OpGroupNonUniformQuadBroadcast].operands.push(OperandId, "X");
    InstructionDesc[OpGroupNonUniformQuadBroadcast].operands.push(OperandId, "'Id'");

    InstructionDesc[OpGroupNonUniformQuadSwap].operands.push(OperandScope, "'Exelwtion'");
    InstructionDesc[OpGroupNonUniformQuadSwap].operands.push(OperandId, "X");
    InstructionDesc[OpGroupNonUniformQuadSwap].operands.push(OperandLiteralNumber, "'Direction'");

    InstructionDesc[OpSubgroupBallotKHR].operands.push(OperandId, "'Predicate'");

    InstructionDesc[OpSubgroupFirstIlwocationKHR].operands.push(OperandId, "'Value'");

    InstructionDesc[OpSubgroupAnyKHR].operands.push(OperandScope, "'Exelwtion'");
    InstructionDesc[OpSubgroupAnyKHR].operands.push(OperandId, "'Predicate'");

    InstructionDesc[OpSubgroupAllKHR].operands.push(OperandScope, "'Exelwtion'");
    InstructionDesc[OpSubgroupAllKHR].operands.push(OperandId, "'Predicate'");

    InstructionDesc[OpSubgroupAllEqualKHR].operands.push(OperandScope, "'Exelwtion'");
    InstructionDesc[OpSubgroupAllEqualKHR].operands.push(OperandId, "'Predicate'");

    InstructionDesc[OpSubgroupReadIlwocationKHR].operands.push(OperandId, "'Value'");
    InstructionDesc[OpSubgroupReadIlwocationKHR].operands.push(OperandId, "'Index'");

    InstructionDesc[OpModuleProcessed].operands.push(OperandLiteralString, "'process'");

    InstructionDesc[OpGroupIAddNonUniformAMD].operands.push(OperandScope, "'Exelwtion'");
    InstructionDesc[OpGroupIAddNonUniformAMD].operands.push(OperandGroupOperation, "'Operation'");
    InstructionDesc[OpGroupIAddNonUniformAMD].operands.push(OperandId, "'X'");

    InstructionDesc[OpGroupFAddNonUniformAMD].operands.push(OperandScope, "'Exelwtion'");
    InstructionDesc[OpGroupFAddNonUniformAMD].operands.push(OperandGroupOperation, "'Operation'");
    InstructionDesc[OpGroupFAddNonUniformAMD].operands.push(OperandId, "'X'");

    InstructionDesc[OpGroupUMinNonUniformAMD].operands.push(OperandScope, "'Exelwtion'");
    InstructionDesc[OpGroupUMinNonUniformAMD].operands.push(OperandGroupOperation, "'Operation'");
    InstructionDesc[OpGroupUMinNonUniformAMD].operands.push(OperandId, "'X'");

    InstructionDesc[OpGroupSMinNonUniformAMD].operands.push(OperandScope, "'Exelwtion'");
    InstructionDesc[OpGroupSMinNonUniformAMD].operands.push(OperandGroupOperation, "'Operation'");
    InstructionDesc[OpGroupSMinNonUniformAMD].operands.push(OperandId, "X");

    InstructionDesc[OpGroupFMinNonUniformAMD].operands.push(OperandScope, "'Exelwtion'");
    InstructionDesc[OpGroupFMinNonUniformAMD].operands.push(OperandGroupOperation, "'Operation'");
    InstructionDesc[OpGroupFMinNonUniformAMD].operands.push(OperandId, "X");

    InstructionDesc[OpGroupUMaxNonUniformAMD].operands.push(OperandScope, "'Exelwtion'");
    InstructionDesc[OpGroupUMaxNonUniformAMD].operands.push(OperandGroupOperation, "'Operation'");
    InstructionDesc[OpGroupUMaxNonUniformAMD].operands.push(OperandId, "X");

    InstructionDesc[OpGroupSMaxNonUniformAMD].operands.push(OperandScope, "'Exelwtion'");
    InstructionDesc[OpGroupSMaxNonUniformAMD].operands.push(OperandGroupOperation, "'Operation'");
    InstructionDesc[OpGroupSMaxNonUniformAMD].operands.push(OperandId, "X");

    InstructionDesc[OpGroupFMaxNonUniformAMD].operands.push(OperandScope, "'Exelwtion'");
    InstructionDesc[OpGroupFMaxNonUniformAMD].operands.push(OperandGroupOperation, "'Operation'");
    InstructionDesc[OpGroupFMaxNonUniformAMD].operands.push(OperandId, "X");

    InstructionDesc[OpFragmentMaskFetchAMD].operands.push(OperandId, "'Image'");
    InstructionDesc[OpFragmentMaskFetchAMD].operands.push(OperandId, "'Coordinate'");

    InstructionDesc[OpFragmentFetchAMD].operands.push(OperandId, "'Image'");
    InstructionDesc[OpFragmentFetchAMD].operands.push(OperandId, "'Coordinate'");
    InstructionDesc[OpFragmentFetchAMD].operands.push(OperandId, "'Fragment Index'");

    InstructionDesc[OpGroupNonUniformPartitionLW].operands.push(OperandId, "X");

    InstructionDesc[OpTypeAccelerationStructureKHR].setResultAndType(true, false);

    InstructionDesc[OpTraceRayKHR].operands.push(OperandId, "'LW Acceleration Structure'");
    InstructionDesc[OpTraceRayKHR].operands.push(OperandId, "'Ray Flags'");
    InstructionDesc[OpTraceRayKHR].operands.push(OperandId, "'Lwll Mask'");
    InstructionDesc[OpTraceRayKHR].operands.push(OperandId, "'SBT Record Offset'");
    InstructionDesc[OpTraceRayKHR].operands.push(OperandId, "'SBT Record Stride'");
    InstructionDesc[OpTraceRayKHR].operands.push(OperandId, "'Miss Index'");
    InstructionDesc[OpTraceRayKHR].operands.push(OperandId, "'Ray Origin'");
    InstructionDesc[OpTraceRayKHR].operands.push(OperandId, "'TMin'");
    InstructionDesc[OpTraceRayKHR].operands.push(OperandId, "'Ray Direction'");
    InstructionDesc[OpTraceRayKHR].operands.push(OperandId, "'TMax'");
    InstructionDesc[OpTraceRayKHR].operands.push(OperandId, "'Payload'");
    InstructionDesc[OpTraceRayKHR].setResultAndType(false, false);

    InstructionDesc[OpReportIntersectionKHR].operands.push(OperandId, "'Hit Parameter'");
    InstructionDesc[OpReportIntersectionKHR].operands.push(OperandId, "'Hit Kind'");

    InstructionDesc[OpIgnoreIntersectionKHR].setResultAndType(false, false);

    InstructionDesc[OpTerminateRayKHR].setResultAndType(false, false);
    
    InstructionDesc[OpExelwteCallableKHR].operands.push(OperandId, "SBT Record Index");
    InstructionDesc[OpExelwteCallableKHR].operands.push(OperandId, "CallableData ID");
    InstructionDesc[OpExelwteCallableKHR].setResultAndType(false, false);

    // Ray Query
    InstructionDesc[OpTypeAccelerationStructureKHR].setResultAndType(true, false);
    InstructionDesc[OpTypeRayQueryProvisionalKHR].setResultAndType(true, false);

    InstructionDesc[OpRayQueryInitializeKHR].operands.push(OperandId, "'RayQuery'");
    InstructionDesc[OpRayQueryInitializeKHR].operands.push(OperandId, "'AccelerationS'");
    InstructionDesc[OpRayQueryInitializeKHR].operands.push(OperandId, "'RayFlags'");
    InstructionDesc[OpRayQueryInitializeKHR].operands.push(OperandId, "'LwllMask'");
    InstructionDesc[OpRayQueryInitializeKHR].operands.push(OperandId, "'Origin'");
    InstructionDesc[OpRayQueryInitializeKHR].operands.push(OperandId, "'Tmin'");
    InstructionDesc[OpRayQueryInitializeKHR].operands.push(OperandId, "'Direction'");
    InstructionDesc[OpRayQueryInitializeKHR].operands.push(OperandId, "'Tmax'");
    InstructionDesc[OpRayQueryInitializeKHR].setResultAndType(false, false);

    InstructionDesc[OpRayQueryTerminateKHR].operands.push(OperandId, "'RayQuery'");
    InstructionDesc[OpRayQueryTerminateKHR].setResultAndType(false, false);

    InstructionDesc[OpRayQueryGenerateIntersectionKHR].operands.push(OperandId, "'RayQuery'");
    InstructionDesc[OpRayQueryGenerateIntersectionKHR].operands.push(OperandId, "'THit'");
    InstructionDesc[OpRayQueryGenerateIntersectionKHR].setResultAndType(false, false);

    InstructionDesc[OpRayQueryConfirmIntersectionKHR].operands.push(OperandId, "'RayQuery'");
    InstructionDesc[OpRayQueryConfirmIntersectionKHR].setResultAndType(false, false);

    InstructionDesc[OpRayQueryProceedKHR].operands.push(OperandId, "'RayQuery'");
    InstructionDesc[OpRayQueryProceedKHR].setResultAndType(true, true);

    InstructionDesc[OpRayQueryGetIntersectionTypeKHR].operands.push(OperandId, "'RayQuery'");
    InstructionDesc[OpRayQueryGetIntersectionTypeKHR].operands.push(OperandId, "'Committed'");
    InstructionDesc[OpRayQueryGetIntersectionTypeKHR].setResultAndType(true, true);

    InstructionDesc[OpRayQueryGetRayTMinKHR].operands.push(OperandId, "'RayQuery'");
    InstructionDesc[OpRayQueryGetRayTMinKHR].setResultAndType(true, true);

    InstructionDesc[OpRayQueryGetRayFlagsKHR].operands.push(OperandId, "'RayQuery'");
    InstructionDesc[OpRayQueryGetRayFlagsKHR].setResultAndType(true, true);

    InstructionDesc[OpRayQueryGetIntersectionTKHR].operands.push(OperandId, "'RayQuery'");
    InstructionDesc[OpRayQueryGetIntersectionTKHR].operands.push(OperandId, "'Committed'");
    InstructionDesc[OpRayQueryGetIntersectionTKHR].setResultAndType(true, true);

    InstructionDesc[OpRayQueryGetIntersectionInstanceLwstomIndexKHR].operands.push(OperandId, "'RayQuery'");
    InstructionDesc[OpRayQueryGetIntersectionInstanceLwstomIndexKHR].operands.push(OperandId, "'Committed'");
    InstructionDesc[OpRayQueryGetIntersectionInstanceLwstomIndexKHR].setResultAndType(true, true);

    InstructionDesc[OpRayQueryGetIntersectionInstanceIdKHR].operands.push(OperandId, "'RayQuery'");
    InstructionDesc[OpRayQueryGetIntersectionInstanceIdKHR].operands.push(OperandId, "'Committed'");
    InstructionDesc[OpRayQueryGetIntersectionInstanceIdKHR].setResultAndType(true, true);

    InstructionDesc[OpRayQueryGetIntersectionInstanceShaderBindingTableRecordOffsetKHR].operands.push(OperandId, "'RayQuery'");
    InstructionDesc[OpRayQueryGetIntersectionInstanceShaderBindingTableRecordOffsetKHR].operands.push(OperandId, "'Committed'");
    InstructionDesc[OpRayQueryGetIntersectionInstanceShaderBindingTableRecordOffsetKHR].setResultAndType(true, true);

    InstructionDesc[OpRayQueryGetIntersectionGeometryIndexKHR].operands.push(OperandId, "'RayQuery'");
    InstructionDesc[OpRayQueryGetIntersectionGeometryIndexKHR].operands.push(OperandId, "'Committed'");
    InstructionDesc[OpRayQueryGetIntersectionGeometryIndexKHR].setResultAndType(true, true);

    InstructionDesc[OpRayQueryGetIntersectionPrimitiveIndexKHR].operands.push(OperandId, "'RayQuery'");
    InstructionDesc[OpRayQueryGetIntersectionPrimitiveIndexKHR].operands.push(OperandId, "'Committed'");
    InstructionDesc[OpRayQueryGetIntersectionPrimitiveIndexKHR].setResultAndType(true, true);

    InstructionDesc[OpRayQueryGetIntersectionBarycentricsKHR].operands.push(OperandId, "'RayQuery'");
    InstructionDesc[OpRayQueryGetIntersectionBarycentricsKHR].operands.push(OperandId, "'Committed'");
    InstructionDesc[OpRayQueryGetIntersectionBarycentricsKHR].setResultAndType(true, true);

    InstructionDesc[OpRayQueryGetIntersectionFrontFaceKHR].operands.push(OperandId, "'RayQuery'");
    InstructionDesc[OpRayQueryGetIntersectionFrontFaceKHR].operands.push(OperandId, "'Committed'");
    InstructionDesc[OpRayQueryGetIntersectionFrontFaceKHR].setResultAndType(true, true);

    InstructionDesc[OpRayQueryGetIntersectionCandidateAABBOpaqueKHR].operands.push(OperandId, "'RayQuery'");
    InstructionDesc[OpRayQueryGetIntersectionCandidateAABBOpaqueKHR].setResultAndType(true, true);

    InstructionDesc[OpRayQueryGetIntersectionObjectRayDirectionKHR].operands.push(OperandId, "'RayQuery'");
    InstructionDesc[OpRayQueryGetIntersectionObjectRayDirectionKHR].operands.push(OperandId, "'Committed'");
    InstructionDesc[OpRayQueryGetIntersectionObjectRayDirectionKHR].setResultAndType(true, true);

    InstructionDesc[OpRayQueryGetIntersectionObjectRayOriginKHR].operands.push(OperandId, "'RayQuery'");
    InstructionDesc[OpRayQueryGetIntersectionObjectRayOriginKHR].operands.push(OperandId, "'Committed'");
    InstructionDesc[OpRayQueryGetIntersectionObjectRayOriginKHR].setResultAndType(true, true);

    InstructionDesc[OpRayQueryGetWorldRayDirectionKHR].operands.push(OperandId, "'RayQuery'");
    InstructionDesc[OpRayQueryGetWorldRayDirectionKHR].setResultAndType(true, true);

    InstructionDesc[OpRayQueryGetWorldRayOriginKHR].operands.push(OperandId, "'RayQuery'");
    InstructionDesc[OpRayQueryGetWorldRayOriginKHR].setResultAndType(true, true);

    InstructionDesc[OpRayQueryGetIntersectionObjectToWorldKHR].operands.push(OperandId, "'RayQuery'");
    InstructionDesc[OpRayQueryGetIntersectionObjectToWorldKHR].operands.push(OperandId, "'Committed'");
    InstructionDesc[OpRayQueryGetIntersectionObjectToWorldKHR].setResultAndType(true, true);

    InstructionDesc[OpRayQueryGetIntersectionWorldToObjectKHR].operands.push(OperandId, "'RayQuery'");
    InstructionDesc[OpRayQueryGetIntersectionWorldToObjectKHR].operands.push(OperandId, "'Committed'");
    InstructionDesc[OpRayQueryGetIntersectionWorldToObjectKHR].setResultAndType(true, true);

    InstructionDesc[OpImageSampleFootprintLW].operands.push(OperandId, "'Sampled Image'");
    InstructionDesc[OpImageSampleFootprintLW].operands.push(OperandId, "'Coordinate'");
    InstructionDesc[OpImageSampleFootprintLW].operands.push(OperandId, "'Granularity'");
    InstructionDesc[OpImageSampleFootprintLW].operands.push(OperandId, "'Coarse'");
    InstructionDesc[OpImageSampleFootprintLW].operands.push(OperandImageOperands, "", true);
    InstructionDesc[OpImageSampleFootprintLW].operands.push(OperandVariableIds, "", true);
    
    InstructionDesc[OpWritePackedPrimitiveIndices4x8LW].operands.push(OperandId, "'Index Offset'");
    InstructionDesc[OpWritePackedPrimitiveIndices4x8LW].operands.push(OperandId, "'Packed Indices'");

    InstructionDesc[OpTypeCooperativeMatrixLW].operands.push(OperandId, "'Component Type'");
    InstructionDesc[OpTypeCooperativeMatrixLW].operands.push(OperandId, "'Scope'");
    InstructionDesc[OpTypeCooperativeMatrixLW].operands.push(OperandId, "'Rows'");
    InstructionDesc[OpTypeCooperativeMatrixLW].operands.push(OperandId, "'Columns'");

    InstructionDesc[OpCooperativeMatrixLoadLW].operands.push(OperandId, "'Pointer'");
    InstructionDesc[OpCooperativeMatrixLoadLW].operands.push(OperandId, "'Stride'");
    InstructionDesc[OpCooperativeMatrixLoadLW].operands.push(OperandId, "'Column Major'");
    InstructionDesc[OpCooperativeMatrixLoadLW].operands.push(OperandMemoryAccess, "'Memory Access'");
    InstructionDesc[OpCooperativeMatrixLoadLW].operands.push(OperandLiteralNumber, "", true);
    InstructionDesc[OpCooperativeMatrixLoadLW].operands.push(OperandId, "", true);

    InstructionDesc[OpCooperativeMatrixStoreLW].operands.push(OperandId, "'Pointer'");
    InstructionDesc[OpCooperativeMatrixStoreLW].operands.push(OperandId, "'Object'");
    InstructionDesc[OpCooperativeMatrixStoreLW].operands.push(OperandId, "'Stride'");
    InstructionDesc[OpCooperativeMatrixStoreLW].operands.push(OperandId, "'Column Major'");
    InstructionDesc[OpCooperativeMatrixStoreLW].operands.push(OperandMemoryAccess, "'Memory Access'");
    InstructionDesc[OpCooperativeMatrixStoreLW].operands.push(OperandLiteralNumber, "", true);
    InstructionDesc[OpCooperativeMatrixStoreLW].operands.push(OperandId, "", true);

    InstructionDesc[OpCooperativeMatrixMulAddLW].operands.push(OperandId, "'A'");
    InstructionDesc[OpCooperativeMatrixMulAddLW].operands.push(OperandId, "'B'");
    InstructionDesc[OpCooperativeMatrixMulAddLW].operands.push(OperandId, "'C'");

    InstructionDesc[OpCooperativeMatrixLengthLW].operands.push(OperandId, "'Type'");

    InstructionDesc[OpDemoteToHelperIlwocationEXT].setResultAndType(false, false);

    InstructionDesc[OpReadClockKHR].operands.push(OperandScope, "'Scope'");
}

}; // end spv namespace
