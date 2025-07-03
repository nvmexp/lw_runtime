/*
** Copyright (c) 2014-2017 The Khronos Group Inc.
**
** Permission is hereby granted, free of charge, to any person obtaining a copy
** of this software and/or associated documentation files (the "Materials"),
** to deal in the Materials without restriction, including without limitation
** the rights to use, copy, modify, merge, publish, distribute, sublicense,
** and/or sell copies of the Materials, and to permit persons to whom the
** Materials are furnished to do so, subject to the following conditions:
**
** The above copyright notice and this permission notice shall be included in
** all copies or substantial portions of the Materials.
**
** MODIFICATIONS TO THIS FILE MAY MEAN IT NO LONGER ACLWRATELY REFLECTS KHRONOS
** STANDARDS. THE UNMODIFIED, NORMATIVE VERSIONS OF KHRONOS SPECIFICATIONS AND
** HEADER INFORMATION ARE LOCATED AT https://www.khronos.org/registry/
**
** THE MATERIALS ARE PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
** OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
** FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
** THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
** LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
** FROM,OUT OF OR IN CONNECTION WITH THE MATERIALS OR THE USE OR OTHER DEALINGS
** IN THE MATERIALS.
*/

#ifndef GLSLextLW_H
#define GLSLextLW_H

enum BuiltIn;
enum Decoration;
enum Op;
enum Capability;

static const int GLSLextLWVersion = 100;
static const int GLSLextLWRevision = 11;

//SPV_LW_sample_mask_override_coverage
const char* const E_SPV_LW_sample_mask_override_coverage = "SPV_LW_sample_mask_override_coverage";

//SPV_LW_geometry_shader_passthrough
const char* const E_SPV_LW_geometry_shader_passthrough = "SPV_LW_geometry_shader_passthrough";

//SPV_LW_viewport_array2
const char* const E_SPV_LW_viewport_array2 = "SPV_LW_viewport_array2";
const char* const E_ARB_shader_viewport_layer_array = "SPV_ARB_shader_viewport_layer_array";

//SPV_LW_stereo_view_rendering
const char* const E_SPV_LW_stereo_view_rendering = "SPV_LW_stereo_view_rendering";

//SPV_LWX_multiview_per_view_attributes
const char* const E_SPV_LWX_multiview_per_view_attributes = "SPV_LWX_multiview_per_view_attributes";

//SPV_LW_shader_subgroup_partitioned
const char* const E_SPV_LW_shader_subgroup_partitioned = "SPV_LW_shader_subgroup_partitioned";

//SPV_LW_fragment_shader_barycentric
const char* const E_SPV_LW_fragment_shader_barycentric = "SPV_LW_fragment_shader_barycentric";

//SPV_LW_compute_shader_derivatives
const char* const E_SPV_LW_compute_shader_derivatives = "SPV_LW_compute_shader_derivatives";

//SPV_LW_shader_image_footprint
const char* const E_SPV_LW_shader_image_footprint = "SPV_LW_shader_image_footprint";

//SPV_LW_mesh_shader
const char* const E_SPV_LW_mesh_shader = "SPV_LW_mesh_shader";

//SPV_LW_raytracing
const char* const E_SPV_LW_ray_tracing = "SPV_LW_ray_tracing";

//SPV_LW_shading_rate
const char* const E_SPV_LW_shading_rate = "SPV_LW_shading_rate";

//SPV_LW_cooperative_matrix
const char* const E_SPV_LW_cooperative_matrix = "SPV_LW_cooperative_matrix";

//SPV_LW_shader_sm_builtins
const char* const E_SPV_LW_shader_sm_builtins = "SPV_LW_shader_sm_builtins";

#endif  // #ifndef GLSLextLW_H
