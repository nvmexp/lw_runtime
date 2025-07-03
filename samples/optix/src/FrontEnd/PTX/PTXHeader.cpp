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

#include <FrontEnd/PTX/PTXHeader.h>
#include <string.h>

namespace optix {

static const char* ptx_declarations =
    // clang-format off
    "     .version 1.4\n"
    "     .target sm_10\n"

    // Textures
    "     .extern  .func (.reg .f32 %fv0, .reg .f32 %fv1, .reg .f32 %fv2, .reg .f32 %fv3) _rt_texture_get_f_id (.reg .u32 %texid, .reg .u32 %dim, .reg .f32 %x, .reg .f32 %y, .reg .f32 %z, .reg .f32 %w)\n"
    "     .extern  .func (.reg .s32 %iv0, .reg .s32 %iv1, .reg .s32 %iv2, .reg .s32 %iv3) _rt_texture_get_i_id (.reg .u32 %texid, .reg .u32 %dim, .reg .f32 %x, .reg .f32 %y, .reg .f32 %z, .reg .f32 %w)\n"
    "     .extern  .func (.reg .u32 %uv0, .reg .u32 %uv1, .reg .u32 %uv2, .reg .u32 %uv3) _rt_texture_get_u_id (.reg .u32 %texid, .reg .u32 %dim, .reg .f32 %x, .reg .f32 %y, .reg .f32 %z, .reg .f32 %w)\n"
    "     .extern  .func (.reg .f32 %fv0, .reg .f32 %fv1, .reg .f32 %fv2, .reg .f32 %fv3) _rt_texture_get_fetch_id (.reg .u32 %texid, .reg .u32 %dim, .reg .s32 %x, .reg .s32 %y, .reg .s32 %z, .reg .s32 %w)\n"
    "     .extern  .func (.reg .u32 %uv0, .reg .u32 %uv1, .reg .u32 %uv2) _rt_texture_get_size_id (.reg .u32 %texid)\n"
    "     .extern  .func (.reg .f32 %fv0, .reg .f32 %fv1, .reg .f32 %fv2, .reg .f32 %fv3) _rt_texture_get_gather_id (.reg .u32 %texid, .reg .u32 %dim, .reg .f32 %x, .reg .f32 %y, .reg .u32 %comp)\n"
    "     .extern  .func (.reg .f32 %fv0, .reg .f32 %fv1, .reg .f32 %fv2, .reg .f32 %fv3) _rt_texture_get_base_id (.reg .u32 %texid, .reg .u32 %dim, .reg .f32 %x, .reg .f32 %y, .reg .f32 %z, .reg .u32 %layer)\n"
    "     .extern  .func (.reg .f32 %fv0, .reg .f32 %fv1, .reg .f32 %fv2, .reg .f32 %fv3) _rt_texture_get_level_id (.reg .u32 %texid, .reg .u32 %dim, .reg .f32 %x, .reg .f32 %y, .reg .f32 %z, .reg .u32 %layer, .reg .f32 %level)\n"
    "     .extern  .func (.reg .f32 %fv0, .reg .f32 %fv1, .reg .f32 %fv2, .reg .f32 %fv3) _rt_texture_get_grad_id (.reg .u32 %texid, .reg .u32 %dim, .reg .f32 %x, .reg .f32 %y, .reg .f32 %z, .reg .u32 %layer, .reg .f32 %dPdx_x, .reg .f32 %dPdx_y, .reg .f32 %dPdx_z, .reg .f32 %dPdy_x, .reg .f32 %dPdy_y, .reg .f32 %dPdy_z)\n"

    // Buffers
    "     .extern  .func (.reg .u32 $ptr) _rt_buffer_get       (.reg .u32 $buffer, .reg .u32 $dim, .reg .u32 $elementsize, .reg .u32 $i0, .reg .u32 $i1, .reg .u32 $i2, .reg .u32 $i3)\n"
    "     .extern  .func (.reg .u32 $ptr) _rt_buffer_get_id    (.reg .s32 $id, .reg .u32 $dim, .reg .u32 $elementsize, .reg .u32 $i0, .reg .u32 $i1, .reg .u32 $i2, .reg .u32 $i3)\n"
    "     .extern  .func (.reg .u32 $dim0, .reg .u32 $dim1, .reg .u32 $dim2, .reg .u32 $dim3) _rt_buffer_get_size (.reg .u32 $buffer, .reg .u32 $dim, .reg .u32 $elementsize)\n"
    "     .extern  .func (.reg .u32 $dim0, .reg .u32 $dim1, .reg .u32 $dim2, .reg .u32 $dim3) _rt_buffer_get_id_size (.reg .s32 $id, .reg .u32 $dim, .reg .u32 $elementsize)\n"
    "     .extern  .func (.reg .u64 $ptr) _rt_buffer_get_64       (.reg .u64 $buffer, .reg .u32 $dim, .reg .u32 $elementsize, .reg .u64 $i0, .reg .u64 $i1, .reg .u64 $i2, .reg .u64 $i3)\n"
    "     .extern  .func (.reg .u64 $ptr) _rt_buffer_get_id_64    (.reg .s32 $id, .reg .u32 $dim, .reg .u32 $elementsize, .reg .u64 $i0, .reg .u64 $i1, .reg .u64 $i2, .reg .u64 $i3)\n"
    "     .extern  .func (.reg .u64 $dim0, .reg .u64 $dim1, .reg .u64 $dim2, .reg .u64 $dim3) _rt_buffer_get_size_64 (.reg .u64 $buffer, .reg .u32 $dim, .reg .u32 $elementsize)\n"
    "     .extern  .func (.reg .u64 $dim0, .reg .u64 $dim1, .reg .u64 $dim2, .reg .u64 $dim3) _rt_buffer_get_id_size_64 (.reg .s32 $id, .reg .u32 $dim, .reg .u32 $elementsize)\n"

    // Demand loaded buffers
    "     .extern  .func (.reg .u32 $bool) _rt_load_or_request_64 (.reg .u64 $buffer, .reg .u32 $dim, .reg .u32 $elementsize, .reg .u64 $i0, .reg .u64 $i1, .reg .u64 $i2, .reg .u64 $i3, .reg .u64 $ptr)\n"

    // Demand loaded textures
    "     .extern  .func (.reg .f32 %fv0, .reg .f32 %fv1, .reg .f32 %fv2, .reg .f32 %fv3) _rt_texture_load_or_request_f_id (.reg .u32 %texid, .reg .u32 %dim, .reg .f32 %x, .reg .f32 %y, .reg .f32 %z, .reg .f32 %w, .reg .u64 $isResident)\n"
    "     .extern  .func (.reg .u32 %uv0, .reg .u32 %uv1, .reg .u32 %uv2, .reg .u32 %uv3) _rt_texture_load_or_request_u_id (.reg .u32 %texid, .reg .u32 %dim, .reg .f32 %x, .reg .f32 %y, .reg .f32 %z, .reg .f32 %w, .reg .u64 $isResident)\n"
    "     .extern  .func (.reg .s32 %iv0, .reg .s32 %iv1, .reg .s32 %iv2, .reg .s32 %iv3) _rt_texture_load_or_request_i_id (.reg .u32 %texid, .reg .u32 %dim, .reg .f32 %x, .reg .f32 %y, .reg .f32 %z, .reg .f32 %w, .reg .u64 $isResident)\n"
    "     .extern  .func (.reg .f32 %fv0, .reg .f32 %fv1, .reg .f32 %fv2, .reg .f32 %fv3) _rt_texture_lod_load_or_request_f_id (.reg .u32 %texid, .reg .u32 %dim, .reg .f32 %x, .reg .f32 %y, .reg .f32 %z, .reg .f32 %w, .reg .f32 %level, .reg .u64 $isResident)\n"
    "     .extern  .func (.reg .u32 %uv0, .reg .u32 %uv1, .reg .u32 %uv2, .reg .u32 %uv3) _rt_texture_lod_load_or_request_u_id (.reg .u32 %texid, .reg .u32 %dim, .reg .f32 %x, .reg .f32 %y, .reg .f32 %z, .reg .f32 %w, .reg .f32 %level, .reg .u64 $isResident)\n"
    "     .extern  .func (.reg .s32 %iv0, .reg .s32 %iv1, .reg .s32 %iv2, .reg .s32 %iv3) _rt_texture_lod_load_or_request_i_id (.reg .u32 %texid, .reg .u32 %dim, .reg .f32 %x, .reg .f32 %y, .reg .f32 %z, .reg .f32 %w, .reg .f32 %level, .reg .u64 $isResident)\n"
    "     .extern  .func (.reg .f32 %fv0, .reg .f32 %fv1, .reg .f32 %fv2, .reg .f32 %fv3) _rt_texture_grad_load_or_request_f_id (.reg .u32 %texid, .reg .u32 %dim, .reg .f32 %x, .reg .f32 %y, .reg .f32 %z, .reg .f32 %w, .reg .f32 %dPdx_x, .reg .f32 %dPdx_y, .reg .f32 %dPdx_z, .reg .f32 %dPdy_x, .reg .f32 %dPdy_y, .reg .f32 %dPdy_z, .reg .u64 $isResident)\n"
    "     .extern  .func (.reg .u32 %uv0, .reg .u32 %uv1, .reg .u32 %uv2, .reg .u32 %uv3) _rt_texture_grad_load_or_request_u_id (.reg .u32 %texid, .reg .u32 %dim, .reg .f32 %x, .reg .f32 %y, .reg .f32 %z, .reg .f32 %w, .reg .f32 %dPdx_x, .reg .f32 %dPdx_y, .reg .f32 %dPdx_z, .reg .f32 %dPdy_x, .reg .f32 %dPdy_y, .reg .f32 %dPdy_z, .reg .u64 $isResident)\n"
    "     .extern  .func (.reg .s32 %iv0, .reg .s32 %iv1, .reg .s32 %iv2, .reg .s32 %iv3) _rt_texture_grad_load_or_request_i_id (.reg .u32 %texid, .reg .u32 %dim, .reg .f32 %x, .reg .f32 %y, .reg .f32 %z, .reg .f32 %w, .reg .f32 %dPdx_x, .reg .f32 %dPdx_y, .reg .f32 %dPdx_z, .reg .f32 %dPdy_x, .reg .f32 %dPdy_y, .reg .f32 %dPdy_z, .reg .u64 $isResident)\n"

    // Ray data
    "     .extern  .func (.reg .u32 $ptr) _rt_raydata_get (.reg .u32 $raydata)\n"
    "     .extern  .func (.reg .u64 $ptr) _rt_raydata_get_64 (.reg .u32 $raydata)\n"

    // Semantic values
    "     .extern  .func (.reg .u32 $u0, .reg .u32 $u1, .reg .u32 $u2) _rt_semantic_value_get_uint3 (.reg .u32 $id)\n"
    "     .extern  .func (.reg .f32 $o0, .reg .f32 $o1, .reg .f32 $o2, .reg .f32 $d0, .reg .f32 $d1, .reg .f32 $d2, .reg .u32 $ray_type, .reg .f32 $tmin, .reg .f32 $tmax) _rt_semantic_value_get_Ray (.reg .u32 $id)\n"
    "     .extern  .func (.reg .f32 $ptr) _rt_semantic_value_get_float (.reg .u32 $id)\n"

    // Functions
    "     .extern  .func (.reg .u32 $ptr) _rt_callable_program_from_id (.reg .s32 $id)\n"
    "     .extern  .func (.reg .u64 $ptr) _rt_callable_program_from_id_64 (.reg .u32 $id)\n"
    "     .extern  .func (.reg .u32 $ptr) _rt_callable_program_from_id_v2 (.reg .s32 $id, .reg .u32 $csid)\n"
    "     .extern  .func (.reg .u64 $ptr) _rt_callable_program_from_id_v2_64 (.reg .u32 $id, .reg .u64 $csid)\n"
    "     .extern  .func (.reg .u32 $pptr) _rt_pickle_pointer (.reg .u32 $ptr)\n"
    "     .extern  .func (.reg .u32 $pptr) _rt_unpickle_pointer (.reg .u32 $pptr)\n"
    "     .extern  .func (.reg .u32 $pptr) _rt_pickle_pointer_64 (.reg .u64 $ptr)\n"
    "     .extern  .func (.reg .u64 $ptr) _rt_unpickle_pointer_64 (.reg .u32 $pptr)\n"
    // rt_trace backwards compatibility for ptx generated with pre OptiX 6.0
    "     .extern  .func _rt_trace (.reg .u32 $node, .reg .f32 $ox, .reg .f32 $oy, .reg .f32 $oz, .reg .f32 $dx, .reg .f32 $dy, .reg .f32 $dz, .reg .u32 $raytype, .reg .f32 $tmin, .reg .f32 $tmax, .reg .u32 $prd, .reg .u32 $prd_size)\n"
    "     .extern  .func _rt_trace_64 (.reg .u32 $node, .reg .f32 $ox, .reg .f32 $oy, .reg .f32 $oz, .reg .f32 $dx, .reg .f32 $dy, .reg .f32 $dz, .reg .u32 $raytype, .reg .f32 $tmin, .reg .f32 $tmax, .reg .u64 $prd, .reg .u32 $prd_size)\n"
    "     .extern  .func _rt_trace_with_time_64 (.reg .u32 $node, .reg .f32 $ox, .reg .f32 $oy, .reg .f32 $oz, .reg .f32 $dx, .reg .f32 $dy, .reg .f32 $dz, .reg .u32 $raytype, .reg .f32 $tmin, .reg .f32 $tmax, .reg .f32 $time, .reg .u64 $prd, .reg .u32 $prd_size)\n"
    // the only two trace functions available starting from OptiX 6.0
    "     .extern  .func _rt_trace_mask_flags_64 (.reg .u32 $node, .reg .f32 $ox, .reg .f32 $oy, .reg .f32 $oz, .reg .f32 $dx, .reg .f32 $dy, .reg .f32 $dz, .reg .u32 $raytype, .reg .f32 $tmin, .reg .f32 $tmax, .reg .u32 $mask, .reg .u32 $flags, .reg .u64 $prd, .reg .u32 $prd_size)\n"
    "     .extern  .func _rt_trace_time_mask_flags_64 (.reg .u32 $node, .reg .f32 $ox, .reg .f32 $oy, .reg .f32 $oz, .reg .f32 $dx, .reg .f32 $dy, .reg .f32 $dz, .reg .u32 $raytype, .reg .f32 $tmin, .reg .f32 $tmax, .reg .f32 $time, .reg .u32 $mask, .reg .u32 $flags, .reg .u64 $prd, .reg .u32 $prd_size)\n"
    "     .extern  .func (.reg .u32 $bool) _rt_potential_intersection (.reg .f32 $t)\n"
    "     .extern  .func (.reg .u32 $bool) _rt_report_intersection (.reg .u32 $matl)\n"
    "     .extern  .func () _rt_ignore_intersection ()\n"
    "     .extern  .func () _rt_terminate_ray ()\n"
    "     .extern  .func () _rt_yield_to_host ()\n"
    "     .extern  .func () _rt_intersect_child (.reg .u32 $index)\n"
    "     .extern  .func (.reg .f32 $b0, .reg .f32 $b1, .reg .f32 $b2, .reg .f32 $b3) _rt_transform_tuple(.reg .u32 $id, .reg .f32 $a0, .reg .f32 $a1, .reg .f32 $a2, .reg .f32 $a3)\n"
    "     .extern  .func ( .reg .f32 $m00, .reg .f32 $m01, .reg .f32 $m02, .reg .f32 $m03, .reg .f32 $m10, .reg .f32 $m11, .reg .f32 $m12, .reg .f32 $m13, .reg .f32 $m20, .reg .f32 $m21, .reg .f32 $m22, .reg .f32 $m23, .reg .f32 $m30, .reg .f32 $m31, .reg .f32 $m32, .reg .f32 $m33) _rt_get_transform(.reg .u32 $kind )\n"
    "     .extern  .func (.reg .u32 $ret) _rt_get_primitive_index()\n"
    "     .extern  .func (.reg .u32 $ret) _rt_is_triangle_hit()\n"
    "     .extern  .func (.reg .u32 $ret) _rt_is_triangle_hit_back_face()\n"
    "     .extern  .func (.reg .u32 $ret) _rt_is_triangle_hit_front_face()\n"
    "     .extern  .func (.reg .f32 $f0, .reg .f32 $f1) _rt_get_triangle_barycentrics()\n"
    "     .extern  .func (.reg .u32 $ret) _rt_get_lowest_group_child_index()\n"
    "     .extern  .func () _rt_throw (.reg .u32 $code)\n"
    "     .extern  .func (.reg .u32 $ret) _rt_get_exception_code()\n"
    "     .extern  .func (.reg .u32 $ret) _rt_get_entry_point_index()\n"
    "     .extern  .func (.reg .u32 $ret) _rt_get_ray_flags()\n"
    "     .extern  .func (.reg .u32 $ret) _rt_get_ray_mask()\n"

    // Internal functions
    "     .extern  .func (.reg .u32 $ret) _rti_get_instance_flags()\n"
    "     .extern  .func (.reg .u32 $bool) _rti_report_full_intersection_ff (.reg .f32 $t, .reg .u32 $matl, .reg .u32 $hitkind, .reg .f32 $f0, .reg .f32 $f1)\n"
    "     .extern  .func (.reg .u32 $ret) _rti_get_function_id(.reg .u32 $value)\n"
    "     .extern  .func (.reg .u32 $ret) _rti_set_function_id(.reg .u32 $value)\n"
    "     .extern  .func (.reg .u64 $addr, .reg .u32 $size, .reg .u32 $type) _rti_object_records_set(.reg .u64 $ptr)\n"
    "     .extern  .func () _rti_save_stack(.reg .u32 $buffer, .reg .u32 $threadidx)\n"
    "     .extern  .func () _rti_load_stack(.reg .u32 $buffer, .reg .u32 $threadidx)\n"
    "     .extern  .func () _rti_save_stack_64 (.reg .u64 $buffer, .reg .u32 $threadidx)\n"
    "     .extern  .func () _rti_load_stack_64 (.reg .u64 $buffer, .reg .u32 $threadidx)\n"
    "     .extern  .func () _rti_reset_stack()\n"
    "     .extern  .func () _rti_intersect_node (.reg .u32 $child)\n"
    "     .extern  .func () _rti_intersect_primitive (.reg .u32 $child, .reg .u32 $primitive)\n"
    "     .extern  .func () _rti_compute_geometry_instance_aabb (.reg .u32 $instance, .reg .u32 $primitive, .reg .u32 $aabb)\n"
    "     .extern  .func () _rti_compute_geometry_instance_aabb_64 (.reg .u32 $instance, .reg .u32 $primitive, .reg .u32 $motionStep, .reg .u64 $aabb)\n"
    "     .extern  .func () _rti_compute_group_child_aabb_64 (.reg .u32 $group, .reg .u32 $primitive, .reg .u64 $aabb)\n"
    "     .extern  .func () _rti_gather_motion_aabbs_64 (.reg .u32 $group, .reg .u64 $aabb)\n"
    "     .extern  .func (.reg .u32 $count) _rti_get_primitive_index_offset (.reg .u32 $child)\n"
    "     .extern  .func () _rti_get_aabb_request(.reg .u64 $ptr)\n"
    "     .extern  .func () _rti_main_switch ()\n"
    "     .extern  .func () _rti_yield ()\n"
    "     .extern  .func () _rti_set_lwrrent_acceleration ()\n"
    "     .extern  .func () _rti_handle_transform_node ()\n"
    "     .extern  .func (.reg .u64 $ptr) _rti_get_status_return()\n"
    "     .extern  .func () _rti_profile_event (.reg .u32 $index)\n"

    // Triangle data access
    "     .extern  .func (.reg .s32 $i0) _rti_get_geometry_triangles_vertexBufferID()\n"
    "     .extern  .func (.reg .s64 $i0) _rti_get_geometry_triangles_vertexBufferOffset()\n"
    "     .extern  .func (.reg .u64 $i0) _rti_get_geometry_triangles_vertexBufferStride()\n"
    "     .extern  .func (.reg .s32 $i0) _rti_get_geometry_triangles_vertexBufferID()\n"
    "     .extern  .func (.reg .s32 $i0) _rti_get_geometry_triangles_indexBufferID()\n"
    "     .extern  .func (.reg .s64 $i0) _rti_get_geometry_triangles_indexBufferOffset()\n"
    "     .extern  .func (.reg .u64 $i0) _rti_get_geometry_triangles_indexBufferStride()\n"

    // Motion triangle data access.
    "     .extern  .func (.reg .u64 $i0) _rti_get_motion_geometry_triangles_vertexBufferMotionStride()\n"
    "     .extern  .func (.reg .s32 $i0) _rti_get_motion_geometry_triangles_motionNumIntervals()\n"
    "     .extern  .func (.reg .f32 $f0) _rti_get_motion_geometry_triangles_timeBegin()\n"
    "     .extern  .func (.reg .f32 $f0) _rti_get_motion_geometry_triangles_timeEnd()\n"
    "     .extern  .func (.reg .s32 $i0) _rti_get_motion_geometry_triangles_motionBorderModeBegin()\n"
    "     .extern  .func (.reg .s32 $i0) _rti_get_motion_geometry_triangles_motionBorderModeEnd()\n"

    // Printing
    "     .extern  .func () _rt_print_write32 (.reg .u32 $val, .reg .u32 $off)\n"
    "     .extern  .func (.reg .u32 $ret) _rt_print_active()\n"
    "     .extern  .func (.reg .u32 $ret) _rt_print_start(.reg .u32 $fmt, .reg .u32 $sz)\n"
    "     .extern  .func (.reg .u32 $ret) _rt_print_start_64(.reg .u64 $fmt, .reg .u32 $sz)\n"


    // Statistics
    "     .extern  .func () _rti_statistic_add_uint (.reg .u32 $ptr, .reg .u32 value)\n"
    "     .extern  .func () _rti_statistic_add_float (.reg .u32 $ptr, .reg .f32 value)\n"
    "     .extern  .func () _rti_statistic_vector_add_int (.reg .u32 $ptr, .reg .s32 value, .reg .u32 idx, .reg .u32 n)\n"
    "     .extern  .func () _rti_statistic_vector_add_uint (.reg .u32 $ptr, .reg .u32 value, .reg .u32 idx, .reg .u32 n)\n"
    "     .extern  .func () _rti_statistic_vector_add_float (.reg .u32 $ptr, .reg .f32 value, .reg .u32 idx, .reg .u32 n)\n"
    "     .extern  .func (.reg .s32 $value) _rti_statistic_vector_get_int (.reg .u32 $ptr, .reg .u32 idx, .reg .u32 n)\n"
    "     .extern  .func (.reg .u32 $value) _rti_statistic_vector_get_uint (.reg .u32 $ptr, .reg .u32 idx, .reg .u32 n)\n"
    "     .extern  .func (.reg .f32 $value) _rti_statistic_vector_get_float (.reg .u32 $ptr, .reg .u32 idx, .reg .u32 n)\n"
    "     .extern  .func (.reg .s32 $value) _rti_statistic_get_int (.reg .u32 $ptr)\n"
    "     .extern  .func (.reg .u32 $value) _rti_statistic_get_uint (.reg .u32 $ptr)\n"
    "     .extern  .func (.reg .f32 $value) _rti_statistic_get_float (.reg .u32 $ptr)\n"
    "     .extern  .func () _rti_statistic_add_int_64 (.reg .u64 $ptr, .reg .u32 value)\n"
    "     .extern  .func () _rti_statistic_add_uint_64 (.reg .u64 $ptr, .reg .u32 value)\n"
    "     .extern  .func () _rti_statistic_add_uint64_64 (.reg .u64 $ptr, .reg .u32 value)\n"
    "     .extern  .func () _rti_statistic_add_float_64 (.reg .u64 $ptr, .reg .f32 value)\n"
    "     .extern  .func () _rti_statistic_vector_add_int_64 (.reg .u64 $ptr, .reg .s32 value, .reg .u32 idx, .reg .u32 n)\n"
    "     .extern  .func () _rti_statistic_vector_add_uint_64 (.reg .u64 $ptr, .reg .u32 value, .reg .u32 idx, .reg .u32 n)\n"
    "     .extern  .func () _rti_statistic_vector_add_uint64_64 (.reg .u64 $ptr, .reg .u32 value, .reg .u32 idx, .reg .u32 n)\n"
    "     .extern  .func () _rti_statistic_vector_add_float_64 (.reg .u64 $ptr, .reg .f32 value, .reg .u32 idx, .reg .u32 n)\n"
    "     .extern  .func (.reg .s32 $value) _rti_statistic_vector_get_int_64 (.reg .u64 $ptr, .reg .u32 idx, .reg .u32 n)\n"
    "     .extern  .func (.reg .u32 $value) _rti_statistic_vector_get_uint_64 (.reg .u64 $ptr, .reg .u32 idx, .reg .u32 n)\n"
    "     .extern  .func (.reg .u32 $value) _rti_statistic_vector_get_uint64_64 (.reg .u64 $ptr, .reg .u32 idx, .reg .u32 n)\n"
    "     .extern  .func (.reg .f32 $value) _rti_statistic_vector_get_float_64 (.reg .u64 $ptr, .reg .u32 idx, .reg .u32 n)\n"
    "     .extern  .func (.reg .s32 $value) _rti_statistic_get_int_64 (.reg .u64 $ptr)\n"
    "     .extern  .func (.reg .u32 $value) _rti_statistic_get_uint_64 (.reg .u64 $ptr)\n"
    "     .extern  .func (.reg .u32 $value) _rti_statistic_get_uint64_64 (.reg .u64 $ptr)\n"
    "     .extern  .func (.reg .f32 $value) _rti_statistic_get_float_64 (.reg .u64 $ptr)\n"


    // Optix 7 interface
    "     .extern  .func (.reg .u64 $value) _optix_get_sbt_data_ptr_64 ()\n"
    "     .extern  .func (.reg .u32 $value) _optix_get_launch_index_x ()\n"
    "     .extern  .func (.reg .u32 $value) _optix_get_launch_index_y ()\n"
    "     .extern  .func (.reg .u32 $value) _optix_get_launch_index_z ()\n"
    "     .extern  .func (.reg .u32 $value) _optix_get_launch_dimension_x ()\n"
    "     .extern  .func (.reg .u32 $value) _optix_get_launch_dimension_y ()\n"
    "     .extern  .func (.reg .u32 $value) _optix_get_launch_dimension_z ()\n"
    "     .extern  .func (.reg .u32 $value) _optix_read_instance_id()\n"
    "     .extern  .func (.reg .u32 $value) _optix_read_instance_idx()\n"
    "     .extern  .func (.reg .u32 $value) _optix_read_primitive_idx()\n"
    "     .extern  .func (.reg .u64 $value) _optix_read_prim_va()\n"
    "     .extern  .func (.reg .f32 $value) _optix_read_key_time()\n"
    "     .extern  .func (.reg .u32 $value) _optix_read_sbt_gas_idx()\n"
    "     .extern  .func () _optix_ignore_intersection()\n"
    "     .extern  .func () _optix_terminate_ray()\n"
    "     .extern  .func (.reg .u32 $value) _optix_get_ray_flags()\n"
    "     .extern  .func (.reg .u32 $value) _optix_get_ray_visibility_mask()\n"
    "     .extern  .func (.reg .f32 $value) _optix_get_ray_tmin()\n"
    "     .extern  .func (.reg .f32 $value) _optix_get_ray_tmax()\n"
    "     .extern  .func (.reg .f32 $value) _optix_get_ray_time()\n"
    "     .extern  .func (.reg .f32 $value) _optix_get_world_ray_origin_x()\n"
    "     .extern  .func (.reg .f32 $value) _optix_get_world_ray_origin_y()\n"
    "     .extern  .func (.reg .f32 $value) _optix_get_world_ray_origin_z()\n"
    "     .extern  .func (.reg .f32 $value) _optix_get_world_ray_direction_x()\n"
    "     .extern  .func (.reg .f32 $value) _optix_get_world_ray_direction_y()\n"
    "     .extern  .func (.reg .f32 $value) _optix_get_world_ray_direction_z()\n"
    "     .extern  .func (.reg .f32 $value) _optix_get_object_ray_origin_x()\n"
    "     .extern  .func (.reg .f32 $value) _optix_get_object_ray_origin_y()\n"
    "     .extern  .func (.reg .f32 $value) _optix_get_object_ray_origin_z()\n"
    "     .extern  .func (.reg .f32 $value) _optix_get_object_ray_direction_x()\n"
    "     .extern  .func (.reg .f32 $value) _optix_get_object_ray_direction_y()\n"
    "     .extern  .func (.reg .f32 $value) _optix_get_object_ray_direction_z()\n"
    "     .extern  .func (.reg .u32 $value) _optix_get_hit_kind()\n"
    "     .extern  .func (.reg .u32 $value) _optix_get_primitive_type_from_hit_kind(.reg .u32 $hitKind)\n"
    "     .extern  .func (.reg .u32 $value) _optix_get_backface_from_hit_kind(.reg .u32 $hitKind)\n"

    "     .extern  .func (.reg .u32 $value) _optix_get_transform_list_size()\n"
    "     .extern  .func (.reg .u64 $value) _optix_get_transform_list_handle(.reg .u32 $index)\n"
    "     .extern  .func (.reg .u32 $type) _optix_get_transform_type_from_handle(.reg .u64 $handle)\n"

    "     .extern  .func (.reg .u64 $handle) _optix_get_instance_traversable_from_ias(.reg .u64 $handle, .reg .u32 $iidx )\n"

    "     .extern  .func (.reg .f32 $v0x, .reg .f32 $v0y, .reg .f32 $v0z, .reg .f32 $v1x, .reg .f32 $v1y, .reg .f32 $v1z, .reg .f32 $v2x, .reg .f32 $v2y, .reg .f32 $v2z ) _optix_get_triangle_vertex_data(.reg .u64 $handle, .reg .u32 $pidx, .reg .u32 $sidx, .reg .f32 $time )\n"
    "     .extern  .func (.reg .f32 $v0x, .reg .f32 $v0y, .reg .f32 $v0z, .reg .f32 $v0w, .reg .f32 $v1x, .reg .f32 $v1y, .reg .f32 $v1z, .reg .f32 $v1w ) _optix_get_linear_lwrve_vertex_data(.reg .u64 $handle, .reg .u32 $pidx, .reg .u32 $sidx, .reg .f32 $time )\n"
    "     .extern  .func (.reg .f32 $v0x, .reg .f32 $v0y, .reg .f32 $v0z, .reg .f32 $v0w, .reg .f32 $v1x, .reg .f32 $v1y, .reg .f32 $v1z, .reg .f32 $v1w, .reg .f32 $v2x, .reg .f32 $v2y, .reg .f32 $v2z, .reg .f32 $v2w ) _optix_get_quadratic_bspline_vertex_data(.reg .u64 $handle, .reg .u32 $pidx, .reg .u32 $sidx, .reg .f32 $time )\n"
    "     .extern  .func (.reg .f32 $v0x, .reg .f32 $v0y, .reg .f32 $v0z, .reg .f32 $v0w, .reg .f32 $v1x, .reg .f32 $v1y, .reg .f32 $v1z, .reg .f32 $v1w, .reg .f32 $v2x, .reg .f32 $v2y, .reg .f32 $v2z, .reg .f32 $v2w, .reg .f32 $v3x, .reg .f32 $v3y, .reg .f32 $v3z, .reg .f32 $v3w ) _optix_get_lwbic_bspline_vertex_data(.reg .u64 $handle, .reg .u32 $pidx, .reg .u32 $sidx, .reg .f32 $time )\n"
    "     .extern  .func (.reg .f32 $v0x, .reg .f32 $v0y, .reg .f32 $v0z, .reg .f32 $v0w, .reg .f32 $v1x, .reg .f32 $v1y, .reg .f32 $v1z, .reg .f32 $v1w, .reg .f32 $v2x, .reg .f32 $v2y, .reg .f32 $v2z, .reg .f32 $v2w, .reg .f32 $v3x, .reg .f32 $v3y, .reg .f32 $v3z, .reg .f32 $v3w ) _optix_get_catmullrom_vertex_data(.reg .u64 $handle, .reg .u32 $pidx, .reg .u32 $sidx, .reg .f32 $time )\n"
    "     .extern  .func (.reg .f32 $v0x, .reg .f32 $v0y, .reg .f32 $v0z, .reg .f32 $v0w ) _optix_get_sphere_data(.reg .u64 $handle, .reg .u32 $pidx, .reg .u32 $sidx, .reg .f32 $time )\n"

    "     .extern  .func (.reg .u32 $type) _optix_get_instance_id_from_handle(.reg .u64 $handle)\n"
    "     .extern  .func (.reg .u64 $type) _optix_get_instance_child_from_handle(.reg .u64 $handle)\n"
    "     .extern  .func (.reg .u64 $type) _optix_get_instance_transform_from_handle(.reg .u64 $handle)\n"
    "     .extern  .func (.reg .u64 $type) _optix_get_instance_ilwerse_transform_from_handle(.reg .u64 $handle)\n"
    "     .extern  .func (.reg .u64 $type) _optix_get_static_transform_from_handle(.reg .u64 $handle)\n"
    "     .extern  .func (.reg .u64 $type) _optix_get_matrix_motion_transform_from_handle(.reg .u64 $handle)\n"
    "     .extern  .func (.reg .u64 $type) _optix_get_srt_motion_transform_from_handle(.reg .u64 $handle)\n"

    "     .extern  .func (.reg .u64 $type) _optix_get_gas_ptr(.reg .u64 $handle)\n"

    "     .extern  .func (.reg .u64 $type) _optix_get_gas_traversable_handle()\n"

    "     .extern  .func (.reg .f32 $type) _optix_get_gas_motion_time_begin(.reg .u64 $handle)\n"
    "     .extern  .func (.reg .f32 $type) _optix_get_gas_motion_time_end(.reg .u64 $handle)\n"
    "     .extern  .func (.reg .u32 $type) _optix_get_gas_motion_step_count(.reg .u64 $handle)\n"

    "     .extern  .func (.reg .u32 $value) _optix_undef_value()\n"

    "     .extern  .func () _optix_set_payload_0(.reg .u32 $value)\n"
    "     .extern  .func () _optix_set_payload_1(.reg .u32 $value)\n"
    "     .extern  .func () _optix_set_payload_2(.reg .u32 $value)\n"
    "     .extern  .func () _optix_set_payload_3(.reg .u32 $value)\n"
    "     .extern  .func () _optix_set_payload_4(.reg .u32 $value)\n"
    "     .extern  .func () _optix_set_payload_5(.reg .u32 $value)\n"
    "     .extern  .func () _optix_set_payload_6(.reg .u32 $value)\n"
    "     .extern  .func () _optix_set_payload_7(.reg .u32 $value)\n"
    "     .extern  .func () _optix_set_payload(.reg .u32 $index, .reg .u32 $value)\n"
    "     .extern  .func (.reg .u32 $value) _optix_get_payload_0()\n"
    "     .extern  .func (.reg .u32 $value) _optix_get_payload_1()\n"
    "     .extern  .func (.reg .u32 $value) _optix_get_payload_2()\n"
    "     .extern  .func (.reg .u32 $value) _optix_get_payload_3()\n"
    "     .extern  .func (.reg .u32 $value) _optix_get_payload_4()\n"
    "     .extern  .func (.reg .u32 $value) _optix_get_payload_5()\n"
    "     .extern  .func (.reg .u32 $value) _optix_get_payload_6()\n"
    "     .extern  .func (.reg .u32 $value) _optix_get_payload_7()\n"
    "     .extern  .func (.reg .u32 $value) _optix_get_payload(.reg .u32 $index)\n"

    "     .extern  .func () _optix_set_payload_types(.reg .u32 $type)\n"

    "     .extern  .func (.reg .u32 $value) _optix_get_attribute_0()\n"
    "     .extern  .func (.reg .u32 $value) _optix_get_attribute_1()\n"
    "     .extern  .func (.reg .u32 $value) _optix_get_attribute_2()\n"
    "     .extern  .func (.reg .u32 $value) _optix_get_attribute_3()\n"
    "     .extern  .func (.reg .u32 $value) _optix_get_attribute_4()\n"
    "     .extern  .func (.reg .u32 $value) _optix_get_attribute_5()\n"
    "     .extern  .func (.reg .u32 $value) _optix_get_attribute_6()\n"
    "     .extern  .func (.reg .u32 $value) _optix_get_attribute_7()\n"

    "     .extern  .func (.reg .u32 $value) _optix_report_intersection_0(.reg .f32 $hitT, .reg .u32 $hitKind)\n"
    "     .extern  .func (.reg .u32 $value) _optix_report_intersection_1(.reg .f32 $hitT, .reg .u32 $hitKind, .reg .u32 $a0)\n"
    "     .extern  .func (.reg .u32 $value) _optix_report_intersection_2(.reg .f32 $hitT, .reg .u32 $hitKind, .reg .u32 $a0, .reg .u32 $a1)\n"
    "     .extern  .func (.reg .u32 $value) _optix_report_intersection_3(.reg .f32 $hitT, .reg .u32 $hitKind, .reg .u32 $a0, .reg .u32 $a1, .reg .u32 $a2)\n"
    "     .extern  .func (.reg .u32 $value) _optix_report_intersection_4(.reg .f32 $hitT, .reg .u32 $hitKind, .reg .u32 $a0, .reg .u32 $a1, .reg .u32 $a2, .reg .u32 $a3)\n"
    "     .extern  .func (.reg .u32 $value) _optix_report_intersection_5(.reg .f32 $hitT, .reg .u32 $hitKind, .reg .u32 $a0, .reg .u32 $a1, .reg .u32 $a2, .reg .u32 $a3, .reg .u32 $a4)\n"
    "     .extern  .func (.reg .u32 $value) _optix_report_intersection_6(.reg .f32 $hitT, .reg .u32 $hitKind, .reg .u32 $a0, .reg .u32 $a1, .reg .u32 $a2, .reg .u32 $a3, .reg .u32 $a4, .reg .u32 $a5)\n"
    "     .extern  .func (.reg .u32 $value) _optix_report_intersection_7(.reg .f32 $hitT, .reg .u32 $hitKind, .reg .u32 $a0, .reg .u32 $a1, .reg .u32 $a2, .reg .u32 $a3, .reg .u32 $a4, .reg .u32 $a5, .reg .u32 $a6)\n"
    "     .extern  .func (.reg .u32 $value) _optix_report_intersection_8(.reg .f32 $hitT, .reg .u32 $hitKind, .reg .u32 $a0, .reg .u32 $a1, .reg .u32 $a2, .reg .u32 $a3, .reg .u32 $a4, .reg .u32 $a5, .reg .u32 $a6, .reg .u32 $a7)\n"

    "     .extern  .func () _optix_trace_0( .reg .u64 handle, .reg .f32 rayDirectionX, .reg .f32 rayDirectionY, .reg .f32 rayDirectionZ, .reg .f32 rayOriginX, .reg .f32 rayOriginY, .reg .f32 rayOriginZ, .reg .f32 tmin, .reg .f32 tmax, .reg .f32 rayTime, .reg .u32 visibilityMask, .reg .u32 rayFlags, .reg .u32 SBToffset, .reg .u32 SBTstride, .reg .u32 missSBTIndex )\n"
    "     .extern  .func (.reg .u32 p0) _optix_trace_1( .reg .u64 handle, .reg .f32 rayDirectionX, .reg .f32 rayDirectionY, .reg .f32 rayDirectionZ, .reg .f32 rayOriginX, .reg .f32 rayOriginY, .reg .f32 rayOriginZ, .reg .f32 tmin, .reg .f32 tmax, .reg .f32 rayTime, .reg .u32 visibilityMask, .reg .u32 rayFlags, .reg .u32 SBToffset, .reg .u32 SBTstride, .reg .u32 missSBTIndex, .reg .u32 p0 )\n"
    "     .extern  .func (.reg .u32 p0, .reg .u32 p1) _optix_trace_2( .reg .u64 handle, .reg .f32 rayDirectionX, .reg .f32 rayDirectionY, .reg .f32 rayDirectionZ, .reg .f32 rayOriginX, .reg .f32 rayOriginY, .reg .f32 rayOriginZ, .reg .f32 tmin, .reg .f32 tmax, .reg .f32 rayTime, .reg .u32 visibilityMask, .reg .u32 rayFlags, .reg .u32 SBToffset, .reg .u32 SBTstride, .reg .u32 missSBTIndex, .reg .u32 p0, .reg .u32 p1 )\n"
    "     .extern  .func (.reg .u32 p0, .reg .u32 p1, .reg .u32 p2) _optix_trace_3( .reg .u64 handle, .reg .f32 rayDirectionX, .reg .f32 rayDirectionY, .reg .f32 rayDirectionZ, .reg .f32 rayOriginX, .reg .f32 rayOriginY, .reg .f32 rayOriginZ, .reg .f32 tmin, .reg .f32 tmax, .reg .f32 rayTime, .reg .u32 visibilityMask, .reg .u32 rayFlags, .reg .u32 SBToffset, .reg .u32 SBTstride, .reg .u32 missSBTIndex, .reg .u32 p0, .reg .u32 p1, .reg .u32 p2 )\n"
    "     .extern  .func (.reg .u32 p0, .reg .u32 p1, .reg .u32 p2, .reg .u32 p3) _optix_trace_4( .reg .u64 handle, .reg .f32 rayDirectionX, .reg .f32 rayDirectionY, .reg .f32 rayDirectionZ, .reg .f32 rayOriginX, .reg .f32 rayOriginY, .reg .f32 rayOriginZ, .reg .f32 tmin, .reg .f32 tmax, .reg .f32 rayTime, .reg .u32 visibilityMask, .reg .u32 rayFlags, .reg .u32 SBToffset, .reg .u32 SBTstride, .reg .u32 missSBTIndex, .reg .u32 p0, .reg .u32 p1, .reg .u32 p2, .reg .u32 p3 )\n"
    "     .extern  .func (.reg .u32 p0, .reg .u32 p1, .reg .u32 p2, .reg .u32 p3, .reg .u32 p4) _optix_trace_5( .reg .u64 handle, .reg .f32 rayDirectionX, .reg .f32 rayDirectionY, .reg .f32 rayDirectionZ, .reg .f32 rayOriginX, .reg .f32 rayOriginY, .reg .f32 rayOriginZ, .reg .f32 tmin, .reg .f32 tmax, .reg .f32 rayTime, .reg .u32 visibilityMask, .reg .u32 rayFlags, .reg .u32 SBToffset, .reg .u32 SBTstride, .reg .u32 missSBTIndex, .reg .u32 p0, .reg .u32 p1, .reg .u32 p2, .reg .u32 p3, .reg .u32 p4 )\n"
    "     .extern  .func (.reg .u32 p0, .reg .u32 p1, .reg .u32 p2, .reg .u32 p3, .reg .u32 p4, .reg .u32 p5) _optix_trace_6( .reg .u64 handle, .reg .f32 rayDirectionX, .reg .f32 rayDirectionY, .reg .f32 rayDirectionZ, .reg .f32 rayOriginX, .reg .f32 rayOriginY, .reg .f32 rayOriginZ, .reg .f32 tmin, .reg .f32 tmax, .reg .f32 rayTime, .reg .u32 visibilityMask, .reg .u32 rayFlags, .reg .u32 SBToffset, .reg .u32 SBTstride, .reg .u32 missSBTIndex, .reg .u32 p0, .reg .u32 p1, .reg .u32 p2, .reg .u32 p3, .reg .u32 p4, .reg .u32 p5 )\n"
    "     .extern  .func (.reg .u32 p0, .reg .u32 p1, .reg .u32 p2, .reg .u32 p3, .reg .u32 p4, .reg .u32 p5, .reg .u32 p6) _optix_trace_7( .reg .u64 handle, .reg .f32 rayDirectionX, .reg .f32 rayDirectionY, .reg .f32 rayDirectionZ, .reg .f32 rayOriginX, .reg .f32 rayOriginY, .reg .f32 rayOriginZ, .reg .f32 tmin, .reg .f32 tmax, .reg .f32 rayTime, .reg .u32 visibilityMask, .reg .u32 rayFlags, .reg .u32 SBToffset, .reg .u32 SBTstride, .reg .u32 missSBTIndex, .reg .u32 p0, .reg .u32 p1, .reg .u32 p2, .reg .u32 p3, .reg .u32 p4, .reg .u32 p5, .reg .u32 p6 )\n"
    "     .extern  .func (.reg .u32 p0, .reg .u32 p1, .reg .u32 p2, .reg .u32 p3, .reg .u32 p4, .reg .u32 p5, .reg .u32 p6, .reg .u32 p7) _optix_trace_8( .reg .u64 handle, .reg .f32 rayDirectionX, .reg .f32 rayDirectionY, .reg .f32 rayDirectionZ, .reg .f32 rayOriginX, .reg .f32 rayOriginY, .reg .f32 rayOriginZ, .reg .f32 tmin, .reg .f32 tmax, .reg .f32 rayTime, .reg .u32 visibilityMask, .reg .u32 rayFlags, .reg .u32 SBToffset, .reg .u32 SBTstride, .reg .u32 missSBTIndex, .reg .u32 p0, .reg .u32 p1, .reg .u32 p2, .reg .u32 p3, .reg .u32 p4, .reg .u32 p5, .reg .u32 p6, .reg .u32 p7 )\n"

    "     .extern  .func (.reg .u32 p0, .reg .u32 p1, .reg .u32 p2, .reg .u32 p3, .reg .u32 p4, .reg .u32 p5, .reg .u32 p6, .reg .u32 p7, .reg .u32 p8, .reg .u32 p9, .reg .u32 p10, .reg .u32 p11, .reg .u32 p12, .reg .u32 p13, .reg .u32 p14, .reg .u32 p15, .reg .u32 p16, .reg .u32 p17, .reg .u32 p18, .reg .u32 p19, .reg .u32 p20, .reg .u32 p21, .reg .u32 p22, .reg .u32 p23, .reg .u32 p24, .reg .u32 p25, .reg .u32 p26, .reg .u32 p27, .reg .u32 p28, .reg .u32 p29, .reg .u32 p30, .reg .u32 p31) _optix_trace_typed_32( .reg .u32 type, .reg .u64 handle, .reg .f32 rayDirectionX, .reg .f32 rayDirectionY, .reg .f32 rayDirectionZ, .reg .f32 rayOriginX, .reg .f32 rayOriginY, .reg .f32 rayOriginZ, .reg .f32 tmin, .reg .f32 tmax, .reg .f32 rayTime, .reg .u32 visibilityMask, .reg .u32 rayFlags, .reg .u32 SBToffset, .reg .u32 SBTstride, .reg .u32 missSBTIndex, .reg .u32 numPayloadValues, .reg .u32 p0, .reg .u32 p1, .reg .u32 p2, .reg .u32 p3, .reg .u32 p4, .reg .u32 p5, .reg .u32 p6, .reg .u32 p7, .reg .u32 p8, .reg .u32 p9, .reg .u32 p10, .reg .u32 p11, .reg .u32 p12, .reg .u32 p13, .reg .u32 p14, .reg .u32 p15, .reg .u32 p16, .reg .u32 p17, .reg .u32 p18, .reg .u32 p19, .reg .u32 p20, .reg .u32 p21, .reg .u32 p22, .reg .u32 p23, .reg .u32 p24, .reg .u32 p25, .reg .u32 p26, .reg .u32 p27, .reg .u32 p28, .reg .u32 p29, .reg .u32 p30, .reg .u32 p31 )\n"

    "     .extern  .func (.reg .f32 $uValue, .reg .f32 $uValue) _optix_get_triangle_barycentrics()\n"

    "     .extern  .func () _optix_throw_exception_0(.reg .s32 $code)\n"
    "     .extern  .func () _optix_throw_exception_1(.reg .s32 $code, .reg .u32 $detail0)\n"
    "     .extern  .func () _optix_throw_exception_2(.reg .s32 $code, .reg .u32 $detail0, .reg .u32 $detail1)\n"
    "     .extern  .func () _optix_throw_exception_3(.reg .s32 $code, .reg .u32 $detail0, .reg .u32 $detail1, .reg .u32 $detail2)\n"
    "     .extern  .func () _optix_throw_exception_4(.reg .s32 $code, .reg .u32 $detail0, .reg .u32 $detail1, .reg .u32 $detail2, .reg .u32 $detail3)\n"
    "     .extern  .func () _optix_throw_exception_5(.reg .s32 $code, .reg .u32 $detail0, .reg .u32 $detail1, .reg .u32 $detail2, .reg .u32 $detail3, .reg .u32 $detail4)\n"
    "     .extern  .func () _optix_throw_exception_6(.reg .s32 $code, .reg .u32 $detail0, .reg .u32 $detail1, .reg .u32 $detail2, .reg .u32 $detail3, .reg .u32 $detail4, .reg .u32 $detail5)\n"
    "     .extern  .func () _optix_throw_exception_7(.reg .s32 $code, .reg .u32 $detail0, .reg .u32 $detail1, .reg .u32 $detail2, .reg .u32 $detail3, .reg .u32 $detail4, .reg .u32 $detail5, .reg .u32 $detail6)\n"
    "     .extern  .func () _optix_throw_exception_8(.reg .s32 $code, .reg .u32 $detail0, .reg .u32 $detail1, .reg .u32 $detail2, .reg .u32 $detail3, .reg .u32 $detail4, .reg .u32 $detail5, .reg .u32 $detail6, .reg .u32 $detail7)\n"

    "     .extern  .func (.reg .s32 $value) _optix_get_exception_code()\n"

    "     .extern  .func (.reg .u32 $value) _optix_get_exception_detail_0()\n"
    "     .extern  .func (.reg .u32 $value) _optix_get_exception_detail_1()\n"
    "     .extern  .func (.reg .u32 $value) _optix_get_exception_detail_2()\n"
    "     .extern  .func (.reg .u32 $value) _optix_get_exception_detail_3()\n"
    "     .extern  .func (.reg .u32 $value) _optix_get_exception_detail_4()\n"
    "     .extern  .func (.reg .u32 $value) _optix_get_exception_detail_5()\n"
    "     .extern  .func (.reg .u32 $value) _optix_get_exception_detail_6()\n"
    "     .extern  .func (.reg .u32 $value) _optix_get_exception_detail_7()\n"

    "     .extern  .func (.reg .u64 $value) _optix_get_exception_ilwalid_traversable()\n"
    "     .extern  .func (.reg .s32 $value) _optix_get_exception_ilwalid_sbt_offset()\n"
    "     .extern  .func (.reg .f32 $rayOriginX, .reg .f32 $rayOriginY, .reg .f32 $rayOriginZ, .reg .f32 $rayDirectionX, .reg .f32 $rayDirectionY, .reg .f32 $rayDirectionZ, .reg .f32 $tmin, .reg .f32 $tmax, .reg .f32 $rayTime) _optix_get_exception_ilwalid_ray()\n"
    "     .extern  .func (.reg .u32 $expected, .reg .u32 $actual, .reg .u32 $sbtIndex, .reg .u64 $calleeName) _optix_get_exception_parameter_mismatch()\n"
    "     .extern  .func (.reg .u64 $lineInfo) _optix_get_exception_line_info()\n"

    "     .extern  .func (.reg .u64 $value) _optix_call_direct_callable(.reg .u32 idx)\n"
    "     .extern  .func (.reg .u64 $value) _optix_call_continuation_callable(.reg .u32 idx)\n"
    "     .extern  .func () _optix_tex_footprint_2d(.reg .u64 sampler, .reg .f32 x, .reg .f32 y, .reg .u32 coarse, .reg .u32 granularity, .reg .u64 singleMipLevelPtr, .reg .u64 resultPtr )\n"
    "     .extern  .func () _optix_tex_footprint_2d_grad(.reg .u64 sampler, .reg .f32 x, .reg .f32 y, .reg .f32 dPdx_x, .reg .f32 dPdx_y, .reg .f32 dPdy_x, .reg .f32 dPdy_y, .reg .u32 coarse, .reg .u32 granularity, .reg .u64 singleMipLevelPtr, .reg .u64 resultPtr )\n"
    "     .extern  .func () _optix_tex_footprint_2d_lod(.reg .u64 sampler, .reg .f32 x, .reg .f32 y, .reg .f32 level, .reg .u32 coarse, .reg .u32 granularity, .reg .u64 singleMipLevelPtr, .reg .u64 resultPtr )\n"
    "     .extern  .func () _optix_tex_footprint_2d_v2     (.reg .u64 tex, .reg .u32 texInfo, .reg .u32 x, .reg .u32 y, .reg .u64 singleMipLevelPtr, .reg .u64 resultPtr )\n"
    "     .extern  .func () _optix_tex_footprint_2d_grad_v2(.reg .u64 tex, .reg .u32 texInfo, .reg .u32 x, .reg .u32 y, .reg .u32 dPdx_x, .reg .u32 dPdx_y, .reg .u32 dPdy_x, .reg .u32 dPdy_y, .reg .u32 coarse, .reg .u64 singleMipLevelPtr, .reg .u64 resultPtr )\n"
    "     .extern  .func () _optix_tex_footprint_2d_lod_v2 (.reg .u64 tex, .reg .u32 texInfo, .reg .u32 x, .reg .u32 y, .reg .u32 level, .reg .u32 coarse, .reg .u64 singleMipLevelPtr, .reg .u64 resultPtr )\n"

    "     .extern  .func (.reg .u32 $value) _optix_private_get_compile_time_constant(.reg .u32 idx)\n"

;
// clang-format on

void getDeclarationsFromPtx( const prodlib::StringView& ptx, std::string& decls )
{
    const char* off = ptx.data();
    const char* end = ptx.data() + ptx.size();
    while( off < end && ( off = prodlib::strNStr( off, "_rti_comment_", end - off ) ) != nullptr )
    {
        const char* begin = off;
        while( off < end && *off != ' ' )
        {
            ++off;
        };

        std::string name;
        name.assign( begin, off );
        decls += "     .extern .func () " + name + " ()\n";
    }
}

std::string createPTXHeaderString( const prodlib::StringView& ptx )
{
    std::string decls( ptx_declarations );
    getDeclarationsFromPtx( ptx, decls );
    return decls;
}

std::string createPTXHeaderString( const std::vector<prodlib::StringView>& ptxStrings )
{
    std::string decls( ptx_declarations );
    for( const prodlib::StringView& ptx : ptxStrings )
        getDeclarationsFromPtx( ptx, decls );
    return decls;
}

std::string retrieveOptixPTXDeclarations()
{
    return ptx_declarations;
}

}  // namespace optix
