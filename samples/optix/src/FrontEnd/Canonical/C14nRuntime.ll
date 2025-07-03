; NOTE: Make sure these types match exactly the ones in the bitcode of CommonRuntime.h

%"struct.cort::CanonicalState" = type { %"struct.cort::Global", %"struct.cort::Raygen", %"struct.cort::Exception", %"struct.cort::TerminateRay", %"struct.cort::Scopes", %"struct.cort::ActiveProgram", %"struct.cort::AabbParameters", %"struct.cort::TraversalParameters", %"struct.cort::IntersectParameters", %"struct.cort::TraceFrame"*, %"struct.cort::RayFrame"* }
%"struct.cort::Global" = type { i8*, %"struct.cort::Buffer"*, %"struct.cort::TextureSampler"*, %"struct.cort::ProgramHeader"*, %"struct.cort::TraversableHeader"*, i16*, i32*, i32*, i64*, i64*, i32, i64*, %"struct.cort::FrameStatus"*, i32, i32, i32, i32, i32, %"struct.cort::uint3", %"struct.cort::uint3", i8, i16, i16, i32, i32, i64, %"struct.cort::AabbRequest", i16, i16, i16, i16, i16, [14 x i8] }
%"struct.cort::Buffer" = type { %"struct.cort::Buffer::DeviceIndependent", %"struct.cort::Buffer::DeviceDependent" }
%"struct.cort::Buffer::DeviceIndependent" = type { %"struct.cort::size3", %"struct.cort::uint3" }
%"struct.cort::size3" = type { i64, i64, i64 }
%"struct.cort::Buffer::DeviceDependent" = type { i8*, i32 }
%"struct.cort::TextureSampler" = type { i32, i32, i32, i32, i32, float, [4 x i8], [4 x i8], [4 x i8], i32, %"struct.cort::TextureSampler::DeviceDependent" }
%"struct.cort::TextureSampler::DeviceDependent" = type { i64, i8*, i8, [7 x i8] }
%"struct.cort::ProgramHeader" = type { %"struct.cort::ProgramHeader::DeviceIndependent", %"struct.cort::ProgramHeader::DeviceDependent" }
%"struct.cort::ProgramHeader::DeviceIndependent" = type { i32 }
%"struct.cort::ProgramHeader::DeviceDependent" = type { i32 }
%"struct.cort::TraversableHeader" = type { i64 }
%"struct.cort::FrameStatus" = type { i32, i8, i32, i32, i32, i32, float, float, float, float, i8*, i8*, i8*, i8* }
%"struct.cort::uint2" = type { i32, i32 }
%"struct.cort::uint3" = type { i32, i32, i32 }
%"struct.cort::AabbRequest" = type { i8, i32, i32, i32, i8, %"struct.cort::Aabb"*, %"struct.cort::uint2"* }
%"struct.cort::Aabb" = type { %"struct.cort::float3", %"struct.cort::float3" }
%"struct.cort::float3" = type { float, float, float }
%"struct.cort::Raygen" = type { %"struct.cort::uint3" }
%"struct.cort::Exception" = type { i32, [12 x i64] }
%"struct.cort::TerminateRay" = type { i8 }
%"struct.cort::Scopes" = type { i32, i32 }
%"struct.cort::ActiveProgram" = type { i32 }
%"struct.cort::AabbParameters" = type { float*, i32, i32 }
%"struct.cort::TraversalParameters" = type { i32 }
%"struct.cort::IntersectParameters" = type { i32 }
%"struct.cort::TraceFrame" = type { float, float, float, i16, i8, i8, i32, i32, i32, i32, i8, i8*, [256 x i8], [256 x i8], [16 x i32], [16 x i32] }
%"struct.cort::RayFrame" = type { %"struct.cort::float3", %"struct.cort::float3", float }
%"struct.cort::MaterialRecord" = type { %"struct.cort::LexicalScopeRecord", [1 x %struct.anon.0] }
%"struct.cort::LexicalScopeRecord" = type { i32 }
%struct.anon.0 = type { i32, i32 }
%"struct.cort::GlobalScopeRecord" = type { %"struct.cort::LexicalScopeRecord", [1 x %struct.anon] }
%struct.anon = type { i32, i32, i32 }
%"struct.cort::GeneralBB" = type { %"struct.cort::float3", %"struct.cort::float3", %"struct.cort::float3", %"struct.cort::float3", i8 }
%"struct.cort::float4" = type { float, float, float, float }
%"struct.cort::float2" = type { float, float }
%"struct.cort::Matrix4x4" = type { [4 x [4 x float]] }
%"struct.cort::OptixRay" = type { %"struct.cort::float3", %"struct.cort::float3", i32, float, float }
%"struct.cort::GeometryInstanceRecord" = type { %"struct.cort::LexicalScopeRecord", i32, i32, [1 x i32] }


declare %"struct.cort::CanonicalState"* @optixi_getState()

; _rti_compute_geometry_instance_aabb_64
declare void @optixi_computeGeometryInstanceAABB(%"struct.cort::CanonicalState"* %state, i32 %GIOffset, i32 %primitive, i32 %motionStep, float* %aabb)
define linkonce_odr void @_rti_compute_geometry_instance_aabb_64( i32 %giOffset, i32 %primitive, i32 %motionStep, i64 %aabb) nounwind alwaysinline {
  %state = call %"struct.cort::CanonicalState"* @optixi_getState()
  %aabbp_gen = inttoptr i64 %aabb to float*
  call void @optixi_computeGeometryInstanceAABB( %"struct.cort::CanonicalState"* %state, i32 %giOffset, i32 %primitive, i32 %motionStep, float* %aabbp_gen )
  ret void
}

; _rti_compute_group_child_aabb_64
declare void @optixi_computeGroupChildAABB(%"struct.cort::CanonicalState"* %state, i32 %groupOffset, i32 %child, float* %aabb)
define linkonce_odr void @_rti_compute_group_child_aabb_64( i32 %groupOffset, i32 %child, i64 %aabb) nounwind alwaysinline {
  %state = call %"struct.cort::CanonicalState"* @optixi_getState()
  %aabbp_gen = inttoptr i64 %aabb to float*
  call void @optixi_computeGroupChildAABB( %"struct.cort::CanonicalState"* %state, i32 %groupOffset, i32 %child, float* %aabbp_gen )
  ret void
}

; _rti_gather_motion_aabbs_64
declare void @optixi_gatherMotionAABBs(%"struct.cort::CanonicalState"* %state, i32 %groupOffset, float* %aabb)
define linkonce_odr void @_rti_gather_motion_aabbs_64( i32 %groupOffset, i64 %aabb) nounwind alwaysinline {
  %state = call %"struct.cort::CanonicalState"* @optixi_getState()
  %aabbp_gen = inttoptr i64 %aabb to float*
  call void @optixi_gatherMotionAABBs( %"struct.cort::CanonicalState"* %state, i32 %groupOffset, float* %aabbp_gen )
  ret void
}

; _rt_get_exception_code
declare i32 @optixi_getExceptionCode(%"struct.cort::CanonicalState"* %state)
define linkonce_odr i32 @_rt_get_exception_code() alwaysinline
{
   %state = call %"struct.cort::CanonicalState"* @optixi_getState()
   %code = call i32 @optixi_getExceptionCode(%"struct.cort::CanonicalState"* %state)
   ret i32 %code
}

; _rti_get_status
declare %"struct.cort::FrameStatus"* @optixi_getFrameStatus(%"struct.cort::CanonicalState"* %state)
define linkonce_odr i64 @_rti_get_status_return() nounwind alwaysinline
{
  %state = call %"struct.cort::CanonicalState"* @optixi_getState()
  %s = call %"struct.cort::FrameStatus"* @optixi_getFrameStatus(%"struct.cort::CanonicalState"* %state)
  %res = ptrtoint %"struct.cort::FrameStatus"* %s to i64
  ret i64 %res
}

; _rti_get_primitive_index_offset
declare i32 @optixi_getPrimitiveIndexOffset(%"struct.cort::CanonicalState"* %state, i32 %GIOffset)
define linkonce_odr i32 @_rti_get_primitive_index_offset( i32 %child ) nounwind alwaysinline
{
  %state = call %"struct.cort::CanonicalState"* @optixi_getState()
  %offset = call i32 @optixi_getPrimitiveIndexOffset(%"struct.cort::CanonicalState"* %state, i32 %child)
  ret i32 %offset
}

; _rti_get_aabb_request
declare void @optixi_getAabbRequest(%"struct.cort::CanonicalState"* %state, %"struct.cort::AabbRequest"* %ptr)
define linkonce_odr void @_rti_get_aabb_request( i64 %iptr ) nounwind alwaysinline
{
  %ptr = inttoptr i64 %iptr to %"struct.cort::AabbRequest"*
  %state = call %"struct.cort::CanonicalState"* @optixi_getState()
  call void @optixi_getAabbRequest(%"struct.cort::CanonicalState"* %state, %"struct.cort::AabbRequest"* %ptr )
  ret void
}

; _rt_ignore_intersection
declare void @optixi_ignoreIntersection(%"struct.cort::CanonicalState"*)
define linkonce_odr void @_rt_ignore_intersection() alwaysinline
{
   %state = call %"struct.cort::CanonicalState"* @optixi_getState()
   call void @optixi_ignoreIntersection(%"struct.cort::CanonicalState"* %state)
   ret void
}

; _rti_intersect_primitive
declare void @optixi_intersectPrimitive(%"struct.cort::CanonicalState"* %state, i32 %GIOffset, i32 %primitiveIndex)
define linkonce_odr void @_rti_intersect_primitive(i32 %child, i32 %primitive) alwaysinline
{
   %state = call %"struct.cort::CanonicalState"* @optixi_getState()
   call void @optixi_intersectPrimitive(%"struct.cort::CanonicalState"* %state, i32 %child, i32 %primitive )
   ret void
}

; _rt_intersect_child
declare void @optixi_intersectChild(%"struct.cort::CanonicalState"* %state, i32 %child)
define linkonce_odr void @_rt_intersect_child( i32 %child ) nounwind alwaysinline
{
   %state = call %"struct.cort::CanonicalState"* @optixi_getState()  
   call void @optixi_intersectChild(%"struct.cort::CanonicalState"* %state, i32 %child)
   ret void
}

; _rti_intersect_node
declare void @optixi_intersectNode(%"struct.cort::CanonicalState"* %state, i32 %child)
define linkonce_odr void @_rti_intersect_node( i32 %child ) nounwind alwaysinline
{
   %state = call %"struct.cort::CanonicalState"* @optixi_getState()
   call void @optixi_intersectNode(%"struct.cort::CanonicalState"* %state, i32 %child)
   ret void
}

; _rt_potential_intersection
declare zeroext i1 @optixi_isPotentialIntersection(%"struct.cort::CanonicalState"* %state, float %t)
define linkonce_odr i32 @_rt_potential_intersection(float %p0 ) alwaysinline
{
   %state = call %"struct.cort::CanonicalState"* @optixi_getState()
   %ret = call i1 @optixi_isPotentialIntersection(%"struct.cort::CanonicalState"* %state, float %p0)
   %retz = zext i1 %ret to i32
   ret i32 %retz
}

; _rt_report_intersection
declare zeroext i1 @optixi_reportIntersection(%"struct.cort::CanonicalState"*, i32, i8)
define linkonce_odr i32 @_rt_report_intersection(i32 %matlIndex) alwaysinline
{
   %state = call %"struct.cort::CanonicalState"* @optixi_getState()
   %ret = call i1 @optixi_reportIntersection(%"struct.cort::CanonicalState"* %state, i32 %matlIndex, i8 0)
   %retz = zext i1 %ret to i32
   ret i32 %retz
}

; _rti_report_full_intersection_ff
declare zeroext i1 @optixi_reportFullIntersection.noUniqueName.float2(%"struct.cort::CanonicalState"*, float, i32, i8, %"struct.cort::float2")
define linkonce_odr i32 @_rti_report_full_intersection_ff(float %t, i32 %matlIndex, i32 %hitKind, float %f0, float %f1) alwaysinline
{
   %state = call %"struct.cort::CanonicalState"* @optixi_getState()
   %hitKind_i8 = trunc i32 %hitKind to i8
   %f  = insertvalue %"struct.cort::float2" undef, float %f0, 0
   %ff = insertvalue %"struct.cort::float2" %f, float %f1, 1
   %ret = call i1 @optixi_reportFullIntersection.noUniqueName.float2(%"struct.cort::CanonicalState"* %state, float %t, i32 %matlIndex, i8 %hitKind_i8, %"struct.cort::float2" %ff )
   %retz = zext i1 %ret to i32
   ret i32 %retz
}

; _rti_set_lwrrent_acceleration
declare void @optixi_setLwrrentAcceleration(%"struct.cort::CanonicalState"*)
define linkonce_odr void @_rti_set_lwrrent_acceleration() alwaysinline
{
   %state = call %"struct.cort::CanonicalState"* @optixi_getState()
   call void @optixi_setLwrrentAcceleration(%"struct.cort::CanonicalState"* %state)
   ret void
}

; _rt_transform_tuple
declare %"struct.cort::float4" @optixi_transformTuple(%"struct.cort::CanonicalState"*, i32, float, float, float, float)
define linkonce_odr { float, float, float, float } @_rt_transform_tuple(i32 %kind, float %x, float %y, float %z, float %w) alwaysinline
{
  %state = call %"struct.cort::CanonicalState"* @optixi_getState()
  %result = call %"struct.cort::float4" @optixi_transformTuple(%"struct.cort::CanonicalState"* %state, i32 %kind, float %x, float %y, float %z, float %w)
  %a = extractvalue %"struct.cort::float4" %result, 0
  %b = extractvalue %"struct.cort::float4" %result, 1
  %c = extractvalue %"struct.cort::float4" %result, 2
  %d = extractvalue %"struct.cort::float4" %result, 3
  %rrrr = insertvalue { float, float, float, float} undef, float %a, 0
  %rrr  = insertvalue { float, float, float, float} %rrrr, float %b, 1
  %rr   = insertvalue { float, float, float, float} %rrr , float %c, 2
  %r    = insertvalue { float, float, float, float} %rr  , float %d, 3
  ret { float, float, float, float } %r
}

; _rt_get_transform
declare %"struct.cort::Matrix4x4" @optixi_getTransform(%"struct.cort::CanonicalState"*, i32)
define linkonce_odr { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } @_rt_get_transform(i32 %kind) alwaysinline
{
  %state = call %"struct.cort::CanonicalState"* @optixi_getState()
  %result = call %"struct.cort::Matrix4x4" @optixi_getTransform(%"struct.cort::CanonicalState"* %state, i32 %kind)

  %matrix = extractvalue %"struct.cort::Matrix4x4" %result, 0

  ; Naming convention: element.row.column
  ; First row.
  %element.0.0 = extractvalue [4 x [4 x float]] %matrix, 0, 0
  %element.0.1 = extractvalue [4 x [4 x float]] %matrix, 0, 1
  %element.0.2 = extractvalue [4 x [4 x float]] %matrix, 0, 2
  %element.0.3 = extractvalue [4 x [4 x float]] %matrix, 0, 3

  ; Second row.
  %element.1.0 = extractvalue [4 x [4 x float]] %matrix, 1, 0
  %element.1.1 = extractvalue [4 x [4 x float]] %matrix, 1, 1
  %element.1.2 = extractvalue [4 x [4 x float]] %matrix, 1, 2
  %element.1.3 = extractvalue [4 x [4 x float]] %matrix, 1, 3

  ; Third row.
  %element.2.0 = extractvalue [4 x [4 x float]] %matrix, 2, 0
  %element.2.1 = extractvalue [4 x [4 x float]] %matrix, 2, 1
  %element.2.2 = extractvalue [4 x [4 x float]] %matrix, 2, 2
  %element.2.3 = extractvalue [4 x [4 x float]] %matrix, 2, 3

  ; Fourth row.
  %element.3.0 = extractvalue [4 x [4 x float]] %matrix, 3, 0
  %element.3.1 = extractvalue [4 x [4 x float]] %matrix, 3, 1
  %element.3.2 = extractvalue [4 x [4 x float]] %matrix, 3, 2
  %element.3.3 = extractvalue [4 x [4 x float]] %matrix, 3, 3

  ; Fill the output.
  ; First row.
  %output.0.0 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } undef, float %element.0.0, 0
  %output.0.1 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %output.0.0, float %element.0.1, 1
  %output.0.2 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %output.0.1, float %element.0.2, 2
  %output.0.3 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %output.0.2, float %element.0.3, 3

  ; Second row.
  %output.1.0 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %output.0.3, float %element.1.0, 4
  %output.1.1 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %output.1.0, float %element.1.1, 5
  %output.1.2 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %output.1.1, float %element.1.2, 6
  %output.1.3 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %output.1.2, float %element.1.3, 7

  ; Third row.
  %output.2.0 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %output.1.3, float %element.2.0, 8
  %output.2.1 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %output.2.0, float %element.2.1, 9
  %output.2.2 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %output.2.1, float %element.2.2, 10
  %output.2.3 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %output.2.2, float %element.2.3, 11

  ; Fourth row.
  %output.3.0 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %output.2.3, float %element.3.0, 12
  %output.3.1 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %output.3.0, float %element.3.1, 13
  %output.3.2 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %output.3.1, float %element.3.2, 14
  %output.3.3 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %output.3.2, float %element.3.3, 15

  ret { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %output.3.3
}

; rti_transform
declare void @optixi_handleTransformNode(%"struct.cort::CanonicalState"*)
define linkonce_odr void @_rti_handle_transform_node() alwaysinline
{
   %state = call %"struct.cort::CanonicalState"* @optixi_getState()
   call void @optixi_handleTransformNode(%"struct.cort::CanonicalState"* %state)
   ret void
}

; _rt_get_primitive_index
declare i32 @optixi_getPrimitiveIndex(%"struct.cort::CanonicalState"* %state)
define linkonce_odr i32 @_rt_get_primitive_index() alwaysinline
{
   %state = call %"struct.cort::CanonicalState"* @optixi_getState()
   %index = call i32 @optixi_getPrimitiveIndex(%"struct.cort::CanonicalState"* %state)
   ret i32 %index
}

; _rt_get_entry_point_index
declare i16 @optixi_getEntryPointIndex(%"struct.cort::CanonicalState"* %state)
define linkonce_odr i32 @_rt_get_entry_point_index() alwaysinline
{
   %state = call %"struct.cort::CanonicalState"* @optixi_getState()
   %index = call i16 @optixi_getEntryPointIndex(%"struct.cort::CanonicalState"* %state)
   %index_ext = zext i16 %index to i32
   ret i32 %index_ext
}

; _rt_is_triangle_hit
declare i1 @optixi_isTriangleHit(%"struct.cort::CanonicalState"* %state)
define linkonce_odr i32 @_rt_is_triangle_hit() alwaysinline
{
   %state = call %"struct.cort::CanonicalState"* @optixi_getState()
   %isTriangleHit = call i1 @optixi_isTriangleHit(%"struct.cort::CanonicalState"* %state)
   %isTriangleHit_ext = zext i1 %isTriangleHit to i32
   ret i32 %isTriangleHit_ext
}

; _rt_is_triangle_hit_back_face
declare i1 @optixi_isTriangleHitBackFace(%"struct.cort::CanonicalState"* %state)
define linkonce_odr i32 @_rt_is_triangle_hit_back_face() alwaysinline
{
   %state = call %"struct.cort::CanonicalState"* @optixi_getState()
   %isTriangleHit = call i1 @optixi_isTriangleHitBackFace(%"struct.cort::CanonicalState"* %state)
   %isTriangleHit_ext = zext i1 %isTriangleHit to i32
   ret i32 %isTriangleHit_ext
}

; _rt_is_triangle_hit_front_face
declare i1 @optixi_isTriangleHitFrontFace(%"struct.cort::CanonicalState"* %state)
define linkonce_odr i32 @_rt_is_triangle_hit_front_face() alwaysinline
{
   %state = call %"struct.cort::CanonicalState"* @optixi_getState()
   %isTriangleHit = call i1 @optixi_isTriangleHitFrontFace(%"struct.cort::CanonicalState"* %state)
   %isTriangleHit_ext = zext i1 %isTriangleHit to i32
   ret i32 %isTriangleHit_ext
}

; _rti_get_instance_flags
declare i32 @optixi_getInstanceFlags(%"struct.cort::CanonicalState"* %state)
define linkonce_odr i32 @_rti_get_instance_flags() alwaysinline
{
   %state = call %"struct.cort::CanonicalState"* @optixi_getState()
   %instanceFlags = call i32 @optixi_getInstanceFlags(%"struct.cort::CanonicalState"* %state)
   ret i32 %instanceFlags
}

; _rt_get_lowest_group_child_index
declare i32 @optixi_getLowestGroupChildIndex(%"struct.cort::CanonicalState"* %state)
define linkonce_odr i32 @_rt_get_lowest_group_child_index() alwaysinline
{
   %state = call %"struct.cort::CanonicalState"* @optixi_getState()
   %index = call i32 @optixi_getLowestGroupChildIndex(%"struct.cort::CanonicalState"* %state)
   ret i32 %index
}

; _rt_get_ray_flags
declare i32 @optixi_getRayFlags(%"struct.cort::CanonicalState"* %state)
define linkonce_odr i32 @_rt_get_ray_flags() alwaysinline
{
   %state = call %"struct.cort::CanonicalState"* @optixi_getState()
   %flags = call i32 @optixi_getRayFlags(%"struct.cort::CanonicalState"* %state)
   ret i32 %flags
}

; _rt_get_ray_mask
declare i32 @optixi_getRayMask(%"struct.cort::CanonicalState"* %state)
define linkonce_odr i32 @_rt_get_ray_mask() alwaysinline
{
   %state = call %"struct.cort::CanonicalState"* @optixi_getState()
   %mask = call i32 @optixi_getRayMask(%"struct.cort::CanonicalState"* %state)
   ret i32 %mask
}

; rti_profile_event
declare void @optixi_profileEvent(%"struct.cort::CanonicalState"*, i32)
define linkonce_odr void @_rti_profile_event(i32 %idx) alwaysinline
{
   %state = call %"struct.cort::CanonicalState"* @optixi_getState()
   call void @optixi_profileEvent(%"struct.cort::CanonicalState"* %state, i32 %idx)
   ret void
}

; rti_throw
declare void @optixi_throw(%"struct.cort::CanonicalState"*, i32)
define linkonce_odr void @_rt_throw(i32 %code) alwaysinline
{
   %state = call %"struct.cort::CanonicalState"* @optixi_getState()
   call void @optixi_throw(%"struct.cort::CanonicalState"* %state, i32 %code)
   ret void
}

declare void @optixi_terminateRay(%"struct.cort::CanonicalState"*)
define linkonce_odr void @_rt_terminate_ray() alwaysinline
{
   %state = call %"struct.cort::CanonicalState"* @optixi_getState()
   call void @optixi_terminateRay(%"struct.cort::CanonicalState"* %state)
   ret void
}

declare %"struct.cort::float2" @optixi_getTriangleBarycentrics(%"struct.cort::CanonicalState"*)
define linkonce_odr { float, float } @_rt_get_triangle_barycentrics() alwaysinline
{
   %state = call %"struct.cort::CanonicalState"* @optixi_getState()
   %val = call %"struct.cort::float2" @optixi_getTriangleBarycentrics(%"struct.cort::CanonicalState"* %state)
   %f0 = extractvalue %"struct.cort::float2" %val, 0
   %f1 = extractvalue %"struct.cort::float2" %val, 1
   %f  = insertvalue { float, float } undef, float %f0, 0
   %ff = insertvalue { float, float } %f, float %f1, 1
   ret { float, float } %ff
}

;; GeometryTriangle record access

declare i32 @optixi_getGeometryTrianglesVertexBufferID(%"struct.cort::CanonicalState"*)
define linkonce_odr i32 @_rti_get_geometry_triangles_vertexBufferID() alwaysinline
{
   %state = call %"struct.cort::CanonicalState"* @optixi_getState()
   %val = call i32 @optixi_getGeometryTrianglesVertexBufferID(%"struct.cort::CanonicalState"* %state)
   ret i32 %val 
}

declare i64 @optixi_getGeometryTrianglesVertexBufferOffset(%"struct.cort::CanonicalState"*)
define linkonce_odr i64 @_rti_get_geometry_triangles_vertexBufferOffset() alwaysinline
{
   %state = call %"struct.cort::CanonicalState"* @optixi_getState()
   %val = call i64 @optixi_getGeometryTrianglesVertexBufferOffset(%"struct.cort::CanonicalState"* %state)
   ret i64 %val 
}

declare i64 @optixi_getGeometryTrianglesVertexBufferStride(%"struct.cort::CanonicalState"*)
define linkonce_odr i64 @_rti_get_geometry_triangles_vertexBufferStride() alwaysinline
{
   %state = call %"struct.cort::CanonicalState"* @optixi_getState()
   %val = call i64 @optixi_getGeometryTrianglesVertexBufferStride(%"struct.cort::CanonicalState"* %state)
   ret i64 %val 
}

declare i32 @optixi_getGeometryTrianglesIndexBufferID(%"struct.cort::CanonicalState"*)
define linkonce_odr i32 @_rti_get_geometry_triangles_indexBufferID() alwaysinline
{
   %state = call %"struct.cort::CanonicalState"* @optixi_getState()
   %val = call i32 @optixi_getGeometryTrianglesIndexBufferID(%"struct.cort::CanonicalState"* %state)
   ret i32 %val 
}

declare i64 @optixi_getGeometryTrianglesIndexBufferOffset(%"struct.cort::CanonicalState"*)
define linkonce_odr i64 @_rti_get_geometry_triangles_indexBufferOffset() alwaysinline
{
   %state = call %"struct.cort::CanonicalState"* @optixi_getState()
   %val = call i64 @optixi_getGeometryTrianglesIndexBufferOffset(%"struct.cort::CanonicalState"* %state)
   ret i64 %val 
}

declare i64 @optixi_getGeometryTrianglesIndexBufferStride(%"struct.cort::CanonicalState"*)
define linkonce_odr i64 @_rti_get_geometry_triangles_indexBufferStride() alwaysinline
{
   %state = call %"struct.cort::CanonicalState"* @optixi_getState()
   %val = call i64 @optixi_getGeometryTrianglesIndexBufferStride(%"struct.cort::CanonicalState"* %state)
   ret i64 %val 
}

;; MotionGeometryTriangle record access

declare i64 @optixi_getMotionGeometryTrianglesVertexBufferMotionStride(%"struct.cort::CanonicalState"*)
define linkonce_odr i64 @_rti_get_motion_geometry_triangles_vertexBufferMotionStride() alwaysinline
{
   %state = call %"struct.cort::CanonicalState"* @optixi_getState()
   %val = call i64 @optixi_getMotionGeometryTrianglesVertexBufferMotionStride(%"struct.cort::CanonicalState"* %state)
   ret i64 %val 
}

declare i32 @optixi_getMotionGeometryTrianglesMotionNumIntervals(%"struct.cort::CanonicalState"*)
define linkonce_odr i32 @_rti_get_motion_geometry_triangles_motionNumIntervals() alwaysinline
{
   %state = call %"struct.cort::CanonicalState"* @optixi_getState()
   %val = call i32 @optixi_getMotionGeometryTrianglesMotionNumIntervals(%"struct.cort::CanonicalState"* %state)
   ret i32 %val 
}

declare float @optixi_getMotionGeometryTrianglesTimeBegin(%"struct.cort::CanonicalState"*)
define linkonce_odr float @_rti_get_motion_geometry_triangles_timeBegin() alwaysinline
{
   %state = call %"struct.cort::CanonicalState"* @optixi_getState()
   %val = call float @optixi_getMotionGeometryTrianglesTimeBegin(%"struct.cort::CanonicalState"* %state)
   ret float %val 
}

declare float @optixi_getMotionGeometryTrianglesTimeEnd(%"struct.cort::CanonicalState"*)
define linkonce_odr float @_rti_get_motion_geometry_triangles_timeEnd() alwaysinline
{
   %state = call %"struct.cort::CanonicalState"* @optixi_getState()
   %val = call float @optixi_getMotionGeometryTrianglesTimeEnd(%"struct.cort::CanonicalState"* %state)
   ret float %val 
}

declare i32 @optixi_getMotionGeometryTrianglesMotionBorderModeBegin(%"struct.cort::CanonicalState"*)
define linkonce_odr i32 @_rti_get_motion_geometry_triangles_motionBorderModeBegin() alwaysinline
{
   %state = call %"struct.cort::CanonicalState"* @optixi_getState()
   %val = call i32 @optixi_getMotionGeometryTrianglesMotionBorderModeBegin(%"struct.cort::CanonicalState"* %state)
   ret i32 %val 
}

declare i32 @optixi_getMotionGeometryTrianglesMotionBorderModeEnd(%"struct.cort::CanonicalState"*)
define linkonce_odr i32 @_rti_get_motion_geometry_triangles_motionBorderModeEnd() alwaysinline
{
   %state = call %"struct.cort::CanonicalState"* @optixi_getState()
   %val = call i32 @optixi_getMotionGeometryTrianglesMotionBorderModeEnd(%"struct.cort::CanonicalState"* %state)
   ret i32 %val 
}

   ;;  left to do:
   ; _rt_print_active
   ; _rt_print_start_64
   ; _rt_print_start
   ; _rt_print_write32

