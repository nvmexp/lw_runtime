; Copyright LWPU Corporation 2016
; TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED
; *AS IS* AND LWPU AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS
; OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY
; AND FITNESS FOR A PARTICULAR PURPOSE.  IN NO EVENT SHALL LWPU OR ITS SUPPLIERS
; BE LIABLE FOR ANY SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES
; WHATSOEVER (INCLUDING, WITHOUT LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS,
; BUSINESS INTERRUPTION, LOSS OF BUSINESS INFORMATION, OR ANY OTHER PELWNIARY LOSS)
; ARISING OUT OF THE USE OF OR INABILITY TO USE THIS SOFTWARE, EVEN IF LWPU HAS
; BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES 

; ------------------------------------------------------------------------------
; Declaration of the rtcore interface functions.
; Some functions declarations are missing because they require name mangling:
; e.g.: @lw.rt.trace.T

attributes #0 = { nounwind readnone alwaysinline } ; Non-volatile functions
attributes #1 = { nounwind readonly alwaysinline }
attributes #2 = { nounwind alwaysinline } ; "volatile" functions, which may return different results on different ilwocations
attributes #3 = { noreturn }
attributes #4 = { noreturn readonly }
attributes #5 = { nounwind readnone noinline }

declare i32 @lw.rt.read.launch.idx.x() #0
declare i32 @lw.rt.read.launch.idx.y() #0

declare i32 @lw.rt.read.launch.dim.x() #0
declare i32 @lw.rt.read.launch.dim.y() #0

declare i64 @lw.rt.read.sbt.data.ptr() #1

declare float @lw.rt.read.world.ray.origin.x() #0
declare float @lw.rt.read.world.ray.origin.y() #0
declare float @lw.rt.read.world.ray.origin.z() #0

declare float @lw.rt.read.world.ray.direction.x() #0
declare float @lw.rt.read.world.ray.direction.y() #0
declare float @lw.rt.read.world.ray.direction.z() #0

declare float @lw.rt.read.object.ray.origin.x() #0
declare float @lw.rt.read.object.ray.origin.y() #0
declare float @lw.rt.read.object.ray.origin.z() #0

declare float @lw.rt.read.object.ray.direction.x() #0
declare float @lw.rt.read.object.ray.direction.y() #0
declare float @lw.rt.read.object.ray.direction.z() #0

declare float @lw.rt.read.ray.tmin() #0
declare float @lw.rt.read.ray.tmax() #1

declare float @lw.rt.read.current.time() #1

declare i32 @lw.rt.read.transform.list.size() #0
declare i64 @lw.rt.read.transform.list.traversable( i32 %index ) #0
declare i32 @lw.rt.read.transform.type.from.traversable( i64 %handle ) #0
declare i64 @lw.rt.read.matrix.motion.transform.from.traversable( i64 %handle ) #0
declare i64 @lw.rt.read.srt.motion.transform.from.traversable( i64 %handle ) #0
declare i64 @lw.rt.read.static.transform.from.traversable( i64 %handle ) #0
declare i64 @lw.rt.read.instance.transform.from.traversable( i64 %handle ) #0
declare i64 @lw.rt.read.instance.ilwerse.transform.from.traversable( i64 %handle ) #0

declare i32 @lw.rt.read.primitive.idx() #0

declare i32 @lw.rt.read.instance.id() #0

declare i32 @lw.rt.read.instance.idx() #0 ; Only 24 bit precision.

declare i8 @lw.rt.read.sbt.record.offset() #0

declare i8 @lw.rt.read.hitkind() #0

declare i32 @lw.rt.read.instance.flags() #0

declare i16 @lw.rt.read.ray.flags() #0

declare i8 @lw.rt.read.ray.mask() #0

declare void @lw.rt.terminate.ray() #3

declare void @lw.rt.ignore.intersection() #3

declare void @lw.rt.throw.exception( i32 %exceptionCode, [23 x i32] %exceptionDetails ) #4

declare i32 @lw.rt.read.exception.code() #0

declare i32 @lw.rt.read.exception.detail( i32 %index ) #0

declare i32 @lw.rt.read.payload.i32( i32 %index ) #1

declare i32 @lw.rt.read.register.attribute.i32( i32 %index ) #0

; ------------------------------------------------------------------------------

declare float @llvm.lwvm.fma.rn.ftz.f(float, float, float) readnone
declare float @llvm.lwvm.mul.rn.ftz.f(float, float) readnone

; ------------------------------------------------------------------------------

%"struct.cort::uint2" = type { i32, i32 }
%"struct.cort::uint4" = type { i32, i32, i32, i32 }
%"struct.cort::float2" = type { float, float }
%"struct.cort::float3" = type { float, float, float }
%"struct.cort::Matrix4x4" = type { [4 x [4 x float]] }
%"struct.cort::OptixRay" = type { %"struct.cort::float3", %"struct.cort::float3", i32, float, float }
%"union.cort::SBTRecordData" = type { %"struct.cort::SBTRecordData::GeometryInstanceDataT" }
%"struct.cort::SBTRecordData::GeometryInstanceDataT" = type { i32, i32, i32 }

; ------------------------------------------------------------------------------
declare <4 x i32> @optix.ptx.ld.global.nc.v4.u32( <4 x i32> addrspace(1)* %address )
define weak %"struct.cort::uint4" @RTX_vectorizedLoadTextureCache( i8* %address )
{
    %vectorAddress = bitcast i8* %address to <4 x i32>*
    %vectorAddressGlobal = addrspacecast <4 x i32>* %vectorAddress to <4 x i32> addrspace(1)*
    %result = call <4 x i32> @optix.ptx.ld.global.nc.v4.u32( <4 x i32> addrspace(1)* %vectorAddressGlobal)
    %value1 = extractelement <4 x i32> %result, i32 0
    %value2 = extractelement <4 x i32> %result, i32 1
    %value3 = extractelement <4 x i32> %result, i32 2
    %value4 = extractelement <4 x i32> %result, i32 3
    %returnStruct1 = insertvalue %"struct.cort::uint4" undef, i32 %value1, 0
    %returnStruct2 = insertvalue %"struct.cort::uint4" %returnStruct1, i32 %value2, 1
    %returnStruct3 = insertvalue %"struct.cort::uint4" %returnStruct2, i32 %value3, 2
    %returnStruct4 = insertvalue %"struct.cort::uint4" %returnStruct3, i32 %value4, 3
    ret %"struct.cort::uint4" %returnStruct4
}

; ------------------------------------------------------------------------------
declare i32 @llvm.lwvm.atomic.rmw.i32.p0i32( i32 %flag, i32 addrspace(0)* %address, i32 %val )
define weak i32 @RTX_atomicOr( i32 addrspace(0)* %address, i32 %value )
{
    ; Flag arg: Set relaxed ordering, GPU scope, add operation
    ; %flag bits:
    ; 0-3   Ordering      Relaxed = 1
    ; 4-7   Scope         GPU = 0
    ; 8-15  Reserved
    ; 16-23 Operation     or = 5
    ; 24-31 Reserved
    ;
    ; ( 1 << 0 ) | ( 0 << 4 ) | ( 5 << 16 ) = 327681
    ; 
    ; Flag doc: https://p4viewer.lwpu.com/get/sw/compiler/docs/UnifiedLWVMIR/asciidoc/html/lwvmIR.html#_atomic_functions
    ;
    %r0 = call i32 @llvm.lwvm.atomic.rmw.i32.p0i32( i32 327681, i32 addrspace(0)* %address, i32 %value )
    ret i32 %r0
}

; ------------------------------------------------------------------------------
define weak %"struct.cort::uint2" @RTX_getLaunchIndex() #0
{
  %x = tail call i32 @lw.rt.read.launch.idx.x() #0
  %y = tail call i32 @lw.rt.read.launch.idx.y() #0
  %first  = insertvalue %"struct.cort::uint2" undef, i32 %x, 0
  %result = insertvalue %"struct.cort::uint2" %first, i32 %y, 1
  ret %"struct.cort::uint2" %result
}

; ------------------------------------------------------------------------------
define %"union.cort::SBTRecordData"* @RTX_getSBTRecordData() #1
{
  %sbt.address.value = tail call i64 @lw.rt.read.sbt.data.ptr() #1
  %sbt.address = inttoptr i64 %sbt.address.value to %"union.cort::SBTRecordData"*
  ret %"union.cort::SBTRecordData"* %sbt.address
}

; NOTE: Must only be called from IS/AH/CH, according to RTCore spec of
;       lw.rt.reat.sbt.record.offset:
; Available in: IS, AH, CH, GN
; Description: Returns the value of sbtRecordOffset passed into the lwrrently
;   active call to lw.rt.trace.T.  Results are undefined if called from
;   GN not called by IS, AH, or CH.
;   The offset's precision is only 4 bit, the return type is i8 only because
;   that is the smallest type available in LWVM-RT.
define i32 @RTX_getSBTRecordOffset() #0
{
  %sbt.record.offset = tail call i8 @lw.rt.read.sbt.record.offset() #0
  %sbt.record.offset.i32 = zext i8 %sbt.record.offset to i32
  ret i32 %sbt.record.offset.i32
}

; ------------------------------------------------------------------------------
; If the SBT Record data changes removing the material offset retrive the same information by reading the OR of the geometry instance.
; define i32 @_ZN4cort17getMaterialHandleEPNS_14CanonicalStateE(%"struct.cort::CanonicalState"*) 
; {
;   %sbt.address.value = tail call i64 @lw.rt.read.sbt.data.ptr() #1
;   %sbt.address = inttoptr i64 %sbt.address.value to %"struct.cort::SBTRecordData"*
;   %sbt.material = getelementptr inbounds %"struct.cort::SBTRecordData"* %sbt.address, i64 0, i32 2
;   %material.offset = load i32* %sbt.material
;   ret i32 %material.offset
; }

;; ------------------------------------------------------------------------------
;define i32 @_ZN4cort17getProgramHandleEPNS_14CanonicalStateE(%"struct.cort::CanonicalState"*)
;{
;  %sbt.address.value = tail call i64 @lw.rt.read.sbt.data.ptr() #1
;  %sbt.address = inttoptr i64 %sbt.address.value to %"struct.cort::SBTRecordData"*
;  %sbt.material = getelementptr inbounds %"struct.cort::SBTRecordData"* %sbt.address, i64 0, i32 0
;  %program.offset = load i32* %sbt.material
;  ret i32 %program.offset
;}
;
;; ------------------------------------------------------------------------------
;define i32 @_ZN4cort25getGeometryInstanceHandleEPNS_14CanonicalStateE(%"struct.cort::CanonicalState"*)
;{
;  %sbt.address.value = tail call i64 @lw.rt.read.sbt.data.ptr() #1
;  %sbt.address = inttoptr i64 %sbt.address.value to %"struct.cort::SBTRecordData"*
;  %sbt.gi = getelementptr inbounds %"struct.cort::SBTRecordData"* %sbt.address, i64 0, i32 1
;  %gi.offset = load i32* %sbt.gi
;  ret i32 %gi.offset
;}
;
;; ------------------------------------------------------------------------------
;define i32 @_ZN4cort7getSkipEPNS_14CanonicalStateE(%"struct.cort::CanonicalState"*)
;{
;  %sbt.address.value = tail call i64 @lw.rt.read.sbt.data.ptr() #1
;  %sbt.address = inttoptr i64 %sbt.address.value to %"struct.cort::SBTRecordData"*
;  %sbt.material = getelementptr inbounds %"struct.cort::SBTRecordData"* %sbt.address, i64 0, i32 3
;  %skip = load i32* %sbt.material
;  ret i32 %skip
;}

; ------------------------------------------------------------------------------
define float @RTX_getLwrrentTmax() #0
{
  %tmax = tail call float @lw.rt.read.ray.tmax()
  ret float %tmax
}

; ------------------------------------------------------------------------------
define float @RTX_getLwrrentTime() #1
{
  %lwrrentTime = tail call float @lw.rt.read.current.time()
  ret float %lwrrentTime
}

; ------------------------------------------------------------------------------
; See OptiX_Programming_Guide section 4.1.6 "Program Variable Transformation"
; CH & MS use World Space Ray
define %"struct.cort::OptixRay" @RTX_getWorldSpaceRay()
{
  %ox = tail call float @lw.rt.read.world.ray.origin.x() #0
  %oy = tail call float @lw.rt.read.world.ray.origin.y() #0
  %oz = tail call float @lw.rt.read.world.ray.origin.z() #0
  %tmin = tail call float @lw.rt.read.ray.tmin() #0
  %dx = tail call float @lw.rt.read.world.ray.direction.x() #0
  %dy = tail call float @lw.rt.read.world.ray.direction.y() #0
  %dz = tail call float @lw.rt.read.world.ray.direction.z() #0
  %tmax = tail call float @lw.rt.read.ray.tmax() #2 
  %rayType = tail call i8 @lw.rt.read.sbt.record.offset() #0
  %rayType.i32 = zext i8 %rayType to i32
  %o1 = insertvalue %"struct.cort::float3" undef, float %ox, 0
  %o2 = insertvalue %"struct.cort::float3" %o1, float %oy, 1
  %o3 = insertvalue %"struct.cort::float3" %o2, float %oz, 2
  %d1 = insertvalue %"struct.cort::float3" undef, float %dx, 0
  %d2 = insertvalue %"struct.cort::float3" %d1, float %dy, 1
  %d3 = insertvalue %"struct.cort::float3" %d2, float %dz, 2
  %ray1 = insertvalue %"struct.cort::OptixRay" undef, %"struct.cort::float3" %o3, 0
  %ray2 = insertvalue %"struct.cort::OptixRay" %ray1, %"struct.cort::float3" %d3, 1
  %ray3 = insertvalue %"struct.cort::OptixRay" %ray2, i32 %rayType.i32, 2 
  %ray4 = insertvalue %"struct.cort::OptixRay" %ray3, float %tmin, 3
  %ray5 = insertvalue %"struct.cort::OptixRay" %ray4, float %tmax, 4
  ret %"struct.cort::OptixRay" %ray5
}

; ------------------------------------------------------------------------------
; See OptiX_Programming_Guide section 4.1.6 "Program Variable Transformation"
; AH, IS, LW all use Object Space Ray
define %"struct.cort::OptixRay" @RTX_getObjectSpaceRay()
{
  %ox = tail call float @lw.rt.read.object.ray.origin.x() #0
  %oy = tail call float @lw.rt.read.object.ray.origin.y() #0
  %oz = tail call float @lw.rt.read.object.ray.origin.z() #0
  %tmin = tail call float @lw.rt.read.ray.tmin() #0
  %dx = tail call float @lw.rt.read.object.ray.direction.x() #0
  %dy = tail call float @lw.rt.read.object.ray.direction.y() #0
  %dz = tail call float @lw.rt.read.object.ray.direction.z() #0
  %tmax = tail call float @lw.rt.read.ray.tmax() #2 
  %rayType = tail call i8 @lw.rt.read.sbt.record.offset() #0
  %rayType.i32 = zext i8 %rayType to i32
  %o1 = insertvalue %"struct.cort::float3" undef, float %ox, 0
  %o2 = insertvalue %"struct.cort::float3" %o1, float %oy, 1
  %o3 = insertvalue %"struct.cort::float3" %o2, float %oz, 2
  %d1 = insertvalue %"struct.cort::float3" undef, float %dx, 0
  %d2 = insertvalue %"struct.cort::float3" %d1, float %dy, 1
  %d3 = insertvalue %"struct.cort::float3" %d2, float %dz, 2
  %ray1 = insertvalue %"struct.cort::OptixRay" undef, %"struct.cort::float3" %o3, 0
  %ray2 = insertvalue %"struct.cort::OptixRay" %ray1, %"struct.cort::float3" %d3, 1
  %ray3 = insertvalue %"struct.cort::OptixRay" %ray2, i32 %rayType.i32, 2 
  %ray4 = insertvalue %"struct.cort::OptixRay" %ray3, float %tmin, 3
  %ray5 = insertvalue %"struct.cort::OptixRay" %ray4, float %tmax, 4
  ret %"struct.cort::OptixRay" %ray5
}

; ------------------------------------------------------------------------------
define i32 @RTX_getPrimitiveIdx() #0
{
entry:
  %primitiveIndex = tail call i32 @lw.rt.read.primitive.idx() #0
  ret i32 %primitiveIndex
}

; ------------------------------------------------------------------------------
define i8 @RTX_getHitKind() #0
{
entry:
  %hitKind = tail call i8 @lw.rt.read.hitkind() #0
  ret i8 %hitKind
}

; ------------------------------------------------------------------------------
define i32 @RTX_getInstanceFlags() #0
{
entry:
  %flags = tail call i32 @lw.rt.read.instance.flags() #0
  ret i32 %flags
}

; ------------------------------------------------------------------------------
define i32 @RTX_getInstanceIndex() #0
{
entry:
  %index = tail call i32 @lw.rt.read.instance.idx() #0
  ret i32 %index
}

; ------------------------------------------------------------------------------
define i16 @RTX_getRayFlags() #0
{
entry:
  %flags = tail call i16 @lw.rt.read.ray.flags() #0
  ret i16 %flags
}

; ------------------------------------------------------------------------------
define i8 @RTX_getRayMask() #0
{
entry:
  %mask = tail call i8 @lw.rt.read.ray.mask() #0
  ret i8 %mask
}

; ------------------------------------------------------------------------------
define i32 @RTX_getTransformListSize() #0{
entry:
  %size = tail call i32 @lw.rt.read.transform.list.size() #0
  ret i32 %size
}

; ------------------------------------------------------------------------------
define i64 @RTX_getTransformListHandle( i32 %index ) #0
{
entry:
  %traversable = tail call i64 @lw.rt.read.transform.list.traversable( i32 %index ) #0
  ret i64 %traversable
}

; ------------------------------------------------------------------------------
define i32 @RTX_getTransformTypeFromHandle( i64 %handle ) #0
{
entry:
  %type = tail call i32 @lw.rt.read.transform.type.from.traversable( i64 %handle ) #0
  ret i32 %type
}

; ------------------------------------------------------------------------------
define i64 @RTX_getMatrixMotionTransformFromHandle( i64 %handle ) #0
{
entry:
  %ptr = tail call i64 @lw.rt.read.matrix.motion.transform.from.traversable( i64 %handle ) #0
  ret i64 %ptr
}

; ------------------------------------------------------------------------------
define i64 @RTX_getSRTMotionTransformFromHandle( i64 %handle ) #0
{
entry:
  %ptr = tail call i64 @lw.rt.read.srt.motion.transform.from.traversable( i64 %handle ) #0
  ret i64 %ptr
}

; ------------------------------------------------------------------------------
define i64 @RTX_getStaticTransformFromHandle( i64 %handle ) #0
{
entry:
  %ptr = tail call i64 @lw.rt.read.static.transform.from.traversable( i64 %handle ) #0
  ret i64 %ptr
}

; ------------------------------------------------------------------------------
define i64 @RTX_getInstanceTransformFromHandle( i64 %handle ) #0
{
entry:
  %ptr = tail call i64 @lw.rt.read.instance.transform.from.traversable( i64 %handle ) #0
  ret i64 %ptr
}

; ------------------------------------------------------------------------------
define i64 @RTX_getInstanceIlwerseTransformFromHandle( i64 %handle ) #0
{
entry:
  %ptr = tail call i64 @lw.rt.read.instance.ilwerse.transform.from.traversable( i64 %handle ) #0
  ret i64 %ptr
}

; ------------------------------------------------------------------------------
define void @RTX_terminateRay()
{
  call void @lw.rt.terminate.ray()
  ret void
}

; ------------------------------------------------------------------------------
define void @RTX_ignoreIntersection()
{
  call void @lw.rt.ignore.intersection()
  ret void
}

; ------------------------------------------------------------------------------
define void @RTX_throwException( i32 %exceptionCode, i32* %datap_in )
{
  %datap = bitcast i32* %datap_in to [23 x i32]*
  %data = load [23 x i32], [23 x i32]* %datap, align 8
  call void @lw.rt.throw.exception( i32 %exceptionCode, [23 x i32] %data )
  ret void
}

; ------------------------------------------------------------------------------
define i1 @RTX_isInfOrNan( float %x ) #0
{
  %isPInfOrNan = fcmp ueq float %x, 0x7FF0000000000000
  %isNInfOrNan = fcmp ueq float %x, 0xFFF0000000000000
  %isInfOrNan = or i1 %isPInfOrNan, %isNInfOrNan
  ret i1 %isInfOrNan
}

; ------------------------------------------------------------------------------
define i1 @RTX_isNan( float %x ) #0
{
  %isNan = fcmp une float %x, %x
  ret i1 %isNan
}

; ------------------------------------------------------------------------------
define i32 @RTX_getExceptionCode()
{
  %code = call i32 @lw.rt.read.exception.code()
  ret i32 %code
}

; ------------------------------------------------------------------------------
define i32 @RTX_getExceptionDetail( i32 %index )
{
  %data = call i32 @lw.rt.read.exception.detail( i32 %index )
  ret i32 %data
}
; ------------------------------------------------------------------------------
define i8* @RTX_getPayloadPointer() #1
{
  %plLo = tail call i32 @lw.rt.read.payload.i32(i32 0) #1
  %plHi = tail call i32 @lw.rt.read.payload.i32(i32 1) #1
  %plPtr1 = insertelement <2 x i32> undef, i32 %plLo, i32 0
  %plPtr2 = insertelement <2 x i32> %plPtr1, i32 %plHi, i32 1
  %pl = bitcast <2 x i32> %plPtr2 to i64
  %plPtr = inttoptr i64 %pl to i8*
  ret i8* %plPtr
}

; ------------------------------------------------------------------------------
define %"struct.cort::float2" @RTX_getTriangleBarycentrics() #0
{
  %a0 = call i32 @lw.rt.read.register.attribute.i32(i32 0) #0
  %a1 = call i32 @lw.rt.read.register.attribute.i32(i32 1) #0
  %f0 = bitcast i32 %a0 to float
  %f1 = bitcast i32 %a1 to float
  %f  = insertvalue %"struct.cort::float2" undef, float %f0, 0
  %ff = insertvalue %"struct.cort::float2" %f, float %f1, 1
  ret %"struct.cort::float2" %ff
}  


