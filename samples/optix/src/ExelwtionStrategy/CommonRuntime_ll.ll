; Copyright LWPU Corporation 2008
; TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED
; *AS IS* AND LWPU AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS
; OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY
; AND FITNESS FOR A PARTICULAR PURPOSE.  IN NO EVENT SHALL LWPU OR ITS SUPPLIERS
; BE LIABLE FOR ANY SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES
; WHATSOEVER (INCLUDING, WITHOUT LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS,
; BUSINESS INTERRUPTION, LOSS OF BUSINESS INFORMATION, OR ANY OTHER PELWNIARY LOSS)
; ARISING OUT OF THE USE OF OR INABILITY TO USE THIS SOFTWARE, EVEN IF LWPU HAS
; BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES 



attributes #0 = { nounwind readnone alwaysinline } ; Non-volatile functions
attributes #1 = { nounwind alwaysinline }          ; "volatile" functions, which may return different results on different ilwocations
attributes #2 = { nounwind readnone }

%"struct.cort::uint2" = type { i32, i32 }
%"struct.cort::uint3" = type { i32, i32, i32 }
%"struct.cort::uint4" = type { i32, i32, i32, i32 }
%"struct.cort::float4" = type { float, float, float, float }


;
; printf
;
declare i32 @vprintf(i8*, i8*) #1
define i32 @_ZN4lwda7vprintfEPKcPc(i8* %p0, i8* %p1) #1
{
  %1 = call i32 @vprintf(i8* %p0, i8* %p1) #1
  ret i32 %1
}

;
; memcpy
;
declare void @llvm.memcpy.p0i8.p0i8.i32(i8*, i8*, i32, i32, i1)
define void @_Z11cort_memcpyPcPKcjj(i8* nocapture noalias %to, i8* nocapture noalias readonly %from, i32 %size, i32 %alignment) #1
{
  switch i32 %alignment, label %align0 [
    i32 16, label %align16
    i32 8, label %align8
    i32 4, label %align4
  ]

  align16:
    call void @llvm.memcpy.p0i8.p0i8.i32(i8* %to, i8* %from, i32 %size, i32 16, i1 false );
    br label %end

  align8:
    call void @llvm.memcpy.p0i8.p0i8.i32(i8* %to, i8* %from, i32 %size, i32 8, i1 false );
    br label %end

  align4:
    call void @llvm.memcpy.p0i8.p0i8.i32(i8* %to, i8* %from, i32 %size, i32 4, i1 false );
    br label %end

  align0:
    call void @llvm.memcpy.p0i8.p0i8.i32(i8* %to, i8* %from, i32 %size, i32 0, i1 false );
    br label %end

  end:
    ret void
}

;
; address space cast
;
define i8 addrspace(0)* @_Z23cort_castConstToGenericPU3AS4c(i8 addrspace(4)* nocapture %addr) #0
{
  %1 = addrspacecast i8 addrspace(4)* %addr to i8 addrspace(0)*
  ret i8 addrspace(0)* %1
}


;
; Atomic Instructions.
;

; atomicAdd
; ------------------------------------------------------------------------------
declare i32 @optix.ptx.atom.add.s32(i32 addrspace(0)*, i32) #1
define i32 @_ZN4cort9atomicAddEPii(i32* nocapture %address, i32 %value) #1 {
  %1 = tail call i32 @optix.ptx.atom.add.s32(i32* %address, i32 %value)
  ret i32 %1
}

declare i32 @optix.ptx.atom.add.u32(i32 addrspace(0)*, i32) #1
define i32 @_ZN4cort9atomicAddEPjj(i32* nocapture %address, i32 %value) #1 {
  %1 = tail call i32 @optix.ptx.atom.add.u32(i32* %address, i32 %value)
  ret i32 %1
}

declare i64 @optix.ptx.atom.add.u64(i64 addrspace(0)*, i64) #1
define i64 @_ZN4cort9atomicAddEPyy(i64* nocapture %address, i64 %value) #1 {
  %1 = tail call i64 @optix.ptx.atom.add.u64(i64* %address, i64 %value)
  ret i64 %1
}

declare float @optix.ptx.atom.add.f32(float addrspace(0)*, float) #1
define float @_ZN4cort9atomicAddEPff(float* nocapture %address, float %value) #1 {
  %1 = tail call float @optix.ptx.atom.add.f32(float* %address, float %value)
  ret float %1
}

declare i32 @optix.ptx.atom.global.add.s32(i32 addrspace(1)*, i32) #1
define i32 @_ZN4lwda9atomicAddEPU3AS1ii(i32 addrspace(1)* nocapture %address, i32 %value) #1 {
  %1 = tail call i32 @optix.ptx.atom.global.add.s32(i32 addrspace(1)* %address, i32 %value)
  ret i32 %1
}

declare i64 @optix.ptx.atom.global.add.u64(i64 addrspace(1)*, i64) #1
define i64 @_ZN4lwda9atomicAddEPU3AS1yy(i64 addrspace(1)* nocapture %address, i64 %value) #1 {
  %1 = tail call i64 @optix.ptx.atom.global.add.u64(i64 addrspace(1)* %address, i64 %value)
  ret i64 %1
}


; atomicSub
; ------------------------------------------------------------------------------
declare i32 @optix.ptx.atom.sub.s32(i32 addrspace(0)*, i32) #1
define i32 @_ZN4cort9atomicSubEPii(i32* nocapture %address, i32 %value) #1 {
  %1 = tail call i32 @optix.ptx.atom.sub.s32(i32* %address, i32 %value)
  ret i32 %1
}

declare i32 @optix.ptx.atom.sub.u32(i32 addrspace(0)*, i32) #1
define i32 @_ZN4cort9atomicSubEPjj(i32* nocapture %address, i32 %value) #1 {
  %1 = tail call i32 @optix.ptx.atom.sub.u32(i32* %address, i32 %value)
  ret i32 %1
}

; atomicExch
; ------------------------------------------------------------------------------
declare i32 @optix.ptx.atom.exch.b32(i32 addrspace(0)*, i32) #1
define i32 @_ZN4cort10atomicExchEPii(i32* nocapture %address, i32 %value) #1 {
  %1 = tail call i32 @optix.ptx.atom.exch.b32(i32* %address, i32 %value)
  ret i32 %1 
}

define i32 @_ZN4cort10atomicExchEPjj(i32* nocapture %address, i32 %value) #1 {
  %1 = tail call i32 @optix.ptx.atom.exch.b32(i32* %address, i32 %value)
  ret i32 %1 
}

declare i64 @optix.ptx.atom.exch.b64(i64 addrspace(0)*, i64) #1
define i64 @_ZN4cort10atomicExchEPyy(i64* nocapture %address, i64 %value) #1 {
  %1 = tail call i64 @optix.ptx.atom.exch.b64(i64* %address, i64 %value)
  ret i64 %1 
}

define float @_ZN4cort10atomicExchEPff(float* nocapture %address, float %value) #1 {
  %1 = bitcast float* %address to i32*
  %2 = fptosi float %value to i32
  %3 = tail call i32 @optix.ptx.atom.exch.b32(i32 addrspace(0)* %1, i32 %2)
  %4 = sitofp i32 %3 to float
  ret  float %4
}

; atomicMax
; ------------------------------------------------------------------------------
declare i32 @optix.ptx.atom.max.s32( i32 addrspace(0)* %p0, i32 %p1 ) #1
define i32 @_ZN4cort9atomicMaxEPii(i32* nocapture %address, i32 %value) #1 {
  %1 = tail call i32 @optix.ptx.atom.max.s32(i32* %address, i32 %value)
  ret i32 %1
}

declare i32 @optix.ptx.atom.max.u32( i32 addrspace(0)* %p0, i32 %p1 ) #1
define i32 @_ZN4cort9atomicMaxEPjj(i32* nocapture %address, i32 %value) #1 {
  %1 = tail call i32 @optix.ptx.atom.max.u32(i32* %address, i32 %value)
  ret i32 %1
}

declare i64 @optix.ptx.atom.max.u64( i64 addrspace(0)* %p0, i64 %p1 ) #1
define i64 @_ZN4cort9atomicMaxEPyy(i64* nocapture %address, i64 %value) #1 {
  %1 = tail call i64 @optix.ptx.atom.max.u64(i64* %address, i64 %value)
  ret i64 %1 
}

; atomicMin
; ------------------------------------------------------------------------------
declare i32 @optix.ptx.atom.min.s32( i32 addrspace(0)* %p0, i32 %p1 ) #1
define i32 @_ZN4cort9atomicMinEPii(i32* nocapture %address, i32 %value) #1 {
  %1 = tail call i32 @optix.ptx.atom.min.s32(i32* %address, i32 %value)
  ret i32 %1
}

declare i32 @optix.ptx.atom.min.u32( i32 addrspace(0)* %p0, i32 %p1 ) #1
define i32 @_ZN4cort9atomicMinEPjj(i32* nocapture %address, i32 %value) #1 {
  %1 = tail call i32 @optix.ptx.atom.min.u32(i32* %address, i32 %value)
  ret i32 %1
}

declare i64 @optix.ptx.atom.min.u64( i64 addrspace(0)* %p0, i64 %p1 ) #1
define i64 @_ZN4cort9atomicMinEPyy(i64* nocapture %address, i64 %value) #1 {
  %1 = tail call i64 @optix.ptx.atom.min.u64(i64* %address, i64 %value)
  ret i64 %1 
}

; atomicInc
; ------------------------------------------------------------------------------
declare i32 @optix.ptx.atom.inc.u32( i32 addrspace(0)* %p0, i32 %p1 ) #1
define i32 @_ZN4cort9atomicIncEPjj(i32* nocapture %address, i32 %value) #1 {
  %1 = tail call i32 @optix.ptx.atom.inc.u32(i32* %address, i32 %value)
  ret i32 %1
}

; atomicDec
; ------------------------------------------------------------------------------
declare i32 @optix.ptx.atom.dec.u32( i32 addrspace(0)* %p0, i32 %p1 ) #1
define i32 @_ZN4cort9atomicDecEPjj(i32* nocapture %address, i32 %value) #1 {
  %1 = tail call i32 @optix.ptx.atom.dec.u32(i32* %address, i32 %value)
  ret i32 %1
}

; atomicCAS
; ------------------------------------------------------------------------------
declare i32 @optix.ptx.atom.cas.b32( i32 addrspace(0)* %p0, i32 %p1, i32 %p2 ) #1
define i32 @_ZN4cort9atomicCASEPiii(i32* nocapture %address, i32 %compare, i32 %value) #1 {
  %1 = tail call i32 @optix.ptx.atom.cas.b32(i32* %address, i32 %compare, i32 %value)
  ret i32 %1
}

define i32 @_ZN4cort9atomicCASEPjjj(i32* nocapture %address, i32 %compare, i32 %value) #1 {
  %1 = tail call i32 @optix.ptx.atom.cas.b32(i32* %address, i32 %compare, i32 %value)
  ret i32 %1
}

declare i64 @optix.ptx.atom.cas.b64( i64 addrspace(0)* %p0, i64 %p1, i64 %p2 ) #1
define i64 @_ZN4cort9atomicCASEPyyy(i64* nocapture %address, i64 %compare, i64 %value) #1 {
  %1 = tail call i64 @optix.ptx.atom.cas.b64(i64* %address, i64 %compare, i64 %value)
  ret i64 %1
}

; atomicAnd
; ------------------------------------------------------------------------------
declare i32 @optix.ptx.atom.and.b32( i32 addrspace(0)* %p0, i32 %p1 ) #1
define i32 @_ZN4cort9atomicAndEPii(i32* nocapture %address, i32 %value) #1 {
  %1 = tail call i32 @optix.ptx.atom.and.b32(i32* %address, i32 %value) 
  ret i32 %1
}

define i32 @_ZN4cort9atomicAndEPjj(i32* nocapture %address, i32 %value) #1 {
  %1 = tail call i32 @optix.ptx.atom.and.b32(i32* %address, i32 %value) 
  ret i32 %1
}

declare i64 @optix.ptx.atom.and.b64( i64 addrspace(0)* %p0, i64 %p1 ) #1 
define i64 @_ZN4cort9atomicAndEPyy(i64* nocapture %address, i64 %value) #1 {
  %1 = tail call i64 @optix.ptx.atom.and.b64(i64* %address, i64 %value)
  ret i64 %1  
}

; atomicOr
; ------------------------------------------------------------------------------
declare i32 @optix.ptx.atom.or.b32( i32 addrspace(0)* %p0, i32 %p1 ) #1
define i32 @_ZN4cort8atomicOrEPii(i32* nocapture %address, i32 %value) #1 {
  %1 = tail call i32 @optix.ptx.atom.or.b32(i32* %address, i32 %value)
  ret i32 %1
}

define i32 @_ZN4cort8atomicOrEPjj(i32* nocapture %address, i32 %value) #1 {
  %1 = tail call i32 @optix.ptx.atom.or.b32(i32* %address, i32 %value)
  ret i32 %1
}

declare i64 @optix.ptx.atom.or.b64( i64 addrspace(0)* %p0, i64 %p1 ) #1 
define i64 @_ZN4cort8atomicOrEPyy(i64* nocapture %address, i64 %value) #1 {
  %1 = tail call i64 @optix.ptx.atom.or.b64(i64* %address, i64 %value)
  ret i64 %1  
}

; atomiXor
; ------------------------------------------------------------------------------
declare i32 @optix.ptx.atom.xor.b32( i32 addrspace(0)* %p0, i32 %p1 ) #1
define i32 @_ZN4cort9atomicXorEPii(i32* nocapture %address, i32 %value) #1 {
  %1 = tail call i32 @optix.ptx.atom.xor.b32(i32* %address, i32 %value)
  ret i32 %1
}

define i32 @_ZN4cort9atomicXorEPjj(i32* nocapture %address, i32 %value) #1 {
  %1 = tail call i32 @optix.ptx.atom.xor.b32(i32* %address, i32 %value)
  ret i32 %1
}

declare i64 @optix.ptx.atom.xor.b64( i64 addrspace(0)* %p0, i64 %p1 ) #1 
define i64 @_ZN4cort9atomicXorEPyy(i64* nocapture %address, i64 %value) #1 {
  %1 = tail call i64 @optix.ptx.atom.xor.b64(i64* %address, i64 %value)
  ret i64 %1  
}

;
; cta/grid dimensions
;

declare i32 @optix.lwvm.read.ptx.sreg.tid.x() #0
declare i32 @optix.lwvm.read.ptx.sreg.tid.y() #0
declare i32 @optix.lwvm.read.ptx.sreg.tid.z() #0

define %"struct.cort::uint3" @_ZN4lwda3tidEv() #0 {
 %1 = tail call i32 @optix.lwvm.read.ptx.sreg.tid.x() #0
 %2 = tail call i32 @optix.lwvm.read.ptx.sreg.tid.y() #0
 %3 = tail call i32 @optix.lwvm.read.ptx.sreg.tid.z() #0
 %4 = insertvalue %"struct.cort::uint3" undef, i32 %1, 0
 %5 = insertvalue %"struct.cort::uint3" %4, i32 %2, 1
 %6 = insertvalue %"struct.cort::uint3" %5, i32 %3, 2
 ret %"struct.cort::uint3" %6
}

declare i32 @optix.lwvm.read.ptx.sreg.ntid.x() #0
declare i32 @optix.lwvm.read.ptx.sreg.ntid.y() #0
declare i32 @optix.lwvm.read.ptx.sreg.ntid.z() #0

define %"struct.cort::uint3" @_ZN4lwda4ntidEv() #0 {
 %1 = tail call i32 @optix.lwvm.read.ptx.sreg.ntid.x() #0
 %2 = tail call i32 @optix.lwvm.read.ptx.sreg.ntid.y() #0
 %3 = tail call i32 @optix.lwvm.read.ptx.sreg.ntid.z() #0
 %4 = insertvalue %"struct.cort::uint3" undef, i32 %1, 0
 %5 = insertvalue %"struct.cort::uint3" %4, i32 %2, 1
 %6 = insertvalue %"struct.cort::uint3" %5, i32 %3, 2
 ret %"struct.cort::uint3" %6
}

declare i32 @optix.lwvm.read.ptx.sreg.ctaid.x() #0
declare i32 @optix.lwvm.read.ptx.sreg.ctaid.y() #0
declare i32 @optix.lwvm.read.ptx.sreg.ctaid.z() #0

define %"struct.cort::uint3" @_ZN4lwda5ctaidEv() #0 {
 %1 = tail call i32 @optix.lwvm.read.ptx.sreg.ctaid.x() #0
 %2 = tail call i32 @optix.lwvm.read.ptx.sreg.ctaid.y() #0
 %3 = tail call i32 @optix.lwvm.read.ptx.sreg.ctaid.z() #0
 %4 = insertvalue %"struct.cort::uint3" undef, i32 %1, 0
 %5 = insertvalue %"struct.cort::uint3" %4, i32 %2, 1
 %6 = insertvalue %"struct.cort::uint3" %5, i32 %3, 2
 ret %"struct.cort::uint3" %6
}

declare i32 @optix.lwvm.read.ptx.sreg.nctaid.x() #0
declare i32 @optix.lwvm.read.ptx.sreg.nctaid.y() #0
declare i32 @optix.lwvm.read.ptx.sreg.nctaid.z() #0

define %"struct.cort::uint3" @_ZN4lwda6nctaidEv() #0 {
 %1 = tail call i32 @optix.lwvm.read.ptx.sreg.nctaid.x() #0
 %2 = tail call i32 @optix.lwvm.read.ptx.sreg.nctaid.y() #0
 %3 = tail call i32 @optix.lwvm.read.ptx.sreg.nctaid.z() #0
 %4 = insertvalue %"struct.cort::uint3" undef, i32 %1, 0
 %5 = insertvalue %"struct.cort::uint3" %4, i32 %2, 1
 %6 = insertvalue %"struct.cort::uint3" %5, i32 %3, 2
 ret %"struct.cort::uint3" %6
}

;
; low-level ids.  Note that some of these are volatile.
;

; warpid is volatile!
declare i32 @optix.lwvm.read.ptx.sreg.warpid() #1
define i32 @_ZN4lwda6warpidEv() #1 {
 %1 = tail call i32 @optix.lwvm.read.ptx.sreg.warpid() #1
 ret i32 %1
}

declare i32 @optix.lwvm.read.ptx.sreg.nwarpid() #0
define i32 @_ZN4lwda7nwarpidEv() #0 {
 %1 = tail call i32 @optix.lwvm.read.ptx.sreg.nwarpid() #0
 ret i32 %1
}

; smid is volatile!
declare i32 @optix.lwvm.read.ptx.sreg.smid() #1
define i32 @_ZN4lwda4smidEv() #1 {
 %1 = tail call i32 @optix.lwvm.read.ptx.sreg.smid() #1
 ret i32 %1
}

declare i32 @optix.lwvm.read.ptx.sreg.nsmid() #0
define i32 @_ZN4lwda5nsmidEv() #0 {
 %1 = tail call i32 @optix.lwvm.read.ptx.sreg.nsmid() #0
 ret i32 %1
}

declare i64 @optix.lwvm.read.ptx.sreg.gridid() #0
define i64 @_ZN4lwda6grididEv() #0 {
 %1 = tail call i64 @optix.lwvm.read.ptx.sreg.gridid() #0
 ret i64 %1
}

;
;  Laneid/lanemask
;

declare i32 @optix.lwvm.read.ptx.sreg.laneid() #0
define i32 @_ZN4lwda6laneidEv() #0 {
 %1 = tail call i32 @optix.lwvm.read.ptx.sreg.laneid() #0
 ret i32 %1
}

declare i32 @optix.lwvm.read.ptx.sreg.lanemask_eq() #0
define i32 @_ZN4lwda11lanemask_eqEv() #0 {
 %1 = tail call i32 @optix.lwvm.read.ptx.sreg.lanemask_eq() #0
 ret i32 %1
}

declare i32 @optix.lwvm.read.ptx.sreg.lanemask_le() #0
define i32 @_ZN4lwda11lanemask_leEv() #0 {
 %1 = tail call i32 @optix.lwvm.read.ptx.sreg.lanemask_le() #0
 ret i32 %1
}

declare i32 @optix.lwvm.read.ptx.sreg.lanemask_lt() #0
define i32 @_ZN4lwda11lanemask_ltEv() #0 {
 %1 = tail call i32 @optix.lwvm.read.ptx.sreg.lanemask_lt() #0
 ret i32 %1
}

declare i32 @optix.lwvm.read.ptx.sreg.lanemask_ge() #0
define i32 @_ZN4lwda11lanemask_geEv() #0 {
 %1 = tail call i32 @optix.lwvm.read.ptx.sreg.lanemask_ge() #0
 ret i32 %1
}

declare i32 @optix.lwvm.read.ptx.sreg.lanemask_gt() #0
define i32 @_ZN4lwda11lanemask_gtEv() #0 {
 %1 = tail call i32 @optix.lwvm.read.ptx.sreg.lanemask_gt() #0
 ret i32 %1
}

;
; Count Leading Zeros
;
declare i32 @optix.ptx.clz.b32(i32) #0
define i32 @_ZN4lwda3clzEj(i32 %value) #0
{
  %r = tail call i32 @optix.ptx.clz.b32(i32 %value) #0
  ret i32 %r
}

;
; Bit twiddling intrinsics
;

declare i32 @optix.ptx.popc.b32(i32) #0
define i32 @_ZN4lwda4popcEj(i32 %x) #0 {
 %r = tail call i32 @optix.ptx.popc.b32(i32 %x) #0
 ret i32 %r
}

declare i32 @optix.ptx.bfind.u32(i32) #0
define i32 @_ZN4lwda3ffsEj(i32 %x) #0 {
 %r = tail call i32 @optix.ptx.bfind.u32(i32 %x) #0
 ret i32 %r
}

define i32 @_ZN4lwda12float_as_intEf(float %f) #0 {
  %r = bitcast float %f to i32
  ret i32 %r
}

define float @_ZN4lwda12int_as_floatEj(i32 %r) #0 {
  %f = bitcast i32 %r to float
  ret float %f
}

declare i32 @optix.ptx.bfe.u32(i32, i32, i32) #0
define i32 @_ZN4lwda3bfeEjii(i32 %val, i32 %pos, i32 %len) #0 {
 %r = tail call i32 @optix.ptx.bfe.u32(i32 %val, i32 %pos, i32 %len) #0
 ret i32 %r
}

declare i32 @optix.ptx.bfi.b32(i32, i32, i32, i32) #0
define i32 @_ZN4lwda3bfiEjjii(i32 %src, i32 %dst, i32 %pos, i32 %len) #0 {
 %r = tail call i32 @optix.ptx.bfi.b32(i32 %src, i32 %dst, i32 %pos, i32 %len) #0
 ret i32 %r
}


declare i32 @optix.ptx.prmt.b32(i32, i32, i32) #0
define i32 @_ZN4lwda7permuteEjjj(i32 %valuesLo, i32 %valuesHi, i32 %indices) #0 {
 %r = tail call i32 @optix.ptx.prmt.b32(i32 %valuesLo, i32 %valuesHi, i32 %indices) #0
 ret i32 %r
}

declare i32 @optix.ptx.slct.s32.s32(i32, i32, i32) #0
define i32 @_ZN4lwda6selectEiii(i32 %a, i32 %b, i32 %c) #0 {
 %r = tail call i32 @optix.ptx.slct.s32.s32(i32 %a, i32 %b, i32 %c) #0
 ret i32 %r
}

;
; Vote: Volatile.
;

declare i32 @optix.ptx.vote.ballot.b32(i1) #1
define i32 @_ZN4lwda6ballotEj(i32 %x) #1 {
 %p = icmp ne i32 %x, 0
 %r = tail call i32 @optix.ptx.vote.ballot.b32(i1 %p) #1
 ret i32 %r
}

;
; Clock. These are also volatile.
;

declare i32 @optix.lwvm.read.ptx.sreg.clock() #1
define i32 @_ZN4lwda5clockEv() #1 {
 %1 = tail call i32 @optix.lwvm.read.ptx.sreg.clock() #1
 ret i32 %1
}

declare i64 @optix.lwvm.read.ptx.sreg.clock64() #1
define i64 @_ZN4lwda7clock64Ev() #1 {
 %1 = tail call i64 @optix.lwvm.read.ptx.sreg.clock64() #1
 ret i64 %1
}

;
; Debug breakpoint. Volatile
;

declare void @optix.ptx.brkpt() #1
define void @_ZN4lwda8dbgbreakEv() #1 {
  call void @optix.ptx.brkpt()
  ret void
}

;
; Shuffle. Volatile ( deprecated as of ptx6.0)
;
declare { i32, i1 } @optix.ptx.shfl.bfly.b32( i32 %p0, i32 %p1, i32 %p2 ) #1
define i32 @_ZN4lwda8shfl_xorEii( i32 %p0, i32 %p1 ) #1 {
  %r0 = call { i32, i1 } @optix.ptx.shfl.bfly.b32( i32 %p0, i32 %p1, i32 31 ) #1
  %r1 = extractvalue { i32, i1 } %r0, 0
  ret i32 %r1
}

declare { i32, i1 } @optix.ptx.shfl.idx.b32( i32 %p0, i32 %p1, i32 %p2 ) #1
define i32 @_ZN4lwda4shflEii( i32 %p0, i32 %p1 ) #1 {
  %r0 = call { i32, i1 } @optix.ptx.shfl.idx.b32( i32 %p0, i32 %p1, i32 31 ) #1
  %r1 = extractvalue { i32, i1 } %r0, 0
  ret i32 %r1
}

;
; Shuffle. Volatile. synchronized ( as of ptx6.0 )
;
declare { i32, i1 } @optix.ptx.shfl.sync.bfly.b32( i32 %p0, i32 %p1, i32 %p2, i32 %memberMask ) #1
define i32 @_ZN4lwda13shfl_xor_syncEiii( i32 %p0, i32 %p1, i32 %memberMask ) #1 {
  %r0 = call { i32, i1 } @optix.ptx.shfl.sync.bfly.b32( i32 %p0, i32 %p1, i32 31, i32 %memberMask ) #1
  %r1 = extractvalue { i32, i1 } %r0, 0
  ret i32 %r1
}

declare { i32, i1 } @optix.ptx.shfl.sync.idx.b32( i32 %p0, i32 %p1, i32 %p2, i32 %memberMask ) #1
define i32 @_ZN4lwda9shfl_syncEiii( i32 %p0, i32 %p1, i32 %memberMask ) #1 {
  %r0 = call { i32, i1 } @optix.ptx.shfl.sync.idx.b32( i32 %p0, i32 %p1, i32 31, i32 %memberMask ) #1
  %r1 = extractvalue { i32, i1 } %r0, 0
  ret i32 %r1
}

;
; synchronization. Volatile
;

declare void @optix.lwca.syncthreads() #1
define void @_ZN4lwda11syncthreadsEv() #1 
{
  call void @optix.lwca.syncthreads()
  ret void
}

;
; Math intrinsics
;

declare float @llvm.sqrt.f32(float) #0
define float @_ZN4cort4sqrtEf(float %f) #0
{
  %s = call float @llvm.sqrt.f32(float %f) #0
  ret float %s
}

define float @_ZN4cort3nanEv() #0
{
  ret float 0x7FFFFFFFE0000000
}

declare i32 @optix.lwvm.min.i(i32, i32) #0
define i32 @_ZN4lwda4miniEii(i32 %x, i32 %y) #0 {
  %1 = tail call i32 @optix.lwvm.min.i(i32 %x, i32 %y) #0
  ret i32 %1
}

declare i32 @optix.lwvm.max.i(i32, i32) #0
define i32 @_ZN4lwda4maxiEii(i32 %x, i32 %y) #0 {
  %1 = tail call i32 @optix.lwvm.max.i(i32 %x, i32 %y) #0
  ret i32 %1
}

declare i32 @optix.lwvm.min.ui(i32, i32) #0
define i32 @_ZN4lwda5minuiEjj(i32 %x, i32 %y) #0 {
  %1 = tail call i32 @optix.lwvm.min.ui(i32 %x, i32 %y) #0
  ret i32 %1
}

declare i32 @optix.lwvm.max.ui(i32, i32) #0
define i32 @_ZN4lwda5maxuiEjj(i32 %x, i32 %y) #0 {
  %1 = tail call i32 @optix.lwvm.max.ui(i32 %x, i32 %y) #0
  ret i32 %1
}

declare float @optix.lwvm.fmin.f(float, float) #0
define float @_ZN4lwda4minfEff(float %x, float %y) #0 {
  %1 = tail call float @optix.lwvm.fmin.f(float %x, float %y) #0
  ret float %1
}

declare float @optix.lwvm.fmax.f(float, float) #0
define float @_ZN4lwda4maxfEff(float %x, float %y) #0 {
  %1 = tail call float @optix.lwvm.fmax.f(float %x, float %y) #0
  ret float %1
}

declare float @optix.lwvm.floor.f(float) #0
define float @_ZN4lwda6floorfEf(float %x) #0 {
  %1 = tail call float @optix.lwvm.floor.f(float %x) #0
  ret float %1
}

declare float @optix.lwvm.ceil.f(float) #0
define float @_ZN4lwda5ceilfEf(float %x) #0 {
  %1 = tail call float @optix.lwvm.ceil.f(float %x) #0
  ret float %1
}

declare float @optix.lwvm.ui2f.rz(i32) #0
define float @_ZN4lwda13uint2float_rzEj(i32 %x) #0 {
  %1 = tail call float @optix.lwvm.ui2f.rz(i32 %x) #0
  ret float %1
}

declare float @optix.lwvm.i2f.rz(i32) #0
define float @_ZN4lwda12int2float_rzEi(i32 %x) #0 {
  %1 = tail call float @optix.lwvm.i2f.rz(i32 %x) #0
  ret float %1
}

declare i32 @optix.lwvm.f2i.rz(float) #0
define i32 @_ZN4lwda12float2int_rzEf(float %x) #0 {
  %1 = tail call i32 @optix.lwvm.f2i.rz(float %x) #0
  ret i32 %1
}

declare i32 @optix.lwvm.f2i.rm(float) #0
define i32 @_ZN4lwda12float2int_rmEf(float %x) #0 {
  %1 = tail call i32 @optix.lwvm.f2i.rm(float %x) #0
  ret i32 %1
}

declare float @optix.lwvm.mul.rz.f(float, float) #0
define float @_ZN4lwda6mul_rzEff(float %x, float %y) #0 {
  %1 = tail call float @optix.lwvm.mul.rz.f(float %x, float %y) #0
  ret float %1
}

declare float @optix.lwvm.add.rz.f(float, float) #0
define float @_ZN4lwda6add_rzEff(float %x, float %y) #0 {
  %1 = tail call float @optix.lwvm.add.rz.f(float %x, float %y) #0
  ret float %1
}

declare float @optix.ptx.rcp.approx.ftz.f32(float) #0
define float @_ZN4lwda3rcpEf(float %x) #0
{
  %1 = tail call float @optix.ptx.rcp.approx.ftz.f32(float %x) #0
  ret float %1
}

declare float @optix.ptx.fma.rn.ftz.f32(float, float, float) #0
define float @_ZN4lwda4fmadEfff(float %a, float %b, float %c) #0 {
  %1 = tail call float @optix.ptx.fma.rn.ftz.f32(float %a, float %b, float %c) #0
  ret float %1
}

declare float @optix.ptx.fma.rp.f32(float, float, float) #0
define float @_ZN4lwda7fmad_rpEfff(float %a, float %b, float %c) #0 {
  %1 = tail call float @optix.ptx.fma.rp.f32(float %a, float %b, float %c) #0
  ret float %1
}

declare float @optix.ptx.fma.rm.f32(float, float, float) #0
define float @_ZN4lwda7fmad_rmEfff(float %a, float %b, float %c) #0 {
  %1 = tail call float @optix.ptx.fma.rm.f32(float %a, float %b, float %c) #0
  ret float %1
}

declare float @optix.ptx.cvt.sat.f32.f32(float) #0
define float @_ZN4lwda8saturateEf(float %x) #0 {
  %1 = tail call float @optix.ptx.cvt.sat.f32.f32(float %x) #0
  ret float %1
}

declare float @llvm.fabs.f32(float) #0
define float @_ZN4lwda4fabsEf(float %x) #0 {
  %1 = tail call float @llvm.fabs.f32(float %x) #0
  ret float %1
}

; The llvm.exp2.f32 intrinsic is not supported by LWVM, so we use inline PTX assembly (see libDevice.ll)
declare float @llvm.exp2.f32(float) #0
declare float @optix.ptx.ex2.approx.f32(float) #0
define float @_ZN4lwda4exp2Ef(float %x) #0 {
  %1 = tail call float @optix.ptx.ex2.approx.f32(float %x) #0
  ret float %1
}

declare float @optix.ptx.lg2.approx.f32(float) #0
define float @_ZN4lwda4log2Ef(float %x) #0 {
  %1 = tail call float @optix.ptx.lg2.approx.f32(float %x) #0
  ret float %1
}

;
; Video intrinsics
;

declare i32 @optix.ptx.vmin.min.s32.s32.s32( i32, i32, i32 ) #0
define i32 @_ZN4lwda6minminEiii(i32 %a, i32 %b, i32 %c) #0 {
  %1 = tail call i32 @optix.ptx.vmin.min.s32.s32.s32( i32 %a, i32 %b, i32 %c ) #0
  ret i32 %1
}

declare i32 @optix.ptx.vmin.max.s32.s32.s32( i32, i32, i32 ) #0
define i32 @_ZN4lwda6minmaxEiii(i32 %a, i32 %b, i32 %c) #0 {
  %1 = tail call i32 @optix.ptx.vmin.max.s32.s32.s32( i32 %a, i32 %b, i32 %c ) #0
  ret i32 %1
}

declare i32 @optix.ptx.vmax.min.s32.s32.s32( i32, i32, i32 ) #0
define i32 @_ZN4lwda6maxminEiii(i32 %a, i32 %b, i32 %c) #0 {
  %1 = tail call i32 @optix.ptx.vmax.min.s32.s32.s32( i32 %a, i32 %b, i32 %c ) #0
  ret i32 %1
}

declare i32 @optix.ptx.vmax.max.s32.s32.s32( i32, i32, i32 ) #0
define i32 @_ZN4lwda6maxmaxEiii(i32 %a, i32 %b, i32 %c) #0 {
  %1 = tail call i32 @optix.ptx.vmax.max.s32.s32.s32( i32 %a, i32 %b, i32 %c ) #0
  ret i32 %1
}

declare i32 @optix.ptx.vshl.clamp.u32.u32.u32.b0(i32, i32) #0
define i32 @_ZN4lwda13vshl_clamp_b0Ejj(i32 %val, i32 %shift) #0 {
 %r = tail call i32 @optix.ptx.vshl.clamp.u32.u32.u32.b0(i32 %val, i32 %shift) #0
 ret i32 %r
}

declare i32 @optix.ptx.vshl.clamp.u32.u32.u32.b1(i32, i32) #0
define i32 @_ZN4lwda13vshl_clamp_b1Ejj(i32 %val, i32 %shift) #0 {
 %r = tail call i32 @optix.ptx.vshl.clamp.u32.u32.u32.b1(i32 %val, i32 %shift) #0
 ret i32 %r
}

declare i32 @optix.ptx.vshl.clamp.u32.u32.u32.b2(i32, i32) #0
define i32 @_ZN4lwda13vshl_clamp_b2Ejj(i32 %val, i32 %shift) #0 {
 %r = tail call i32 @optix.ptx.vshl.clamp.u32.u32.u32.b2(i32 %val, i32 %shift) #0
 ret i32 %r
}

declare i32 @optix.ptx.vshl.clamp.u32.u32.u32.b3(i32, i32) #0
define i32 @_ZN4lwda13vshl_clamp_b3Ejj(i32 %val, i32 %shift) #0 {
 %r = tail call i32 @optix.ptx.vshl.clamp.u32.u32.u32.b3(i32 %val, i32 %shift) #0
 ret i32 %r
}

declare i32 @optix.ptx.vshl.wrap.add.u32.u32.u32.b0.b0(i32, i32, i32) #0
define i32 @_ZN4lwda19vshl_wrap_add_b0_b0Ejjj(i32 %val, i32 %shift, i32 %adden) #0 {
 %r = tail call i32 @optix.ptx.vshl.wrap.add.u32.u32.u32.b0.b0(i32 %val, i32 %shift, i32 %adden) #0
 ret i32 %r
}

declare i32 @optix.ptx.vshl.wrap.add.u32.u32.u32.b1.b1(i32, i32, i32) #0
define i32 @_ZN4lwda19vshl_wrap_add_b1_b1Ejjj(i32 %val, i32 %shift, i32 %adden) #0 {
 %r = tail call i32 @optix.ptx.vshl.wrap.add.u32.u32.u32.b1.b1(i32 %val, i32 %shift, i32 %adden) #0
 ret i32 %r
}

declare i32 @optix.ptx.vshl.wrap.add.u32.u32.u32.b2.b2(i32, i32, i32) #0
define i32 @_ZN4lwda19vshl_wrap_add_b2_b2Ejjj(i32 %val, i32 %shift, i32 %adden) #0 {
 %r = tail call i32 @optix.ptx.vshl.wrap.add.u32.u32.u32.b2.b2(i32 %val, i32 %shift, i32 %adden) #0
 ret i32 %r
}

declare i32 @optix.ptx.vshl.wrap.add.u32.u32.u32.b3.b3(i32, i32, i32) #0
define i32 @_ZN4lwda19vshl_wrap_add_b3_b3Ejjj(i32 %val, i32 %shift, i32 %adden) #0 {
 %r = tail call i32 @optix.ptx.vshl.wrap.add.u32.u32.u32.b3.b3(i32 %val, i32 %shift, i32 %adden) #0
 ret i32 %r
}


;
; Texture implementations for bindless texture.  Implemented in backend
;

define %"struct.cort::float4" @colwert.to.float4( { float, float, float, float } %s ) #0 {
  %rx = extractvalue { float, float, float, float } %s, 0
  %px = insertvalue %"struct.cort::float4" undef, float %rx, 0
  %ry = extractvalue { float, float, float, float } %s, 1
  %py = insertvalue %"struct.cort::float4" %px, float %ry, 1
  %rz = extractvalue { float, float, float, float } %s, 2
  %pz = insertvalue %"struct.cort::float4" %py, float %rz, 2
  %rw = extractvalue { float, float, float, float } %s, 3
  %pw = insertvalue %"struct.cort::float4" %pz, float %rw, 3
  ret %"struct.cort::float4" %pw
}

define %"struct.cort::float4" @colwert.v4.f32.to.float4( <4 x float> %p0 ) #0 {
  %rx = extractelement <4 x float> %p0, i32 0
  %px = insertvalue %"struct.cort::float4" undef, float %rx, 0
  %ry = extractelement <4 x float> %p0, i32 1
  %py = insertvalue %"struct.cort::float4" %px, float %ry, 1
  %rz = extractelement <4 x float> %p0, i32 2
  %pz = insertvalue %"struct.cort::float4" %py, float %rz, 2
  %rw = extractelement <4 x float> %p0, i32 3
  %pw = insertvalue %"struct.cort::float4" %pz, float %rw, 3
  ret %"struct.cort::float4" %pw
}

define %"struct.cort::uint3" @colwert.to.uint3( i32 %w, i32 %h, i32 %d ) #0 {
  %px = insertvalue %"struct.cort::uint3" undef, i32 %w, 0
  %py = insertvalue %"struct.cort::uint3" %px, i32 %h, 1
  %pz = insertvalue %"struct.cort::uint3" %py, i32 %d, 2
  ret %"struct.cort::uint3" %pz
}

declare i32 @optix.lwvm.txq_width(i64 %texref) #0
define weak i32 @_ZN4cort31Texture_getElement_hw_txq_widthEy(i64 %texref) #0
{
  %w = call i32 @optix.lwvm.txq_width(i64 %texref) #0
  ret i32 %w
}

declare i32 @optix.lwvm.txq_height(i64 %texref) #0
define weak i32 @_ZN4cort32Texture_getElement_hw_txq_heightEy(i64 %texref) #0
{
  %h = call i32 @optix.lwvm.txq_height(i64 %texref) #0
  ret i32 %h
}

declare i32 @optix.lwvm.txq_depth(i64 %texref) #0
define weak i32 @_ZN4cort31Texture_getElement_hw_txq_depthEy(i64 %texref) #0
{
  %d = call i32 @optix.lwvm.txq_depth(i64 %texref) #0
  ret i32 %d
}

define weak %"struct.cort::uint3" @_ZN4cort26Texture_getElement_hw_sizeEy(i64 %texref) #0
{
  %w = call i32 @optix.lwvm.txq_width(i64 %texref) #0
  %h = call i32 @optix.lwvm.txq_height(i64 %texref) #0
  %d = call i32 @optix.lwvm.txq_depth(i64 %texref) #0
  %r = call %"struct.cort::uint3" @colwert.to.uint3( i32 %w, i32 %h, i32 %d ) #0
  ret %"struct.cort::uint3" %r
}

declare { float, float, float, float } @optix.lwvm.tex_1d(i64 %texref, float %x) #0
define weak %"struct.cort::float4" @_ZN4cort28Texture_getElement_hw_tex_1dEyf(i64 %texref, float %x) #0
{
  %s = call { float, float, float, float } @optix.lwvm.tex_1d(i64 %texref, float %x) #0
  %r = call %"struct.cort::float4" @colwert.to.float4( { float, float, float, float } %s ) #0
  ret %"struct.cort::float4" %r
}

declare { float, float, float, float } @optix.lwvm.tex_2d(i64 %texref, float, float) #0
define weak %"struct.cort::float4" @_ZN4cort28Texture_getElement_hw_tex_2dEyff(i64 %texref, float %x, float %y ) #0
{
  %s = call { float, float, float, float } @optix.lwvm.tex_2d( i64 %texref, float %x, float %y ) #0
  %r = call %"struct.cort::float4" @colwert.to.float4( { float, float, float, float } %s ) #0
  ret %"struct.cort::float4" %r
}

declare { float, float, float, float } @optix.lwvm.tex_3d(i64 %texref, float, float, float) #0
define weak %"struct.cort::float4" @_ZN4cort28Texture_getElement_hw_tex_3dEyfff(i64 %texref, float %x, float %y, float %z ) #0
{
  %s = call { float, float, float, float } @optix.lwvm.tex_3d( i64 %texref, float %x, float %y, float %z ) #0
  %r = call %"struct.cort::float4" @colwert.to.float4( { float, float, float, float } %s ) #0
  ret %"struct.cort::float4" %r
}

declare < 4 x float > @optix.ptx.nonsparse.tex.a1d.v4.f32.f32( i64 %p0, { i32, float } %p1 ) #1
define weak %"struct.cort::float4" @_ZN4cort29Texture_getElement_hw_tex_a1dEyjf(i64 %texref, i32 %a, float %x) #0
{
  %params0 = insertvalue { i32, float } undef, i32 %a, 0
  %params1 = insertvalue { i32, float } %params0, float %x, 1

  %res = call < 4 x float > @optix.ptx.nonsparse.tex.a1d.v4.f32.f32( i64 %texref, { i32, float } %params1 ) #1
  %final = call %"struct.cort::float4" @colwert.v4.f32.to.float4( < 4 x float > %res ) #0
  ret %"struct.cort::float4" %final
}

declare { float, float, float, float } @optix.lwvm.tex_a2d(i64 %texref, i32, float, float) #0
define weak %"struct.cort::float4" @_ZN4cort29Texture_getElement_hw_tex_a2dEyjff(i64 %texref, i32 %a, float %x, float %y) #0
{
  %s = call { float, float, float, float } @optix.lwvm.tex_a2d( i64 %texref, i32 %a, float %x, float %y ) #0
  %r = call %"struct.cort::float4" @colwert.to.float4( { float, float, float, float } %s ) #0
  ret %"struct.cort::float4" %r
}

declare { float, float, float, float } @optix.lwvm.tex_lwbe(i64 %texref, float, float, float) #0
define weak %"struct.cort::float4" @_ZN4cort30Texture_getElement_hw_tex_lwbeEyfff(i64 %texref, float %x, float %y, float %z) #0
{
  %s = call { float, float, float, float } @optix.lwvm.tex_lwbe( i64 %texref, float %x, float %y, float %z ) #0
  %r = call %"struct.cort::float4" @colwert.to.float4( { float, float, float, float } %s ) #0
  ret %"struct.cort::float4" %r
}

declare < 4 x float > @optix.ptx.nonsparse.tex.alwbe.v4.f32.f32( i64 %p0, { i32, float, float, float } %p1 ) #1
define weak %"struct.cort::float4" @_ZN4cort31Texture_getElement_hw_tex_alwbeEyjfff(i64 %texref, i32 %a, float %x, float %y, float %z) #0
{
  %params0 = insertvalue { i32, float, float, float } undef, i32 %a, 0
  %params1 = insertvalue { i32, float, float, float } %params0, float %x, 1
  %params2 = insertvalue { i32, float, float, float } %params1, float %y, 2
  %params3 = insertvalue { i32, float, float, float } %params2, float %z, 3

  %res = call <4 x float> @optix.ptx.nonsparse.tex.alwbe.v4.f32.f32( i64 %texref, { i32, float, float, float } %params3 ) #1
  %final = call %"struct.cort::float4" @colwert.v4.f32.to.float4( <4 x float> %res ) #0
  ret %"struct.cort::float4" %final
}

declare { float, float, float, float } @optix.lwvm.texfetch_1d(i64 %texref, i32) #0
define weak %"struct.cort::float4" @_ZN4cort33Texture_getElement_hw_texfetch_1dEyi(i64 %texref, i32 %x) #0
{
  %s = call { float, float, float, float } @optix.lwvm.texfetch_1d( i64 %texref, i32 %x ) #0
  %r = call %"struct.cort::float4" @colwert.to.float4( { float, float, float, float } %s ) #0
  ret %"struct.cort::float4" %r
}

declare { float, float, float, float } @optix.lwvm.texfetch_2d(i64 %texref, i32, i32) #0
define weak %"struct.cort::float4" @_ZN4cort33Texture_getElement_hw_texfetch_2dEyii(i64 %texref, i32 %x, i32 %y) #0
{
  %s = call { float, float, float, float } @optix.lwvm.texfetch_2d( i64 %texref, i32 %x, i32 %y ) #0
  %r = call %"struct.cort::float4" @colwert.to.float4( { float, float, float, float } %s ) #0
  ret %"struct.cort::float4" %r
}

declare { float, float, float, float } @optix.lwvm.texfetch_3d(i64 %texref, i32, i32, i32) #0
define weak %"struct.cort::float4" @_ZN4cort33Texture_getElement_hw_texfetch_3dEyiii(i64 %texref, i32 %x, i32 %y, i32 %z) #0
{
  %s = call { float, float, float, float } @optix.lwvm.texfetch_3d( i64 %texref, i32 %x, i32 %y, i32 %z ) #0
  %r = call %"struct.cort::float4" @colwert.to.float4( { float, float, float, float } %s ) #0
  ret %"struct.cort::float4" %r
}

declare { float, float, float, float } @optix.lwvm.texfetch_a1d(i64 %texref, i32, i32) #0
define weak %"struct.cort::float4" @_ZN4cort34Texture_getElement_hw_texfetch_a1dEyji(i64 %texref, i32 %a, i32 %x) #0
{
  %s = call { float, float, float, float } @optix.lwvm.texfetch_a1d( i64 %texref, i32 %a, i32 %x ) #0
  %r = call %"struct.cort::float4" @colwert.to.float4( { float, float, float, float } %s ) #0
  ret %"struct.cort::float4" %r
}

declare { float, float, float, float } @optix.lwvm.texfetch_a2d(i64 %texref, i32, i32, i32) #0
define weak %"struct.cort::float4" @_ZN4cort34Texture_getElement_hw_texfetch_a2dEyjii(i64 %texref, i32 %a, i32 %x, i32 %y) #0
{
  %s = call { float, float, float, float } @optix.lwvm.texfetch_a2d( i64 %texref, i32 %a, i32 %x, i32 %y ) #0
  %r = call %"struct.cort::float4" @colwert.to.float4( { float, float, float, float } %s ) #0
  ret %"struct.cort::float4" %r
}

declare { float, float, float, float } @optix.lwvm.texfetch_2dms(i64 %texref, i32, i32, i32) #0
define weak %"struct.cort::float4" @_ZN4cort35Texture_getElement_hw_texfetch_2dmsEyjii(i64 %texref, i32 %samp, i32 %x, i32 %y) #0
{
  %s = call { float, float, float, float } @optix.lwvm.texfetch_2dms( i64 %texref, i32 %samp, i32 %x, i32 %y ) #0
  %r = call %"struct.cort::float4" @colwert.to.float4( { float, float, float, float } %s ) #0
  ret %"struct.cort::float4" %r
}

declare { float, float, float, float } @optix.lwvm.texfetch_a2dms(i64 %texref, i32, i32, i32, i32) #0
define weak %"struct.cort::float4" @_ZN4cort36Texture_getElement_hw_texfetch_a2dmsEyjjii(i64 %texref, i32 %samp, i32 %a, i32 %x, i32 %y ) #0
{
  %s = call { float, float, float, float } @optix.lwvm.texfetch_a2dms( i64 %texref, i32 %samp, i32 %a, i32 %x, i32 %y ) #0
  %r = call %"struct.cort::float4" @colwert.to.float4( { float, float, float, float } %s ) #0
  ret %"struct.cort::float4" %r
}

declare <4 x float> @optix.ptx.nonsparse.tex.level.1d.v4.f32.f32(i64 %p0, float %p1, float %p2) #1
define weak %"struct.cort::float4" @_ZN4cort33Texture_getElement_hw_texlevel_1dEyff(i64 %texref, float %x, float %level ) #0
{
  %res = call <4 x float> @optix.ptx.nonsparse.tex.level.1d.v4.f32.f32(i64 %texref, float %x, float %level) #1
  %final = call %"struct.cort::float4" @colwert.v4.f32.to.float4( < 4 x float > %res ) #0
  ret %"struct.cort::float4" %final
}

declare { float, float, float, float } @optix.lwvm.texlevel_2d(i64 %texref, float, float, float) #0
define weak %"struct.cort::float4" @_ZN4cort33Texture_getElement_hw_texlevel_2dEyfff(i64 %texref, float %x, float %y, float %level ) #0
{
  %s = call { float, float, float, float } @optix.lwvm.texlevel_2d( i64 %texref, float %x, float %y, float %level ) #0
  %r = call %"struct.cort::float4" @colwert.to.float4( { float, float, float, float } %s ) #0
  ret %"struct.cort::float4" %r
}

declare <4 x float> @optix.ptx.nonsparse.tex.level.3d.v4.f32.f32(i64 %p0, <4 x float> %p1, float %p2) #1
define weak %"struct.cort::float4" @_ZN4cort33Texture_getElement_hw_texlevel_3dEyffff(i64 %texref, float %x, float %y, float %z, float %level ) #0
{
  %coords0 = insertelement < 4 x float > undef, float %x, i32 0
  %coords1 = insertelement < 4 x float > %coords0, float %y, i32 1
  %coords2 = insertelement < 4 x float > %coords1, float %z, i32 2

  %res = call <4 x float> @optix.ptx.nonsparse.tex.level.3d.v4.f32.f32(i64 %texref, <4 x float> %coords2, float %level) #1
  %final = call %"struct.cort::float4" @colwert.v4.f32.to.float4( < 4 x float > %res ) #0
  ret %"struct.cort::float4" %final
}

declare < 4 x float > @optix.ptx.nonsparse.tex.level.a1d.v4.f32.f32( i64 %p0, { i32, float } %p1, float %p2 ) #1
define weak %"struct.cort::float4" @_ZN4cort34Texture_getElement_hw_texlevel_a1dEyjff(i64 %texref, i32 %a, float %x, float %level ) #0
{
  %params0 = insertvalue { i32, float } undef, i32 %a, 0
  %params1 = insertvalue { i32, float } %params0, float %x, 1

  %res = call < 4 x float > @optix.ptx.nonsparse.tex.level.a1d.v4.f32.f32( i64 %texref, { i32, float } %params1, float %level ) #1
  %final = call %"struct.cort::float4" @colwert.v4.f32.to.float4( < 4 x float > %res ) #0
  ret %"struct.cort::float4" %final
}

declare { float, float, float, float } @optix.lwvm.texlevel_a2d(i64 %texref, i32, float, float, float) #0
define weak %"struct.cort::float4" @_ZN4cort34Texture_getElement_hw_texlevel_a2dEyjfff(i64 %texref, i32 %a, float %x, float %y, float %level ) #0
{
  %s = call { float, float, float, float } @optix.lwvm.texlevel_a2d( i64 %texref, i32 %a, float %x, float %y, float %level ) #0
  %r = call %"struct.cort::float4" @colwert.to.float4( { float, float, float, float } %s ) #0
  ret %"struct.cort::float4" %r
}

declare { float, float, float, float } @optix.lwvm.texlevel_lwbe(i64 %texref, float, float, float, float) #0
define weak %"struct.cort::float4" @_ZN4cort35Texture_getElement_hw_texlevel_lwbeEyffff(i64 %texref, float %x, float %y, float %z, float %level ) #0
{
  %s = call { float, float, float, float } @optix.lwvm.texlevel_lwbe( i64 %texref, float %x, float %y, float %z, float %level ) #0
  %r = call %"struct.cort::float4" @colwert.to.float4( { float, float, float, float } %s ) #0
  ret %"struct.cort::float4" %r
}

declare < 4 x float > @optix.ptx.nonsparse.tex.level.alwbe.v4.f32.f32( i64 %p0, { i32, float, float, float } %p1, float %p2 ) #1
define weak %"struct.cort::float4" @_ZN4cort36Texture_getElement_hw_texlevel_alwbeEyjffff(i64 %texref, i32 %a, float %x, float %y, float %z, float %level ) #0
{
  %params0 = insertvalue { i32, float, float, float } undef, i32 %a, 0
  %params1 = insertvalue { i32, float, float, float } %params0, float %x, 1
  %params2 = insertvalue { i32, float, float, float } %params1, float %y, 2
  %params3 = insertvalue { i32, float, float, float } %params2, float %z, 3

  %res = call < 4 x float > @optix.ptx.nonsparse.tex.level.alwbe.v4.f32.f32( i64 %texref, { i32, float, float, float } %params3, float %level ) #1
  %final = call %"struct.cort::float4" @colwert.v4.f32.to.float4( < 4 x float > %res ) #0
  ret %"struct.cort::float4" %final
}

declare <4 x float> @optix.ptx.nonsparse.tex.grad.1d.v4.f32.f32(i64, float, < 1 x float >, < 1 x float >) #0
define weak %"struct.cort::float4" @_ZN4cort32Texture_getElement_hw_texgrad_1dEyfff(i64 %texref, float %x, float %dPdx, float %dPdy ) #0
{
  %dPdx_vec = insertelement < 1 x float > undef, float %dPdx, i32 0
  %dPdy_vec = insertelement < 1 x float > undef, float %dPdy, i32 0
  %res = call <4 x float> @optix.ptx.nonsparse.tex.grad.1d.v4.f32.f32(i64 %texref, float %x, < 1 x float > %dPdx_vec, < 1 x float > %dPdy_vec) #0
  %final = call %"struct.cort::float4" @colwert.v4.f32.to.float4( < 4 x float > %res ) #0
  ret %"struct.cort::float4" %final
}

declare <4 x float> @optix.ptx.nonsparse.tex.grad.2d.v4.f32.f32(i64 %texref, <2 x float> %coords, <2 x float> %ddx, <2 x float> %ddy) #0
define weak %"struct.cort::float4" @_ZN4cort32Texture_getElement_hw_texgrad_2dEyffffff(i64 %texref, float %x, float %y, float %dPdx_x, float %dPdx_y, float %dPdy_x, float %dPdy_y ) #0
{
  %coords0 = insertelement < 2 x float > undef, float %x, i32 0
  %coords1 = insertelement < 2 x float > %coords0, float %y, i32 1

  %dPdx0 = insertelement < 2 x float > undef, float %dPdx_x, i32 0
  %dPdx1 = insertelement < 2 x float > %dPdx0, float %dPdx_y, i32 1

  %dPdy0 = insertelement < 2 x float > undef, float %dPdy_x, i32 0
  %dPdy1 = insertelement < 2 x float > %dPdy0, float %dPdy_y, i32 1

  %res = call < 4 x float > @optix.ptx.nonsparse.tex.grad.2d.v4.f32.f32( i64 %texref, < 2 x float > %coords1, < 2 x float > %dPdx1, < 2 x float > %dPdy1 ) #1
  %final = call %"struct.cort::float4" @colwert.v4.f32.to.float4( < 4 x float > %res ) #0
  ret %"struct.cort::float4" %final
}

declare { <4 x float>, i1 } @optix.ptx.tex.grad.2d.v4.f32.f32(i64, <2 x float>, <2 x float>, <2 x float>) #1
define weak %"struct.cort::float4" @_ZN4cort43Texture_getElement_hw_texgrad_2d_isResidentEyffffffPi(i64 %texref, float %x, float %y, float %dPdx_x, float %dPdx_y, float %dPdy_x, float %dPdy_y, i32* %isResident ) #0
{
  %coords0 = insertelement <2 x float> undef, float %x, i32 0
  %coords1 = insertelement <2 x float> %coords0, float %y, i32 1

  %ddx0 = insertelement <2 x float> undef, float %dPdx_x, i32 0
  %ddx1 = insertelement <2 x float> %ddx0, float %dPdx_y, i32 1

  %ddy0 = insertelement <2 x float> undef, float %dPdy_x, i32 0
  %ddy1 = insertelement <2 x float> %ddy0, float %dPdy_y, i32 1

  %res = call { <4 x float>, i1 } @optix.ptx.tex.grad.2d.v4.f32.f32(i64 %texref, <2 x float> %coords1, <2 x float> %ddx1, <2 x float> %ddy1)

  %texVals = extractvalue { <4 x float>, i1 } %res, 0

  %texVal0 = extractelement <4 x float> %texVals, i32 0
  %texVal1 = extractelement <4 x float> %texVals, i32 1
  %texVal2 = extractelement <4 x float> %texVals, i32 2
  %texVal3 = extractelement <4 x float> %texVals, i32 3

  %texValRes0 = insertvalue %"struct.cort::float4" undef, float %texVal0, 0
  %texValRes1 = insertvalue %"struct.cort::float4" %texValRes0, float %texVal1, 1
  %texValRes2 = insertvalue %"struct.cort::float4" %texValRes1, float %texVal2, 2
  %texValRes3 = insertvalue %"struct.cort::float4" %texValRes2, float %texVal3, 3

  %isResidentResult = extractvalue { <4 x float>, i1 } %res, 1
  %isResidentResultExtended = zext i1 %isResidentResult to i32
  store i32 %isResidentResultExtended, i32* %isResident

  ret %"struct.cort::float4" %texValRes3
}

declare { i32, i32, i32, i32, i1} @optix.lwvm.texgrad_footprint_2d( i32, i64 %texref, float, float, float, float, float, float) #0
define weak %"struct.cort::uint4" @_ZN4cort42Texture_getElement_hw_texgrad_footprint_2dEjyffffffPi( i32 %granularity, i64 %texref, float %x, float %y, float %dPdx_x, float %dPdx_y, float %dPdy_x, float %dPdy_y, i32* %coversSingleMipLevel ) #0
{
  %s = call { i32, i32, i32, i32, i1} @optix.lwvm.texgrad_footprint_2d( i32 %granularity, i64 %texref, float %x, float %y, float %dPdx_x, float %dPdx_y, float %dPdy_x, float %dPdy_y ) #0
  %footprintVal0 = extractvalue {i32, i32 , i32, i32, i1} %s, 0
  %footprintVal1 = extractvalue {i32, i32 , i32, i32, i1} %s, 1
  %footprintVal2 = extractvalue {i32, i32 , i32, i32, i1} %s, 2
  %footprintVal3 = extractvalue {i32, i32 , i32, i32, i1} %s, 3
  %footprintValRes0 = insertvalue %"struct.cort::uint4" undef, i32 %footprintVal0, 0
  %footprintValRes1 = insertvalue %"struct.cort::uint4" %footprintValRes0, i32 %footprintVal1, 1
  %footprintValRes2 = insertvalue %"struct.cort::uint4" %footprintValRes1, i32 %footprintVal2, 2
  %r = insertvalue %"struct.cort::uint4" %footprintValRes2, i32 %footprintVal3, 3
  %coversSingleMipLevelResult = extractvalue {i32, i32, i32, i32, i1} %s, 4
  %coversSingleMipLevelResultExtended = zext i1 %coversSingleMipLevelResult to i32
  store i32 %coversSingleMipLevelResultExtended, i32* %coversSingleMipLevel
  ret %"struct.cort::uint4" %r
}

declare { i32, i32, i32, i32, i1} @optix.lwvm.texgrad_footprint_coarse_2d( i32, i64 %texref, float, float, float, float, float, float) #0
define weak %"struct.cort::uint4" @_ZN4cort49Texture_getElement_hw_texgrad_footprint_coarse_2dEjyffffffPi( i32 %granularity, i64 %texref, float %x, float %y, float %dPdx_x, float %dPdx_y, float %dPdy_x, float %dPdy_y, i32* %coversSingleMipLevel ) #0
{
  %s = call { i32, i32, i32, i32, i1} @optix.lwvm.texgrad_footprint_coarse_2d( i32 %granularity, i64 %texref, float %x, float %y, float %dPdx_x, float %dPdx_y, float %dPdy_x, float %dPdy_y ) #0
  %footprintVal0 = extractvalue {i32, i32 , i32, i32, i1} %s, 0
  %footprintVal1 = extractvalue {i32, i32 , i32, i32, i1} %s, 1
  %footprintVal2 = extractvalue {i32, i32 , i32, i32, i1} %s, 2
  %footprintVal3 = extractvalue {i32, i32 , i32, i32, i1} %s, 3
  %footprintValRes0 = insertvalue %"struct.cort::uint4" undef, i32 %footprintVal0, 0
  %footprintValRes1 = insertvalue %"struct.cort::uint4" %footprintValRes0, i32 %footprintVal1, 1
  %footprintValRes2 = insertvalue %"struct.cort::uint4" %footprintValRes1, i32 %footprintVal2, 2
  %r = insertvalue %"struct.cort::uint4" %footprintValRes2, i32 %footprintVal3, 3
  %coversSingleMipLevelResult = extractvalue {i32, i32, i32, i32, i1} %s, 4
  %coversSingleMipLevelResultExtended = zext i1 %coversSingleMipLevelResult to i32
  store i32 %coversSingleMipLevelResultExtended, i32* %coversSingleMipLevel
  ret %"struct.cort::uint4" %r
}

declare <4 x float> @optix.ptx.nonsparse.tex.grad.3d.v4.f32.f32(i64 %p0, <4 x float> %p1, <4 x float> %p2, <4 x float> %p3) #1
define weak %"struct.cort::float4" @_ZN4cort32Texture_getElement_hw_texgrad_3dEyfffffffff(i64 %texref, float %x, float %y, float %z, float %dPdx_x, float %dPdx_y, float %dPdx_z, float %dPdy_x, float %dPdy_y, float %dPdy_z ) #0
{
  %coords0 = insertelement < 4 x float > undef, float %x, i32 0
  %coords1 = insertelement < 4 x float > %coords0, float %y, i32 1
  %coords2 = insertelement < 4 x float > %coords1, float %z, i32 2

  %dPdx0 = insertelement < 4 x float > undef, float %dPdx_x, i32 0
  %dPdx1 = insertelement < 4 x float > %dPdx0, float %dPdx_y, i32 1
  %dPdx2 = insertelement < 4 x float > %dPdx1, float %dPdx_z, i32 2

  %dPdy0 = insertelement < 4 x float > undef, float %dPdy_x, i32 0
  %dPdy1 = insertelement < 4 x float > %dPdy0, float %dPdy_y, i32 1
  %dPdy2 = insertelement < 4 x float > %dPdy1, float %dPdy_z, i32 2

  %res = call < 4 x float > @optix.ptx.nonsparse.tex.grad.3d.v4.f32.f32( i64 %texref, < 4 x float > %coords2, < 4 x float > %dPdx2, < 4 x float > %dPdy2 ) #1
  %final = call %"struct.cort::float4" @colwert.v4.f32.to.float4( < 4 x float > %res ) #0
  ret %"struct.cort::float4" %final
}

declare < 4 x float > @optix.ptx.nonsparse.tex.grad.a1d.v4.f32.f32( i64 %p0, { i32, float } %p1, < 1 x float > %p2, < 1 x float > %p3 ) #1
define weak %"struct.cort::float4" @_ZN4cort33Texture_getElement_hw_texgrad_a1dEyjfff(i64 %texref, i32 %a, float %x, float %dPdx, float %dPdy ) #0
{
  %params0 = insertvalue { i32, float } undef, i32 %a, 0
  %params1 = insertvalue { i32, float } %params0, float %x, 1

  %dPdx_vec = insertelement < 1 x float > undef, float %dPdx, i32 0
  %dPdy_vec = insertelement < 1 x float > undef, float %dPdy, i32 0

  %res = call < 4 x float > @optix.ptx.nonsparse.tex.grad.a1d.v4.f32.f32( i64 %texref, { i32, float } %params1, < 1 x float > %dPdx_vec, < 1 x float > %dPdy_vec ) #1
  %final = call %"struct.cort::float4" @colwert.v4.f32.to.float4( < 4 x float > %res ) #0
  ret %"struct.cort::float4" %final
}

declare { float, float, float, float } @optix.lwvm.texgrad_a2d(i64 %texref, i32, float, float, float, float, float, float) #0
define weak %"struct.cort::float4" @_ZN4cort33Texture_getElement_hw_texgrad_a2dEyjffffff(i64 %texref, i32 %a, float %x, float %y, float %dPdx_x, float %dPdx_y, float %dPdy_x, float %dPdy_y ) #0
{
  %s = call { float, float, float, float } @optix.lwvm.texgrad_a2d( i64 %texref, i32 %a, float %x, float %y, float %dPdx_x, float %dPdx_y, float %dPdy_x, float %dPdy_y ) #0
  %r = call %"struct.cort::float4" @colwert.to.float4( { float, float, float, float } %s ) #0
  ret %"struct.cort::float4" %r
}

declare { float, float, float, float } @optix.lwvm.texgrad_lwbe(i64 %texref, float, float, float, float, float, float, float) #0
define weak %"struct.cort::float4" @_ZN4cort32Texture_getElement_hw_texgrad_lwbeEyfffffff(i64 %texref, float %x, float %y, float %z, float %dPdx_x, float %dPdx_y, float %dPdy_x, float %dPdy_y ) #0
{
  %s = call { float, float, float, float } @optix.lwvm.texgrad_lwbe( i64 %texref, float %x, float %y, float %z, float %dPdx_x, float %dPdx_y, float %dPdy_x, float %dPdy_y ) #0
  %r = call %"struct.cort::float4" @colwert.to.float4( { float, float, float, float } %s ) #0
  ret %"struct.cort::float4" %r
}

declare { float, float, float, float } @optix.lwvm.texgrad_alwbe(i64 %texref, i32, float, float, float, float, float, float, float ) #0
define weak %"struct.cort::float4" @_ZN4cort32Texture_getElement_hw_texgrad_alwbeEyjfffffff(i64 %texref, i32 %a, float %x, float %y, float %z, float %dPdx_x, float %dPdx_y, float %dPdy_x, float %dPdy_y ) #0
{
  %s = call { float, float, float, float } @optix.lwvm.texgrad_alwbe( i64 %texref, i32 %a, float %x, float %y, float %z, float %dPdx_x, float %dPdx_y, float %dPdy_x, float %dPdy_y ) #0
  %r = call %"struct.cort::float4" @colwert.to.float4( { float, float, float, float } %s ) #0
  ret %"struct.cort::float4" %r
}

declare { < 4 x float >, i1 } @optix.ptx.tld4.2d.r.v4.f32.f32( i64 %p0, < 2 x float > %p1 ) #1
define weak %"struct.cort::float4" @_ZN4cort30Texture_getElement_hw_tld4r_2dEyff(i64 %texref, float %x, float %y) #0
{
  %params0 = insertelement < 2 x float > undef, float %x, i32 0
  %params1 = insertelement < 2 x float > %params0, float %y, i32 1

  %res = call { < 4 x float >, i1 } @optix.ptx.tld4.2d.r.v4.f32.f32( i64 %texref, < 2 x float > %params1 ) #1
  %texRes = extractvalue { < 4 x float >, i1 } %res, 0
  %final = call %"struct.cort::float4" @colwert.v4.f32.to.float4( < 4 x float > %texRes ) #0
  ret %"struct.cort::float4" %final
}

declare { < 4 x float >, i1 } @optix.ptx.tld4.2d.g.v4.f32.f32( i64 %p0, < 2 x float > %p1 ) #1
define weak %"struct.cort::float4" @_ZN4cort30Texture_getElement_hw_tld4g_2dEyff(i64 %texref, float %x, float %y) #0
{
  %params0 = insertelement < 2 x float > undef, float %x, i32 0
  %params1 = insertelement < 2 x float > %params0, float %y, i32 1

  %res = call { < 4 x float >, i1 } @optix.ptx.tld4.2d.g.v4.f32.f32( i64 %texref, < 2 x float > %params1 ) #1
  %texRes = extractvalue { < 4 x float >, i1 } %res, 0
  %final = call %"struct.cort::float4" @colwert.v4.f32.to.float4( < 4 x float > %texRes ) #0
  ret %"struct.cort::float4" %final
}

declare { < 4 x float >, i1 } @optix.ptx.tld4.2d.b.v4.f32.f32( i64 %p0, < 2 x float > %p1 ) #1
define weak %"struct.cort::float4" @_ZN4cort30Texture_getElement_hw_tld4b_2dEyff(i64 %texref, float %x, float %y) #0
{
  %params0 = insertelement < 2 x float > undef, float %x, i32 0
  %params1 = insertelement < 2 x float > %params0, float %y, i32 1

  %res = call { < 4 x float >, i1 } @optix.ptx.tld4.2d.b.v4.f32.f32( i64 %texref, < 2 x float > %params1 ) #1
  %texRes = extractvalue { < 4 x float >, i1 } %res, 0
  %final = call %"struct.cort::float4" @colwert.v4.f32.to.float4( < 4 x float > %texRes ) #0
  ret %"struct.cort::float4" %final
}

declare { < 4 x float >, i1 } @optix.ptx.tld4.2d.a.v4.f32.f32( i64 %p0, < 2 x float > %p1 ) #1
define weak %"struct.cort::float4" @_ZN4cort30Texture_getElement_hw_tld4a_2dEyff(i64 %texref, float %x, float %y) #0
{
  %params0 = insertelement < 2 x float > undef, float %x, i32 0
  %params1 = insertelement < 2 x float > %params0, float %y, i32 1

  %res = call { < 4 x float >, i1 } @optix.ptx.tld4.2d.a.v4.f32.f32( i64 %texref, < 2 x float > %params1 ) #1
  %texRes = extractvalue { < 4 x float >, i1 } %res, 0
  %final = call %"struct.cort::float4" @colwert.v4.f32.to.float4( < 4 x float > %texRes ) #0
  ret %"struct.cort::float4" %final
}

declare { float, float, float, float } @optix.lwvm.tld4r_a2d(i64 %texref, i32, float, float) #0
define weak %"struct.cort::float4" @_ZN4cort31Texture_getElement_hw_tld4r_a2dEyjff(i64 %texref, i32 %a, float %x, float %y) #0
{
  %s = call { float, float, float, float } @optix.lwvm.tld4r_a2d( i64 %texref, i32 %a, float %x, float %y ) #0
  %r = call %"struct.cort::float4" @colwert.to.float4( { float, float, float, float } %s ) #0
  ret %"struct.cort::float4" %r
}

declare { float, float, float, float } @optix.lwvm.tld4g_a2d(i64 %texref, i32, float, float) #0
define weak %"struct.cort::float4" @_ZN4cort31Texture_getElement_hw_tld4g_a2dEyjff(i64 %texref, i32 %a, float %x, float %y) #0
{
  %s = call { float, float, float, float } @optix.lwvm.tld4g_a2d( i64 %texref, i32 %a, float %x, float %y ) #0
  %r = call %"struct.cort::float4" @colwert.to.float4( { float, float, float, float } %s ) #0
  ret %"struct.cort::float4" %r
}

declare { float, float, float, float } @optix.lwvm.tld4b_a2d(i64 %texref, i32, float, float) #0
define weak %"struct.cort::float4" @_ZN4cort31Texture_getElement_hw_tld4b_a2dEyjff(i64 %texref, i32 %a, float %x, float %y) #0
{
  %s = call { float, float, float, float } @optix.lwvm.tld4b_a2d( i64 %texref, i32 %a, float %x, float %y ) #0
  %r = call %"struct.cort::float4" @colwert.to.float4( { float, float, float, float } %s ) #0
  ret %"struct.cort::float4" %r
}

declare { float, float, float, float } @optix.lwvm.tld4a_a2d(i64 %texref, i32, float, float) #0
define weak %"struct.cort::float4" @_ZN4cort31Texture_getElement_hw_tld4a_a2dEyjff(i64 %texref, i32 %a, float %x, float %y) #0
{
  %s = call { float, float, float, float } @optix.lwvm.tld4a_a2d( i64 %texref, i32 %a, float %x, float %y ) #0
  %r = call %"struct.cort::float4" @colwert.to.float4( { float, float, float, float } %s ) #0
  ret %"struct.cort::float4" %r
}

declare { float, float, float, float } @optix.lwvm.tld4r_lwbe(i64 %texref, float, float, float) #0
define weak %"struct.cort::float4" @_ZN4cort32Texture_getElement_hw_tld4r_lwbeEyfff(i64 %texref, float %x, float %y, float %z) #0
{
  %s = call { float, float, float, float } @optix.lwvm.tld4r_lwbe( i64 %texref, float %x, float %y, float %z ) #0
  %r = call %"struct.cort::float4" @colwert.to.float4( { float, float, float, float } %s ) #0
  ret %"struct.cort::float4" %r
}

declare { float, float, float, float } @optix.lwvm.tld4g_lwbe(i64 %texref, float, float, float) #0
define weak %"struct.cort::float4" @_ZN4cort32Texture_getElement_hw_tld4g_lwbeEyfff(i64 %texref, float %x, float %y, float %z) #0
{
  %s = call { float, float, float, float } @optix.lwvm.tld4g_lwbe( i64 %texref, float %x, float %y, float %z ) #0
  %r = call %"struct.cort::float4" @colwert.to.float4( { float, float, float, float } %s ) #0
  ret %"struct.cort::float4" %r
}

declare { float, float, float, float } @optix.lwvm.tld4b_lwbe(i64 %texref, float, float, float) #0
define weak %"struct.cort::float4" @_ZN4cort32Texture_getElement_hw_tld4b_lwbeEyfff(i64 %texref, float %x, float %y, float %z) #0
{
  %s = call { float, float, float, float } @optix.lwvm.tld4b_lwbe( i64 %texref, float %x, float %y, float %z ) #0
  %r = call %"struct.cort::float4" @colwert.to.float4( { float, float, float, float } %s ) #0
  ret %"struct.cort::float4" %r
}

declare { float, float, float, float } @optix.lwvm.tld4a_lwbe(i64 %texref, float, float, float) #0
define weak %"struct.cort::float4" @_ZN4cort32Texture_getElement_hw_tld4a_lwbeEyfff(i64 %texref, float %x, float %y, float %z) #0
{
  %s = call { float, float, float, float } @optix.lwvm.tld4a_lwbe( i64 %texref, float %x, float %y, float %z ) #0
  %r = call %"struct.cort::float4" @colwert.to.float4( { float, float, float, float } %s ) #0
  ret %"struct.cort::float4" %r
}

declare { float, float, float, float } @optix.lwvm.tld4r_alwbe(i64 %texref, i32, float, float, float) #0
define weak %"struct.cort::float4" @_ZN4cort33Texture_getElement_hw_tld4r_alwbeEyjfff(i64 %texref, i32 %a, float %x, float %y, float %z) #0
{
  %s = call { float, float, float, float } @optix.lwvm.tld4r_alwbe( i64 %texref, i32 %a, float %x, float %y, float %z) #0
  %r = call %"struct.cort::float4" @colwert.to.float4( { float, float, float, float } %s ) #0
  ret %"struct.cort::float4" %r
}

declare { float, float, float, float } @optix.lwvm.tld4g_alwbe(i64 %texref, i32, float, float, float) #0
define weak %"struct.cort::float4" @_ZN4cort33Texture_getElement_hw_tld4g_alwbeEyjfff(i64 %texref, i32 %a, float %x, float %y, float %z) #0
{
  %s = call { float, float, float, float } @optix.lwvm.tld4g_alwbe( i64 %texref, i32 %a, float %x, float %y, float %z ) #0
  %r = call %"struct.cort::float4" @colwert.to.float4( { float, float, float, float } %s ) #0
  ret %"struct.cort::float4" %r
}

declare { float, float, float, float } @optix.lwvm.tld4b_alwbe(i64 %texref, i32, float, float, float) #0
define weak %"struct.cort::float4" @_ZN4cort33Texture_getElement_hw_tld4b_alwbeEyjfff(i64 %texref, i32 %a, float %x, float %y, float %z) #0
{
  %s = call { float, float, float, float } @optix.lwvm.tld4b_alwbe( i64 %texref, i32 %a, float %x, float %y, float %z ) #0
  %r = call %"struct.cort::float4" @colwert.to.float4( { float, float, float, float } %s ) #0
  ret %"struct.cort::float4" %r
}

declare { float, float, float, float } @optix.lwvm.tld4a_alwbe(i64 %texref, i32, float, float, float) #0
define weak %"struct.cort::float4" @_ZN4cort33Texture_getElement_hw_tld4a_alwbeEyjfff(i64 %texref, i32 %a, float %x, float %y, float %z) #0
{
  %s = call { float, float, float, float } @optix.lwvm.tld4a_alwbe( i64 %texref, i32 %a, float %x, float %y, float %z ) #0
  %r = call %"struct.cort::float4" @colwert.to.float4( { float, float, float, float } %s ) #0
  ret %"struct.cort::float4" %r
}

;
; texref only function
;
declare i64 @optix.cort.texref2handle( i32 %texref ) #0
define weak %"struct.cort::float4" @_ZN4cort39Texture_getElement_hwtexref_texfetch_1dEji(i32 %texref, i32 %x) #0
{
  %hdl = call i64 @optix.cort.texref2handle( i32 %texref ) #0
  %s = call { float, float, float, float } @optix.lwvm.texfetch_1d( i64 %hdl, i32 %x ) #0
  %r = call %"struct.cort::float4" @colwert.to.float4( { float, float, float, float } %s ) #0
  ret %"struct.cort::float4" %r
}


