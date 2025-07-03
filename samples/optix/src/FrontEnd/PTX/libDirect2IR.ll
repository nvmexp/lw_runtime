;; Copyright (c) 2021, LWPU CORPORATION.
;; TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED
;; *AS IS* AND LWPU AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS
;; OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY
;; AND FITNESS FOR A PARTICULAR PURPOSE.  IN NO EVENT SHALL LWPU OR ITS SUPPLIERS
;; BE LIABLE FOR ANY SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES
;; WHATSOEVER (INCLUDING, WITHOUT LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS,
;; BUSINESS INTERRUPTION, LOSS OF BUSINESS INFORMATION, OR ANY OTHER PELWNIARY LOSS)
;; ARISING OUT OF THE USE OF OR INABILITY TO USE THIS SOFTWARE, EVEN IF LWPU HAS
;; BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES

;-------------------------------------------------------------------------------
; This file contains wrappers to colwert OptiX PTX intrinsics to LLVM IR
; functions. The intrinsics these wrappers use are supported in Direct2IR, but
; not LWPTX, so we only link them in when we're using the D2IR backend.
;-------------------------------------------------------------------------------

attributes #0 = { nounwind readnone alwaysinline } ; Non-volatile functions
attributes #1 = { nounwind alwaysinline }          ; "volatile" functions, which may return different results on different ilwocations
attributes #2 = { nounwind readnone }
attributes #4 = { nounwind }                       ; for texture intrinsics - they MUST not have readnone

;-------------------------------------------------------------------------------
; Mad


declare i32 @llvm.lwvm.mad.hi.sat.s32(i32 %a, i32 %b, i32 %c)

define linkonce_odr i32 @optix.ptx.mad.hi.sat.s32( i32 %a, i32 %b, i32 %c ) #0 {
  %ret = call i32 @llvm.lwvm.mad.hi.sat.s32(i32 %a, i32 %b, i32 %c)
  ret i32 %ret
}

;-------------------------------------------------------------------------------
; Exit

declare void @llvm.lwvm.exit() noreturn

define linkonce void @optix.ptx.exit() nounwind alwaysinline noreturn {
    call void @llvm.lwvm.exit() 
    ret void
}

;-------------------------------------------------------------------------------
; Trap

declare void @llvm.trap() noreturn nounwind

define linkonce_odr void @optix.ptx.trap() nounwind alwaysinline noreturn {
    call void @llvm.trap()
    ret void
}

;-------------------------------------------------------------------------------
; pmevent

; NOTE: This intrinsic always takes its argument as a mask (as opposed to the
;       PTX instruction, which can take an index or a mask)
declare void @llvm.lwvm.pm.trigger(i32 %mask)

define linkonce_odr void @optix.ptx.pmevent(i32 %which) #1 {
  start:
    switch i32 %which, label %default [
      i32  0, label %pm0
      i32  1, label %pm1
      i32  2, label %pm2
      i32  3, label %pm3
      i32  4, label %pm4
      i32  5, label %pm5
      i32  6, label %pm6
      i32  7, label %pm7
      i32  8, label %pm8
      i32  9, label %pm9
      i32 10, label %pm10
      i32 11, label %pm11
      i32 12, label %pm12
      i32 13, label %pm13
      i32 14, label %pm14
      i32 15, label %pm15
      ]
    ret void
  default:
    call void @llvm.trap()
    ret void
  pm0:
    call void @llvm.lwvm.pm.trigger(i32 1)
    ret void
  pm1:
    call void @llvm.lwvm.pm.trigger(i32 2)
    ret void
  pm2:
    call void @llvm.lwvm.pm.trigger(i32 4)
    ret void
  pm3:
    call void @llvm.lwvm.pm.trigger(i32 8)
    ret void
  pm4:
    call void @llvm.lwvm.pm.trigger(i32 16)
    ret void
  pm5:
    call void @llvm.lwvm.pm.trigger(i32 32)
    ret void
  pm6:
    call void @llvm.lwvm.pm.trigger(i32 64)
    ret void
  pm7:
    call void @llvm.lwvm.pm.trigger(i32 128)
    ret void
  pm8:
    call void @llvm.lwvm.pm.trigger(i32 256)
    ret void
  pm9:
    call void @llvm.lwvm.pm.trigger(i32 512)
    ret void
  pm10:
    call void @llvm.lwvm.pm.trigger(i32 1024)
    ret void
  pm11:
    call void @llvm.lwvm.pm.trigger(i32 2048)
    ret void
  pm12:
    call void @llvm.lwvm.pm.trigger(i32 4096)
    ret void
  pm13:
    call void @llvm.lwvm.pm.trigger(i32 8192)
    ret void
  pm14:
    call void @llvm.lwvm.pm.trigger(i32 16384)
    ret void
  pm15:
    call void @llvm.lwvm.pm.trigger(i32 32768)
    ret void
}

define linkonce_odr void @optix.ptx.pmevent.mask(i32) #1 {
    call void @llvm.lwvm.pm.trigger(i32 %0)
    ret void
}

;-------------------------------------------------------------------------------
; Floating point negation

; NOTE: We subtract from -0 to make sure we change the sign bit if we're
; negating 0 or negative 0 (for IEEE 754 compatibility)

declare half @llvm.lwvm.sub.ftz.f16(i32 %flags, half %x, half %y) #0
define linkonce half @optix.ptx.neg.ftz.f16( half %p0 ) #0 {
    ; Mode: 0x1=1 [Round]
    %r0 = call half @llvm.lwvm.sub.ftz.f16( i32 1, half -0.000000e+00, half %p0 ) #0
    ret half %r0
}

declare half @llvm.lwvm.sub.f16(i32 %flags, half %x, half %y) #0
define linkonce half @optix.ptx.neg.f16( half %p0 ) #0 {
    ; Mode: 0x1=1 [Round]
    %r0 = call half @llvm.lwvm.sub.f16( i32 1, half -0.000000e+00, half %p0 ) #0
    ret half %r0
}

declare <2 x half> @llvm.lwvm.sub.ftz.v2f16(i32 %flags, <2 x half> %x, <2 x half> %y) #0
define linkonce <2 x half> @optix.ptx.neg.ftz.f16x2( <2 x half> %p0 ) #0 {
    %zeroes_1 = insertelement <2 x half> undef, half -0.000000e+00, i32 0
    %zeroes_2 = insertelement <2 x half> %zeroes_1, half -0.000000e+00, i32 1
    ; Mode: 0x1=1 [Round]
    %r0 = call <2 x half> @llvm.lwvm.sub.ftz.v2f16( i32 1, <2 x half> %zeroes_2, <2 x half> %p0 ) #0
    ret <2 x half> %r0
}

declare <2 x half> @llvm.lwvm.sub.v2f16(i32 %flags, <2 x half> %x, <2 x half> %y) #0
define linkonce <2 x half> @optix.ptx.neg.f16x2( <2 x half> %p0 ) #0 {
    %zeroes_1 = insertelement <2 x half> undef, half -0.000000e+00, i32 0
    %zeroes_2 = insertelement <2 x half> %zeroes_1, half -0.000000e+00, i32 1
    ; Mode: 0x1=1 [Round]
    %r0 = call <2 x half> @llvm.lwvm.sub.v2f16( i32 1, <2 x half> %zeroes_2, <2 x half> %p0 ) #0
    ret <2 x half> %r0
}

declare float @llvm.lwvm.sub.ftz.f32(i32 %flags, float %x, float %y) #0
define linkonce float @optix.ptx.neg.ftz.f32( float %p0 ) #0 {
    ; Mode: 0x1=1 [Round]
    %r0 = call float @llvm.lwvm.sub.ftz.f32( i32 1, float -0.000000e+00, float %p0 ) #0
    ret float %r0
}

declare float @llvm.lwvm.sub.f32(i32 %flags, float %x, float %y) #0
define linkonce float @optix.ptx.neg.f32( float %p0 ) #0 {
    ; Mode: 0x1=1 [Round]
    %r0 = call float @llvm.lwvm.sub.f32( i32 1, float -0.000000e+00, float %p0 ) #0
    ret float %r0
}

declare double @llvm.lwvm.sub.f64(i32 %flags, double %x, double %y) #0
define linkonce double @optix.ptx.neg.f64( double %p0 ) #0 {
    ; Mode: 0x1=1 [Round]
    %r0 = call double @llvm.lwvm.sub.f64( i32 1, double -0.000000e+00, double %p0 ) #0
    ret double %r0
}

;-------------------------------------------------------------------------------
; Texture query intrinsics

declare i32 @llvm.lwvm.tex.query( i32 %mode, i64 %texref, i32 %idx ) #4
define linkonce_odr i32 @optix.lwvm.txq_width(i64 %texref) #0 {
  ; Mode: 5 = 3D, width, no LOD
  %w = call i32 @llvm.lwvm.tex.query( i32 5, i64 %texref, i32 undef ) #4
  ret i32 %w
}

define linkonce_odr i32 @optix.lwvm.txq_height(i64 %texref) #0 {
  ; Mode: 22 = 3D, height, no LOD
  %h = call i32 @llvm.lwvm.tex.query( i32 22, i64 %texref, i32 undef ) #4
  ret i32 %h
}

define linkonce_odr i32 @optix.lwvm.txq_depth(i64 %texref) #0 {
  ; Mode: 37 = 3D, depth, no LOD
  %d = call i32 @llvm.lwvm.tex.query( i32 37, i64 %texref, i32 undef ) #4
  ret i32 %d
}

;-------------------------------------------------------------------------------
; activemask

declare {i32, i1} @llvm.lwvm.vote( i32 %mode, i1 %pred )

define linkonce i32 @optix.ptx.activemask.b32(  ) #1 {
    ; Emulate activemask by calling vote and returning all active
    ; threads.
    ; Mode argument doesn't matter, since we won't be looking at the
    ; resulting boolean value.
    ; We pass 1 as the predicate, so all active threads will set their
    ; position in the resulting mask to 1.
    %res = call {i32, i1} @llvm.lwvm.vote( i32 1, i1 1 ) #4
    %r0 = extractvalue {i32, i1} %res, 0
    ret i32 %r0
}

;-------------------------------------------------------------------------------
; Voting

define linkonce i1 @optix.ptx.vote.all.pred( i1 %p0 ) #1 {
    %res = call {i32, i1} @llvm.lwvm.vote( i32 0, i1 %p0 ) #4
    %r0 = extractvalue {i32, i1} %res, 1
    ret i1 %r0
}

define linkonce i1 @optix.ptx.vote.any.pred( i1 %p0 ) #1 {
    %res = call {i32, i1} @llvm.lwvm.vote( i32 1, i1 %p0 ) #4
    %r0 = extractvalue {i32, i1} %res, 1
    ret i1 %r0
}

define linkonce i1 @optix.ptx.vote.uni.pred( i1 %p0 ) #1 {
    %res = call {i32, i1} @llvm.lwvm.vote( i32 2, i1 %p0 ) #4
    %r0 = extractvalue {i32, i1} %res, 1
    ret i1 %r0
}

define linkonce i32 @optix.ptx.vote.ballot.b32( i1 %p0 ) #1 {
    ; The Mode argument doesn't matter here.
    %res = call {i32, i1} @llvm.lwvm.vote( i32 1, i1 %p0 ) #4
    %r0 = extractvalue {i32, i1} %res, 0
    ret i32 %r0
}

;-------------------------------------------------------------------------------
; Synchronized voting

declare {i32, i1} @llvm.lwvm.vote.sync(i32 %membermask, i32 %mode, i1 %predicate)

define linkonce i32 @optix.ptx.vote.sync.ballot.b32( i1 %p0, i32 %am ) #1 {
    ; The Mode argument doesn't matter here.
    %res = call {i32, i1} @llvm.lwvm.vote.sync(i32 %am, i32 1, i1 %p0 ) #4
    %r0 = extractvalue {i32, i1} %res, 0
    ret i32 %r0
}

define linkonce i1 @optix.ptx.vote.sync.all.pred( i1 %p0, i32 %am ) #1 {
    %res = call {i32, i1} @llvm.lwvm.vote.sync(i32 %am, i32 0, i1 %p0 ) #4
    %r0 = extractvalue {i32, i1} %res, 1
    ret i1 %r0
}
  
define linkonce i1 @optix.ptx.vote.sync.any.pred( i1 %p0, i32 %am ) #1 {
    %res = call {i32, i1} @llvm.lwvm.vote.sync(i32 %am, i32 1, i1 %p0 ) #4
    %r0 = extractvalue {i32, i1} %res, 1
    ret i1 %r0
}

define linkonce i1 @optix.ptx.vote.sync.uni.pred( i1 %p0, i32 %am ) #1 {
    %res = call {i32, i1} @llvm.lwvm.vote.sync(i32 %am, i32 2, i1 %p0 ) #4
    %r0 = extractvalue {i32, i1} %res, 1
    ret i1 %r0
}

;-------------------------------------------------------------------------------
; Match

declare i32 @llvm.lwvm.match.any.sync.i32( i32 %membermask, i32 %value )
define linkonce i32 @optix.ptx.match.sync.any.b32( i32 %value, i32 %membermask ) #1 {
    %r0 = call i32 @llvm.lwvm.match.any.sync.i32( i32 %membermask, i32 %value ) #4
    ret i32 %r0
}

declare i32 @llvm.lwvm.match.any.sync.i64( i32 %membermask, i64 %value )
define linkonce i32 @optix.ptx.match.sync.any.b64( i64 %value,  i32 %membermask ) #1 {
    %r0 = call i32 @llvm.lwvm.match.any.sync.i64( i32 %membermask,  i64 %value ) #4
    ret i32 %r0
}

declare {i32, i1} @llvm.lwvm.match.all.sync.i32( i32 %membermask, i32 %value )
define linkonce { i32, i1 } @optix.ptx.match.sync.all.b32( i32 %value, i32 %membermask ) #1 {
    %r0 = call {i32, i1} @llvm.lwvm.match.all.sync.i32( i32 %membermask,  i32 %value ) #4
    ret { i32, i1 } %r0
}

declare {i32, i1} @llvm.lwvm.match.all.sync.i64( i32 %membermask, i64 %value )
define linkonce { i32, i1 } @optix.ptx.match.sync.all.b64( i64 %value, i32 %membermask ) #1 {
    %r0 = call {i32, i1} @llvm.lwvm.match.all.sync.i64( i32 %membermask,  i64 %value ) #4
    ret { i32, i1 } %r0
}

;-------------------------------------------------------------------------------
; shfl

declare { i32, i1 } @llvm.lwvm.shfl.sync.i32(i32 %membermask, i32 %mode, i32 %a, i32 %b, i32 %c)

define linkonce { i32, i1 } @optix.ptx.shfl.sync.idx.b32( i32 %a, i32 %b, i32 %c, i32 %membermask ) #1 {
    ; Mode 0 = IDX
    %r0 = call { i32, i1 } @llvm.lwvm.shfl.sync.i32( i32 %membermask, i32 0, i32 %a, i32 %b, i32 %c ) #4
    ret { i32, i1 } %r0
}

define linkonce { i32, i1 } @optix.ptx.shfl.sync.up.b32( i32 %a, i32 %b, i32 %c, i32 %membermask ) #1 {
    ; Mode 1 = UP
    %r0 = call { i32, i1 } @llvm.lwvm.shfl.sync.i32( i32 %membermask, i32 1, i32 %a, i32 %b, i32 %c ) #4
    ret { i32, i1 } %r0
}

define linkonce { i32, i1 } @optix.ptx.shfl.sync.down.b32( i32 %a, i32 %b, i32 %c, i32 %membermask ) #1 {
    ; Mode 2 = DOWN
    %r0 = call { i32, i1 } @llvm.lwvm.shfl.sync.i32( i32 %membermask, i32 2, i32 %a, i32 %b, i32 %c ) #4
    ret { i32, i1 } %r0
}

define linkonce { i32, i1 } @optix.ptx.shfl.sync.bfly.b32( i32 %a, i32 %b, i32 %c, i32 %membermask ) #1 {
    ; Mode 3 = BFLY
    %r0 = call { i32, i1 } @llvm.lwvm.shfl.sync.i32( i32 %membermask, i32 3, i32 %a, i32 %b, i32 %c ) #4
    ret { i32, i1 } %r0
}

;-------------------------------------------------------------------------------
; shfl (emulate non-sync versions)

define linkonce { i32, i1 } @optix.ptx.shfl.idx.b32( i32 %a, i32 %b, i32 %c ) #1 {
    %membermask = call i32 @optix.ptx.activemask.b32() #0
    ; Mode 0 = IDX
    %r0 = call { i32, i1 } @llvm.lwvm.shfl.sync.i32( i32 %membermask, i32 0, i32 %a, i32 %b, i32 %c ) #4
    ret { i32, i1 } %r0
}

define linkonce { i32, i1 } @optix.ptx.shfl.up.b32( i32 %a, i32 %b, i32 %c ) #1 {
    %membermask = call i32 @optix.ptx.activemask.b32() #0
    ; Mode 1 = UP
    %r0 = call { i32, i1 } @llvm.lwvm.shfl.sync.i32( i32 %membermask, i32 1, i32 %a, i32 %b, i32 %c ) #4
    ret { i32, i1 } %r0
}

define linkonce { i32, i1 } @optix.ptx.shfl.down.b32( i32 %a, i32 %b, i32 %c ) #1 {
    %membermask = call i32 @optix.ptx.activemask.b32() #0
    ; Mode 2 = DOWN
    %r0 = call { i32, i1 } @llvm.lwvm.shfl.sync.i32( i32 %membermask, i32 2, i32 %a, i32 %b, i32 %c ) #4
    ret { i32, i1 } %r0
}

define linkonce { i32, i1 } @optix.ptx.shfl.bfly.b32( i32 %a, i32 %b, i32 %c ) #1 {
    %membermask = call i32 @optix.ptx.activemask.b32() #0
    ; Mode 3 = BFLY
    %r0 = call { i32, i1 } @llvm.lwvm.shfl.sync.i32( i32 %membermask, i32 3, i32 %a, i32 %b, i32 %c ) #4
    ret { i32, i1 } %r0
}

;-------------------------------------------------------------------------------
; Special register access

declare i32 @llvm.lwvm.read.sreg.i32(i32 %num) #1
declare i64 @llvm.lwvm.read.sreg.i64(i32 %num) #1

define linkonce_odr i32 @optix.lwvm.read.ptx.sreg.clock() #1 {
    ; 82 = SR_CLOCK_LO
    %r = call i32 @llvm.lwvm.read.sreg.i32(i32 82) #1
    ret i32 %r
}

define linkonce_odr i32 @optix.lwvm.read.ptx.sreg.clock_hi() #1 {
    ; 83 = SR_CLOCK_HI
    %r = call i32 @llvm.lwvm.read.sreg.i32(i32 83) #1
    ret i32 %r
}

define linkonce_odr i64 @optix.lwvm.read.ptx.sreg.clock64() #1 {
    ; 82 = SR_CLOCK_LO, 83 = SR_CLOCK_HI
    %lo = call i32 @llvm.lwvm.read.sreg.i32(i32 82) #1
    %hi = call i32 @llvm.lwvm.read.sreg.i32(i32 83) #1
    %v0 = insertelement <2 x i32> undef, i32 %lo, i32 1
    %v1 = insertelement <2 x i32> %v0, i32 %hi, i32 0
    %r = bitcast <2 x i32> %v1 to i64
    ret i64 %r
}

define linkonce_odr i64 @optix.lwvm.read.ptx.sreg.globaltimer() #1 {
    ; 84 = SR_GLOBAL_TIMER_LO, 85 = SR_GLOBAL_TIMER_HI
    %lo = call i32 @llvm.lwvm.read.sreg.i32(i32 84) #1
    %hi = call i32 @llvm.lwvm.read.sreg.i32(i32 85) #1
    %v0 = insertelement <2 x i32> undef, i32 %lo, i32 1
    %v1 = insertelement <2 x i32> %v0, i32 %hi, i32 0
    %r = bitcast <2 x i32> %v1 to i64
    ret i64 %r
}

define linkonce_odr i32 @optix.lwvm.read.ptx.sreg.globaltimer_lo() #1 {
    ; 84 = SR_GLOBAL_TIMER_LO
    %r = call i32 @llvm.lwvm.read.sreg.i32(i32 84) #1
    ret i32 %r
}

define linkonce_odr i32 @optix.lwvm.read.ptx.sreg.globaltimer_hi() #1 {
    ; 85 = SR_GLOBAL_TIMER_HI
    %r = call i32 @llvm.lwvm.read.sreg.i32(i32 85) #1
    ret i32 %r
}

define linkonce_odr i32 @optix.lwvm.read.ptx.sreg.laneid() #1 {
    ; 12 = SR_THREADINWARP (synonymous with laneid)
    %r = call i32 @llvm.lwvm.read.sreg.i32(i32 12) #0
    ret i32 %r
}

; warpid is volatile
define linkonce_odr i32 @optix.lwvm.read.ptx.sreg.warpid() #1 {
    ; 14 = SR_WARPID
    %r = call i32 @llvm.lwvm.read.sreg.i32(i32 14) #1
    ret i32 %r
}

define linkonce_odr i32 @optix.lwvm.read.ptx.sreg.tid.x() #0 {
    ; 38 = SR_TID_X
    %r = call i32 @llvm.lwvm.read.sreg.i32(i32 38) #0
    ret i32 %r
}

define linkonce_odr i32 @optix.lwvm.read.ptx.sreg.tid.y() #0 {
    ; 39 = SR_TID_Y
    %r = call i32 @llvm.lwvm.read.sreg.i32(i32 39) #0
    ret i32 %r
}

define linkonce_odr i32 @optix.lwvm.read.ptx.sreg.tid.z() #0 {
    ; 40 = SR_TID_Z
    %r = call i32 @llvm.lwvm.read.sreg.i32(i32 40) #0
    ret i32 %r
}

define linkonce_odr <4 x i32> @optix.lwvm.read.ptx.sreg.tid() #0 {
    %x = call i32 @llvm.lwvm.read.sreg.i32(i32 38) #0
    %y = call i32 @llvm.lwvm.read.sreg.i32(i32 39) #0
    %z = call i32 @llvm.lwvm.read.sreg.i32(i32 40) #0
    %v0 = insertelement <4 x i32> undef, i32 %x, i32 0
    %v1 = insertelement <4 x i32> %v0, i32 %y, i32 1
    %v2 = insertelement <4 x i32> %v1, i32 %z, i32 2
    ret <4 x i32> %v2
}

define linkonce_odr i32 @optix.lwvm.read.ptx.sreg.ntid.x() #0 {
    ; 41 = SR_NTID_X
    %r = call i32 @llvm.lwvm.read.sreg.i32(i32 41) #0
    ret i32 %r
}

define linkonce_odr i32 @optix.lwvm.read.ptx.sreg.ntid.y() #0 {
    ; 42 = SR_NTID_Y
    %r = call i32 @llvm.lwvm.read.sreg.i32(i32 42) #0
    ret i32 %r
}

define linkonce_odr i32 @optix.lwvm.read.ptx.sreg.ntid.z() #0 {
    ; 43 = SR_NTID_Z
    %r = call i32 @llvm.lwvm.read.sreg.i32(i32 43) #0
    ret i32 %r
}

define linkonce_odr <4 x i32> @optix.lwvm.read.ptx.sreg.ntid() #0 {
    %x = call i32 @llvm.lwvm.read.sreg.i32(i32 41) #0
    %y = call i32 @llvm.lwvm.read.sreg.i32(i32 42) #0
    %z = call i32 @llvm.lwvm.read.sreg.i32(i32 43) #0
    %v0 = insertelement <4 x i32> undef, i32 %x, i32 0
    %v1 = insertelement <4 x i32> %v0, i32 %y, i32 1
    %v2 = insertelement <4 x i32> %v1, i32 %z, i32 2
    ret <4 x i32> %v2
}

define linkonce_odr i32 @optix.lwvm.read.ptx.sreg.ctaid.x() #0 {
    ; 44 = SR_CTAID_X
    %r = call i32 @llvm.lwvm.read.sreg.i32(i32 44) #0
    ret i32 %r
}

define linkonce_odr i32 @optix.lwvm.read.ptx.sreg.ctaid.y() #0 {
    ; 45 = SR_CTAID_Y
    %r = call i32 @llvm.lwvm.read.sreg.i32(i32 45) #0
    ret i32 %r
}

define linkonce_odr i32 @optix.lwvm.read.ptx.sreg.ctaid.z() #0 {
    ; 46 = SR_CTAID_Z
    %r = call i32 @llvm.lwvm.read.sreg.i32(i32 46) #0
    ret i32 %r
}

define linkonce_odr <4 x i32> @optix.lwvm.read.ptx.sreg.ctaid() #0 {
    %x = call i32 @llvm.lwvm.read.sreg.i32(i32 44) #0
    %y = call i32 @llvm.lwvm.read.sreg.i32(i32 45) #0
    %z = call i32 @llvm.lwvm.read.sreg.i32(i32 46) #0
    %v0 = insertelement <4 x i32> undef, i32 %x, i32 0
    %v1 = insertelement <4 x i32> %v0, i32 %y, i32 1
    %v2 = insertelement <4 x i32> %v1, i32 %z, i32 2
    ret <4 x i32> %v2
}

define linkonce_odr i32 @optix.lwvm.read.ptx.sreg.nctaid.x() #0 {
    ; 47 = SR_NCTAID_X
    %r = call i32 @llvm.lwvm.read.sreg.i32(i32 47) #0
    ret i32 %r
}

define linkonce_odr i32 @optix.lwvm.read.ptx.sreg.nctaid.y() #0 {
    ; 48 = SR_NCTAID_Y
    %r = call i32 @llvm.lwvm.read.sreg.i32(i32 48) #0
    ret i32 %r
}

define linkonce_odr i32 @optix.lwvm.read.ptx.sreg.nctaid.z() #0 {
    ; 49 = SR_NCTAID_Z
    %r = call i32 @llvm.lwvm.read.sreg.i32(i32 49) #0
    ret i32 %r
}

define linkonce_odr <4 x i32> @optix.lwvm.read.ptx.sreg.nctaid() #0 {
    %x = call i32 @llvm.lwvm.read.sreg.i32(i32 47) #0
    %y = call i32 @llvm.lwvm.read.sreg.i32(i32 48) #0
    %z = call i32 @llvm.lwvm.read.sreg.i32(i32 49) #0
    %v0 = insertelement <4 x i32> undef, i32 %x, i32 0
    %v1 = insertelement <4 x i32> %v0, i32 %y, i32 1
    %v2 = insertelement <4 x i32> %v1, i32 %z, i32 2
    ret <4 x i32> %v2
}

; smid is volatile
define linkonce_odr i32 @optix.lwvm.read.ptx.sreg.smid() #1 {
    ; 53 = SR_SMID
    %r = call i32 @llvm.lwvm.read.sreg.i32(i32 53) #1
    ret i32 %r
}

define linkonce_odr i32 @optix.lwvm.read.ptx.sreg.lanemask_eq() #0 {
    ; 65 = SR_THREADEQMASK
    %r = call i32 @llvm.lwvm.read.sreg.i32(i32 65) #0
    ret i32 %r
}

define linkonce_odr i32 @optix.lwvm.read.ptx.sreg.lanemask_lt() #0 {
    ; 66 = SR_THREADLTMASK
    %r = call i32 @llvm.lwvm.read.sreg.i32(i32 66) #0
    ret i32 %r
}

define linkonce_odr i32 @optix.lwvm.read.ptx.sreg.lanemask_le() #0 {
    ; 67 = SR_THREADLEMASK
    %r = call i32 @llvm.lwvm.read.sreg.i32(i32 67) #0
    ret i32 %r
}

define linkonce_odr i32 @optix.lwvm.read.ptx.sreg.lanemask_gt() #0 {
    ; 68 = SR_THREADGTMASK
    %r = call i32 @llvm.lwvm.read.sreg.i32(i32 68) #0
    ret i32 %r
}

define linkonce_odr i32 @optix.lwvm.read.ptx.sreg.lanemask_ge() #0 {
    ; 69 = SR_THREADGEMASK
    %r = call i32 @llvm.lwvm.read.sreg.i32(i32 69) #0
    ret i32 %r
}

; gridid is volatile
define linkonce_odr i64 @optix.lwvm.read.ptx.sreg.gridid() #1 {
    ; 96 = SR_GRIDID64
    %r = call i64 @llvm.lwvm.read.sreg.i64(i32 96) #1
    ret i64 %r
}

;-------------------------------------------------------------------------------
; 1D Texture access

declare <4 x float> @llvm.lwvm.tex.fetch.v4f32(i32 %mode, i64 %texref, i64 %samp, float %x, float %y, float %z, i32 %array_idx, float %lev) #4
define linkonce_odr { float, float, float, float } @optix.lwvm.tex_1d(i64 %texref, float %x) #0 {
  ; For some reason using the 1D dimensionality mode causes this call to return incorrect results, so we use 2D mode instead.
  ; mode 67 = 2d and bindless texture sampler
  %ffff = call <4 x float> @llvm.lwvm.tex.fetch.v4f32(i32 67, i64 %texref, i64 undef, float %x, float 0.0, float 0.0, i32 undef, float undef) #4
  %v1 = extractelement <4 x float> %ffff, i32 0
  %v2 = extractelement <4 x float> %ffff, i32 1
  %v3 = extractelement <4 x float> %ffff, i32 2
  %v4 = extractelement <4 x float> %ffff, i32 3
  %s1 = insertvalue {float, float, float, float} undef, float %v1, 0
  %s2 = insertvalue {float, float, float, float} %s1, float %v2, 1
  %s3 = insertvalue {float, float, float, float} %s2, float %v3, 2
  %s4 = insertvalue {float, float, float, float} %s3, float %v4, 3
  ret { float, float, float, float } %s4
}

declare <4 x float> @llvm.lwvm.tex.load.v4f32( i32 %mode, i64 %tex, i64 %samp, i32 %coord_x, i32 %coord_y, i32 %coord_z, i32 %array_idx, i32 %lod, i32 %ms_pos ) #4
define linkonce_odr { float, float, float, float } @optix.lwvm.texfetch_1d(i64 %texref, i32 %x) #0 {
  ; mode 64 = 1d and bindless texture sampler
  %ffff = call <4 x float>  @llvm.lwvm.tex.load.v4f32(i32 64, i64 %texref, i64 undef, i32 %x, i32 0, i32 0, i32 undef, i32 undef, i32 undef) #4

  %v1 = extractelement <4 x float> %ffff, i32 0
  %v2 = extractelement <4 x float> %ffff, i32 1
  %v3 = extractelement <4 x float> %ffff, i32 2
  %v4 = extractelement <4 x float> %ffff, i32 3
  %s1 = insertvalue {float, float, float, float} undef, float %v1, 0
  %s2 = insertvalue {float, float, float, float} %s1, float %v2, 1
  %s3 = insertvalue {float, float, float, float} %s2, float %v3, 2
  %s4 = insertvalue {float, float, float, float} %s3, float %v4, 3

  ret { float, float, float, float } %s4
}

;-------------------------------------------------------------------------------
; 2D Texture access

define linkonce_odr { float, float, float, float } @optix.lwvm.tex_2d(i64 %texref, float %x, float %y) #0 {
  ; Mode: 0x3=3 [2D texture]
  %ffff = call <4 x float> @llvm.lwvm.tex.fetch.v4f32(i32 3, i64 %texref, i64 undef, float %x, float %y, float undef, i32 undef, float undef) #4
  %v1 = extractelement <4 x float> %ffff, i32 0
  %v2 = extractelement <4 x float> %ffff, i32 1
  %v3 = extractelement <4 x float> %ffff, i32 2
  %v4 = extractelement <4 x float> %ffff, i32 3
  %s1 = insertvalue {float, float, float, float} undef, float %v1, 0
  %s2 = insertvalue {float, float, float, float} %s1, float %v2, 1
  %s3 = insertvalue {float, float, float, float} %s2, float %v3, 2
  %s4 = insertvalue {float, float, float, float} %s3, float %v4, 3
  ret { float, float, float, float } %s4
}

define linkonce_odr { float, float, float, float } @optix.lwvm.texfetch_2d(i64 %texref, i32 %x, i32 %y) #0 {
  ; Mode: 0x3=3 [2D texture]
  %ffff = call <4 x float>  @llvm.lwvm.tex.load.v4f32(i32 3, i64 %texref, i64 undef, i32 %x, i32 %y, i32 undef, i32 undef, i32 undef, i32 undef) #4

  %v1 = extractelement <4 x float> %ffff, i32 0
  %v2 = extractelement <4 x float> %ffff, i32 1
  %v3 = extractelement <4 x float> %ffff, i32 2
  %v4 = extractelement <4 x float> %ffff, i32 3
  %s1 = insertvalue {float, float, float, float} undef, float %v1, 0
  %s2 = insertvalue {float, float, float, float} %s1, float %v2, 1
  %s3 = insertvalue {float, float, float, float} %s2, float %v3, 2
  %s4 = insertvalue {float, float, float, float} %s3, float %v4, 3

  ret { float, float, float, float } %s4
}

define linkonce_odr { float, float, float, float } @optix.lwvm.texlevel_2d(i64 %texref, float %x, float %y, float %lev) #0 {
  ; Mode: 0x23=35 [2D texture, absolute LOD]
  %ffff = call <4 x float> @llvm.lwvm.tex.fetch.v4f32(i32 35, i64 %texref, i64 undef, float %x, float %y, float undef, i32 undef, float %lev) #4
  %v1 = extractelement <4 x float> %ffff, i32 0
  %v2 = extractelement <4 x float> %ffff, i32 1
  %v3 = extractelement <4 x float> %ffff, i32 2
  %v4 = extractelement <4 x float> %ffff, i32 3
  %s1 = insertvalue {float, float, float, float} undef, float %v1, 0
  %s2 = insertvalue {float, float, float, float} %s1, float %v2, 1
  %s3 = insertvalue {float, float, float, float} %s2, float %v3, 2
  %s4 = insertvalue {float, float, float, float} %s3, float %v4, 3
  ret { float, float, float, float } %s4
}


declare <4 x float> @llvm.lwvm.tex.fetch.grad.v4f32(i32 %mode, i64 %texref, i64 %samp, float %x, float %y, float %z, i32 %array_idx, float %dsdx, float %dsdy, float %dsdz, float %dtdx, float %dtdy, float %dtdz) #4

declare { <4 x i32>, i1 } @llvm.lwvm.tex.footprint.grad.v4i32(i32 %mode, i64 %tex, i64 %samp, float %coord_x, float %coord_y, float %coord_z, i32 %array_idx, float %dsdx, float %dsdy, float %dsdz, float %dtdx, float %dtdy, float %dtdz, i32 %granularity) #4
define linkonce_odr { i32, i32, i32, i32, i1 } @optix.lwvm.texgrad_footprint_2d(i32 %granularity, i64 %texref, float %x, float %y, float %dpdx_x, float %dpdx_y, float %dpdy_x, float %dpdy_y) #0 {
  ; Mode: 0x3=3 [2D texture]
  %iiiib = call {<4 x i32>, i1} @llvm.lwvm.tex.footprint.grad.v4i32(i32 3, i64 %texref, i64 undef, float %x, float %y, float undef, i32 undef, float %dpdx_x, float %dpdx_y, float undef, float %dpdy_x, float %dpdy_y, float undef, i32 %granularity) #4
  %footprintResult = extractvalue {<4 x i32>, i1} %iiiib, 0
  %i0 = extractelement <4 x i32> %footprintResult, i32 0
  %i1 = extractelement <4 x i32> %footprintResult, i32 1
  %i2 = extractelement <4 x i32> %footprintResult, i32 2
  %i3 = extractelement <4 x i32> %footprintResult, i32 3
  %isSingleMipLevel = extractvalue {<4 x i32>, i1} %iiiib, 1
  %ret0 = insertvalue {i32, i32, i32, i32, i1} undef, i32 %i0, 0
  %ret1 = insertvalue {i32, i32, i32, i32, i1} %ret0, i32 %i1, 1
  %ret2 = insertvalue {i32, i32, i32, i32, i1} %ret1, i32 %i2, 2
  %ret3 = insertvalue {i32, i32, i32, i32, i1} %ret2, i32 %i3, 3
  %ret4 = insertvalue {i32, i32, i32, i32, i1} %ret3, i1 %isSingleMipLevel, 4
  ret { i32, i32, i32, i32, i1 } %ret4
}

define linkonce_odr { i32, i32, i32, i32, i1 } @optix.lwvm.texgrad_footprint_coarse_2d(i32 %granularity, i64 %texref, float %x, float %y, float %dpdx_x, float %dpdx_y, float %dpdy_x, float %dpdy_y) #0 {
  ; Mode: 0x4003=16387 [2D texture, coarse level footprint]
  %iiiib = call {<4 x i32>, i1} @llvm.lwvm.tex.footprint.grad.v4i32(i32 16387, i64 %texref, i64 undef, float %x, float %y, float undef, i32 undef, float %dpdx_x, float %dpdx_y, float undef, float %dpdy_x, float %dpdy_y, float undef, i32 %granularity) #4
  %footprintResult = extractvalue {<4 x i32>, i1} %iiiib, 0
  %i0 = extractelement <4 x i32> %footprintResult, i32 0
  %i1 = extractelement <4 x i32> %footprintResult, i32 1
  %i2 = extractelement <4 x i32> %footprintResult, i32 2
  %i3 = extractelement <4 x i32> %footprintResult, i32 3
  %isSingleMipLevel = extractvalue {<4 x i32>, i1} %iiiib, 1
  %ret0 = insertvalue {i32, i32, i32, i32, i1} undef, i32 %i0, 0
  %ret1 = insertvalue {i32, i32, i32, i32, i1} %ret0, i32 %i1, 1
  %ret2 = insertvalue {i32, i32, i32, i32, i1} %ret1, i32 %i2, 2
  %ret3 = insertvalue {i32, i32, i32, i32, i1} %ret2, i32 %i3, 3
  %ret4 = insertvalue {i32, i32, i32, i32, i1} %ret3, i1 %isSingleMipLevel, 4
  ret { i32, i32, i32, i32, i1 } %ret4
}

;-------------------------------------------------------------------------------
; 2D Array Texture access

define linkonce_odr { float, float, float, float } @optix.lwvm.tex_a2d(i64 %texref, i32 %a, float %x, float %y) #0 {
  ; Mode: 0x4=4 [2D array texture]
  %ffff = call <4 x float> @llvm.lwvm.tex.fetch.v4f32(i32 4, i64 %texref, i64 undef, float %x, float %y, float undef, i32 %a, float undef) #4
  %v1 = extractelement <4 x float> %ffff, i32 0
  %v2 = extractelement <4 x float> %ffff, i32 1
  %v3 = extractelement <4 x float> %ffff, i32 2
  %v4 = extractelement <4 x float> %ffff, i32 3
  %s1 = insertvalue {float, float, float, float} undef, float %v1, 0
  %s2 = insertvalue {float, float, float, float} %s1, float %v2, 1
  %s3 = insertvalue {float, float, float, float} %s2, float %v3, 2
  %s4 = insertvalue {float, float, float, float} %s3, float %v4, 3
  ret { float, float, float, float } %s4
}

define linkonce_odr { float, float, float, float } @optix.lwvm.texlevel_a2d(i64 %texref, i32 %a, float %x, float %y, float %lev) #0 {
  ; Mode: 0x24=36 [2D array texture, absolute LOD]
  %ffff = call <4 x float> @llvm.lwvm.tex.fetch.v4f32(i32 36, i64 %texref, i64 undef, float %x, float %y, float undef, i32 %a, float %lev) #4
  %v1 = extractelement <4 x float> %ffff, i32 0
  %v2 = extractelement <4 x float> %ffff, i32 1
  %v3 = extractelement <4 x float> %ffff, i32 2
  %v4 = extractelement <4 x float> %ffff, i32 3
  %s1 = insertvalue {float, float, float, float} undef, float %v1, 0
  %s2 = insertvalue {float, float, float, float} %s1, float %v2, 1
  %s3 = insertvalue {float, float, float, float} %s2, float %v3, 2
  %s4 = insertvalue {float, float, float, float} %s3, float %v4, 3
  ret { float, float, float, float } %s4
}

define linkonce_odr  { float, float, float, float } @optix.lwvm.texgrad_a2d(i64 %texref, i32 %a, float %x, float %y, float %dpdx_x, float %dpdx_y, float %dpdy_x, float %dpdy_y) #0 {
  ; Mode: 0x4=4 [2D array texture]
  %ffff = call <4 x float> @llvm.lwvm.tex.fetch.grad.v4f32(i32 4, i64 %texref, i64 undef, float %x, float %y, float undef, i32 %a, float %dpdx_x, float %dpdx_y, float undef, float %dpdy_x, float %dpdy_y, float undef) #4
  %v1 = extractelement <4 x float> %ffff, i32 0
  %v2 = extractelement <4 x float> %ffff, i32 1
  %v3 = extractelement <4 x float> %ffff, i32 2
  %v4 = extractelement <4 x float> %ffff, i32 3
  %s1 = insertvalue {float, float, float, float} undef, float %v1, 0
  %s2 = insertvalue {float, float, float, float} %s1, float %v2, 1
  %s3 = insertvalue {float, float, float, float} %s2, float %v3, 2
  %s4 = insertvalue {float, float, float, float} %s3, float %v4, 3
  ret { float, float, float, float } %s4
}

;-------------------------------------------------------------------------------
; 3D Texture access

define linkonce_odr { float, float, float, float } @optix.lwvm.tex_3d(i64 %texref, float %x, float %y, float %z) #0 {
  ; Mode: 0x5=5 [3D texture]
  %ffff = call <4 x float> @llvm.lwvm.tex.fetch.v4f32(i32 5, i64 %texref, i64 undef, float %x, float %y, float %z, i32 undef, float undef) #4
  %v1 = extractelement <4 x float> %ffff, i32 0
  %v2 = extractelement <4 x float> %ffff, i32 1
  %v3 = extractelement <4 x float> %ffff, i32 2
  %v4 = extractelement <4 x float> %ffff, i32 3
  %s1 = insertvalue {float, float, float, float} undef, float %v1, 0
  %s2 = insertvalue {float, float, float, float} %s1, float %v2, 1
  %s3 = insertvalue {float, float, float, float} %s2, float %v3, 2
  %s4 = insertvalue {float, float, float, float} %s3, float %v4, 3
  ret { float, float, float, float } %s4
}

define linkonce_odr { float, float, float, float } @optix.lwvm.texfetch_3d(i64 %texref, i32 %x, i32 %y, i32 %z) #0 {
  ; Mode: 0x5=5 [3D texture]
  %ffff = call <4 x float>  @llvm.lwvm.tex.load.v4f32(i32 5, i64 %texref, i64 undef, i32 %x, i32 %y, i32 %z, i32 undef, i32 undef, i32 undef) #4

  %v1 = extractelement <4 x float> %ffff, i32 0
  %v2 = extractelement <4 x float> %ffff, i32 1
  %v3 = extractelement <4 x float> %ffff, i32 2
  %v4 = extractelement <4 x float> %ffff, i32 3
  %s1 = insertvalue {float, float, float, float} undef, float %v1, 0
  %s2 = insertvalue {float, float, float, float} %s1, float %v2, 1
  %s3 = insertvalue {float, float, float, float} %s2, float %v3, 2
  %s4 = insertvalue {float, float, float, float} %s3, float %v4, 3

  ret { float, float, float, float } %s4
}

;-------------------------------------------------------------------------------
; Lwbe Texture access

define linkonce_odr { float, float, float, float } @optix.lwvm.tex_lwbe(i64 %texref, float %x, float %y, float %z) #0 {
  ; Mode: 0x7=7 [lwbe texture]
  %ffff = call <4 x float> @llvm.lwvm.tex.fetch.v4f32(i32 7, i64 %texref, i64 undef, float %x, float %y, float %z, i32 undef, float undef) #4
  %v1 = extractelement <4 x float> %ffff, i32 0
  %v2 = extractelement <4 x float> %ffff, i32 1
  %v3 = extractelement <4 x float> %ffff, i32 2
  %v4 = extractelement <4 x float> %ffff, i32 3
  %s1 = insertvalue {float, float, float, float} undef, float %v1, 0
  %s2 = insertvalue {float, float, float, float} %s1, float %v2, 1
  %s3 = insertvalue {float, float, float, float} %s2, float %v3, 2
  %s4 = insertvalue {float, float, float, float} %s3, float %v4, 3
  ret { float, float, float, float } %s4
}

define linkonce_odr { float, float, float, float } @optix.lwvm.texlevel_lwbe(i64 %texref, float %x, float %y, float %z, float %lev) #0 {
  ; Mode: 0x27=39 [lwbe texture, absolute LOD]
  %ffff = call <4 x float> @llvm.lwvm.tex.fetch.v4f32(i32 39, i64 %texref, i64 undef, float %x, float %y, float %z, i32 undef, float %lev) #4
  %v1 = extractelement <4 x float> %ffff, i32 0
  %v2 = extractelement <4 x float> %ffff, i32 1
  %v3 = extractelement <4 x float> %ffff, i32 2
  %v4 = extractelement <4 x float> %ffff, i32 3
  %s1 = insertvalue {float, float, float, float} undef, float %v1, 0
  %s2 = insertvalue {float, float, float, float} %s1, float %v2, 1
  %s3 = insertvalue {float, float, float, float} %s2, float %v3, 2
  %s4 = insertvalue {float, float, float, float} %s3, float %v4, 3
  ret { float, float, float, float } %s4
}

;-------------------------------------------------------------------------------
; bfind (ported from PTX macros at
; drivers/compiler/gpulibs/emulib/ptx-src/ptxMacros/ptx-builtins.ptx)

declare i32 @llvm.lwvm.flo.u.i32(i32, i1)
declare i32 @llvm.lwvm.flo.s.i32(i32, i1)

; Function Attrs: alwaysinline nounwind
define linkonce i32 @optix.ptx.bfind.u32(i32) #1 {
  %2 = call i32 @llvm.lwvm.flo.u.i32(i32 %0, i1 false)
  ret i32 %2
}

; Function Attrs: alwaysinline nounwind
define linkonce i32 @optix.ptx.bfind.s32(i32) #1 {
  %2 = call i32 @llvm.lwvm.flo.s.i32(i32 %0, i1 false)
  ret i32 %2
}

; Function Attrs: alwaysinline nounwind
define linkonce i32 @optix.ptx.bfind.shiftamt.u32(i32) #1 {
  %2 = call i32 @llvm.lwvm.flo.u.i32(i32 %0, i1 true)
  ret i32 %2
}

; Function Attrs: alwaysinline nounwind
define linkonce i32 @optix.ptx.bfind.shiftamt.s32(i32) #1 {
  %2 = call i32 @llvm.lwvm.flo.s.i32(i32 %0, i1 true)
  ret i32 %2
}

declare i32 @llvm.lwvm.unpack.hi.i32.i64(i64)
declare i32 @llvm.lwvm.unpack.lo.i32.i64(i64)

define linkonce i32 @optix.ptx.bfind.shiftamt.u64(i64 %p0) #1 {
  %topBits = call i32 @llvm.lwvm.unpack.hi.i32.i64(i64 %p0)
  %topBitsRes = call i32 @optix.ptx.bfind.shiftamt.u32(i32 %topBits)

  %bottomBits = call i32 @llvm.lwvm.unpack.lo.i32.i64(i64 %p0)
  %bottomBitsRes = call i32 @optix.ptx.bfind.shiftamt.u32(i32 %bottomBits)

  %topBitsContainRes = icmp ne i32 %topBitsRes, -1
  br i1 %topBitsContainRes, label %topBitsContainResExit, label %bottomBitsContainResExit

topBitsContainResExit:
  ret i32 %topBitsRes

bottomBitsContainResExit:
  %shiftedBottomBits = add i32 %bottomBitsRes, 32
  %bottomBitsContainNonSign = icmp ne i32 %bottomBitsRes, -1
  %returlwal = select i1 %bottomBitsContainNonSign, i32 %shiftedBottomBits, i32 -1
  ret i32 %returlwal
}

define linkonce i32 @optix.ptx.bfind.u64(i64 %p0) #1 {
  %topBits = call i32 @llvm.lwvm.unpack.hi.i32.i64(i64 %p0)
  %topBitsRes = call i32 @optix.ptx.bfind.u32(i32 %topBits)

  %bottomBitsContainRes = icmp eq i32 %topBitsRes, -1
  br i1 %bottomBitsContainRes, label %bottomBitsContainResExit, label %topBitsContainResExit

topBitsContainResExit:
  %shiftedTopBits = add i32 %topBitsRes, 32
  ret i32 %shiftedTopBits

bottomBitsContainResExit:
  %bottomBits = call i32 @llvm.lwvm.unpack.lo.i32.i64(i64 %p0)
  %bottomBitsRes = call i32 @optix.ptx.bfind.u32(i32 %bottomBits)
  ret i32 %bottomBitsRes
}

define linkonce i32 @optix.ptx.bfind.shiftamt.s64(i64 %p0) #1 {
  %topBits = call i32 @llvm.lwvm.unpack.hi.i32.i64(i64 %p0)
  %topBitsRes = call i32 @optix.ptx.bfind.shiftamt.s32(i32 %topBits)

  %bottomBits = call i32 @llvm.lwvm.unpack.lo.i32.i64(i64 %p0)

  %topBitsAreNegative = icmp slt i32 %topBits, 0
  br i1 %topBitsAreNegative, label %topBitsNegative, label %topBitsPositive

topBitsNegative:
  %negatedBottomBits = xor i32 %bottomBits, -1
  %bottomBitsRes0 = call i32 @optix.ptx.bfind.shiftamt.u32(i32 %negatedBottomBits)
  br label %bottomBitsComputed

topBitsPositive:
  %bottomBitsRes1 = call i32 @optix.ptx.bfind.shiftamt.u32(i32 %bottomBits)
  br label %bottomBitsComputed

bottomBitsComputed:
  %bottomBitsRes = phi i32 [ %bottomBitsRes0, %topBitsNegative ], [ %bottomBitsRes1, %topBitsPositive ]

  %topBitsContainRes = icmp ne i32 %topBitsRes, -1
  br i1 %topBitsContainRes, label %topBitsContainResExit, label %bottomBitsContainResExit
  
topBitsContainResExit:
  ret i32 %topBitsRes

bottomBitsContainResExit:
  %shiftedBottomBits = add i32 %bottomBitsRes, 32
  %bottomBitsContainNonSign = icmp ne i32 %bottomBitsRes, -1
  %returlwal = select i1 %bottomBitsContainNonSign, i32 %shiftedBottomBits, i32 -1
  ret i32 %returlwal
}

define linkonce i32 @optix.ptx.bfind.s64(i64 %p0) #1 {
  %topBits = call i32 @llvm.lwvm.unpack.hi.i32.i64(i64 %p0)
  %topBitsRes = call i32 @optix.ptx.bfind.s32(i32 %topBits)

  %bottomBits = call i32 @llvm.lwvm.unpack.lo.i32.i64(i64 %p0)

  %bottomBitsContainRes = icmp eq i32 %topBitsRes, -1
  br i1 %bottomBitsContainRes, label %bottomBitsContainResExit, label %topBitsContainResExit

topBitsContainResExit:
  %shiftedTopBits = add i32 %topBitsRes, 32
  ret i32 %shiftedTopBits

bottomBitsContainResExit:
  %topBitsAreNegative = icmp slt i32 %topBits, 0
  br i1 %topBitsAreNegative, label %topBitsNegative, label %topBitsPositive

topBitsNegative:
  %negatedBottomBits = xor i32 %bottomBits, -1
  %bottomBitsRes0 = call i32 @optix.ptx.bfind.u32(i32 %negatedBottomBits)
  ret i32 %bottomBitsRes0

topBitsPositive:
  %bottomBitsRes1 = call i32 @optix.ptx.bfind.u32(i32 %bottomBits)
  ret i32 %bottomBitsRes1
}
