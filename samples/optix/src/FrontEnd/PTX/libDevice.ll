;; Copyright (c) 2017, LWPU CORPORATION.
;; TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED
;; *AS IS* AND LWPU AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS
;; OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY
;; AND FITNESS FOR A PARTICULAR PURPOSE.  IN NO EVENT SHALL LWPU OR ITS SUPPLIERS
;; BE LIABLE FOR ANY SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES
;; WHATSOEVER (INCLUDING, WITHOUT LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS,
;; BUSINESS INTERRUPTION, LOSS OF BUSINESS INFORMATION, OR ANY OTHER PELWNIARY LOSS)
;; ARISING OUT OF THE USE OF OR INABILITY TO USE THIS SOFTWARE, EVEN IF LWPU HAS
;; BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES

attributes #0 = { nounwind readnone alwaysinline } ; Non-volatile functions
attributes #1 = { nounwind alwaysinline }          ; "volatile" functions, which may return different results on different ilwocations
attributes #2 = { nounwind readnone }
attributes #4 = { nounwind }                       ; for texture intrinsics - they MUST not have readnone

;-------------------------------------------------------------------------------
; Most instructions are defined in the auto-generated
; ptx_instructions.ll but these need definitions here.

declare void @llvm.lwca.syncthreads()
define linkonce_odr void @optix.lwca.syncthreads() #1 {
    call void @llvm.lwca.syncthreads()
    ret void
}

;-------------------------------------------------------------------------------
; Provide definitions of commonly used instructions to avoid inline
; assembly, resulting in cleaner output.  Eventually move to
; generating the library at runtime to avoid inline asm in most cases.

;-------------------------------------------------------------------------------
; Multiply

declare float @llvm.lwvm.mul.rn.ftz.f(float %x, float %y) #0
define linkonce float @optix.ptx.mul.ftz.f32( float %x, float %y ) #0 {
    %r0 = call float @llvm.lwvm.mul.rn.ftz.f( float %x, float %y ) #0
    ret float %r0
}

define linkonce float @optix.ptx.mul.rn.ftz.f32( float %x, float %y ) #0 {
    %r0 = call float @llvm.lwvm.mul.rn.ftz.f( float %x, float %y ) #0
    ret float %r0
}

declare float @llvm.lwvm.mul.rn.f(float %x, float %y) #0
define linkonce float @optix.ptx.mul.f32( float %x, float %y ) #0 {
    %r0 = call float @llvm.lwvm.mul.rn.f( float %x, float %y ) #0
    ret float %r0
}

declare float @llvm.lwvm.mul.rz.f(float %x, float %y) #0
define linkonce_odr float @optix.lwvm.mul.rz.f(float %x, float %y) #0 {
    %1 = tail call float @llvm.lwvm.mul.rz.f(float %x, float %y) #0
    ret float %1
}

define linkonce float @optix.ptx.mul.rn.f32( float %x, float %y ) #0 {
    %r0 = call float @llvm.lwvm.mul.rn.f( float %x, float %y ) #0
    ret float %r0
}

;-------------------------------------------------------------------------------
; Wide multiply

define linkonce i32 @optix.ptx.mul.wide.s16( i16 %p0, i16 %p1 ) #0 {
    %awide = sext i16 %p0 to i32
    %bwide = sext i16 %p1 to i32
    %r = mul i32 %awide, %bwide
    ret i32 %r
}

define linkonce i64 @optix.ptx.mul.wide.s32( i32 %p0, i32 %p1 ) #0 {
    %awide = sext i32 %p0 to i64
    %bwide = sext i32 %p1 to i64
    %r = mul i64 %awide, %bwide
    ret i64 %r
}

define linkonce i32 @optix.ptx.mul.wide.u16( i16 %p0, i16 %p1 ) #0 {
    %awide = zext i16 %p0 to i32
    %bwide = zext i16 %p1 to i32
    %r = mul i32 %awide, %bwide
    ret i32 %r
}

define linkonce i64 @optix.ptx.mul.wide.u32( i32 %p0, i32 %p1 ) #0 {
    %awide = zext i32 %p0 to i64
    %bwide = zext i32 %p1 to i64
    %r = mul i64 %awide, %bwide
    ret i64 %r
}

;-------------------------------------------------------------------------------
; Divide

declare float @llvm.lwvm.div.rn.f( float %p0, float %p1 ) #0
define linkonce float @optix.ptx.div.rn.f32( float %p0, float %p1 ) #0 {
    %r0 = call float @llvm.lwvm.div.rn.f( float %p0, float %p1 ) #0
    ret float %r0
}

declare float @llvm.lwvm.div.approx.f( float %p0, float %p1 ) #0
define linkonce float @optix.ptx.div.approx.f32( float %p0, float %p1 ) #0 {
    %r0 = call float @llvm.lwvm.div.approx.f( float %p0, float %p1 ) #0
    ret float %r0
}

declare float @llvm.lwvm.div.approx.ftz.f( float %p0, float %p1 ) #0
define linkonce float @optix.ptx.div.approx.ftz.f32( float %p0, float %p1 ) #0 {
    %r0 = call float @llvm.lwvm.div.approx.ftz.f( float %p0, float %p1 ) #0
    ret float %r0
}

;-------------------------------------------------------------------------------
; Add

declare float @llvm.lwvm.add.rn.ftz.f(float %x, float %y) #0
define linkonce float @optix.ptx.add.ftz.f32( float %x, float %y ) #0 {
    %r0 = call float @llvm.lwvm.add.rn.ftz.f( float %x, float %y ) #0
    ret float %r0
}

define linkonce float @optix.ptx.add.rn.ftz.f32( float %x, float %y ) #0 {
    %r0 = call float @llvm.lwvm.add.rn.ftz.f( float %x, float %y ) #0
    ret float %r0
}

declare float @llvm.lwvm.add.rn.f(float %x, float %y) #0
define linkonce float @optix.ptx.add.f32( float %x, float %y ) #0 {
    %r0 = call float @llvm.lwvm.add.rn.f( float %x, float %y ) #0
    ret float %r0
}

define linkonce float @optix.ptx.add.rn.f32( float %x, float %y ) #0 {
    %r0 = call float @llvm.lwvm.add.rn.f( float %x, float %y ) #0
    ret float %r0
}

declare float @llvm.lwvm.add.rz.f(float %x, float %y) #0
define linkonce_odr float @optix.lwvm.add.rz.f(float %x, float %y) #0 {
    %1 = tail call float @llvm.lwvm.add.rz.f(float %x, float %y) #0
    ret float %1
}

;-------------------------------------------------------------------------------
; Sub

; The 'flags' parameter is used to communicate e.g. the rounding mode, see here:
; https:;;p4viewer.lwpu.com/get/sw/compiler/docs/UnifiedLWVMIR/asciidoc/html/lwvmIR.html#_math_intrinsic_functions
; https:;;p4viewer.lwpu.com/get/sw/compiler/docs/UnifiedLWVMIR/asciidoc/html/lwvmIR.html#rounding_mode_encoding
; i32 1 corresponds to RN rounding mode.
declare float @llvm.lwvm.sub.ftz.f32(i32 %flags, float %x, float %y) #0
define linkonce float @optix.ptx.sub.ftz.f32( float %x, float %y ) #0 {
    %r0 = call float @llvm.lwvm.sub.ftz.f32( i32 1, float %x, float %y ) #0
    ret float %r0
}

;-------------------------------------------------------------------------------
; Sin / Cos

declare float @llvm.lwvm.cos.approx.ftz.f( float %p0 ) #0
define linkonce float @optix.ptx.cos.approx.ftz.f32( float %p0 ) #0 {
   %r0 = call float @llvm.lwvm.cos.approx.ftz.f( float %p0 ) #0
   ret float %r0
}

declare float @llvm.lwvm.cos.approx.f( float %p0 ) #0
define linkonce float @optix.ptx.cos.approx.f32( float %p0 ) #0 {
   %r0 = call float @llvm.lwvm.cos.approx.f( float %p0 ) #0
   ret float %r0
}

declare float @llvm.lwvm.sin.approx.ftz.f( float %p0 ) #0
define linkonce float @optix.ptx.sin.approx.ftz.f32( float %p0 ) #0 {
   %r0 = call float @llvm.lwvm.sin.approx.ftz.f( float %p0 ) #0
   ret float %r0
}

declare float @llvm.lwvm.sin.approx.f( float %p0 ) #0
define linkonce float @optix.ptx.sin.approx.f32( float %p0 ) #0 {
   %r0 = call float @llvm.lwvm.sin.approx.f( float %p0 ) #0
   ret float %r0
}

;-------------------------------------------------------------------------------
; Absolute value

declare i32 @llvm.lwvm.abs.i( i32 %p0 ) #0
define linkonce i32 @optix.ptx.abs.s32( i32 %p0 ) #0 {
    %r0 = call i32 @llvm.lwvm.abs.i( i32 %p0 ) #0
    ret i32 %r0
}

declare float @llvm.lwvm.fabs.f( float %p0 ) #0
define linkonce float @optix.ptx.abs.f32( float %p0 ) #0 {
    %r0 = call float @llvm.lwvm.fabs.f( float %p0 ) #0
    ret float %r0
}

declare float @llvm.lwvm.fabs.ftz.f( float %p0 ) #0
define linkonce float @optix.ptx.abs.ftz.f32( float %p0 ) #0 {
    %r0 = call float @llvm.lwvm.fabs.ftz.f( float %p0 ) #0
    ret float %r0
}

;-------------------------------------------------------------------------------
; Square root

declare float @llvm.lwvm.sqrt.approx.ftz.f( float %p0 ) #0
define linkonce float @optix.ptx.sqrt.approx.ftz.f32( float %p0 ) #0 {
    %r0 = call float @llvm.lwvm.sqrt.approx.ftz.f( float %p0 ) #0
    ret float %r0
}

declare float @llvm.lwvm.sqrt.rn.ftz.f( float %p0 ) #0
define linkonce float @optix.ptx.sqrt.rn.ftz.f32( float %p0 ) #0 {
    %r0 = call float @llvm.lwvm.sqrt.rn.ftz.f( float %p0 ) #0
    ret float %r0
}

declare float @llvm.lwvm.sqrt.rn.f( float %p0 ) #0
define linkonce float @optix.ptx.sqrt.rn.f32( float %p0 ) #0 {
    %r0 = call float @llvm.lwvm.sqrt.rn.f( float %p0 ) #0
    ret float %r0
}

declare float @llvm.lwvm.sqrt.rp.f( float %p0 ) #0
define linkonce float @optix.ptx.sqrt.rp.f32( float %p0 ) #0 {
    %r0 = call float @llvm.lwvm.sqrt.rp.f( float %p0 ) #0
    ret float %r0
}

declare float @llvm.lwvm.sqrt.rp.ftz.f( float %p0 ) #0
define linkonce float @optix.ptx.sqrt.rp.ftz.f32( float %p0 ) #0 {
    %r0 = call float @llvm.lwvm.sqrt.rp.ftz.f( float %p0 ) #0
    ret float %r0
}

declare float @llvm.lwvm.rsqrt.approx.f( float %p0 ) #0
define linkonce float @optix.ptx.rsqrt.approx.f32( float %p0 ) #0 {
    %r0 = call float @llvm.lwvm.rsqrt.approx.f( float %p0 ) #0
    ret float %r0
}

declare float @llvm.lwvm.rsqrt.approx.ftz.f( float %p0 ) #0
define linkonce float @optix.ptx.rsqrt.approx.ftz.f32( float %p0 ) #0 {
    %r0 = call float @llvm.lwvm.rsqrt.approx.ftz.f( float %p0 ) #0
    ret float %r0
}

;-------------------------------------------------------------------------------
; ex2 / lg2

declare float @llvm.lwvm.ex2.approx.f( float %p0 ) #0
define linkonce float @optix.ptx.ex2.approx.f32( float %p0 ) #0 {
    %r0 = call float @llvm.lwvm.ex2.approx.f( float %p0 ) #0
    ret float %r0
}

declare float @llvm.lwvm.ex2.approx.ftz.f( float %p0 ) #0
define linkonce float @optix.ptx.ex2.approx.ftz.f32( float %p0 ) #0 {
    %r0 = call float @llvm.lwvm.ex2.approx.ftz.f( float %p0 ) #0
    ret float %r0
}

declare float @llvm.lwvm.lg2.approx.ftz.f( float %p0 ) #0
define linkonce float @optix.ptx.lg2.approx.ftz.f32( float %p0 ) #0 {
    %r0 = call float @llvm.lwvm.lg2.approx.ftz.f( float %p0 ) #0
    ret float %r0
}

declare float @llvm.lwvm.lg2.approx.f( float %p0 ) #0
define linkonce float @optix.ptx.lg2.approx.f32( float %p0 ) #0 {
    %r0 = call float @llvm.lwvm.lg2.approx.f( float %p0 ) #0
    ret float %r0
}

;-------------------------------------------------------------------------------
; Saturate

declare float @llvm.lwvm.saturate.ftz.f( float %p0 ) #0
define linkonce float @optix.ptx.cvt.ftz.sat.f32.f32( float %p0 ) #0 {
    ; Mode: 0x7=7 [flush to 0, saturate, round to even (rn)]
    %r0 = call float @llvm.lwvm.saturate.ftz.f( float %p0 ) #0
    ret float %r0
}

declare float @llvm.lwvm.saturate.f( float %p0 ) #0
define linkonce float @optix.ptx.cvt.sat.f32.f32( float %p0 ) #0 {
    %r0 = call float @llvm.lwvm.saturate.f( float %p0 ) #0
    ret float %r0
}

;-------------------------------------------------------------------------------
; FMA

declare float @llvm.lwvm.fma.rn.f(float %x, float %y, float %w) #0
define linkonce float @optix.ptx.fma.rn.f32( float %x, float %y, float %w ) #0 {
    %r0 = call float @llvm.lwvm.fma.rn.f( float %x, float %y, float %w ) #0
    ret float %r0
}

declare float @llvm.lwvm.fma.rn.ftz.f(float %x, float %y, float %w) #0
define linkonce float @optix.ptx.fma.rn.ftz.f32( float %x, float %y, float %w ) #0 {
    %r0 = call float @llvm.lwvm.fma.rn.ftz.f( float %x, float %y, float %w ) #0
    ret float %r0
}

;-------------------------------------------------------------------------------
; Max / min

declare float @llvm.lwvm.fmax.ftz.f(float %x, float %y) #0
define linkonce float @optix.ptx.max.ftz.f32( float %x, float %y ) #0 {
    %r0 = call float @llvm.lwvm.fmax.ftz.f( float %x, float %y ) #0
    ret float %r0
}

declare float @llvm.lwvm.fmin.ftz.f(float %x, float %y) #0
define linkonce float @optix.ptx.min.ftz.f32( float %x, float %y ) #0 {
    %r0 = call float @llvm.lwvm.fmin.ftz.f( float %x, float %y ) #0
    ret float %r0
}

declare i32 @llvm.lwvm.min.i(i32 %x, i32 %y) #0
define linkonce_odr i32 @optix.lwvm.min.i(i32 %x, i32 %y) #0 {
    %1 = tail call i32 @llvm.lwvm.min.i(i32 %x, i32 %y) #0
    ret i32 %1
}

declare i32 @llvm.lwvm.max.i(i32 %x, i32 %y) #0
define linkonce_odr i32 @optix.lwvm.max.i(i32 %x, i32 %y) #0 {
    %1 = tail call i32 @llvm.lwvm.max.i(i32 %x, i32 %y) #0
    ret i32 %1
}

declare i32 @llvm.lwvm.min.ui(i32 %x, i32 %y) #0
define linkonce_odr i32 @optix.lwvm.min.ui(i32 %x, i32 %y) #0 {
    %1 = tail call i32 @llvm.lwvm.min.ui(i32 %x, i32 %y) #0
    ret i32 %1
}

declare i32 @llvm.lwvm.max.ui(i32 %x, i32 %y) #0
define linkonce_odr i32 @optix.lwvm.max.ui(i32 %x, i32 %y) #0 {
    %1 = tail call i32 @llvm.lwvm.max.ui(i32 %x, i32 %y) #0
    ret i32 %1
}

declare float @llvm.lwvm.fmin.f(float %x, float %y) #0
define linkonce_odr float @optix.lwvm.fmin.f(float %x, float %y) #0 {
    %1 = tail call float @llvm.lwvm.fmin.f(float %x, float %y) #0
    ret float %1
}

declare float @llvm.lwvm.fmax.f(float %x, float %y) #0
define linkonce_odr float @optix.lwvm.fmax.f(float %x, float %y) #0 {
    %1 = tail call float @llvm.lwvm.fmax.f(float %x, float %y) #0
    ret float %1
}

;-------------------------------------------------------------------------------
; Floor / ceil

declare float @llvm.lwvm.floor.f(float %x) #0
define linkonce_odr float @optix.lwvm.floor.f(float %x) #0 {
    %1 = tail call float @llvm.lwvm.floor.f(float %x) #0
    ret float %1
}

declare float @llvm.lwvm.ceil.f(float %x) #0
define linkonce_odr float @optix.lwvm.ceil.f(float %x) #0 {
    %1 = tail call float @llvm.lwvm.ceil.f(float %x) #0
    ret float %1
}

;-------------------------------------------------------------------------------
; Colwersions

declare float @llvm.lwvm.ui2f.rz(i32 %x) #0
define linkonce_odr float @optix.lwvm.ui2f.rz(i32 %x) #0 {
    %1 = tail call float @llvm.lwvm.ui2f.rz(i32 %x) #0
    ret float %1
}

declare float @llvm.lwvm.i2f.rz(i32 %x) #0
define linkonce_odr float @optix.lwvm.i2f.rz(i32 %x) #0 {
    %1 = tail call float @llvm.lwvm.i2f.rz(i32 %x) #0
    ret float %1
}

declare i32 @llvm.lwvm.f2i.rz(float %x) #0
define linkonce_odr i32 @optix.lwvm.f2i.rz(float %x) #0 {
    %1 = tail call i32 @llvm.lwvm.f2i.rz(float %x) #0
    ret i32 %1
}

declare i32 @llvm.lwvm.f2i.rm(float %x) #0
define linkonce_odr i32 @optix.lwvm.f2i.rm(float %x) #0 {
    %1 = tail call i32 @llvm.lwvm.f2i.rm(float %x) #0
    ret i32 %1
}

