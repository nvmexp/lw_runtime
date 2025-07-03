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

;
; llvm intrinsics to be linked in the ptx front end.  The remainder will be linked in the
; LLVMtoPTX backend.  Functions in here should have "linkonce_odr" linkage.  The ODR
; (C++'s One Definition Rule) linkage allows for better inlining support, because the
; definition of the function can't be overriden during later linking.
;

attributes #0 = { nounwind readnone alwaysinline } ; Non-volatile functions
attributes #1 = { nounwind alwaysinline }          ; "volatile" functions, which may return different results on different ilwocations
attributes #2 = { nounwind readonly alwaysinline }
attributes #3 = { nounwind }

define linkonce_odr i8 @optix.ptx.cvt.u8.u32(i32 %p0) #0 {
    %p1 = trunc i32 %p0 to i8
    ret i8 %p1
}

define linkonce_odr i32 @optix.ptx.cvt.u32.u8(i8 %p0) #0 {
    %p1 = zext i8 %p0 to i32
    ret i32 %p1
}

define linkonce_odr i8 @optix.ptx.cvt.s8.s32(i32 %p0) #0 {
    %p1 = trunc i32 %p0 to i8
    ret i8 %p1
}

define linkonce_odr i64 @optix.ptx.cvt.u64.u8(i8 %p0) #0 {
    %p1 = zext i8 %p0 to i64
    ret i64 %p1
}

define linkonce_odr i64 @optix.ptx.cvt.u64.u32(i32 %p0) #0 {
    %p1 = zext i32 %p0 to i64
    ret i64 %p1
}

define linkonce_odr i32 @optix.ptx.cvt.s32.s16(i16 %p0) #0 {
    %p1 = sext i16 %p0 to i32
    ret i32 %p1
}

define linkonce_odr i64 @optix.ptx.cvt.u64.s32(i32 %p0) #0 {
    %p1 = sext i32 %p0 to i64
    ret i64 %p1
}

define linkonce_odr i32 @optix.ptx.cvt.u32.u64(i64 %p0) #0 {
    %p1 = trunc i64 %p0 to i32
    ret i32 %p1
}

define linkonce_odr i32 @optix.ptx.cvt.s32.u64(i64 %p0) #0 {
    %p1 = trunc i64 %p0 to i32
    ret i32 %p1
}

define linkonce_odr i64 @optix.ptx.cvt.s64.u32(i32 %p0) #0 {
    %p1 = zext i32 %p0 to i64
    ret i64 %p1
}

define linkonce_odr i64 @optix.ptx.cvt.s64.s32(i32 %p0) #0 {
    %p1 = sext i32 %p0 to i64
    ret i64 %p1
}

define linkonce_odr i32 @optix.ptx.cvt.u32.s64(i64 %p0) #0 {
    %p1 = trunc i64 %p0 to i32
    ret i32 %p1
}

define linkonce_odr i32 @optix.ptx.cvt.u32.u16(i16 %p0) #0 {
    %p1 = zext i16 %p0 to i32
    ret i32 %p1
}

define linkonce_odr i16 @optix.ptx.cvt.u16.u32(i32 %p0) #0 {
    %p1 = trunc i32 %p0 to i16
    ret i16 %p1
}

define linkonce_odr i32 @optix.ptx.cvt.s32.s64(i64 %p0) #0 {
    %p1 = trunc i64 %p0 to i32
    ret i32 %p1
}

define linkonce_odr i16 @optix.ptx.cvt.u16.u64(i64 %p0) #0 {
    %p1 = trunc i64 %p0 to i16
    ret i16 %p1
}

;
; mov
;


define linkonce_odr i1 @optix.ptx.mov.pred( i1 %p0 ) #0 {
    ret i1 %p0
}

define linkonce_odr i16 @optix.ptx.mov.u16( i16 %p0 ) #0 {
    ret i16 %p0
}

define linkonce_odr i16 @optix.ptx.mov.s16( i16 %p0 ) #0 {
    ret i16 %p0
}

define linkonce_odr i16 @optix.ptx.mov.b16( i16 %p0 ) #0 {
    ret i16 %p0
}

define linkonce_odr i32 @optix.ptx.mov.u32( i32 %p0 ) #0 {
    ret i32 %p0
}

define linkonce_odr i32 @optix.ptx.mov.s32( i32 %p0 ) #0 {
    ret i32 %p0
}

define linkonce_odr i32 @optix.ptx.mov.b32( i32 %p0 ) #0 {
    ret i32 %p0
}

define linkonce_odr float @optix.ptx.mov.f32( float %p0 ) #0 {
    ret float %p0
}

define linkonce_odr i64 @optix.ptx.mov.u64( i64 %p0 ) #0 {
    ret i64 %p0
}

define linkonce_odr i64 @optix.ptx.mov.s64( i64 %p0 ) #0 {
    ret i64 %p0
}

define linkonce_odr i64 @optix.ptx.mov.b64( i64 %p0 ) #0 {
    ret i64 %p0
}

define linkonce_odr double @optix.ptx.mov.f64( double %p0 ) #0 {
    ret double %p0
}

;
; add/sub/neg
;

define linkonce_odr i16 @optix.ptx.add.s16( i16 %p0, i16 %p1 ) #0 {
    %r = add i16 %p0, %p1
    ret i16 %r
}

define linkonce_odr i16 @optix.ptx.add.u16( i16 %p0, i16 %p1 ) #0 {
    %r = add i16 %p0, %p1
    ret i16 %r
}

define linkonce_odr i32 @optix.ptx.add.s32( i32 %p0, i32 %p1 ) #0 {
    %r = add i32 %p0, %p1
    ret i32 %r
}

define linkonce_odr i32 @optix.ptx.add.u32( i32 %p0, i32 %p1 ) #0 {
    %r = add i32 %p0, %p1
    ret i32 %r
}

define linkonce_odr i64 @optix.ptx.add.s64( i64 %p0, i64 %p1 ) #0 {
    %r = add i64 %p0, %p1
    ret i64 %r
}

define linkonce_odr i64 @optix.ptx.add.u64( i64 %p0, i64 %p1 ) #0 {
    %r = add i64 %p0, %p1
    ret i64 %r
}

define linkonce_odr i16 @optix.ptx.sub.s16( i16 %p0, i16 %p1 ) #0 {
    %r = sub i16 %p0, %p1
    ret i16 %r
}

define linkonce_odr i16 @optix.ptx.sub.u16( i16 %p0, i16 %p1 ) #0 {
    %r = sub i16 %p0, %p1
    ret i16 %r
}

define linkonce_odr i32 @optix.ptx.sub.s32( i32 %p0, i32 %p1 ) #0 {
    %r = sub i32 %p0, %p1
    ret i32 %r
}

define linkonce_odr i32 @optix.ptx.sub.u32( i32 %p0, i32 %p1 ) #0 {
    %r = sub i32 %p0, %p1
    ret i32 %r
}

define linkonce_odr i64 @optix.ptx.sub.s64( i64 %p0, i64 %p1 ) #0 {
    %r = sub i64 %p0, %p1
    ret i64 %r
}

define linkonce_odr i64 @optix.ptx.sub.u64( i64 %p0, i64 %p1 ) #0 {
    %r = sub i64 %p0, %p1
    ret i64 %r
}

define linkonce_odr i16 @optix.ptx.neg.s16( i16 %p0 ) #0 {
    %r = sub i16 0, %p0
    ret i16 %r
}

define linkonce_odr i32 @optix.ptx.neg.s32( i32 %p0 ) #0 {
    %r = sub i32 0, %p0
    ret i32 %r
}

define linkonce_odr i64 @optix.ptx.neg.s64( i64 %p0 ) #0 {
    %r = sub i64 0, %p0
    ret i64 %r
}

;
; min/max
;

define linkonce_odr i16 @optix.ptx.min.u16( i16 %p0, i16 %p1 ) #0 {
    %lt = icmp ult i16 %p0, %p1
    %r = select i1 %lt, i16 %p0, i16 %p1
    ret i16 %r
}

define linkonce_odr i16 @optix.ptx.min.s16( i16 %p0, i16 %p1 ) #0 {
    %lt = icmp slt i16 %p0, %p1
    %r = select i1 %lt, i16 %p0, i16 %p1
    ret i16 %r
}

define linkonce_odr i16 @optix.ptx.max.u16( i16 %p0, i16 %p1 ) #0 {
    %gt = icmp ugt i16 %p0, %p1
    %r = select i1 %gt, i16 %p0, i16 %p1
    ret i16 %r
}

define linkonce_odr i16 @optix.ptx.max.s16( i16 %p0, i16 %p1 ) #0 {
    %gt = icmp sgt i16 %p0, %p1
    %r = select i1 %gt, i16 %p0, i16 %p1
    ret i16 %r
}

define linkonce_odr i32 @optix.ptx.min.u32( i32 %p0, i32 %p1 ) #0 {
    %lt = icmp ult i32 %p0, %p1
    %r = select i1 %lt, i32 %p0, i32 %p1
    ret i32 %r
}

define linkonce_odr i32 @optix.ptx.min.s32( i32 %p0, i32 %p1 ) #0 {
    %lt = icmp slt i32 %p0, %p1
    %r = select i1 %lt, i32 %p0, i32 %p1
    ret i32 %r
}

define linkonce_odr i32 @optix.ptx.max.u32( i32 %p0, i32 %p1 ) #0 {
    %gt = icmp ugt i32 %p0, %p1
    %r = select i1 %gt, i32 %p0, i32 %p1
    ret i32 %r
}

define linkonce_odr i32 @optix.ptx.max.s32( i32 %p0, i32 %p1 ) #0 {
    %gt = icmp sgt i32 %p0, %p1
    %r = select i1 %gt, i32 %p0, i32 %p1
    ret i32 %r
}

define linkonce_odr i64 @optix.ptx.min.u64( i64 %p0, i64 %p1 ) #0 {
    %lt = icmp ult i64 %p0, %p1
    %r = select i1 %lt, i64 %p0, i64 %p1
    ret i64 %r
}

define linkonce_odr i64 @optix.ptx.min.s64( i64 %p0, i64 %p1 ) #0 {
    %lt = icmp slt i64 %p0, %p1
    %r = select i1 %lt, i64 %p0, i64 %p1
    ret i64 %r
}

define linkonce_odr i64 @optix.ptx.max.u64( i64 %p0, i64 %p1 ) #0 {
    %gt = icmp ugt i64 %p0, %p1
    %r = select i1 %gt, i64 %p0, i64 %p1
    ret i64 %r
}

define linkonce_odr i64 @optix.ptx.max.s64( i64 %p0, i64 %p1 ) #0 {
    %gt = icmp sgt i64 %p0, %p1
    %r = select i1 %gt, i64 %p0, i64 %p1
    ret i64 %r
}

;
; multiply
;

; 24 bit multiply
define linkonce_odr i32 @optix.ptx.mul24.lo.u32( i32 %p0, i32 %p1 ) #0 {
    ; Zero out top 8 bits of  parameters.
    %a = and i32 %p0, 16777215
    %b = and i32 %p1, 16777215
    %res = mul i32 %a, %b
    ; Zero out top 8 bits of result
    %ret = and i32 %res, 16777215
    ret i32 %ret
}

define linkonce_odr i16 @optix.ptx.mul.lo.u16( i16 %p0, i16 %p1 ) #0 {
    %r = mul i16 %p0, %p1
    ret i16 %r
}

define linkonce_odr i16 @optix.ptx.mul.lo.s16( i16 %p0, i16 %p1 ) #0 {
    %r = mul i16 %p0, %p1
    ret i16 %r
}

define linkonce_odr i32 @optix.ptx.mul.lo.u32( i32 %p0, i32 %p1 ) #0 {
    %r = mul i32 %p0, %p1
    ret i32 %r
}

define linkonce_odr i32 @optix.ptx.mul.lo.s32( i32 %p0, i32 %p1 ) #0 {
    %r = mul i32 %p0, %p1
    ret i32 %r
}

define linkonce_odr i64 @optix.ptx.mul.lo.u64( i64 %p0, i64 %p1 ) #0 {
    %r = mul i64 %p0, %p1
    ret i64 %r
}

define linkonce_odr i64 @optix.ptx.mul.lo.s64( i64 %p0, i64 %p1 ) #0 {
    %r = mul i64 %p0, %p1
    ret i64 %r
}

define linkonce_odr i64 @optix.ptx.mul.wide.u32( i32 %p0, i32 %p1 ) #0 {
    %s0 = zext i32 %p0 to i64
    %s1 = zext i32 %p1 to i64
    %r = mul i64 %s0, %s1
    ret i64 %r
}

define linkonce_odr i64 @optix.ptx.mul.wide.s32( i32 %p0, i32 %p1 ) #0 {
    %s0 = sext i32 %p0 to i64
    %s1 = sext i32 %p1 to i64
    %r = mul i64 %s0, %s1
    ret i64 %r
}

;
; mad
;

define linkonce_odr i32 @optix.ptx.mad.wide.u16( i16 %a, i16 %b, i32 %c ) #0 {
  %aExtended = zext i16 %a to i32
  %bExtended = zext i16 %b to i32

  %product = mul i32 %aExtended, %bExtended
  %sum = add i32 %product, %c

  ret i32 %sum
}

define linkonce_odr i32 @optix.ptx.mad.wide.s16( i16 %a, i16 %b, i32 %c ) #0 {
  %aExtended = sext i16 %a to i32
  %bExtended = sext i16 %b to i32

  %product = mul i32 %aExtended, %bExtended
  %sum = add i32 %product, %c

  ret i32 %sum
}

;
; divide
;

define linkonce_odr i16 @optix.ptx.div.s16( i16 %p0, i16 %p1 ) #0 {
    %r0 = sdiv i16 %p0, %p1
    ret i16 %r0
}

define linkonce_odr i16 @optix.ptx.div.u16( i16 %p0, i16 %p1 ) #0 {
    %r0 = udiv i16 %p0, %p1
    ret i16 %r0
}

define linkonce_odr i32 @optix.ptx.div.s32( i32 %p0, i32 %p1 ) #0 {
    %r0 = sdiv i32 %p0, %p1
    ret i32 %r0
}

define linkonce_odr i32 @optix.ptx.div.u32( i32 %p0, i32 %p1 ) #0 {
    %r0 = udiv i32 %p0, %p1
    ret i32 %r0
}

define linkonce_odr i64 @optix.ptx.div.s64( i64 %p0, i64 %p1 ) #0 {
    %r0 = sdiv i64 %p0, %p1
    ret i64 %r0
}

define linkonce_odr i64 @optix.ptx.div.u64( i64 %p0, i64 %p1 ) #0 {
    %r0 = udiv i64 %p0, %p1
    ret i64 %r0
}

;
; remainder
;

define linkonce_odr i16 @optix.ptx.rem.u16( i16 %p0, i16 %p1 ) #0 {
    %rem = urem i16 %p0, %p1
    ret i16 %rem
}

define linkonce_odr i16 @optix.ptx.rem.s16( i16 %p0, i16 %p1 ) #0 {
    %rem = srem i16 %p0, %p1
    ret i16 %rem
}

define linkonce_odr i32 @optix.ptx.rem.u32( i32 %p0, i32 %p1 ) #0 {
    %rem = urem i32 %p0, %p1
    ret i32 %rem
}

define linkonce_odr i32 @optix.ptx.rem.s32( i32 %p0, i32 %p1 ) #0 {
    %rem = srem i32 %p0, %p1
    ret i32 %rem
}

define linkonce_odr i64 @optix.ptx.rem.u64( i64 %p0, i64 %p1 ) #0 {
    %rem = urem i64 %p0, %p1
    ret i64 %rem
}

define linkonce_odr i64 @optix.ptx.rem.s64( i64 %p0, i64 %p1 ) #0 {
    %rem = srem i64 %p0, %p1
    ret i64 %rem
}


;
; lops/shifts
;

define linkonce_odr i1 @optix.ptx.or.pred(i1 %a, i1 %b) #0
{
  %r = or i1 %a, %b
  ret i1 %r
}
  
define linkonce_odr i1 @optix.ptx.and.pred(i1 %a, i1 %b) #0
{
  %r = and i1 %a, %b
  ret i1 %r
}
  
define linkonce_odr i1 @optix.ptx.xor.pred(i1 %a, i1 %b) #0
{
  %r = xor i1 %a, %b
  ret i1 %r
}

define linkonce_odr i1 @optix.ptx.not.pred( i1 %a ) #0
{
  %r = xor i1 %a, true
  ret i1 %r
}

define linkonce_odr i16 @optix.ptx.and.b16( i16 %p0, i16 %p1 ) #0 {
    %r = and i16 %p0, %p1
    ret i16 %r
}

define linkonce_odr i16 @optix.ptx.or.b16( i16 %p0, i16 %p1 ) #0 {
    %r = or i16 %p0, %p1
    ret i16 %r
}

define linkonce_odr i16 @optix.ptx.xor.b16( i16 %p0, i16 %p1 ) #0 {
    %r = xor i16 %p0, %p1
    ret i16 %r
}

define linkonce_odr i16 @optix.ptx.not.b16( i16 %p0 ) #0 {
    %r = xor i16 %p0, -1
    ret i16 %r
}
  
define linkonce_odr i32 @optix.ptx.and.b32( i32 %p0, i32 %p1 ) #0 {
    %r = and i32 %p0, %p1
    ret i32 %r
}

define linkonce_odr i32 @optix.ptx.or.b32( i32 %p0, i32 %p1 ) #0 {
    %r = or i32 %p0, %p1
    ret i32 %r
}

define linkonce_odr i32 @optix.ptx.xor.b32( i32 %p0, i32 %p1 ) #0 {
    %r = xor i32 %p0, %p1
    ret i32 %r
}

define linkonce_odr i32 @optix.ptx.not.b32( i32 %p0 ) #0 {
    %r = xor i32 %p0, -1
    ret i32 %r
}

define linkonce_odr i64 @optix.ptx.and.b64( i64 %p0, i64 %p1 ) #0 {
    %r = and i64 %p0, %p1
    ret i64 %r
}

define linkonce_odr i64 @optix.ptx.or.b64( i64 %p0, i64 %p1 ) #0 {
    %r = or i64 %p0, %p1
    ret i64 %r
}

define linkonce_odr i64 @optix.ptx.xor.b64( i64 %p0, i64 %p1 ) #0 {
    %r = xor i64 %p0, %p1
    ret i64 %r
}

define linkonce_odr i64 @optix.ptx.not.b64( i64 %p0 ) #0 {
    %r = xor i64 %p0, -1
    ret i64 %r
}

define linkonce_odr i16 @optix.ptx.shr.b16( i16 %p0, i32 %p1 ) #0 {
  %t = trunc i32 %p1 to i16
  %r = lshr i16 %p0, %t
  ret i16 %r
}

define linkonce_odr i16 @optix.ptx.shr.u16( i16 %p0, i32 %p1 ) #0 {
  %t = trunc i32 %p1 to i16
  %r = lshr i16 %p0, %t
  ret i16 %r
}

; LLVM returns a poison value (0) when the shift amount is greater than or
; equal to the data width, while PTX clamps the shift amount instead. In
; practice, this means PTX returns -1 when the input value is negative and the
; shift amount is clamped, while LLVM returns 0 in the same case.
;
; To correct for this, we must add clamp emulation to all signed
; implementations of shr. When the shift amount is >= the data width and the
; input is negative, we must return -1.

define linkonce_odr i16 @optix.ptx.shr.s16( i16 %p0, i32 %p1 ) #0 {
  %t = trunc i32 %p1 to i16

  %shouldClamp = icmp uge i16 %t, 16
  %isNegative = icmp slt i16 %p0, 0
  %retNegativeOne = and i1 %shouldClamp, %isNegative

  %r = ashr i16 %p0, %t

  %retVal = select i1 %retNegativeOne, i16 -1, i16 %r

  ret i16 %retVal
}

define linkonce_odr i32 @optix.ptx.shr.b32( i32 %p0, i32 %p1 ) #0 {
  %r = lshr i32 %p0, %p1
  ret i32 %r
}

define linkonce_odr i32 @optix.ptx.shr.u32( i32 %p0, i32 %p1 ) #0 {
  %r = lshr i32 %p0, %p1
  ret i32 %r
}

define linkonce_odr i32 @optix.ptx.shr.s32( i32 %p0, i32 %p1 ) #0 {
  %shouldClamp = icmp uge i32 %p1, 32
  %isNegative = icmp slt i32 %p0, 0
  %retNegativeOne = and i1 %shouldClamp, %isNegative

  %r = ashr i32 %p0, %p1

  %retVal = select i1 %retNegativeOne, i32 -1, i32 %r

  ret i32 %retVal
}

define linkonce_odr i64 @optix.ptx.shr.b64( i64 %p0, i32 %p1 ) #0 {
  %s = zext i32 %p1 to i64
  %r = lshr i64 %p0, %s
  ret i64 %r
}

define linkonce_odr i64 @optix.ptx.shr.u64( i64 %p0, i32 %p1 ) #0 {
  %s = zext i32 %p1 to i64
  %r = lshr i64 %p0, %s
  ret i64 %r
}

define linkonce_odr i64 @optix.ptx.shr.s64( i64 %p0, i32 %p1 ) #0 {
  %s = zext i32 %p1 to i64

  %shouldClamp = icmp uge i64 %s, 64
  %isNegative = icmp slt i64 %p0, 0
  %retNegativeOne = and i1 %shouldClamp, %isNegative

  %r = ashr i64 %p0, %s

  %retVal = select i1 %retNegativeOne, i64 -1, i64 %r

  ret i64 %retVal
}

define linkonce_odr i16 @optix.ptx.shl.b16( i16 %p0, i32 %p1 ) #0 {
  %t = trunc i32 %p1 to i16
  %r = shl i16 %p0, %t
  ret i16 %r
}

define linkonce_odr i16 @optix.ptx.shl.u16( i16 %p0, i32 %p1 ) #0 {
  %t = trunc i32 %p1 to i16
  %r = shl i16 %p0, %t
  ret i16 %r
}

define linkonce_odr i16 @optix.ptx.shl.s16( i16 %p0, i32 %p1 ) #0 {
  %t = trunc i32 %p1 to i16
  %r = shl i16 %p0, %t
  ret i16 %r
}

define linkonce_odr i32 @optix.ptx.shl.b32( i32 %p0, i32 %p1 ) #0 {
  %r = shl i32 %p0, %p1
  ret i32 %r
}

define linkonce_odr i32 @optix.ptx.shl.u32( i32 %p0, i32 %p1 ) #0 {
  %r = shl i32 %p0, %p1
  ret i32 %r
}

define linkonce_odr i32 @optix.ptx.shl.s32( i32 %p0, i32 %p1 ) #0 {
  %r = shl i32 %p0, %p1
  ret i32 %r
}

define linkonce_odr i64 @optix.ptx.shl.b64( i64 %p0, i32 %p1 ) #0 {
  %s = zext i32 %p1 to i64
  %r = shl i64 %p0, %s
  ret i64 %r
}

define linkonce_odr i64 @optix.ptx.shl.u64( i64 %p0, i32 %p1 ) #0 {
  %s = zext i32 %p1 to i64
  %r = shl i64 %p0, %s
  ret i64 %r
}

define linkonce_odr i64 @optix.ptx.shl.s64( i64 %p0, i32 %p1 ) #0 {
  %s = zext i32 %p1 to i64
  %r = shl i64 %p0, %s
  ret i64 %r
}

;
; selp
;

define linkonce_odr float @optix.ptx.selp.f32(float %a, float %b, i1 %c) #0
{
  %r = select i1 %c, float %a, float %b
  ret float %r
}

define linkonce_odr i16 @optix.ptx.selp.b16(i16 %a, i16 %b, i1 %c) #0
{
  %r = select i1 %c, i16 %a, i16 %b
  ret i16 %r  
}

define linkonce_odr i16 @optix.ptx.selp.u16(i16 %a, i16 %b, i1 %c) #0
{
  %r = select i1 %c, i16 %a, i16 %b
  ret i16 %r  
}

define linkonce_odr i16 @optix.ptx.selp.s16(i16 %a, i16 %b, i1 %c) #0
{
  %r = select i1 %c, i16 %a, i16 %b
  ret i16 %r  
}

define linkonce_odr i32 @optix.ptx.selp.b32(i32 %a, i32 %b, i1 %c) #0
{
  %r = select i1 %c, i32 %a, i32 %b
  ret i32 %r  
}

define linkonce_odr i32 @optix.ptx.selp.u32(i32 %a, i32 %b, i1 %c) #0
{
  %r = select i1 %c, i32 %a, i32 %b
  ret i32 %r  
}

define linkonce_odr i32 @optix.ptx.selp.s32(i32 %a, i32 %b, i1 %c) #0
{
  %r = select i1 %c, i32 %a, i32 %b
  ret i32 %r  
}

define linkonce_odr i64 @optix.ptx.selp.b64(i64 %a, i64 %b, i1 %c) #0
{
  %r = select i1 %c, i64 %a, i64 %b
  ret i64 %r  
}

define linkonce_odr i64 @optix.ptx.selp.u64(i64 %a, i64 %b, i1 %c) #0
{
  %r = select i1 %c, i64 %a, i64 %b
  ret i64 %r  
}

define linkonce_odr i64 @optix.ptx.selp.s64(i64 %a, i64 %b, i1 %c) #0
{
  %r = select i1 %c, i64 %a, i64 %b
  ret i64 %r  
}

;
; count leading zeroes
;

declare i32 @llvm.ctlz.i32( i32 %c, i1 %iszeroundef ) #0
define linkonce_odr i32 @optix.ptx.clz.b32( i32 %c ) #0 {
    %r0 = call i32 @llvm.ctlz.i32( i32 %c, i1 false )
    ret i32 %r0
}

declare i64 @llvm.ctlz.i64( i64 %c, i1 %iszeroundef ) #0
define linkonce_odr i32 @optix.ptx.clz.b64( i64 %c ) #0 {
    %r0 = call i64 @llvm.ctlz.i64( i64 %c, i1 false )
    %r1 = trunc i64 %r0 to i32
    ret i32 %r1
}

;
; Set
;

define linkonce_odr i32 @optix.ptx.set.eq.ftz.s32.f32( float %p0, float %p1 ) #0 {
   %pred = fcmp oeq float %p0, %p1
   %r = select i1 %pred, i32 4294967295, i32 0
   ret i32 %r
}

define linkonce_odr i32 @optix.ptx.set.ge.ftz.s32.f32( float %p0, float %p1 ) #0 {
   %pred = fcmp oge float %p0, %p1
   %r = select i1 %pred, i32 4294967295, i32 0
   ret i32 %r
}

define linkonce_odr i32 @optix.ptx.set.gt.ftz.s32.f32( float %p0, float %p1 ) #0 {
   %pred = fcmp ogt float %p0, %p1
   %r = select i1 %pred, i32 4294967295, i32 0
   ret i32 %r
}

define linkonce_odr i32 @optix.ptx.set.lt.ftz.s32.f32( float %p0, float %p1 ) #0 {
   %pred = fcmp olt float %p0, %p1
   %r = select i1 %pred, i32 4294967295, i32 0
   ret i32 %r
}

define linkonce_odr i32 @optix.ptx.set.eq.ftz.u32.f32( float %p0, float %p1 ) #0 {
   %pred = fcmp oeq float %p0, %p1
   %r = select i1 %pred, i32 4294967295, i32 0
   ret i32 %r
}

define linkonce_odr i32 @optix.ptx.set.neu.ftz.u32.f32( float %p0, float %p1 ) #0 {
   %pred = fcmp une float %p0, %p1
   %r = select i1 %pred, i32 4294967295, i32 0
   ret i32 %r
}

define linkonce_odr i32 @optix.ptx.set.ge.ftz.u32.f32( float %p0, float %p1 ) #0 {
   %pred = fcmp oge float %p0, %p1
   %r = select i1 %pred, i32 4294967295, i32 0
   ret i32 %r
}

define linkonce_odr i32 @optix.ptx.set.gt.ftz.u32.f32( float %p0, float %p1 ) #0 {
   %pred = fcmp ogt float %p0, %p1
   %r = select i1 %pred, i32 4294967295, i32 0
   ret i32 %r
}

define linkonce_odr i32 @optix.ptx.set.le.ftz.u32.f32( float %p0, float %p1 ) #0 {
   %pred = fcmp ole float %p0, %p1
   %r = select i1 %pred, i32 4294967295, i32 0
   ret i32 %r
}

define linkonce_odr i32 @optix.ptx.set.lt.ftz.u32.f32( float %p0, float %p1 ) #0 {
   %pred = fcmp olt float %p0, %p1
   %r = select i1 %pred, i32 4294967295, i32 0
   ret i32 %r
}

define linkonce_odr i32 @optix.ptx.set.eq.s32.s32( i32 %p0, i32 %p1 ) #0 {
   %pred = icmp eq i32 %p0, %p1
   %r = select i1 %pred, i32 4294967295, i32 0
   ret i32 %r
}

define linkonce_odr i32 @optix.ptx.set.ne.s32.s32( i32 %p0, i32 %p1 ) #0 {
   %pred = icmp ne i32 %p0, %p1
   %r = select i1 %pred, i32 4294967295, i32 0
   ret i32 %r
}

define linkonce_odr i32 @optix.ptx.set.ge.s32.s32( i32 %p0, i32 %p1 ) #0 {
   %pred = icmp sge i32 %p0, %p1
   %r = select i1 %pred, i32 4294967295, i32 0
   ret i32 %r
}

define linkonce_odr i32 @optix.ptx.set.gt.s32.s32( i32 %p0, i32 %p1 ) #0 {
   %pred = icmp sgt i32 %p0, %p1
   %r = select i1 %pred, i32 4294967295, i32 0
   ret i32 %r
}

define linkonce_odr i32 @optix.ptx.set.lt.s32.s32( i32 %p0, i32 %p1 ) #0 {
   %pred = icmp slt i32 %p0, %p1
   %r = select i1 %pred, i32 4294967295, i32 0
   ret i32 %r
}

define linkonce_odr i32 @optix.ptx.set.eq.u32.u32( i32 %p0, i32 %p1 ) #0 {
   %pred = icmp eq i32 %p0, %p1
   %r = select i1 %pred, i32 4294967295, i32 0
   ret i32 %r
}

define linkonce_odr i32 @optix.ptx.set.ne.u32.u32( i32 %p0, i32 %p1 ) #0 {
   %pred = icmp ne i32 %p0, %p1
   %r = select i1 %pred, i32 4294967295, i32 0
   ret i32 %r
}

define linkonce_odr i32 @optix.ptx.set.le.u32.u32( i32 %p0, i32 %p1 ) #0 {
   %pred = icmp ule i32 %p0, %p1
   %r = select i1 %pred, i32 4294967295, i32 0
   ret i32 %r
}

define linkonce_odr i32 @optix.ptx.set.gt.u32.u32( i32 %p0, i32 %p1 ) #0 {
   %pred = icmp ugt i32 %p0, %p1
   %r = select i1 %pred, i32 4294967295, i32 0
   ret i32 %r
}

define linkonce_odr i32 @optix.ptx.set.eq.u32.s32( i32 %p0, i32 %p1 ) #0 {
   %pred = icmp eq i32 %p0, %p1
   %r = select i1 %pred, i32 4294967295, i32 0
   ret i32 %r
}

define linkonce_odr i32 @optix.ptx.set.ne.u32.s32( i32 %p0, i32 %p1 ) #0 {
   %pred = icmp ne i32 %p0, %p1
   %r = select i1 %pred, i32 4294967295, i32 0
   ret i32 %r
}

define linkonce_odr i32 @optix.ptx.set.le.u32.s32( i32 %p0, i32 %p1 ) #0 {
   %pred = icmp sle i32 %p0, %p1
   %r = select i1 %pred, i32 4294967295, i32 0
   ret i32 %r
}

define linkonce_odr i32 @optix.ptx.set.lt.u32.s32( i32 %p0, i32 %p1 ) #0 {
   %pred = icmp slt i32 %p0, %p1
   %r = select i1 %pred, i32 4294967295, i32 0
   ret i32 %r
}

define linkonce_odr i32 @optix.ptx.set.ge.u32.s32( i32 %p0, i32 %p1 ) #0 {
   %pred = icmp sge i32 %p0, %p1
   %r = select i1 %pred, i32 4294967295, i32 0
   ret i32 %r
}

define linkonce_odr i32 @optix.ptx.set.gt.u32.s32( i32 %p0, i32 %p1 ) #0 {
   %pred = icmp sgt i32 %p0, %p1
   %r = select i1 %pred, i32 4294967295, i32 0
   ret i32 %r
}


define linkonce_odr i32 @optix.ptx.set.eq.u32.u64( i64 %p0, i64 %p1 ) #0 {
   %pred = icmp eq i64 %p0, %p1
   %r = select i1 %pred, i32 4294967295, i32 0
   ret i32 %r
}

define linkonce_odr i32 @optix.ptx.set.le.u32.u64( i64 %p0, i64 %p1 ) #0 {
   %pred = icmp ule i64 %p0, %p1
   %r = select i1 %pred, i32 4294967295, i32 0
   ret i32 %r
}

define linkonce_odr i32 @optix.ptx.set.gt.u32.u64( i64 %p0, i64 %p1 ) #0 {
   %pred = icmp ugt i64 %p0, %p1
   %r = select i1 %pred, i32 4294967295, i32 0
   ret i32 %r
}

;
; slct
;

define linkonce_odr float @optix.ptx.slct.f32.s32( float %p0, float %p1, i32 %p3 ) #0 {
   %pred = icmp sge i32 %p3, 0
   %ret = select i1 %pred, float %p0, float %p1
   ret float %ret
}

define linkonce_odr i32 @optix.ptx.slct.s32.s32( i32 %p0, i32 %p1, i32 %p3 ) #0 {
   %pred = icmp sge i32 %p3, 0
   %ret = select i1 %pred, i32 %p0, i32 %p1
   ret i32 %ret
}

define linkonce_odr i32 @optix.ptx.slct.u32.s32( i32 %p0, i32 %p1, i32 %p3 ) #0 {
   %pred = icmp uge i32 %p3, 0
   %ret = select i1 %pred, i32 %p0, i32 %p1
   ret i32 %ret
}

;
; setp
;

define linkonce_odr {i1, i1} @optix.ptx.setp.eq.b16( i16 %p0, i16 %p1 ) #0 {
   %pred = icmp eq i16 %p0, %p1
   %v0 = insertvalue {i1, i1} undef, i1 %pred, 0
   %not = xor i1 %pred, true
   %v1 = insertvalue {i1, i1} %v0, i1 %not, 1
   ret {i1, i1} %v1
}

define linkonce_odr {i1, i1} @optix.ptx.setp.eq.u16( i16 %p0, i16 %p1 ) #0 {
   %pred = icmp eq i16 %p0, %p1
   %v0 = insertvalue {i1, i1} undef, i1 %pred, 0
   %not = xor i1 %pred, true
   %v1 = insertvalue {i1, i1} %v0, i1 %not, 1
   ret {i1, i1} %v1
}

define linkonce_odr {i1, i1} @optix.ptx.setp.ne.b16( i16 %p0, i16 %p1 ) #0 {
   %pred = icmp ne i16 %p0, %p1
   %v0 = insertvalue {i1, i1} undef, i1 %pred, 0
   %not = xor i1 %pred, true
   %v1 = insertvalue {i1, i1} %v0, i1 %not, 1
   ret {i1, i1} %v1
}

define linkonce_odr {i1, i1} @optix.ptx.setp.ne.u16( i16 %p0, i16 %p1 ) #0 {
   %pred = icmp ne i16 %p0, %p1
   %v0 = insertvalue {i1, i1} undef, i1 %pred, 0
   %not = xor i1 %pred, true
   %v1 = insertvalue {i1, i1} %v0, i1 %not, 1
   ret {i1, i1} %v1
}

define linkonce_odr {i1, i1} @optix.ptx.setp.ne.s16( i16 %p0, i16 %p1 ) #0 {
   %pred = icmp ne i16 %p0, %p1
   %v0 = insertvalue {i1, i1} undef, i1 %pred, 0
   %not = xor i1 %pred, true
   %v1 = insertvalue {i1, i1} %v0, i1 %not, 1
   ret {i1, i1} %v1
}

define linkonce_odr {i1, i1} @optix.ptx.setp.lt.u16( i16 %p0, i16 %p1 ) #0 {
   %pred = icmp ult i16 %p0, %p1
   %v0 = insertvalue {i1, i1} undef, i1 %pred, 0
   %not = xor i1 %pred, true
   %v1 = insertvalue {i1, i1} %v0, i1 %not, 1
   ret {i1, i1} %v1
}

define linkonce_odr {i1, i1} @optix.ptx.setp.lt.s16( i16 %p0, i16 %p1 ) #0 {
   %pred = icmp slt i16 %p0, %p1
   %v0 = insertvalue {i1, i1} undef, i1 %pred, 0
   %not = xor i1 %pred, true
   %v1 = insertvalue {i1, i1} %v0, i1 %not, 1
   ret {i1, i1} %v1
}

define linkonce_odr {i1, i1} @optix.ptx.setp.gt.u16( i16 %p0, i16 %p1 ) #0 {
   %pred = icmp ugt i16 %p0, %p1
   %v0 = insertvalue {i1, i1} undef, i1 %pred, 0
   %not = xor i1 %pred, true
   %v1 = insertvalue {i1, i1} %v0, i1 %not, 1
   ret {i1, i1} %v1
}

define linkonce_odr {i1, i1} @optix.ptx.setp.gt.s16( i16 %p0, i16 %p1 ) #0 {
   %pred = icmp sgt i16 %p0, %p1
   %v0 = insertvalue {i1, i1} undef, i1 %pred, 0
   %not = xor i1 %pred, true
   %v1 = insertvalue {i1, i1} %v0, i1 %not, 1
   ret {i1, i1} %v1
}

define linkonce_odr {i1, i1} @optix.ptx.setp.ge.s16( i16 %p0, i16 %p1 ) #0 {
   %pred = icmp sge i16 %p0, %p1
   %v0 = insertvalue {i1, i1} undef, i1 %pred, 0
   %not = xor i1 %pred, true
   %v1 = insertvalue {i1, i1} %v0, i1 %not, 1
   ret {i1, i1} %v1
}

define linkonce_odr {i1, i1} @optix.ptx.setp.eq.b32( i32 %p0, i32 %p1 ) #0 {
   %pred = icmp eq i32 %p0, %p1
   %v0 = insertvalue {i1, i1} undef, i1 %pred, 0
   %not = xor i1 %pred, true
   %v1 = insertvalue {i1, i1} %v0, i1 %not, 1
   ret {i1, i1} %v1
}

define linkonce_odr {i1, i1} @optix.ptx.setp.eq.u32( i32 %p0, i32 %p1 ) #0 {
   %pred = icmp eq i32 %p0, %p1
   %v0 = insertvalue {i1, i1} undef, i1 %pred, 0
   %not = xor i1 %pred, true
   %v1 = insertvalue {i1, i1} %v0, i1 %not, 1
   ret {i1, i1} %v1
}

define linkonce_odr {i1, i1} @optix.ptx.setp.eq.s16( i16 %p0, i16 %p1 ) #0 {
   %pred = icmp eq i16 %p0, %p1
   %v0 = insertvalue {i1, i1} undef, i1 %pred, 0
   %not = xor i1 %pred, true
   %v1 = insertvalue {i1, i1} %v0, i1 %not, 1
   ret {i1, i1} %v1
}

define linkonce_odr {i1, i1} @optix.ptx.setp.eq.s32( i32 %p0, i32 %p1 ) #0 {
   %pred = icmp eq i32 %p0, %p1
   %v0 = insertvalue {i1, i1} undef, i1 %pred, 0
   %not = xor i1 %pred, true
   %v1 = insertvalue {i1, i1} %v0, i1 %not, 1
   ret {i1, i1} %v1
}

define linkonce_odr {i1, i1} @optix.ptx.setp.ne.b32( i32 %p0, i32 %p1 ) #0 {
   %pred = icmp ne i32 %p0, %p1
   %v0 = insertvalue {i1, i1} undef, i1 %pred, 0
   %not = xor i1 %pred, true
   %v1 = insertvalue {i1, i1} %v0, i1 %not, 1
   ret {i1, i1} %v1
}

define linkonce_odr {i1, i1} @optix.ptx.setp.ne.u32( i32 %p0, i32 %p1 ) #0 {
   %pred = icmp ne i32 %p0, %p1
   %v0 = insertvalue {i1, i1} undef, i1 %pred, 0
   %not = xor i1 %pred, true
   %v1 = insertvalue {i1, i1} %v0, i1 %not, 1
   ret {i1, i1} %v1
}

define linkonce_odr {i1, i1} @optix.ptx.setp.ne.s32( i32 %p0, i32 %p1 ) #0 {
   %pred = icmp ne i32 %p0, %p1
   %v0 = insertvalue {i1, i1} undef, i1 %pred, 0
   %not = xor i1 %pred, true
   %v1 = insertvalue {i1, i1} %v0, i1 %not, 1
   ret {i1, i1} %v1
}

define linkonce_odr {i1, i1} @optix.ptx.setp.le.u32( i32 %p0, i32 %p1 ) #0 {
   %pred = icmp ule i32 %p0, %p1
   %v0 = insertvalue {i1, i1} undef, i1 %pred, 0
   %not = xor i1 %pred, true
   %v1 = insertvalue {i1, i1} %v0, i1 %not, 1
   ret {i1, i1} %v1
}

define linkonce_odr {i1, i1} @optix.ptx.setp.le.s32( i32 %p0, i32 %p1 ) #0 {
   %pred = icmp sle i32 %p0, %p1
   %v0 = insertvalue {i1, i1} undef, i1 %pred, 0
   %not = xor i1 %pred, true
   %v1 = insertvalue {i1, i1} %v0, i1 %not, 1
   ret {i1, i1} %v1
}

define linkonce_odr {i1, i1} @optix.ptx.setp.lt.u32( i32 %p0, i32 %p1 ) #0 {
   %pred = icmp ult i32 %p0, %p1
   %v0 = insertvalue {i1, i1} undef, i1 %pred, 0
   %not = xor i1 %pred, true
   %v1 = insertvalue {i1, i1} %v0, i1 %not, 1
   ret {i1, i1} %v1
}

define linkonce_odr {i1, i1} @optix.ptx.setp.lt.s32( i32 %p0, i32 %p1 ) #0 {
   %pred = icmp slt i32 %p0, %p1
   %v0 = insertvalue {i1, i1} undef, i1 %pred, 0
   %not = xor i1 %pred, true
   %v1 = insertvalue {i1, i1} %v0, i1 %not, 1
   ret {i1, i1} %v1
}

define linkonce_odr {i1, i1} @optix.ptx.setp.ge.u32( i32 %p0, i32 %p1 ) #0 {
   %pred = icmp uge i32 %p0, %p1
   %v0 = insertvalue {i1, i1} undef, i1 %pred, 0
   %not = xor i1 %pred, true
   %v1 = insertvalue {i1, i1} %v0, i1 %not, 1
   ret {i1, i1} %v1
}

define linkonce_odr {i1, i1} @optix.ptx.setp.ge.s32( i32 %p0, i32 %p1 ) #0 {
   %pred = icmp sge i32 %p0, %p1
   %v0 = insertvalue {i1, i1} undef, i1 %pred, 0
   %not = xor i1 %pred, true
   %v1 = insertvalue {i1, i1} %v0, i1 %not, 1
   ret {i1, i1} %v1
}

define linkonce_odr {i1, i1} @optix.ptx.setp.gt.u32( i32 %p0, i32 %p1 ) #0 {
   %pred = icmp ugt i32 %p0, %p1
   %v0 = insertvalue {i1, i1} undef, i1 %pred, 0
   %not = xor i1 %pred, true
   %v1 = insertvalue {i1, i1} %v0, i1 %not, 1
   ret {i1, i1} %v1
}

define linkonce_odr {i1, i1} @optix.ptx.setp.gt.s32( i32 %p0, i32 %p1 ) #0 {
   %pred = icmp sgt i32 %p0, %p1
   %v0 = insertvalue {i1, i1} undef, i1 %pred, 0
   %not = xor i1 %pred, true
   %v1 = insertvalue {i1, i1} %v0, i1 %not, 1
   ret {i1, i1} %v1
}

define linkonce_odr {i1, i1} @optix.ptx.setp.eq.b64( i64 %p0, i64 %p1 ) #0 {
   %pred = icmp eq i64 %p0, %p1
   %v0 = insertvalue {i1, i1} undef, i1 %pred, 0
   %not = xor i1 %pred, true
   %v1 = insertvalue {i1, i1} %v0, i1 %not, 1
   ret {i1, i1} %v1
}

define linkonce_odr {i1, i1} @optix.ptx.setp.eq.u64( i64 %p0, i64 %p1 ) #0 {
   %pred = icmp eq i64 %p0, %p1
   %v0 = insertvalue {i1, i1} undef, i1 %pred, 0
   %not = xor i1 %pred, true
   %v1 = insertvalue {i1, i1} %v0, i1 %not, 1
   ret {i1, i1} %v1
}

define linkonce_odr {i1, i1} @optix.ptx.setp.ne.u64( i64 %p0, i64 %p1 ) #0 {
   %pred = icmp ne i64 %p0, %p1
   %v0 = insertvalue {i1, i1} undef, i1 %pred, 0
   %not = xor i1 %pred, true
   %v1 = insertvalue {i1, i1} %v0, i1 %not, 1
   ret {i1, i1} %v1
}

define linkonce_odr {i1, i1} @optix.ptx.setp.eq.s64( i64 %p0, i64 %p1 ) #0 {
   %pred = icmp eq i64 %p0, %p1
   %v0 = insertvalue {i1, i1} undef, i1 %pred, 0
   %not = xor i1 %pred, true
   %v1 = insertvalue {i1, i1} %v0, i1 %not, 1
   ret {i1, i1} %v1
}

define linkonce_odr {i1, i1} @optix.ptx.setp.gt.s64( i64 %p0, i64 %p1 ) #0 {
   %pred = icmp sgt i64 %p0, %p1
   %v0 = insertvalue {i1, i1} undef, i1 %pred, 0
   %not = xor i1 %pred, true
   %v1 = insertvalue {i1, i1} %v0, i1 %not, 1
   ret {i1, i1} %v1
}

define linkonce_odr {i1, i1} @optix.ptx.setp.eq.f32( float %p0, float %p1 ) #0 {
   %pred = fcmp oeq float %p0, %p1
   %v0 = insertvalue {i1, i1} undef, i1 %pred, 0
   %not = xor i1 %pred, true
   %v1 = insertvalue {i1, i1} %v0, i1 %not, 1
   ret {i1, i1} %v1
}

define linkonce_odr {i1, i1} @optix.ptx.setp.leu.f32( float %p0, float %p1 ) #0 {
   %pred = fcmp ule float %p0, %p1
   %v0 = insertvalue {i1, i1} undef, i1 %pred, 0
   %not = xor i1 %pred, true
   %v1 = insertvalue {i1, i1} %v0, i1 %not, 1
   ret {i1, i1} %v1
}

define linkonce_odr {i1, i1} @optix.ptx.setp.geu.f32( float %p0, float %p1 ) #0 {
   %pred = fcmp uge float %p0, %p1
   %v0 = insertvalue {i1, i1} undef, i1 %pred, 0
   %not = xor i1 %pred, true
   %v1 = insertvalue {i1, i1} %v0, i1 %not, 1
   ret {i1, i1} %v1
}

define linkonce_odr {i1, i1} @optix.ptx.setp.eq.f64( double %p0, double %p1 ) #0 {
   %pred = fcmp oeq double %p0, %p1
   %v0 = insertvalue {i1, i1} undef, i1 %pred, 0
   %not = xor i1 %pred, true
   %v1 = insertvalue {i1, i1} %v0, i1 %not, 1
   ret {i1, i1} %v1
}

define linkonce_odr {i1, i1} @optix.ptx.setp.gt.f64( double %p0, double %p1 ) #0 {
   %pred = fcmp ogt double %p0, %p1
   %v0 = insertvalue {i1, i1} undef, i1 %pred, 0
   %not = xor i1 %pred, true
   %v1 = insertvalue {i1, i1} %v0, i1 %not, 1
   ret {i1, i1} %v1
}

define linkonce_odr {i1, i1} @optix.ptx.setp.ne.b64( i64 %p0, i64 %p1 ) #0 {
   %pred = icmp ne i64 %p0, %p1
   %v0 = insertvalue {i1, i1} undef, i1 %pred, 0
   %not = xor i1 %pred, true
   %v1 = insertvalue {i1, i1} %v0, i1 %not, 1
   ret {i1, i1} %v1
}

define linkonce_odr {i1, i1} @optix.ptx.setp.le.u64( i64 %p0, i64 %p1 ) #0 {
   %pred = icmp ule i64 %p0, %p1
   %v0 = insertvalue {i1, i1} undef, i1 %pred, 0
   %not = xor i1 %pred, true
   %v1 = insertvalue {i1, i1} %v0, i1 %not, 1
   ret {i1, i1} %v1
}

define linkonce_odr {i1, i1} @optix.ptx.setp.ne.s64( i64 %p0, i64 %p1 ) #0 {
   %pred = icmp ne i64 %p0, %p1
   %v0 = insertvalue {i1, i1} undef, i1 %pred, 0
   %not = xor i1 %pred, true
   %v1 = insertvalue {i1, i1} %v0, i1 %not, 1
   ret {i1, i1} %v1
}

define linkonce_odr {i1, i1} @optix.ptx.setp.le.s64( i64 %p0, i64 %p1 ) #0 {
   %pred = icmp sle i64 %p0, %p1
   %v0 = insertvalue {i1, i1} undef, i1 %pred, 0
   %not = xor i1 %pred, true
   %v1 = insertvalue {i1, i1} %v0, i1 %not, 1
   ret {i1, i1} %v1
}

define linkonce_odr {i1, i1} @optix.ptx.setp.lt.u64( i64 %p0, i64 %p1 ) #0 {
   %pred = icmp ult i64 %p0, %p1
   %v0 = insertvalue {i1, i1} undef, i1 %pred, 0
   %not = xor i1 %pred, true
   %v1 = insertvalue {i1, i1} %v0, i1 %not, 1
   ret {i1, i1} %v1
}

define linkonce_odr {i1, i1} @optix.ptx.setp.lt.s64( i64 %p0, i64 %p1 ) #0 {
   %pred = icmp slt i64 %p0, %p1
   %v0 = insertvalue {i1, i1} undef, i1 %pred, 0
   %not = xor i1 %pred, true
   %v1 = insertvalue {i1, i1} %v0, i1 %not, 1
   ret {i1, i1} %v1
}

define linkonce_odr {i1, i1} @optix.ptx.setp.lt.f64( double %p0, double %p1 ) #0 {
   %pred = fcmp olt double %p0, %p1
   %v0 = insertvalue {i1, i1} undef, i1 %pred, 0
   %not = xor i1 %pred, true
   %v1 = insertvalue {i1, i1} %v0, i1 %not, 1
   ret {i1, i1} %v1
}

define linkonce_odr {i1, i1} @optix.ptx.setp.gt.u64( i64 %p0, i64 %p1 ) #0 {
   %pred = icmp ugt i64 %p0, %p1
   %v0 = insertvalue {i1, i1} undef, i1 %pred, 0
   %not = xor i1 %pred, true
   %v1 = insertvalue {i1, i1} %v0, i1 %not, 1
   ret {i1, i1} %v1
}

define linkonce_odr {i1, i1} @optix.ptx.setp.ge.u64( i64 %p0, i64 %p1 ) #0 {
   %pred = icmp uge i64 %p0, %p1
   %v0 = insertvalue {i1, i1} undef, i1 %pred, 0
   %not = xor i1 %pred, true
   %v1 = insertvalue {i1, i1} %v0, i1 %not, 1
   ret {i1, i1} %v1
}

define linkonce_odr {i1, i1} @optix.ptx.setp.ge.s64( i64 %p0, i64 %p1 ) #0 {
   %pred = icmp sge i64 %p0, %p1
   %v0 = insertvalue {i1, i1} undef, i1 %pred, 0
   %not = xor i1 %pred, true
   %v1 = insertvalue {i1, i1} %v0, i1 %not, 1
   ret {i1, i1} %v1
}

;
; testp
;

define linkonce_odr i1 @optix.ptx.testp.notanumber.f32( float %p0 ) #0 {
   %pred = fcmp ord float %p0, 0.0
   %not = xor i1 %pred, true
   ret i1 %not
}

; TODO: How to perform FTZ?
define linkonce_odr {i1, i1} @optix.ptx.setp.nan.ftz.f32( float %p0, float %p1 ) #0 {
   %ilwersepred = fcmp ord float %p0, %p1
   %pred = xor i1 %ilwersepred, true
   %v0 = insertvalue {i1, i1} undef, i1 %pred, 0
   %not = xor i1 %pred, true
   %v1 = insertvalue {i1, i1} %v0, i1 %not, 1
   ret {i1, i1} %v1
}

define linkonce_odr {i1, i1} @optix.ptx.setp.nan.f32( float %p0, float %p1 ) #0 {
   %ilwersepred = fcmp ord float %p0, %p1
   %pred = xor i1 %ilwersepred, true
   %v0 = insertvalue {i1, i1} undef, i1 %pred, 0
   %not = xor i1 %pred, true
   %v1 = insertvalue {i1, i1} %v0, i1 %not, 1
   ret {i1, i1} %v1
}

define linkonce_odr {i1, i1} @optix.ptx.setp.num.f32( float %p0, float %p1 ) #0 {
   %pred = fcmp ord float %p0, %p1
   %v0 = insertvalue {i1, i1} undef, i1 %pred, 0
   %not = xor i1 %pred, true
   %v1 = insertvalue {i1, i1} %v0, i1 %not, 1
   ret {i1, i1} %v1
}

;
; Cvta
; 

; Global to generic
define linkonce_odr i64 @optix.ptx.cvta.global.u64( i64 %p0 ) #0 {
  %v0 = inttoptr i64 %p0 to i8 addrspace(1)*
  %v1 = addrspacecast i8 addrspace(1)* %v0 to i8*
  %global_as_generic = ptrtoint i8* %v1 to i64
  ret i64 %global_as_generic
}

; Local to Generic
define linkonce_odr i64 @optix.ptx.cvta.local.u64( i64 %p0 ) #0 {
  %v0 = inttoptr i64 %p0 to i8 addrspace(5)*
  %v1 = addrspacecast i8 addrspace(5)* %v0 to i8*
  %local_as_generic = ptrtoint i8* %v1 to i64
  ret i64 %local_as_generic
}

; Constant to Generic
define linkonce_odr i64 @optix.ptx.cvta.const.u64( i64 %p0 ) #0 {
  %v0 = inttoptr i64 %p0 to i8 addrspace(4)*
  %v1 = addrspacecast i8 addrspace(4)* %v0 to i8*
  %constant_as_generic = ptrtoint i8* %v1 to i64
  ret i64 %constant_as_generic
}

; Generic to local
define linkonce_odr i64 @optix.ptx.cvta.to.local.u64( i64 %p0 ) #0 {
  %v0 = inttoptr i64 %p0 to i8*
  %v1 = addrspacecast i8* %v0 to i8 addrspace(5)*
  %generic_as_local = ptrtoint i8 addrspace(5)* %v1 to i64
  ret i64 %generic_as_local
}

; Generic to global
define linkonce_odr i64 @optix.ptx.cvta.to.global.u64( i64 %p0 ) #0 {
  %v0 = inttoptr i64 %p0 to i8*
  %v1 = addrspacecast i8* %v0 to i8 addrspace(1)*
  %generic_as_global = ptrtoint i8 addrspace(1)* %v1 to i64
  ret i64 %generic_as_global
}

; Generic to const
define linkonce_odr i64 @optix.ptx.cvta.to.const.u64( i64 %p0 ) #0 {
  %v0 = inttoptr i64 %p0 to i8*
  %v1 = addrspacecast i8* %v0 to i8 addrspace(4)*
  %generic_as_const = ptrtoint i8 addrspace(4)* %v1 to i64
  ret i64 %generic_as_const
}

;
; Sync
;

declare { i32, i1 } @llvm.lwvm.vote.sync(i32, i32, i1) #3
define linkonce_odr i1 @optix.ptx.vote.sync.all.pred( i1 %p0, i32 %p1 ) #1 {
    %1 = call { i32, i1 } @llvm.lwvm.vote.sync(i32 %p1, i32 0, i1 %p0)
    %2 = extractvalue { i32, i1 } %1, 1
    ret i1 %2
}

define linkonce_odr i1 @optix.ptx.vote.sync.any.pred( i1 %p0, i32 %p1 ) #1 {
    %1 = call { i32, i1 } @llvm.lwvm.vote.sync(i32 %p1, i32 1, i1 %p0)
    %2 = extractvalue { i32, i1 } %1, 1
    ret i1 %2
}

define linkonce_odr i1 @optix.ptx.vote.sync.uni.pred( i1 %p0, i32 %p1 ) #1 {
    %1 = call { i32, i1 } @llvm.lwvm.vote.sync(i32 %p1, i32 2, i1 %p0)
    %2 = extractvalue { i32, i1 } %1, 1
    ret i1 %2
}

define linkonce_odr i32 @optix.ptx.vote.sync.ballot.b32( i1 %p0, i32 %p1 ) #1 {
    %1 = call { i32, i1 } @llvm.lwvm.vote.sync(i32 %p1, i32 3, i1 %p0)
    %2 = extractvalue { i32, i1 } %1, 0
    ret i32 %2
}


declare { i32, i1 } @llvm.lwvm.match.all.sync.i32(i32, i32) #3
define linkonce_odr { i32, i1 } @optix.ptx.match.sync.all.b32( i32 %p0, i32 %p1 ) #1 {
    %1 = call { i32, i1 } @llvm.lwvm.match.all.sync.i32(i32 %p1, i32 %p0)
    ret { i32, i1 } %1
}

declare { i32, i1 } @llvm.lwvm.match.all.sync.i64(i32, i64) #3
define linkonce_odr { i32, i1 } @optix.ptx.match.sync.all.b64( i64 %p0, i32 %p1 ) #1 {
    %1 = call { i32, i1 } @llvm.lwvm.match.all.sync.i64(i32 %p1, i64 %p0)
    ret { i32, i1 } %1
}

declare i32 @llvm.lwvm.match.any.sync.i32(i32, i32) #3
define linkonce_odr i32 @optix.ptx.match.sync.any.b32( i32 %p0, i32 %p1 ) #1 {
    %1 = call i32 @llvm.lwvm.match.any.sync.i32(i32 %p1, i32 %p0)
    ret i32 %1
}

declare i32 @llvm.lwvm.match.any.sync.i64(i32, i64) #3
define linkonce_odr i32 @optix.ptx.match.sync.any.b64( i64 %p0, i32 %p1 ) #1 {
    %1 = call i32 @llvm.lwvm.match.any.sync.i64(i32 %p1, i64 %p0)
    ret i32 %1
}

; Note: The shfl sync PTX function is declared as readnone by the PTX FrontEnd:
; static bool isReadNone( ptxInstructionTemplate tmplate ) returns true for it.
; This is a bug which causes it being optimized away if the return value
; is unused. Unfortunately, Iray is "using" that bug because it calls shfl.sync.idx.b32 in
; a way that causes a compile error during validation if it is not optimized away. 
; Declaring the LWVM intrinsic as readnone does not work as it is changed back 
; by llvm-as, so we cannot do these at the moment:
;declare { i32, i1 } @llvm.lwvm.shfl.sync.i32(i32, i32, i32, i32, i32) #3
;define linkonce_odr { i32, i1 } @optix.ptx.shfl.sync.idx.b32( i32 %p0, i32 %p1, i32 %p2, i32 %p3 ) #1 {
;    %1 = call { i32, i1 } @llvm.lwvm.shfl.sync.i32(i32 %p3, i32 0, i32 %p0, i32 %p1, i32 %p2)
;    ret { i32, i1 } %1
;}
;
;define linkonce_odr { i32, i1 } @optix.ptx.shfl.sync.up.b32( i32 %p0, i32 %p1, i32 %p2, i32 %p3 ) #1 {
;    %1 = call { i32, i1 } @llvm.lwvm.shfl.sync.i32(i32 %p3, i32 1, i32 %p0, i32 %p1, i32 %p2)
;    ret { i32, i1 } %1
;}
;
;define linkonce_odr { i32, i1 } @optix.ptx.shfl.sync.down.b32( i32 %p0, i32 %p1, i32 %p2, i32 %p3 ) #1 {
;    %1 = call { i32, i1 } @llvm.lwvm.shfl.sync.i32(i32 %p3, i32 2, i32 %p0, i32 %p1, i32 %p2)
;    ret { i32, i1 } %1
;}
;
;define linkonce_odr { i32, i1 } @optix.ptx.shfl.sync.bfly.b32( i32 %p0, i32 %p1, i32 %p2, i32 %p3 ) #1 {
;    %1 = call { i32, i1 } @llvm.lwvm.shfl.sync.i32(i32 %p3, i32 3, i32 %p0, i32 %p1, i32 %p2)
;    ret { i32, i1 } %1
;}

; 
; Ld and st
;

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; auto-generated
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
define linkonce_odr i8 @optix.ptx.ld.b8( i8* %p0 ) #2 {
  %val = load i8, i8* %p0, align 1
  ret i8 %val
}

define linkonce_odr i8 @optix.ptx.ldu.b8( i8* %p0 ) #2 {
  %val = load i8, i8* %p0, align 1
  ret i8 %val
}

define linkonce_odr i8 @optix.ptx.ld.volatile.b8( i8* %p0 ) #1 {
  %val = load volatile i8, i8* %p0, align 1
  ret i8 %val
}

define linkonce_odr void @optix.ptx.st.b8( i8* %p0, i8 %p1) #1 {
  store i8 %p1, i8* %p0
  ret void
}

define linkonce_odr i8 @optix.ptx.ld.u8( i8* %p0 ) #2 {
  %val = load i8, i8* %p0, align 1
  ret i8 %val
}

define linkonce_odr i8 @optix.ptx.ldu.u8( i8* %p0 ) #2 {
  %val = load i8, i8* %p0, align 1
  ret i8 %val
}

define linkonce_odr i8 @optix.ptx.ld.volatile.u8( i8* %p0 ) #1 {
  %val = load volatile i8, i8* %p0, align 1
  ret i8 %val
}

define linkonce_odr void @optix.ptx.st.u8( i8* %p0, i8 %p1) #1 {
  store i8 %p1, i8* %p0
  ret void
}

define linkonce_odr i8 @optix.ptx.ld.s8( i8* %p0 ) #2 {
  %val = load i8, i8* %p0, align 1
  ret i8 %val
}

define linkonce_odr i8 @optix.ptx.ldu.s8( i8* %p0 ) #2 {
  %val = load i8, i8* %p0, align 1
  ret i8 %val
}

define linkonce_odr i8 @optix.ptx.ld.volatile.s8( i8* %p0 ) #1 {
  %val = load volatile i8, i8* %p0, align 1
  ret i8 %val
}

define linkonce_odr void @optix.ptx.st.s8( i8* %p0, i8 %p1) #1 {
  store i8 %p1, i8* %p0
  ret void
}

define linkonce_odr i16 @optix.ptx.ld.b16( i16* %p0 ) #2 {
  %val = load i16, i16* %p0, align 2
  ret i16 %val
}

define linkonce_odr i16 @optix.ptx.ldu.b16( i16* %p0 ) #2 {
  %val = load i16, i16* %p0, align 2
  ret i16 %val
}

define linkonce_odr i16 @optix.ptx.ld.volatile.b16( i16* %p0 ) #1 {
  %val = load volatile i16, i16* %p0, align 2
  ret i16 %val
}

define linkonce_odr void @optix.ptx.st.b16( i16* %p0, i16 %p1) #1 {
  store i16 %p1, i16* %p0
  ret void
}

define linkonce_odr i16 @optix.ptx.ld.u16( i16* %p0 ) #2 {
  %val = load i16, i16* %p0, align 2
  ret i16 %val
}

define linkonce_odr i16 @optix.ptx.ldu.u16( i16* %p0 ) #2 {
  %val = load i16, i16* %p0, align 2
  ret i16 %val
}

define linkonce_odr i16 @optix.ptx.ld.volatile.u16( i16* %p0 ) #1 {
  %val = load volatile i16, i16* %p0, align 2
  ret i16 %val
}

define linkonce_odr void @optix.ptx.st.u16( i16* %p0, i16 %p1) #1 {
  store i16 %p1, i16* %p0
  ret void
}

define linkonce_odr i16 @optix.ptx.ld.s16( i16* %p0 ) #2 {
  %val = load i16, i16* %p0, align 2
  ret i16 %val
}

define linkonce_odr i16 @optix.ptx.ldu.s16( i16* %p0 ) #2 {
  %val = load i16, i16* %p0, align 2
  ret i16 %val
}

define linkonce_odr i16 @optix.ptx.ld.volatile.s16( i16* %p0 ) #1 {
  %val = load volatile i16, i16* %p0, align 2
  ret i16 %val
}

define linkonce_odr void @optix.ptx.st.s16( i16* %p0, i16 %p1) #1 {
  store i16 %p1, i16* %p0
  ret void
}

define linkonce_odr i32 @optix.ptx.ld.b32( i32* %p0 ) #2 {
  %val = load i32, i32* %p0, align 4
  ret i32 %val
}

define linkonce_odr i32 @optix.ptx.ldu.b32( i32* %p0 ) #2 {
  %val = load i32, i32* %p0, align 4
  ret i32 %val
}

define linkonce_odr i32 @optix.ptx.ld.volatile.b32( i32* %p0 ) #1 {
  %val = load volatile i32, i32* %p0, align 4
  ret i32 %val
}

define linkonce_odr void @optix.ptx.st.b32( i32* %p0, i32 %p1) #1 {
  store i32 %p1, i32* %p0
  ret void
}

define linkonce_odr i32 @optix.ptx.ld.u32( i32* %p0 ) #2 {
  %val = load i32, i32* %p0, align 4
  ret i32 %val
}

define linkonce_odr i32 @optix.ptx.ldu.u32( i32* %p0 ) #2 {
  %val = load i32, i32* %p0, align 4
  ret i32 %val
}

define linkonce_odr i32 @optix.ptx.ld.volatile.u32( i32* %p0 ) #1 {
  %val = load volatile i32, i32* %p0, align 4
  ret i32 %val
}

define linkonce_odr void @optix.ptx.st.u32( i32* %p0, i32 %p1) #1 {
  store i32 %p1, i32* %p0
  ret void
}

define linkonce_odr i32 @optix.ptx.ld.s32( i32* %p0 ) #2 {
  %val = load i32, i32* %p0, align 4
  ret i32 %val
}

define linkonce_odr i32 @optix.ptx.ldu.s32( i32* %p0 ) #2 {
  %val = load i32, i32* %p0, align 4
  ret i32 %val
}

define linkonce_odr i32 @optix.ptx.ld.volatile.s32( i32* %p0 ) #1 {
  %val = load volatile i32, i32* %p0, align 4
  ret i32 %val
}

define linkonce_odr void @optix.ptx.st.s32( i32* %p0, i32 %p1) #1 {
  store i32 %p1, i32* %p0
  ret void
}

define linkonce_odr i64 @optix.ptx.ld.b64( i64* %p0 ) #2 {
  %val = load i64, i64* %p0, align 8
  ret i64 %val
}

define linkonce_odr i64 @optix.ptx.ldu.b64( i64* %p0 ) #2 {
  %val = load i64, i64* %p0, align 8
  ret i64 %val
}

define linkonce_odr i64 @optix.ptx.ld.volatile.b64( i64* %p0 ) #1 {
  %val = load volatile i64, i64* %p0, align 8
  ret i64 %val
}

define linkonce_odr void @optix.ptx.st.b64( i64* %p0, i64 %p1) #1 {
  store i64 %p1, i64* %p0
  ret void
}

define linkonce_odr i64 @optix.ptx.ld.u64( i64* %p0 ) #2 {
  %val = load i64, i64* %p0, align 8
  ret i64 %val
}

define linkonce_odr i64 @optix.ptx.ldu.u64( i64* %p0 ) #2 {
  %val = load i64, i64* %p0, align 8
  ret i64 %val
}

define linkonce_odr i64 @optix.ptx.ld.volatile.u64( i64* %p0 ) #1 {
  %val = load volatile i64, i64* %p0, align 8
  ret i64 %val
}

define linkonce_odr void @optix.ptx.st.u64( i64* %p0, i64 %p1) #1 {
  store i64 %p1, i64* %p0
  ret void
}

define linkonce_odr i64 @optix.ptx.ld.s64( i64* %p0 ) #2 {
  %val = load i64, i64* %p0, align 8
  ret i64 %val
}

define linkonce_odr i64 @optix.ptx.ldu.s64( i64* %p0 ) #2 {
  %val = load i64, i64* %p0, align 8
  ret i64 %val
}

define linkonce_odr i64 @optix.ptx.ld.volatile.s64( i64* %p0 ) #1 {
  %val = load volatile i64, i64* %p0, align 8
  ret i64 %val
}

define linkonce_odr void @optix.ptx.st.s64( i64* %p0, i64 %p1) #1 {
  store i64 %p1, i64* %p0
  ret void
}

define linkonce_odr float @optix.ptx.ld.f32( float* %p0 ) #2 {
  %val = load float, float* %p0, align 4
  ret float %val
}

define linkonce_odr float @optix.ptx.ldu.f32( float* %p0 ) #2 {
  %val = load float, float* %p0, align 4
  ret float %val
}

define linkonce_odr float @optix.ptx.ld.volatile.f32( float* %p0 ) #1 {
  %val = load volatile float, float* %p0, align 4
  ret float %val
}

define linkonce_odr void @optix.ptx.st.f32( float* %p0, float %p1) #1 {
  store float %p1, float* %p0
  ret void
}

define linkonce_odr double @optix.ptx.ld.f64( double* %p0 ) #2 {
  %val = load double, double* %p0, align 8
  ret double %val
}

define linkonce_odr double @optix.ptx.ldu.f64( double* %p0 ) #2 {
  %val = load double, double* %p0, align 8
  ret double %val
}

define linkonce_odr double @optix.ptx.ld.volatile.f64( double* %p0 ) #1 {
  %val = load volatile double, double* %p0, align 8
  ret double %val
}

define linkonce_odr void @optix.ptx.st.f64( double* %p0, double %p1) #1 {
  store double %p1, double* %p0
  ret void
}

define linkonce_odr <2 x i8> @optix.ptx.ld.v2.b8( <2 x i8>* %p0 ) #2 {
  %val = load <2 x i8>, <2 x i8>* %p0, align 2
  ret <2 x i8> %val
}

define linkonce_odr <2 x i8> @optix.ptx.ldu.v2.b8( <2 x i8>* %p0 ) #2 {
  %val = load <2 x i8>, <2 x i8>* %p0, align 2
  ret <2 x i8> %val
}

define linkonce_odr <2 x i8> @optix.ptx.ld.volatile.v2.b8( <2 x i8>* %p0 ) #1 {
  %val = load volatile <2 x i8>, <2 x i8>* %p0, align 2
  ret <2 x i8> %val
}

define linkonce_odr void @optix.ptx.st.v2.b8( <2 x i8>* %p0, <2 x i8> %p1) #1 {
  store <2 x i8> %p1, <2 x i8>* %p0
  ret void
}

define linkonce_odr <2 x i8> @optix.ptx.ld.v2.u8( <2 x i8>* %p0 ) #2 {
  %val = load <2 x i8>, <2 x i8>* %p0, align 2
  ret <2 x i8> %val
}

define linkonce_odr <2 x i8> @optix.ptx.ldu.v2.u8( <2 x i8>* %p0 ) #2 {
  %val = load <2 x i8>, <2 x i8>* %p0, align 2
  ret <2 x i8> %val
}

define linkonce_odr <2 x i8> @optix.ptx.ld.volatile.v2.u8( <2 x i8>* %p0 ) #1 {
  %val = load volatile <2 x i8>, <2 x i8>* %p0, align 2
  ret <2 x i8> %val
}

define linkonce_odr void @optix.ptx.st.v2.u8( <2 x i8>* %p0, <2 x i8> %p1) #1 {
  store <2 x i8> %p1, <2 x i8>* %p0
  ret void
}

define linkonce_odr <2 x i8> @optix.ptx.ld.v2.s8( <2 x i8>* %p0 ) #2 {
  %val = load <2 x i8>, <2 x i8>* %p0, align 2
  ret <2 x i8> %val
}

define linkonce_odr <2 x i8> @optix.ptx.ldu.v2.s8( <2 x i8>* %p0 ) #2 {
  %val = load <2 x i8>, <2 x i8>* %p0, align 2
  ret <2 x i8> %val
}

define linkonce_odr <2 x i8> @optix.ptx.ld.volatile.v2.s8( <2 x i8>* %p0 ) #1 {
  %val = load volatile <2 x i8>, <2 x i8>* %p0, align 2
  ret <2 x i8> %val
}

define linkonce_odr void @optix.ptx.st.v2.s8( <2 x i8>* %p0, <2 x i8> %p1) #1 {
  store <2 x i8> %p1, <2 x i8>* %p0
  ret void
}

define linkonce_odr <2 x i16> @optix.ptx.ld.v2.b16( <2 x i16>* %p0 ) #2 {
  %val = load <2 x i16>, <2 x i16>* %p0, align 4
  ret <2 x i16> %val
}

define linkonce_odr <2 x i16> @optix.ptx.ldu.v2.b16( <2 x i16>* %p0 ) #2 {
  %val = load <2 x i16>, <2 x i16>* %p0, align 4
  ret <2 x i16> %val
}

define linkonce_odr <2 x i16> @optix.ptx.ld.volatile.v2.b16( <2 x i16>* %p0 ) #1 {
  %val = load volatile <2 x i16>, <2 x i16>* %p0, align 4
  ret <2 x i16> %val
}

define linkonce_odr void @optix.ptx.st.v2.b16( <2 x i16>* %p0, <2 x i16> %p1) #1 {
  store <2 x i16> %p1, <2 x i16>* %p0
  ret void
}

define linkonce_odr <2 x i16> @optix.ptx.ld.v2.u16( <2 x i16>* %p0 ) #2 {
  %val = load <2 x i16>, <2 x i16>* %p0, align 4
  ret <2 x i16> %val
}

define linkonce_odr <2 x i16> @optix.ptx.ldu.v2.u16( <2 x i16>* %p0 ) #2 {
  %val = load <2 x i16>, <2 x i16>* %p0, align 4
  ret <2 x i16> %val
}

define linkonce_odr <2 x i16> @optix.ptx.ld.volatile.v2.u16( <2 x i16>* %p0 ) #1 {
  %val = load volatile <2 x i16>, <2 x i16>* %p0, align 4
  ret <2 x i16> %val
}

define linkonce_odr void @optix.ptx.st.v2.u16( <2 x i16>* %p0, <2 x i16> %p1) #1 {
  store <2 x i16> %p1, <2 x i16>* %p0
  ret void
}

define linkonce_odr <2 x i16> @optix.ptx.ld.v2.s16( <2 x i16>* %p0 ) #2 {
  %val = load <2 x i16>, <2 x i16>* %p0, align 4
  ret <2 x i16> %val
}

define linkonce_odr <2 x i16> @optix.ptx.ldu.v2.s16( <2 x i16>* %p0 ) #2 {
  %val = load <2 x i16>, <2 x i16>* %p0, align 4
  ret <2 x i16> %val
}

define linkonce_odr <2 x i16> @optix.ptx.ld.volatile.v2.s16( <2 x i16>* %p0 ) #1 {
  %val = load volatile <2 x i16>, <2 x i16>* %p0, align 4
  ret <2 x i16> %val
}

define linkonce_odr void @optix.ptx.st.v2.s16( <2 x i16>* %p0, <2 x i16> %p1) #1 {
  store <2 x i16> %p1, <2 x i16>* %p0
  ret void
}

define linkonce_odr <2 x i32> @optix.ptx.ld.v2.b32( <2 x i32>* %p0 ) #2 {
  %val = load <2 x i32>, <2 x i32>* %p0, align 8
  ret <2 x i32> %val
}

define linkonce_odr <2 x i32> @optix.ptx.ldu.v2.b32( <2 x i32>* %p0 ) #2 {
  %val = load <2 x i32>, <2 x i32>* %p0, align 8
  ret <2 x i32> %val
}

define linkonce_odr <2 x i32> @optix.ptx.ld.volatile.v2.b32( <2 x i32>* %p0 ) #1 {
  %val = load volatile <2 x i32>, <2 x i32>* %p0, align 8
  ret <2 x i32> %val
}

define linkonce_odr void @optix.ptx.st.v2.b32( <2 x i32>* %p0, <2 x i32> %p1) #1 {
  store <2 x i32> %p1, <2 x i32>* %p0
  ret void
}

define linkonce_odr <2 x i32> @optix.ptx.ld.v2.u32( <2 x i32>* %p0 ) #2 {
  %val = load <2 x i32>, <2 x i32>* %p0, align 8
  ret <2 x i32> %val
}

define linkonce_odr <2 x i32> @optix.ptx.ldu.v2.u32( <2 x i32>* %p0 ) #2 {
  %val = load <2 x i32>, <2 x i32>* %p0, align 8
  ret <2 x i32> %val
}

define linkonce_odr <2 x i32> @optix.ptx.ld.volatile.v2.u32( <2 x i32>* %p0 ) #1 {
  %val = load volatile <2 x i32>, <2 x i32>* %p0, align 8
  ret <2 x i32> %val
}

define linkonce_odr void @optix.ptx.st.v2.u32( <2 x i32>* %p0, <2 x i32> %p1) #1 {
  store <2 x i32> %p1, <2 x i32>* %p0
  ret void
}

define linkonce_odr <2 x i32> @optix.ptx.ld.v2.s32( <2 x i32>* %p0 ) #2 {
  %val = load <2 x i32>, <2 x i32>* %p0, align 8
  ret <2 x i32> %val
}

define linkonce_odr <2 x i32> @optix.ptx.ldu.v2.s32( <2 x i32>* %p0 ) #2 {
  %val = load <2 x i32>, <2 x i32>* %p0, align 8
  ret <2 x i32> %val
}

define linkonce_odr <2 x i32> @optix.ptx.ld.volatile.v2.s32( <2 x i32>* %p0 ) #1 {
  %val = load volatile <2 x i32>, <2 x i32>* %p0, align 8
  ret <2 x i32> %val
}

define linkonce_odr void @optix.ptx.st.v2.s32( <2 x i32>* %p0, <2 x i32> %p1) #1 {
  store <2 x i32> %p1, <2 x i32>* %p0
  ret void
}

define linkonce_odr <2 x i64> @optix.ptx.ld.v2.b64( <2 x i64>* %p0 ) #2 {
  %val = load <2 x i64>, <2 x i64>* %p0, align 16
  ret <2 x i64> %val
}

define linkonce_odr <2 x i64> @optix.ptx.ldu.v2.b64( <2 x i64>* %p0 ) #2 {
  %val = load <2 x i64>, <2 x i64>* %p0, align 16
  ret <2 x i64> %val
}

define linkonce_odr <2 x i64> @optix.ptx.ld.volatile.v2.b64( <2 x i64>* %p0 ) #1 {
  %val = load volatile <2 x i64>, <2 x i64>* %p0, align 16
  ret <2 x i64> %val
}

define linkonce_odr void @optix.ptx.st.v2.b64( <2 x i64>* %p0, <2 x i64> %p1) #1 {
  store <2 x i64> %p1, <2 x i64>* %p0
  ret void
}

define linkonce_odr <2 x i64> @optix.ptx.ld.v2.u64( <2 x i64>* %p0 ) #2 {
  %val = load <2 x i64>, <2 x i64>* %p0, align 16
  ret <2 x i64> %val
}

define linkonce_odr <2 x i64> @optix.ptx.ldu.v2.u64( <2 x i64>* %p0 ) #2 {
  %val = load <2 x i64>, <2 x i64>* %p0, align 16
  ret <2 x i64> %val
}

define linkonce_odr <2 x i64> @optix.ptx.ld.volatile.v2.u64( <2 x i64>* %p0 ) #1 {
  %val = load volatile <2 x i64>, <2 x i64>* %p0, align 16
  ret <2 x i64> %val
}

define linkonce_odr void @optix.ptx.st.v2.u64( <2 x i64>* %p0, <2 x i64> %p1) #1 {
  store <2 x i64> %p1, <2 x i64>* %p0
  ret void
}

define linkonce_odr <2 x i64> @optix.ptx.ld.v2.s64( <2 x i64>* %p0 ) #2 {
  %val = load <2 x i64>, <2 x i64>* %p0, align 16
  ret <2 x i64> %val
}

define linkonce_odr <2 x i64> @optix.ptx.ldu.v2.s64( <2 x i64>* %p0 ) #2 {
  %val = load <2 x i64>, <2 x i64>* %p0, align 16
  ret <2 x i64> %val
}

define linkonce_odr <2 x i64> @optix.ptx.ld.volatile.v2.s64( <2 x i64>* %p0 ) #1 {
  %val = load volatile <2 x i64>, <2 x i64>* %p0, align 16
  ret <2 x i64> %val
}

define linkonce_odr void @optix.ptx.st.v2.s64( <2 x i64>* %p0, <2 x i64> %p1) #1 {
  store <2 x i64> %p1, <2 x i64>* %p0
  ret void
}

define linkonce_odr <2 x float> @optix.ptx.ld.v2.f32( <2 x float>* %p0 ) #2 {
  %val = load <2 x float>, <2 x float>* %p0, align 8
  ret <2 x float> %val
}

define linkonce_odr <2 x float> @optix.ptx.ldu.v2.f32( <2 x float>* %p0 ) #2 {
  %val = load <2 x float>, <2 x float>* %p0, align 8
  ret <2 x float> %val
}

define linkonce_odr <2 x float> @optix.ptx.ld.volatile.v2.f32( <2 x float>* %p0 ) #1 {
  %val = load volatile <2 x float>, <2 x float>* %p0, align 8
  ret <2 x float> %val
}

define linkonce_odr void @optix.ptx.st.v2.f32( <2 x float>* %p0, <2 x float> %p1) #1 {
  store <2 x float> %p1, <2 x float>* %p0
  ret void
}

define linkonce_odr <2 x double> @optix.ptx.ld.v2.f64( <2 x double>* %p0 ) #2 {
  %val = load <2 x double>, <2 x double>* %p0, align 16
  ret <2 x double> %val
}

define linkonce_odr <2 x double> @optix.ptx.ldu.v2.f64( <2 x double>* %p0 ) #2 {
  %val = load <2 x double>, <2 x double>* %p0, align 16
  ret <2 x double> %val
}

define linkonce_odr <2 x double> @optix.ptx.ld.volatile.v2.f64( <2 x double>* %p0 ) #1 {
  %val = load volatile <2 x double>, <2 x double>* %p0, align 16
  ret <2 x double> %val
}

define linkonce_odr void @optix.ptx.st.v2.f64( <2 x double>* %p0, <2 x double> %p1) #1 {
  store <2 x double> %p1, <2 x double>* %p0
  ret void
}

define linkonce_odr <4 x i8> @optix.ptx.ld.v4.b8( <4 x i8>* %p0 ) #2 {
  %val = load <4 x i8>, <4 x i8>* %p0, align 4
  ret <4 x i8> %val
}

define linkonce_odr <4 x i8> @optix.ptx.ldu.v4.b8( <4 x i8>* %p0 ) #2 {
  %val = load <4 x i8>, <4 x i8>* %p0, align 4
  ret <4 x i8> %val
}

define linkonce_odr <4 x i8> @optix.ptx.ld.volatile.v4.b8( <4 x i8>* %p0 ) #1 {
  %val = load volatile <4 x i8>, <4 x i8>* %p0, align 4
  ret <4 x i8> %val
}

define linkonce_odr void @optix.ptx.st.v4.b8( <4 x i8>* %p0, <4 x i8> %p1) #1 {
  store <4 x i8> %p1, <4 x i8>* %p0
  ret void
}

define linkonce_odr <4 x i8> @optix.ptx.ld.v4.u8( <4 x i8>* %p0 ) #2 {
  %val = load <4 x i8>, <4 x i8>* %p0, align 4
  ret <4 x i8> %val
}

define linkonce_odr <4 x i8> @optix.ptx.ldu.v4.u8( <4 x i8>* %p0 ) #2 {
  %val = load <4 x i8>, <4 x i8>* %p0, align 4
  ret <4 x i8> %val
}

define linkonce_odr <4 x i8> @optix.ptx.ld.volatile.v4.u8( <4 x i8>* %p0 ) #1 {
  %val = load volatile <4 x i8>, <4 x i8>* %p0, align 4
  ret <4 x i8> %val
}

define linkonce_odr void @optix.ptx.st.v4.u8( <4 x i8>* %p0, <4 x i8> %p1) #1 {
  store <4 x i8> %p1, <4 x i8>* %p0
  ret void
}

define linkonce_odr <4 x i8> @optix.ptx.ld.v4.s8( <4 x i8>* %p0 ) #2 {
  %val = load <4 x i8>, <4 x i8>* %p0, align 4
  ret <4 x i8> %val
}

define linkonce_odr <4 x i8> @optix.ptx.ldu.v4.s8( <4 x i8>* %p0 ) #2 {
  %val = load <4 x i8>, <4 x i8>* %p0, align 4
  ret <4 x i8> %val
}

define linkonce_odr <4 x i8> @optix.ptx.ld.volatile.v4.s8( <4 x i8>* %p0 ) #1 {
  %val = load volatile <4 x i8>, <4 x i8>* %p0, align 4
  ret <4 x i8> %val
}

define linkonce_odr void @optix.ptx.st.v4.s8( <4 x i8>* %p0, <4 x i8> %p1) #1 {
  store <4 x i8> %p1, <4 x i8>* %p0
  ret void
}

define linkonce_odr <4 x i16> @optix.ptx.ld.v4.b16( <4 x i16>* %p0 ) #2 {
  %val = load <4 x i16>, <4 x i16>* %p0, align 8
  ret <4 x i16> %val
}

define linkonce_odr <4 x i16> @optix.ptx.ldu.v4.b16( <4 x i16>* %p0 ) #2 {
  %val = load <4 x i16>, <4 x i16>* %p0, align 8
  ret <4 x i16> %val
}

define linkonce_odr <4 x i16> @optix.ptx.ld.volatile.v4.b16( <4 x i16>* %p0 ) #1 {
  %val = load volatile <4 x i16>, <4 x i16>* %p0, align 8
  ret <4 x i16> %val
}

define linkonce_odr void @optix.ptx.st.v4.b16( <4 x i16>* %p0, <4 x i16> %p1) #1 {
  store <4 x i16> %p1, <4 x i16>* %p0
  ret void
}

define linkonce_odr <4 x i16> @optix.ptx.ld.v4.u16( <4 x i16>* %p0 ) #2 {
  %val = load <4 x i16>, <4 x i16>* %p0, align 8
  ret <4 x i16> %val
}

define linkonce_odr <4 x i16> @optix.ptx.ldu.v4.u16( <4 x i16>* %p0 ) #2 {
  %val = load <4 x i16>, <4 x i16>* %p0, align 8
  ret <4 x i16> %val
}

define linkonce_odr <4 x i16> @optix.ptx.ld.volatile.v4.u16( <4 x i16>* %p0 ) #1 {
  %val = load volatile <4 x i16>, <4 x i16>* %p0, align 8
  ret <4 x i16> %val
}

define linkonce_odr void @optix.ptx.st.v4.u16( <4 x i16>* %p0, <4 x i16> %p1) #1 {
  store <4 x i16> %p1, <4 x i16>* %p0
  ret void
}

define linkonce_odr <4 x i16> @optix.ptx.ld.v4.s16( <4 x i16>* %p0 ) #2 {
  %val = load <4 x i16>, <4 x i16>* %p0, align 8
  ret <4 x i16> %val
}

define linkonce_odr <4 x i16> @optix.ptx.ldu.v4.s16( <4 x i16>* %p0 ) #2 {
  %val = load <4 x i16>, <4 x i16>* %p0, align 8
  ret <4 x i16> %val
}

define linkonce_odr <4 x i16> @optix.ptx.ld.volatile.v4.s16( <4 x i16>* %p0 ) #1 {
  %val = load volatile <4 x i16>, <4 x i16>* %p0, align 8
  ret <4 x i16> %val
}

define linkonce_odr void @optix.ptx.st.v4.s16( <4 x i16>* %p0, <4 x i16> %p1) #1 {
  store <4 x i16> %p1, <4 x i16>* %p0
  ret void
}

define linkonce_odr <4 x i32> @optix.ptx.ld.v4.b32( <4 x i32>* %p0 ) #2 {
  %val = load <4 x i32>, <4 x i32>* %p0, align 16
  ret <4 x i32> %val
}

define linkonce_odr <4 x i32> @optix.ptx.ldu.v4.b32( <4 x i32>* %p0 ) #2 {
  %val = load <4 x i32>, <4 x i32>* %p0, align 16
  ret <4 x i32> %val
}

define linkonce_odr <4 x i32> @optix.ptx.ld.volatile.v4.b32( <4 x i32>* %p0 ) #1 {
  %val = load volatile <4 x i32>, <4 x i32>* %p0, align 16
  ret <4 x i32> %val
}

define linkonce_odr void @optix.ptx.st.v4.b32( <4 x i32>* %p0, <4 x i32> %p1) #1 {
  store <4 x i32> %p1, <4 x i32>* %p0
  ret void
}

define linkonce_odr <4 x i32> @optix.ptx.ld.v4.u32( <4 x i32>* %p0 ) #2 {
  %val = load <4 x i32>, <4 x i32>* %p0, align 16
  ret <4 x i32> %val
}

define linkonce_odr <4 x i32> @optix.ptx.ldu.v4.u32( <4 x i32>* %p0 ) #2 {
  %val = load <4 x i32>, <4 x i32>* %p0, align 16
  ret <4 x i32> %val
}

define linkonce_odr <4 x i32> @optix.ptx.ld.volatile.v4.u32( <4 x i32>* %p0 ) #1 {
  %val = load volatile <4 x i32>, <4 x i32>* %p0, align 16
  ret <4 x i32> %val
}

define linkonce_odr void @optix.ptx.st.v4.u32( <4 x i32>* %p0, <4 x i32> %p1) #1 {
  store <4 x i32> %p1, <4 x i32>* %p0
  ret void
}

define linkonce_odr <4 x i32> @optix.ptx.ld.v4.s32( <4 x i32>* %p0 ) #2 {
  %val = load <4 x i32>, <4 x i32>* %p0, align 16
  ret <4 x i32> %val
}

define linkonce_odr <4 x i32> @optix.ptx.ldu.v4.s32( <4 x i32>* %p0 ) #2 {
  %val = load <4 x i32>, <4 x i32>* %p0, align 16
  ret <4 x i32> %val
}

define linkonce_odr <4 x i32> @optix.ptx.ld.volatile.v4.s32( <4 x i32>* %p0 ) #1 {
  %val = load volatile <4 x i32>, <4 x i32>* %p0, align 16
  ret <4 x i32> %val
}

define linkonce_odr void @optix.ptx.st.v4.s32( <4 x i32>* %p0, <4 x i32> %p1) #1 {
  store <4 x i32> %p1, <4 x i32>* %p0
  ret void
}

define linkonce_odr <4 x i64> @optix.ptx.ld.v4.b64( <4 x i64>* %p0 ) #2 {
  %val = load <4 x i64>, <4 x i64>* %p0, align 32
  ret <4 x i64> %val
}

define linkonce_odr <4 x i64> @optix.ptx.ldu.v4.b64( <4 x i64>* %p0 ) #2 {
  %val = load <4 x i64>, <4 x i64>* %p0, align 32
  ret <4 x i64> %val
}

define linkonce_odr <4 x i64> @optix.ptx.ld.volatile.v4.b64( <4 x i64>* %p0 ) #1 {
  %val = load volatile <4 x i64>, <4 x i64>* %p0, align 32
  ret <4 x i64> %val
}

define linkonce_odr void @optix.ptx.st.v4.b64( <4 x i64>* %p0, <4 x i64> %p1) #1 {
  store <4 x i64> %p1, <4 x i64>* %p0
  ret void
}

define linkonce_odr <4 x i64> @optix.ptx.ld.v4.u64( <4 x i64>* %p0 ) #2 {
  %val = load <4 x i64>, <4 x i64>* %p0, align 32
  ret <4 x i64> %val
}

define linkonce_odr <4 x i64> @optix.ptx.ldu.v4.u64( <4 x i64>* %p0 ) #2 {
  %val = load <4 x i64>, <4 x i64>* %p0, align 32
  ret <4 x i64> %val
}

define linkonce_odr <4 x i64> @optix.ptx.ld.volatile.v4.u64( <4 x i64>* %p0 ) #1 {
  %val = load volatile <4 x i64>, <4 x i64>* %p0, align 32
  ret <4 x i64> %val
}

define linkonce_odr void @optix.ptx.st.v4.u64( <4 x i64>* %p0, <4 x i64> %p1) #1 {
  store <4 x i64> %p1, <4 x i64>* %p0
  ret void
}

define linkonce_odr <4 x i64> @optix.ptx.ld.v4.s64( <4 x i64>* %p0 ) #2 {
  %val = load <4 x i64>, <4 x i64>* %p0, align 32
  ret <4 x i64> %val
}

define linkonce_odr <4 x i64> @optix.ptx.ldu.v4.s64( <4 x i64>* %p0 ) #2 {
  %val = load <4 x i64>, <4 x i64>* %p0, align 32
  ret <4 x i64> %val
}

define linkonce_odr <4 x i64> @optix.ptx.ld.volatile.v4.s64( <4 x i64>* %p0 ) #1 {
  %val = load volatile <4 x i64>, <4 x i64>* %p0, align 32
  ret <4 x i64> %val
}

define linkonce_odr void @optix.ptx.st.v4.s64( <4 x i64>* %p0, <4 x i64> %p1) #1 {
  store <4 x i64> %p1, <4 x i64>* %p0
  ret void
}

define linkonce_odr <4 x float> @optix.ptx.ld.v4.f32( <4 x float>* %p0 ) #2 {
  %val = load <4 x float>, <4 x float>* %p0, align 16
  ret <4 x float> %val
}

define linkonce_odr <4 x float> @optix.ptx.ldu.v4.f32( <4 x float>* %p0 ) #2 {
  %val = load <4 x float>, <4 x float>* %p0, align 16
  ret <4 x float> %val
}

define linkonce_odr <4 x float> @optix.ptx.ld.volatile.v4.f32( <4 x float>* %p0 ) #1 {
  %val = load volatile <4 x float>, <4 x float>* %p0, align 16
  ret <4 x float> %val
}

define linkonce_odr void @optix.ptx.st.v4.f32( <4 x float>* %p0, <4 x float> %p1) #1 {
  store <4 x float> %p1, <4 x float>* %p0
  ret void
}

define linkonce_odr <4 x double> @optix.ptx.ld.v4.f64( <4 x double>* %p0 ) #2 {
  %val = load <4 x double>, <4 x double>* %p0, align 32
  ret <4 x double> %val
}

define linkonce_odr <4 x double> @optix.ptx.ldu.v4.f64( <4 x double>* %p0 ) #2 {
  %val = load <4 x double>, <4 x double>* %p0, align 32
  ret <4 x double> %val
}

define linkonce_odr <4 x double> @optix.ptx.ld.volatile.v4.f64( <4 x double>* %p0 ) #1 {
  %val = load volatile <4 x double>, <4 x double>* %p0, align 32
  ret <4 x double> %val
}

define linkonce_odr void @optix.ptx.st.v4.f64( <4 x double>* %p0, <4 x double> %p1) #1 {
  store <4 x double> %p1, <4 x double>* %p0
  ret void
}

define linkonce_odr i8 @optix.ptx.ld.global.b8( i8 addrspace(1)* %p0 ) #2 {
  %val = load i8, i8 addrspace(1)* %p0, align 1
  ret i8 %val
}

define linkonce_odr i8 @optix.ptx.ldu.global.b8( i8 addrspace(1)* %p0 ) #2 {
  %val = load i8, i8 addrspace(1)* %p0, align 1
  ret i8 %val
}

define linkonce_odr i8 @optix.ptx.ld.volatile.global.b8( i8 addrspace(1)* %p0 ) #1 {
  %val = load volatile i8, i8 addrspace(1)* %p0, align 1
  ret i8 %val
}

define linkonce_odr void @optix.ptx.st.global.b8( i8 addrspace(1)* %p0, i8 %p1) #1 {
  store i8 %p1, i8 addrspace(1)* %p0
  ret void
}

define linkonce_odr i8 @optix.ptx.ld.global.u8( i8 addrspace(1)* %p0 ) #2 {
  %val = load i8, i8 addrspace(1)* %p0, align 1
  ret i8 %val
}

define linkonce_odr i8 @optix.ptx.ldu.global.u8( i8 addrspace(1)* %p0 ) #2 {
  %val = load i8, i8 addrspace(1)* %p0, align 1
  ret i8 %val
}

define linkonce_odr i8 @optix.ptx.ld.volatile.global.u8( i8 addrspace(1)* %p0 ) #1 {
  %val = load volatile i8, i8 addrspace(1)* %p0, align 1
  ret i8 %val
}

define linkonce_odr void @optix.ptx.st.global.u8( i8 addrspace(1)* %p0, i8 %p1) #1 {
  store i8 %p1, i8 addrspace(1)* %p0
  ret void
}

define linkonce_odr i8 @optix.ptx.ld.global.s8( i8 addrspace(1)* %p0 ) #2 {
  %val = load i8, i8 addrspace(1)* %p0, align 1
  ret i8 %val
}

define linkonce_odr i8 @optix.ptx.ldu.global.s8( i8 addrspace(1)* %p0 ) #2 {
  %val = load i8, i8 addrspace(1)* %p0, align 1
  ret i8 %val
}

define linkonce_odr i8 @optix.ptx.ld.volatile.global.s8( i8 addrspace(1)* %p0 ) #1 {
  %val = load volatile i8, i8 addrspace(1)* %p0, align 1
  ret i8 %val
}

define linkonce_odr void @optix.ptx.st.global.s8( i8 addrspace(1)* %p0, i8 %p1) #1 {
  store i8 %p1, i8 addrspace(1)* %p0
  ret void
}

define linkonce_odr i16 @optix.ptx.ld.global.b16( i16 addrspace(1)* %p0 ) #2 {
  %val = load i16, i16 addrspace(1)* %p0, align 2
  ret i16 %val
}

define linkonce_odr i16 @optix.ptx.ldu.global.b16( i16 addrspace(1)* %p0 ) #2 {
  %val = load i16, i16 addrspace(1)* %p0, align 2
  ret i16 %val
}

define linkonce_odr i16 @optix.ptx.ld.volatile.global.b16( i16 addrspace(1)* %p0 ) #1 {
  %val = load volatile i16, i16 addrspace(1)* %p0, align 2
  ret i16 %val
}

define linkonce_odr void @optix.ptx.st.global.b16( i16 addrspace(1)* %p0, i16 %p1) #1 {
  store i16 %p1, i16 addrspace(1)* %p0
  ret void
}

define linkonce_odr i16 @optix.ptx.ld.global.u16( i16 addrspace(1)* %p0 ) #2 {
  %val = load i16, i16 addrspace(1)* %p0, align 2
  ret i16 %val
}

define linkonce_odr i16 @optix.ptx.ldu.global.u16( i16 addrspace(1)* %p0 ) #2 {
  %val = load i16, i16 addrspace(1)* %p0, align 2
  ret i16 %val
}

define linkonce_odr i16 @optix.ptx.ld.volatile.global.u16( i16 addrspace(1)* %p0 ) #1 {
  %val = load volatile i16, i16 addrspace(1)* %p0, align 2
  ret i16 %val
}

define linkonce_odr void @optix.ptx.st.global.u16( i16 addrspace(1)* %p0, i16 %p1) #1 {
  store i16 %p1, i16 addrspace(1)* %p0
  ret void
}

define linkonce_odr i16 @optix.ptx.ld.global.s16( i16 addrspace(1)* %p0 ) #2 {
  %val = load i16, i16 addrspace(1)* %p0, align 2
  ret i16 %val
}

define linkonce_odr i16 @optix.ptx.ldu.global.s16( i16 addrspace(1)* %p0 ) #2 {
  %val = load i16, i16 addrspace(1)* %p0, align 2
  ret i16 %val
}

define linkonce_odr i16 @optix.ptx.ld.volatile.global.s16( i16 addrspace(1)* %p0 ) #1 {
  %val = load volatile i16, i16 addrspace(1)* %p0, align 2
  ret i16 %val
}

define linkonce_odr void @optix.ptx.st.global.s16( i16 addrspace(1)* %p0, i16 %p1) #1 {
  store i16 %p1, i16 addrspace(1)* %p0
  ret void
}

define linkonce_odr i32 @optix.ptx.ld.global.b32( i32 addrspace(1)* %p0 ) #2 {
  %val = load i32, i32 addrspace(1)* %p0, align 4
  ret i32 %val
}

define linkonce_odr i32 @optix.ptx.ldu.global.b32( i32 addrspace(1)* %p0 ) #2 {
  %val = load i32, i32 addrspace(1)* %p0, align 4
  ret i32 %val
}

define linkonce_odr i32 @optix.ptx.ld.volatile.global.b32( i32 addrspace(1)* %p0 ) #1 {
  %val = load volatile i32, i32 addrspace(1)* %p0, align 4
  ret i32 %val
}

define linkonce_odr void @optix.ptx.st.global.b32( i32 addrspace(1)* %p0, i32 %p1) #1 {
  store i32 %p1, i32 addrspace(1)* %p0
  ret void
}

define linkonce_odr i32 @optix.ptx.ld.global.u32( i32 addrspace(1)* %p0 ) #2 {
  %val = load i32, i32 addrspace(1)* %p0, align 4
  ret i32 %val
}

define linkonce_odr i32 @optix.ptx.ldu.global.u32( i32 addrspace(1)* %p0 ) #2 {
  %val = load i32, i32 addrspace(1)* %p0, align 4
  ret i32 %val
}

define linkonce_odr i32 @optix.ptx.ld.volatile.global.u32( i32 addrspace(1)* %p0 ) #1 {
  %val = load volatile i32, i32 addrspace(1)* %p0, align 4
  ret i32 %val
}

define linkonce_odr void @optix.ptx.st.global.u32( i32 addrspace(1)* %p0, i32 %p1) #1 {
  store i32 %p1, i32 addrspace(1)* %p0
  ret void
}

define linkonce_odr i32 @optix.ptx.ld.global.s32( i32 addrspace(1)* %p0 ) #2 {
  %val = load i32, i32 addrspace(1)* %p0, align 4
  ret i32 %val
}

define linkonce_odr i32 @optix.ptx.ldu.global.s32( i32 addrspace(1)* %p0 ) #2 {
  %val = load i32, i32 addrspace(1)* %p0, align 4
  ret i32 %val
}

define linkonce_odr i32 @optix.ptx.ld.volatile.global.s32( i32 addrspace(1)* %p0 ) #1 {
  %val = load volatile i32, i32 addrspace(1)* %p0, align 4
  ret i32 %val
}

define linkonce_odr void @optix.ptx.st.global.s32( i32 addrspace(1)* %p0, i32 %p1) #1 {
  store i32 %p1, i32 addrspace(1)* %p0
  ret void
}

define linkonce_odr i64 @optix.ptx.ld.global.b64( i64 addrspace(1)* %p0 ) #2 {
  %val = load i64, i64 addrspace(1)* %p0, align 8
  ret i64 %val
}

define linkonce_odr i64 @optix.ptx.ldu.global.b64( i64 addrspace(1)* %p0 ) #2 {
  %val = load i64, i64 addrspace(1)* %p0, align 8
  ret i64 %val
}

define linkonce_odr i64 @optix.ptx.ld.volatile.global.b64( i64 addrspace(1)* %p0 ) #1 {
  %val = load volatile i64, i64 addrspace(1)* %p0, align 8
  ret i64 %val
}

define linkonce_odr void @optix.ptx.st.global.b64( i64 addrspace(1)* %p0, i64 %p1) #1 {
  store i64 %p1, i64 addrspace(1)* %p0
  ret void
}

define linkonce_odr i64 @optix.ptx.ld.global.u64( i64 addrspace(1)* %p0 ) #2 {
  %val = load i64, i64 addrspace(1)* %p0, align 8
  ret i64 %val
}

define linkonce_odr i64 @optix.ptx.ldu.global.u64( i64 addrspace(1)* %p0 ) #2 {
  %val = load i64, i64 addrspace(1)* %p0, align 8
  ret i64 %val
}

define linkonce_odr i64 @optix.ptx.ld.volatile.global.u64( i64 addrspace(1)* %p0 ) #1 {
  %val = load volatile i64, i64 addrspace(1)* %p0, align 8
  ret i64 %val
}

define linkonce_odr void @optix.ptx.st.global.u64( i64 addrspace(1)* %p0, i64 %p1) #1 {
  store i64 %p1, i64 addrspace(1)* %p0
  ret void
}

define linkonce_odr i64 @optix.ptx.ld.global.s64( i64 addrspace(1)* %p0 ) #2 {
  %val = load i64, i64 addrspace(1)* %p0, align 8
  ret i64 %val
}

define linkonce_odr i64 @optix.ptx.ldu.global.s64( i64 addrspace(1)* %p0 ) #2 {
  %val = load i64, i64 addrspace(1)* %p0, align 8
  ret i64 %val
}

define linkonce_odr i64 @optix.ptx.ld.volatile.global.s64( i64 addrspace(1)* %p0 ) #1 {
  %val = load volatile i64, i64 addrspace(1)* %p0, align 8
  ret i64 %val
}

define linkonce_odr void @optix.ptx.st.global.s64( i64 addrspace(1)* %p0, i64 %p1) #1 {
  store i64 %p1, i64 addrspace(1)* %p0
  ret void
}

define linkonce_odr float @optix.ptx.ld.global.f32( float addrspace(1)* %p0 ) #2 {
  %val = load float, float addrspace(1)* %p0, align 4
  ret float %val
}

define linkonce_odr float @optix.ptx.ldu.global.f32( float addrspace(1)* %p0 ) #2 {
  %val = load float, float addrspace(1)* %p0, align 4
  ret float %val
}

define linkonce_odr float @optix.ptx.ld.volatile.global.f32( float addrspace(1)* %p0 ) #1 {
  %val = load volatile float, float addrspace(1)* %p0, align 4
  ret float %val
}

define linkonce_odr void @optix.ptx.st.global.f32( float addrspace(1)* %p0, float %p1) #1 {
  store float %p1, float addrspace(1)* %p0
  ret void
}

define linkonce_odr double @optix.ptx.ld.global.f64( double addrspace(1)* %p0 ) #2 {
  %val = load double, double addrspace(1)* %p0, align 8
  ret double %val
}

define linkonce_odr double @optix.ptx.ldu.global.f64( double addrspace(1)* %p0 ) #2 {
  %val = load double, double addrspace(1)* %p0, align 8
  ret double %val
}

define linkonce_odr double @optix.ptx.ld.volatile.global.f64( double addrspace(1)* %p0 ) #1 {
  %val = load volatile double, double addrspace(1)* %p0, align 8
  ret double %val
}

define linkonce_odr void @optix.ptx.st.global.f64( double addrspace(1)* %p0, double %p1) #1 {
  store double %p1, double addrspace(1)* %p0
  ret void
}

define linkonce_odr <2 x i8> @optix.ptx.ld.global.v2.b8( <2 x i8> addrspace(1)* %p0 ) #2 {
  %val = load <2 x i8>, <2 x i8> addrspace(1)* %p0, align 2
  ret <2 x i8> %val
}

define linkonce_odr <2 x i8> @optix.ptx.ldu.global.v2.b8( <2 x i8> addrspace(1)* %p0 ) #2 {
  %val = load <2 x i8>, <2 x i8> addrspace(1)* %p0, align 2
  ret <2 x i8> %val
}

define linkonce_odr <2 x i8> @optix.ptx.ld.volatile.global.v2.b8( <2 x i8> addrspace(1)* %p0 ) #1 {
  %val = load volatile <2 x i8>, <2 x i8> addrspace(1)* %p0, align 2
  ret <2 x i8> %val
}

define linkonce_odr void @optix.ptx.st.global.v2.b8( <2 x i8> addrspace(1)* %p0, <2 x i8> %p1) #1 {
  store <2 x i8> %p1, <2 x i8> addrspace(1)* %p0
  ret void
}

define linkonce_odr <2 x i8> @optix.ptx.ld.global.v2.u8( <2 x i8> addrspace(1)* %p0 ) #2 {
  %val = load <2 x i8>, <2 x i8> addrspace(1)* %p0, align 2
  ret <2 x i8> %val
}

define linkonce_odr <2 x i8> @optix.ptx.ldu.global.v2.u8( <2 x i8> addrspace(1)* %p0 ) #2 {
  %val = load <2 x i8>, <2 x i8> addrspace(1)* %p0, align 2
  ret <2 x i8> %val
}

define linkonce_odr <2 x i8> @optix.ptx.ld.volatile.global.v2.u8( <2 x i8> addrspace(1)* %p0 ) #1 {
  %val = load volatile <2 x i8>, <2 x i8> addrspace(1)* %p0, align 2
  ret <2 x i8> %val
}

define linkonce_odr void @optix.ptx.st.global.v2.u8( <2 x i8> addrspace(1)* %p0, <2 x i8> %p1) #1 {
  store <2 x i8> %p1, <2 x i8> addrspace(1)* %p0
  ret void
}

define linkonce_odr <2 x i8> @optix.ptx.ld.global.v2.s8( <2 x i8> addrspace(1)* %p0 ) #2 {
  %val = load <2 x i8>, <2 x i8> addrspace(1)* %p0, align 2
  ret <2 x i8> %val
}

define linkonce_odr <2 x i8> @optix.ptx.ldu.global.v2.s8( <2 x i8> addrspace(1)* %p0 ) #2 {
  %val = load <2 x i8>, <2 x i8> addrspace(1)* %p0, align 2
  ret <2 x i8> %val
}

define linkonce_odr <2 x i8> @optix.ptx.ld.volatile.global.v2.s8( <2 x i8> addrspace(1)* %p0 ) #1 {
  %val = load volatile <2 x i8>, <2 x i8> addrspace(1)* %p0, align 2
  ret <2 x i8> %val
}

define linkonce_odr void @optix.ptx.st.global.v2.s8( <2 x i8> addrspace(1)* %p0, <2 x i8> %p1) #1 {
  store <2 x i8> %p1, <2 x i8> addrspace(1)* %p0
  ret void
}

define linkonce_odr <2 x i16> @optix.ptx.ld.global.v2.b16( <2 x i16> addrspace(1)* %p0 ) #2 {
  %val = load <2 x i16>, <2 x i16> addrspace(1)* %p0, align 4
  ret <2 x i16> %val
}

define linkonce_odr <2 x i16> @optix.ptx.ldu.global.v2.b16( <2 x i16> addrspace(1)* %p0 ) #2 {
  %val = load <2 x i16>, <2 x i16> addrspace(1)* %p0, align 4
  ret <2 x i16> %val
}

define linkonce_odr <2 x i16> @optix.ptx.ld.volatile.global.v2.b16( <2 x i16> addrspace(1)* %p0 ) #1 {
  %val = load volatile <2 x i16>, <2 x i16> addrspace(1)* %p0, align 4
  ret <2 x i16> %val
}

define linkonce_odr void @optix.ptx.st.global.v2.b16( <2 x i16> addrspace(1)* %p0, <2 x i16> %p1) #1 {
  store <2 x i16> %p1, <2 x i16> addrspace(1)* %p0
  ret void
}

define linkonce_odr <2 x i16> @optix.ptx.ld.global.v2.u16( <2 x i16> addrspace(1)* %p0 ) #2 {
  %val = load <2 x i16>, <2 x i16> addrspace(1)* %p0, align 4
  ret <2 x i16> %val
}

define linkonce_odr <2 x i16> @optix.ptx.ldu.global.v2.u16( <2 x i16> addrspace(1)* %p0 ) #2 {
  %val = load <2 x i16>, <2 x i16> addrspace(1)* %p0, align 4
  ret <2 x i16> %val
}

define linkonce_odr <2 x i16> @optix.ptx.ld.volatile.global.v2.u16( <2 x i16> addrspace(1)* %p0 ) #1 {
  %val = load volatile <2 x i16>, <2 x i16> addrspace(1)* %p0, align 4
  ret <2 x i16> %val
}

define linkonce_odr void @optix.ptx.st.global.v2.u16( <2 x i16> addrspace(1)* %p0, <2 x i16> %p1) #1 {
  store <2 x i16> %p1, <2 x i16> addrspace(1)* %p0
  ret void
}

define linkonce_odr <2 x i16> @optix.ptx.ld.global.v2.s16( <2 x i16> addrspace(1)* %p0 ) #2 {
  %val = load <2 x i16>, <2 x i16> addrspace(1)* %p0, align 4
  ret <2 x i16> %val
}

define linkonce_odr <2 x i16> @optix.ptx.ldu.global.v2.s16( <2 x i16> addrspace(1)* %p0 ) #2 {
  %val = load <2 x i16>, <2 x i16> addrspace(1)* %p0, align 4
  ret <2 x i16> %val
}

define linkonce_odr <2 x i16> @optix.ptx.ld.volatile.global.v2.s16( <2 x i16> addrspace(1)* %p0 ) #1 {
  %val = load volatile <2 x i16>, <2 x i16> addrspace(1)* %p0, align 4
  ret <2 x i16> %val
}

define linkonce_odr void @optix.ptx.st.global.v2.s16( <2 x i16> addrspace(1)* %p0, <2 x i16> %p1) #1 {
  store <2 x i16> %p1, <2 x i16> addrspace(1)* %p0
  ret void
}

define linkonce_odr <2 x i32> @optix.ptx.ld.global.v2.b32( <2 x i32> addrspace(1)* %p0 ) #2 {
  %val = load <2 x i32>, <2 x i32> addrspace(1)* %p0, align 8
  ret <2 x i32> %val
}

define linkonce_odr <2 x i32> @optix.ptx.ldu.global.v2.b32( <2 x i32> addrspace(1)* %p0 ) #2 {
  %val = load <2 x i32>, <2 x i32> addrspace(1)* %p0, align 8
  ret <2 x i32> %val
}

define linkonce_odr <2 x i32> @optix.ptx.ld.volatile.global.v2.b32( <2 x i32> addrspace(1)* %p0 ) #1 {
  %val = load volatile <2 x i32>, <2 x i32> addrspace(1)* %p0, align 8
  ret <2 x i32> %val
}

define linkonce_odr void @optix.ptx.st.global.v2.b32( <2 x i32> addrspace(1)* %p0, <2 x i32> %p1) #1 {
  store <2 x i32> %p1, <2 x i32> addrspace(1)* %p0
  ret void
}

define linkonce_odr <2 x i32> @optix.ptx.ld.global.v2.u32( <2 x i32> addrspace(1)* %p0 ) #2 {
  %val = load <2 x i32>, <2 x i32> addrspace(1)* %p0, align 8
  ret <2 x i32> %val
}

define linkonce_odr <2 x i32> @optix.ptx.ldu.global.v2.u32( <2 x i32> addrspace(1)* %p0 ) #2 {
  %val = load <2 x i32>, <2 x i32> addrspace(1)* %p0, align 8
  ret <2 x i32> %val
}

define linkonce_odr <2 x i32> @optix.ptx.ld.volatile.global.v2.u32( <2 x i32> addrspace(1)* %p0 ) #1 {
  %val = load volatile <2 x i32>, <2 x i32> addrspace(1)* %p0, align 8
  ret <2 x i32> %val
}

define linkonce_odr void @optix.ptx.st.global.v2.u32( <2 x i32> addrspace(1)* %p0, <2 x i32> %p1) #1 {
  store <2 x i32> %p1, <2 x i32> addrspace(1)* %p0
  ret void
}

define linkonce_odr <2 x i32> @optix.ptx.ld.global.v2.s32( <2 x i32> addrspace(1)* %p0 ) #2 {
  %val = load <2 x i32>, <2 x i32> addrspace(1)* %p0, align 8
  ret <2 x i32> %val
}

define linkonce_odr <2 x i32> @optix.ptx.ldu.global.v2.s32( <2 x i32> addrspace(1)* %p0 ) #2 {
  %val = load <2 x i32>, <2 x i32> addrspace(1)* %p0, align 8
  ret <2 x i32> %val
}

define linkonce_odr <2 x i32> @optix.ptx.ld.volatile.global.v2.s32( <2 x i32> addrspace(1)* %p0 ) #1 {
  %val = load volatile <2 x i32>, <2 x i32> addrspace(1)* %p0, align 8
  ret <2 x i32> %val
}

define linkonce_odr void @optix.ptx.st.global.v2.s32( <2 x i32> addrspace(1)* %p0, <2 x i32> %p1) #1 {
  store <2 x i32> %p1, <2 x i32> addrspace(1)* %p0
  ret void
}

define linkonce_odr <2 x i64> @optix.ptx.ld.global.v2.b64( <2 x i64> addrspace(1)* %p0 ) #2 {
  %val = load <2 x i64>, <2 x i64> addrspace(1)* %p0, align 16
  ret <2 x i64> %val
}

define linkonce_odr <2 x i64> @optix.ptx.ldu.global.v2.b64( <2 x i64> addrspace(1)* %p0 ) #2 {
  %val = load <2 x i64>, <2 x i64> addrspace(1)* %p0, align 16
  ret <2 x i64> %val
}

define linkonce_odr <2 x i64> @optix.ptx.ld.volatile.global.v2.b64( <2 x i64> addrspace(1)* %p0 ) #1 {
  %val = load volatile <2 x i64>, <2 x i64> addrspace(1)* %p0, align 16
  ret <2 x i64> %val
}

define linkonce_odr void @optix.ptx.st.global.v2.b64( <2 x i64> addrspace(1)* %p0, <2 x i64> %p1) #1 {
  store <2 x i64> %p1, <2 x i64> addrspace(1)* %p0
  ret void
}

define linkonce_odr <2 x i64> @optix.ptx.ld.global.v2.u64( <2 x i64> addrspace(1)* %p0 ) #2 {
  %val = load <2 x i64>, <2 x i64> addrspace(1)* %p0, align 16
  ret <2 x i64> %val
}

define linkonce_odr <2 x i64> @optix.ptx.ldu.global.v2.u64( <2 x i64> addrspace(1)* %p0 ) #2 {
  %val = load <2 x i64>, <2 x i64> addrspace(1)* %p0, align 16
  ret <2 x i64> %val
}

define linkonce_odr <2 x i64> @optix.ptx.ld.volatile.global.v2.u64( <2 x i64> addrspace(1)* %p0 ) #1 {
  %val = load volatile <2 x i64>, <2 x i64> addrspace(1)* %p0, align 16
  ret <2 x i64> %val
}

define linkonce_odr void @optix.ptx.st.global.v2.u64( <2 x i64> addrspace(1)* %p0, <2 x i64> %p1) #1 {
  store <2 x i64> %p1, <2 x i64> addrspace(1)* %p0
  ret void
}

define linkonce_odr <2 x i64> @optix.ptx.ld.global.v2.s64( <2 x i64> addrspace(1)* %p0 ) #2 {
  %val = load <2 x i64>, <2 x i64> addrspace(1)* %p0, align 16
  ret <2 x i64> %val
}

define linkonce_odr <2 x i64> @optix.ptx.ldu.global.v2.s64( <2 x i64> addrspace(1)* %p0 ) #2 {
  %val = load <2 x i64>, <2 x i64> addrspace(1)* %p0, align 16
  ret <2 x i64> %val
}

define linkonce_odr <2 x i64> @optix.ptx.ld.volatile.global.v2.s64( <2 x i64> addrspace(1)* %p0 ) #1 {
  %val = load volatile <2 x i64>, <2 x i64> addrspace(1)* %p0, align 16
  ret <2 x i64> %val
}

define linkonce_odr void @optix.ptx.st.global.v2.s64( <2 x i64> addrspace(1)* %p0, <2 x i64> %p1) #1 {
  store <2 x i64> %p1, <2 x i64> addrspace(1)* %p0
  ret void
}

define linkonce_odr <2 x float> @optix.ptx.ld.global.v2.f32( <2 x float> addrspace(1)* %p0 ) #2 {
  %val = load <2 x float>, <2 x float> addrspace(1)* %p0, align 8
  ret <2 x float> %val
}

define linkonce_odr <2 x float> @optix.ptx.ldu.global.v2.f32( <2 x float> addrspace(1)* %p0 ) #2 {
  %val = load <2 x float>, <2 x float> addrspace(1)* %p0, align 8
  ret <2 x float> %val
}

define linkonce_odr <2 x float> @optix.ptx.ld.volatile.global.v2.f32( <2 x float> addrspace(1)* %p0 ) #1 {
  %val = load volatile <2 x float>, <2 x float> addrspace(1)* %p0, align 8
  ret <2 x float> %val
}

define linkonce_odr void @optix.ptx.st.global.v2.f32( <2 x float> addrspace(1)* %p0, <2 x float> %p1) #1 {
  store <2 x float> %p1, <2 x float> addrspace(1)* %p0
  ret void
}

define linkonce_odr <2 x double> @optix.ptx.ld.global.v2.f64( <2 x double> addrspace(1)* %p0 ) #2 {
  %val = load <2 x double>, <2 x double> addrspace(1)* %p0, align 16
  ret <2 x double> %val
}

define linkonce_odr <2 x double> @optix.ptx.ldu.global.v2.f64( <2 x double> addrspace(1)* %p0 ) #2 {
  %val = load <2 x double>, <2 x double> addrspace(1)* %p0, align 16
  ret <2 x double> %val
}

define linkonce_odr <2 x double> @optix.ptx.ld.volatile.global.v2.f64( <2 x double> addrspace(1)* %p0 ) #1 {
  %val = load volatile <2 x double>, <2 x double> addrspace(1)* %p0, align 16
  ret <2 x double> %val
}

define linkonce_odr void @optix.ptx.st.global.v2.f64( <2 x double> addrspace(1)* %p0, <2 x double> %p1) #1 {
  store <2 x double> %p1, <2 x double> addrspace(1)* %p0
  ret void
}

define linkonce_odr <4 x i8> @optix.ptx.ld.global.v4.b8( <4 x i8> addrspace(1)* %p0 ) #2 {
  %val = load <4 x i8>, <4 x i8> addrspace(1)* %p0, align 4
  ret <4 x i8> %val
}

define linkonce_odr <4 x i8> @optix.ptx.ldu.global.v4.b8( <4 x i8> addrspace(1)* %p0 ) #2 {
  %val = load <4 x i8>, <4 x i8> addrspace(1)* %p0, align 4
  ret <4 x i8> %val
}

define linkonce_odr <4 x i8> @optix.ptx.ld.volatile.global.v4.b8( <4 x i8> addrspace(1)* %p0 ) #1 {
  %val = load volatile <4 x i8>, <4 x i8> addrspace(1)* %p0, align 4
  ret <4 x i8> %val
}

define linkonce_odr void @optix.ptx.st.global.v4.b8( <4 x i8> addrspace(1)* %p0, <4 x i8> %p1) #1 {
  store <4 x i8> %p1, <4 x i8> addrspace(1)* %p0
  ret void
}

define linkonce_odr <4 x i8> @optix.ptx.ld.global.v4.u8( <4 x i8> addrspace(1)* %p0 ) #2 {
  %val = load <4 x i8>, <4 x i8> addrspace(1)* %p0, align 4
  ret <4 x i8> %val
}

define linkonce_odr <4 x i8> @optix.ptx.ldu.global.v4.u8( <4 x i8> addrspace(1)* %p0 ) #2 {
  %val = load <4 x i8>, <4 x i8> addrspace(1)* %p0, align 4
  ret <4 x i8> %val
}

define linkonce_odr <4 x i8> @optix.ptx.ld.volatile.global.v4.u8( <4 x i8> addrspace(1)* %p0 ) #1 {
  %val = load volatile <4 x i8>, <4 x i8> addrspace(1)* %p0, align 4
  ret <4 x i8> %val
}

define linkonce_odr void @optix.ptx.st.global.v4.u8( <4 x i8> addrspace(1)* %p0, <4 x i8> %p1) #1 {
  store <4 x i8> %p1, <4 x i8> addrspace(1)* %p0
  ret void
}

define linkonce_odr <4 x i8> @optix.ptx.ld.global.v4.s8( <4 x i8> addrspace(1)* %p0 ) #2 {
  %val = load <4 x i8>, <4 x i8> addrspace(1)* %p0, align 4
  ret <4 x i8> %val
}

define linkonce_odr <4 x i8> @optix.ptx.ldu.global.v4.s8( <4 x i8> addrspace(1)* %p0 ) #2 {
  %val = load <4 x i8>, <4 x i8> addrspace(1)* %p0, align 4
  ret <4 x i8> %val
}

define linkonce_odr <4 x i8> @optix.ptx.ld.volatile.global.v4.s8( <4 x i8> addrspace(1)* %p0 ) #1 {
  %val = load volatile <4 x i8>, <4 x i8> addrspace(1)* %p0, align 4
  ret <4 x i8> %val
}

define linkonce_odr void @optix.ptx.st.global.v4.s8( <4 x i8> addrspace(1)* %p0, <4 x i8> %p1) #1 {
  store <4 x i8> %p1, <4 x i8> addrspace(1)* %p0
  ret void
}

define linkonce_odr <4 x i16> @optix.ptx.ld.global.v4.b16( <4 x i16> addrspace(1)* %p0 ) #2 {
  %val = load <4 x i16>, <4 x i16> addrspace(1)* %p0, align 8
  ret <4 x i16> %val
}

define linkonce_odr <4 x i16> @optix.ptx.ldu.global.v4.b16( <4 x i16> addrspace(1)* %p0 ) #2 {
  %val = load <4 x i16>, <4 x i16> addrspace(1)* %p0, align 8
  ret <4 x i16> %val
}

define linkonce_odr <4 x i16> @optix.ptx.ld.volatile.global.v4.b16( <4 x i16> addrspace(1)* %p0 ) #1 {
  %val = load volatile <4 x i16>, <4 x i16> addrspace(1)* %p0, align 8
  ret <4 x i16> %val
}

define linkonce_odr void @optix.ptx.st.global.v4.b16( <4 x i16> addrspace(1)* %p0, <4 x i16> %p1) #1 {
  store <4 x i16> %p1, <4 x i16> addrspace(1)* %p0
  ret void
}

define linkonce_odr <4 x i16> @optix.ptx.ld.global.v4.u16( <4 x i16> addrspace(1)* %p0 ) #2 {
  %val = load <4 x i16>, <4 x i16> addrspace(1)* %p0, align 8
  ret <4 x i16> %val
}

define linkonce_odr <4 x i16> @optix.ptx.ldu.global.v4.u16( <4 x i16> addrspace(1)* %p0 ) #2 {
  %val = load <4 x i16>, <4 x i16> addrspace(1)* %p0, align 8
  ret <4 x i16> %val
}

define linkonce_odr <4 x i16> @optix.ptx.ld.volatile.global.v4.u16( <4 x i16> addrspace(1)* %p0 ) #1 {
  %val = load volatile <4 x i16>, <4 x i16> addrspace(1)* %p0, align 8
  ret <4 x i16> %val
}

define linkonce_odr void @optix.ptx.st.global.v4.u16( <4 x i16> addrspace(1)* %p0, <4 x i16> %p1) #1 {
  store <4 x i16> %p1, <4 x i16> addrspace(1)* %p0
  ret void
}

define linkonce_odr <4 x i16> @optix.ptx.ld.global.v4.s16( <4 x i16> addrspace(1)* %p0 ) #2 {
  %val = load <4 x i16>, <4 x i16> addrspace(1)* %p0, align 8
  ret <4 x i16> %val
}

define linkonce_odr <4 x i16> @optix.ptx.ldu.global.v4.s16( <4 x i16> addrspace(1)* %p0 ) #2 {
  %val = load <4 x i16>, <4 x i16> addrspace(1)* %p0, align 8
  ret <4 x i16> %val
}

define linkonce_odr <4 x i16> @optix.ptx.ld.volatile.global.v4.s16( <4 x i16> addrspace(1)* %p0 ) #1 {
  %val = load volatile <4 x i16>, <4 x i16> addrspace(1)* %p0, align 8
  ret <4 x i16> %val
}

define linkonce_odr void @optix.ptx.st.global.v4.s16( <4 x i16> addrspace(1)* %p0, <4 x i16> %p1) #1 {
  store <4 x i16> %p1, <4 x i16> addrspace(1)* %p0
  ret void
}

define linkonce_odr <4 x i32> @optix.ptx.ld.global.v4.b32( <4 x i32> addrspace(1)* %p0 ) #2 {
  %val = load <4 x i32>, <4 x i32> addrspace(1)* %p0, align 16
  ret <4 x i32> %val
}

define linkonce_odr <4 x i32> @optix.ptx.ldu.global.v4.b32( <4 x i32> addrspace(1)* %p0 ) #2 {
  %val = load <4 x i32>, <4 x i32> addrspace(1)* %p0, align 16
  ret <4 x i32> %val
}

define linkonce_odr <4 x i32> @optix.ptx.ld.volatile.global.v4.b32( <4 x i32> addrspace(1)* %p0 ) #1 {
  %val = load volatile <4 x i32>, <4 x i32> addrspace(1)* %p0, align 16
  ret <4 x i32> %val
}

define linkonce_odr void @optix.ptx.st.global.v4.b32( <4 x i32> addrspace(1)* %p0, <4 x i32> %p1) #1 {
  store <4 x i32> %p1, <4 x i32> addrspace(1)* %p0
  ret void
}

define linkonce_odr <4 x i32> @optix.ptx.ld.global.v4.u32( <4 x i32> addrspace(1)* %p0 ) #2 {
  %val = load <4 x i32>, <4 x i32> addrspace(1)* %p0, align 16
  ret <4 x i32> %val
}

define linkonce_odr <4 x i32> @optix.ptx.ldu.global.v4.u32( <4 x i32> addrspace(1)* %p0 ) #2 {
  %val = load <4 x i32>, <4 x i32> addrspace(1)* %p0, align 16
  ret <4 x i32> %val
}

define linkonce_odr <4 x i32> @optix.ptx.ld.volatile.global.v4.u32( <4 x i32> addrspace(1)* %p0 ) #1 {
  %val = load volatile <4 x i32>, <4 x i32> addrspace(1)* %p0, align 16
  ret <4 x i32> %val
}

define linkonce_odr void @optix.ptx.st.global.v4.u32( <4 x i32> addrspace(1)* %p0, <4 x i32> %p1) #1 {
  store <4 x i32> %p1, <4 x i32> addrspace(1)* %p0
  ret void
}

define linkonce_odr <4 x i32> @optix.ptx.ld.global.v4.s32( <4 x i32> addrspace(1)* %p0 ) #2 {
  %val = load <4 x i32>, <4 x i32> addrspace(1)* %p0, align 16
  ret <4 x i32> %val
}

define linkonce_odr <4 x i32> @optix.ptx.ldu.global.v4.s32( <4 x i32> addrspace(1)* %p0 ) #2 {
  %val = load <4 x i32>, <4 x i32> addrspace(1)* %p0, align 16
  ret <4 x i32> %val
}

define linkonce_odr <4 x i32> @optix.ptx.ld.volatile.global.v4.s32( <4 x i32> addrspace(1)* %p0 ) #1 {
  %val = load volatile <4 x i32>, <4 x i32> addrspace(1)* %p0, align 16
  ret <4 x i32> %val
}

define linkonce_odr void @optix.ptx.st.global.v4.s32( <4 x i32> addrspace(1)* %p0, <4 x i32> %p1) #1 {
  store <4 x i32> %p1, <4 x i32> addrspace(1)* %p0
  ret void
}

define linkonce_odr <4 x i64> @optix.ptx.ld.global.v4.b64( <4 x i64> addrspace(1)* %p0 ) #2 {
  %val = load <4 x i64>, <4 x i64> addrspace(1)* %p0, align 32
  ret <4 x i64> %val
}

define linkonce_odr <4 x i64> @optix.ptx.ldu.global.v4.b64( <4 x i64> addrspace(1)* %p0 ) #2 {
  %val = load <4 x i64>, <4 x i64> addrspace(1)* %p0, align 32
  ret <4 x i64> %val
}

define linkonce_odr <4 x i64> @optix.ptx.ld.volatile.global.v4.b64( <4 x i64> addrspace(1)* %p0 ) #1 {
  %val = load volatile <4 x i64>, <4 x i64> addrspace(1)* %p0, align 32
  ret <4 x i64> %val
}

define linkonce_odr void @optix.ptx.st.global.v4.b64( <4 x i64> addrspace(1)* %p0, <4 x i64> %p1) #1 {
  store <4 x i64> %p1, <4 x i64> addrspace(1)* %p0
  ret void
}

define linkonce_odr <4 x i64> @optix.ptx.ld.global.v4.u64( <4 x i64> addrspace(1)* %p0 ) #2 {
  %val = load <4 x i64>, <4 x i64> addrspace(1)* %p0, align 32
  ret <4 x i64> %val
}

define linkonce_odr <4 x i64> @optix.ptx.ldu.global.v4.u64( <4 x i64> addrspace(1)* %p0 ) #2 {
  %val = load <4 x i64>, <4 x i64> addrspace(1)* %p0, align 32
  ret <4 x i64> %val
}

define linkonce_odr <4 x i64> @optix.ptx.ld.volatile.global.v4.u64( <4 x i64> addrspace(1)* %p0 ) #1 {
  %val = load volatile <4 x i64>, <4 x i64> addrspace(1)* %p0, align 32
  ret <4 x i64> %val
}

define linkonce_odr void @optix.ptx.st.global.v4.u64( <4 x i64> addrspace(1)* %p0, <4 x i64> %p1) #1 {
  store <4 x i64> %p1, <4 x i64> addrspace(1)* %p0
  ret void
}

define linkonce_odr <4 x i64> @optix.ptx.ld.global.v4.s64( <4 x i64> addrspace(1)* %p0 ) #2 {
  %val = load <4 x i64>, <4 x i64> addrspace(1)* %p0, align 32
  ret <4 x i64> %val
}

define linkonce_odr <4 x i64> @optix.ptx.ldu.global.v4.s64( <4 x i64> addrspace(1)* %p0 ) #2 {
  %val = load <4 x i64>, <4 x i64> addrspace(1)* %p0, align 32
  ret <4 x i64> %val
}

define linkonce_odr <4 x i64> @optix.ptx.ld.volatile.global.v4.s64( <4 x i64> addrspace(1)* %p0 ) #1 {
  %val = load volatile <4 x i64>, <4 x i64> addrspace(1)* %p0, align 32
  ret <4 x i64> %val
}

define linkonce_odr void @optix.ptx.st.global.v4.s64( <4 x i64> addrspace(1)* %p0, <4 x i64> %p1) #1 {
  store <4 x i64> %p1, <4 x i64> addrspace(1)* %p0
  ret void
}

define linkonce_odr <4 x float> @optix.ptx.ld.global.v4.f32( <4 x float> addrspace(1)* %p0 ) #2 {
  %val = load <4 x float>, <4 x float> addrspace(1)* %p0, align 16
  ret <4 x float> %val
}

define linkonce_odr <4 x float> @optix.ptx.ldu.global.v4.f32( <4 x float> addrspace(1)* %p0 ) #2 {
  %val = load <4 x float>, <4 x float> addrspace(1)* %p0, align 16
  ret <4 x float> %val
}

define linkonce_odr <4 x float> @optix.ptx.ld.volatile.global.v4.f32( <4 x float> addrspace(1)* %p0 ) #1 {
  %val = load volatile <4 x float>, <4 x float> addrspace(1)* %p0, align 16
  ret <4 x float> %val
}

define linkonce_odr void @optix.ptx.st.global.v4.f32( <4 x float> addrspace(1)* %p0, <4 x float> %p1) #1 {
  store <4 x float> %p1, <4 x float> addrspace(1)* %p0
  ret void
}

define linkonce_odr <4 x double> @optix.ptx.ld.global.v4.f64( <4 x double> addrspace(1)* %p0 ) #2 {
  %val = load <4 x double>, <4 x double> addrspace(1)* %p0, align 32
  ret <4 x double> %val
}

define linkonce_odr <4 x double> @optix.ptx.ldu.global.v4.f64( <4 x double> addrspace(1)* %p0 ) #2 {
  %val = load <4 x double>, <4 x double> addrspace(1)* %p0, align 32
  ret <4 x double> %val
}

define linkonce_odr <4 x double> @optix.ptx.ld.volatile.global.v4.f64( <4 x double> addrspace(1)* %p0 ) #1 {
  %val = load volatile <4 x double>, <4 x double> addrspace(1)* %p0, align 32
  ret <4 x double> %val
}

define linkonce_odr void @optix.ptx.st.global.v4.f64( <4 x double> addrspace(1)* %p0, <4 x double> %p1) #1 {
  store <4 x double> %p1, <4 x double> addrspace(1)* %p0
  ret void
}

define linkonce_odr i8 @optix.ptx.ld.shared.b8( i8 addrspace(3)* %p0 ) #2 {
  %val = load i8, i8 addrspace(3)* %p0, align 1
  ret i8 %val
}

define linkonce_odr i8 @optix.ptx.ldu.shared.b8( i8 addrspace(3)* %p0 ) #2 {
  %val = load i8, i8 addrspace(3)* %p0, align 1
  ret i8 %val
}

define linkonce_odr i8 @optix.ptx.ld.volatile.shared.b8( i8 addrspace(3)* %p0 ) #1 {
  %val = load volatile i8, i8 addrspace(3)* %p0, align 1
  ret i8 %val
}

define linkonce_odr void @optix.ptx.st.shared.b8( i8 addrspace(3)* %p0, i8 %p1) #1 {
  store i8 %p1, i8 addrspace(3)* %p0
  ret void
}

define linkonce_odr i8 @optix.ptx.ld.shared.u8( i8 addrspace(3)* %p0 ) #2 {
  %val = load i8, i8 addrspace(3)* %p0, align 1
  ret i8 %val
}

define linkonce_odr i8 @optix.ptx.ldu.shared.u8( i8 addrspace(3)* %p0 ) #2 {
  %val = load i8, i8 addrspace(3)* %p0, align 1
  ret i8 %val
}

define linkonce_odr i8 @optix.ptx.ld.volatile.shared.u8( i8 addrspace(3)* %p0 ) #1 {
  %val = load volatile i8, i8 addrspace(3)* %p0, align 1
  ret i8 %val
}

define linkonce_odr void @optix.ptx.st.shared.u8( i8 addrspace(3)* %p0, i8 %p1) #1 {
  store i8 %p1, i8 addrspace(3)* %p0
  ret void
}

define linkonce_odr i8 @optix.ptx.ld.shared.s8( i8 addrspace(3)* %p0 ) #2 {
  %val = load i8, i8 addrspace(3)* %p0, align 1
  ret i8 %val
}

define linkonce_odr i8 @optix.ptx.ldu.shared.s8( i8 addrspace(3)* %p0 ) #2 {
  %val = load i8, i8 addrspace(3)* %p0, align 1
  ret i8 %val
}

define linkonce_odr i8 @optix.ptx.ld.volatile.shared.s8( i8 addrspace(3)* %p0 ) #1 {
  %val = load volatile i8, i8 addrspace(3)* %p0, align 1
  ret i8 %val
}

define linkonce_odr void @optix.ptx.st.shared.s8( i8 addrspace(3)* %p0, i8 %p1) #1 {
  store i8 %p1, i8 addrspace(3)* %p0
  ret void
}

define linkonce_odr i16 @optix.ptx.ld.shared.b16( i16 addrspace(3)* %p0 ) #2 {
  %val = load i16, i16 addrspace(3)* %p0, align 2
  ret i16 %val
}

define linkonce_odr i16 @optix.ptx.ldu.shared.b16( i16 addrspace(3)* %p0 ) #2 {
  %val = load i16, i16 addrspace(3)* %p0, align 2
  ret i16 %val
}

define linkonce_odr i16 @optix.ptx.ld.volatile.shared.b16( i16 addrspace(3)* %p0 ) #1 {
  %val = load volatile i16, i16 addrspace(3)* %p0, align 2
  ret i16 %val
}

define linkonce_odr void @optix.ptx.st.shared.b16( i16 addrspace(3)* %p0, i16 %p1) #1 {
  store i16 %p1, i16 addrspace(3)* %p0
  ret void
}

define linkonce_odr i16 @optix.ptx.ld.shared.u16( i16 addrspace(3)* %p0 ) #2 {
  %val = load i16, i16 addrspace(3)* %p0, align 2
  ret i16 %val
}

define linkonce_odr i16 @optix.ptx.ldu.shared.u16( i16 addrspace(3)* %p0 ) #2 {
  %val = load i16, i16 addrspace(3)* %p0, align 2
  ret i16 %val
}

define linkonce_odr i16 @optix.ptx.ld.volatile.shared.u16( i16 addrspace(3)* %p0 ) #1 {
  %val = load volatile i16, i16 addrspace(3)* %p0, align 2
  ret i16 %val
}

define linkonce_odr void @optix.ptx.st.shared.u16( i16 addrspace(3)* %p0, i16 %p1) #1 {
  store i16 %p1, i16 addrspace(3)* %p0
  ret void
}

define linkonce_odr i16 @optix.ptx.ld.shared.s16( i16 addrspace(3)* %p0 ) #2 {
  %val = load i16, i16 addrspace(3)* %p0, align 2
  ret i16 %val
}

define linkonce_odr i16 @optix.ptx.ldu.shared.s16( i16 addrspace(3)* %p0 ) #2 {
  %val = load i16, i16 addrspace(3)* %p0, align 2
  ret i16 %val
}

define linkonce_odr i16 @optix.ptx.s16.volatile.shared.ld( i16 addrspace(3)* %p0 ) #1 {
  %val = load volatile i16, i16 addrspace(3)* %p0, align 2
  ret i16 %val
}

define linkonce_odr void @optix.ptx.st.shared.s16( i16 addrspace(3)* %p0, i16 %p1) #1 {
  store i16 %p1, i16 addrspace(3)* %p0
  ret void
}

define linkonce_odr i32 @optix.ptx.ld.shared.b32( i32 addrspace(3)* %p0 ) #2 {
  %val = load i32, i32 addrspace(3)* %p0, align 4
  ret i32 %val
}

define linkonce_odr i32 @optix.ptx.ldu.shared.b32( i32 addrspace(3)* %p0 ) #2 {
  %val = load i32, i32 addrspace(3)* %p0, align 4
  ret i32 %val
}

define linkonce_odr i32 @optix.ptx.ld.volatile.shared.b32( i32 addrspace(3)* %p0 ) #1 {
  %val = load volatile i32, i32 addrspace(3)* %p0, align 4
  ret i32 %val
}

define linkonce_odr void @optix.ptx.st.shared.b32( i32 addrspace(3)* %p0, i32 %p1) #1 {
  store i32 %p1, i32 addrspace(3)* %p0
  ret void
}

define linkonce_odr i32 @optix.ptx.ld.shared.u32( i32 addrspace(3)* %p0 ) #2 {
  %val = load i32, i32 addrspace(3)* %p0, align 4
  ret i32 %val
}

define linkonce_odr i32 @optix.ptx.ldu.shared.u32( i32 addrspace(3)* %p0 ) #2 {
  %val = load i32, i32 addrspace(3)* %p0, align 4
  ret i32 %val
}

define linkonce_odr i32 @optix.ptx.ld.volatile.shared.u32( i32 addrspace(3)* %p0 ) #1 {
  %val = load volatile i32, i32 addrspace(3)* %p0, align 4
  ret i32 %val
}

define linkonce_odr void @optix.ptx.st.shared.u32( i32 addrspace(3)* %p0, i32 %p1) #1 {
  store i32 %p1, i32 addrspace(3)* %p0
  ret void
}

define linkonce_odr i32 @optix.ptx.ld.shared.s32( i32 addrspace(3)* %p0 ) #2 {
  %val = load i32, i32 addrspace(3)* %p0, align 4
  ret i32 %val
}

define linkonce_odr i32 @optix.ptx.ldu.shared.s32( i32 addrspace(3)* %p0 ) #2 {
  %val = load i32, i32 addrspace(3)* %p0, align 4
  ret i32 %val
}

define linkonce_odr i32 @optix.ptx.ld.volatile.shared.s32( i32 addrspace(3)* %p0 ) #1 {
  %val = load volatile i32, i32 addrspace(3)* %p0, align 4
  ret i32 %val
}

define linkonce_odr void @optix.ptx.st.shared.s32( i32 addrspace(3)* %p0, i32 %p1) #1 {
  store i32 %p1, i32 addrspace(3)* %p0
  ret void
}

define linkonce_odr i64 @optix.ptx.ld.shared.b64( i64 addrspace(3)* %p0 ) #2 {
  %val = load i64, i64 addrspace(3)* %p0, align 8
  ret i64 %val
}

define linkonce_odr i64 @optix.ptx.ldu.shared.b64( i64 addrspace(3)* %p0 ) #2 {
  %val = load i64, i64 addrspace(3)* %p0, align 8
  ret i64 %val
}

define linkonce_odr i64 @optix.ptx.ld.volatile.shared.b64( i64 addrspace(3)* %p0 ) #1 {
  %val = load volatile i64, i64 addrspace(3)* %p0, align 8
  ret i64 %val
}

define linkonce_odr void @optix.ptx.st.shared.b64( i64 addrspace(3)* %p0, i64 %p1) #1 {
  store i64 %p1, i64 addrspace(3)* %p0
  ret void
}

define linkonce_odr i64 @optix.ptx.ld.shared.u64( i64 addrspace(3)* %p0 ) #2 {
  %val = load i64, i64 addrspace(3)* %p0, align 8
  ret i64 %val
}

define linkonce_odr i64 @optix.ptx.ldu.shared.u64( i64 addrspace(3)* %p0 ) #2 {
  %val = load i64, i64 addrspace(3)* %p0, align 8
  ret i64 %val
}

define linkonce_odr i64 @optix.ptx.ld.volatile.shared.u64( i64 addrspace(3)* %p0 ) #1 {
  %val = load volatile i64, i64 addrspace(3)* %p0, align 8
  ret i64 %val
}

define linkonce_odr void @optix.ptx.st.shared.u64( i64 addrspace(3)* %p0, i64 %p1) #1 {
  store i64 %p1, i64 addrspace(3)* %p0
  ret void
}

define linkonce_odr i64 @optix.ptx.ld.shared.s64( i64 addrspace(3)* %p0 ) #2 {
  %val = load i64, i64 addrspace(3)* %p0, align 8
  ret i64 %val
}

define linkonce_odr i64 @optix.ptx.ldu.shared.s64( i64 addrspace(3)* %p0 ) #2 {
  %val = load i64, i64 addrspace(3)* %p0, align 8
  ret i64 %val
}

define linkonce_odr i64 @optix.ptx.ld.volatile.shared.s64( i64 addrspace(3)* %p0 ) #1 {
  %val = load volatile i64, i64 addrspace(3)* %p0, align 8
  ret i64 %val
}

define linkonce_odr void @optix.ptx.st.shared.s64( i64 addrspace(3)* %p0, i64 %p1) #1 {
  store i64 %p1, i64 addrspace(3)* %p0
  ret void
}

define linkonce_odr float @optix.ptx.ld.shared.f32( float addrspace(3)* %p0 ) #2 {
  %val = load float, float addrspace(3)* %p0, align 4
  ret float %val
}

define linkonce_odr float @optix.ptx.ldu.shared.f32( float addrspace(3)* %p0 ) #2 {
  %val = load float, float addrspace(3)* %p0, align 4
  ret float %val
}

define linkonce_odr float @optix.ptx.ld.volatile.shared.f32( float addrspace(3)* %p0 ) #1 {
  %val = load volatile float, float addrspace(3)* %p0, align 4
  ret float %val
}

define linkonce_odr void @optix.ptx.st.shared.f32( float addrspace(3)* %p0, float %p1) #1 {
  store float %p1, float addrspace(3)* %p0
  ret void
}

define linkonce_odr double @optix.ptx.ld.shared.f64( double addrspace(3)* %p0 ) #2 {
  %val = load double, double addrspace(3)* %p0, align 8
  ret double %val
}

define linkonce_odr double @optix.ptx.ldu.shared.f64( double addrspace(3)* %p0 ) #2 {
  %val = load double, double addrspace(3)* %p0, align 8
  ret double %val
}

define linkonce_odr double @optix.ptx.ld.volatile.shared.f64( double addrspace(3)* %p0 ) #1 {
  %val = load volatile double, double addrspace(3)* %p0, align 8
  ret double %val
}

define linkonce_odr void @optix.ptx.st.shared.f64( double addrspace(3)* %p0, double %p1) #1 {
  store double %p1, double addrspace(3)* %p0
  ret void
}

define linkonce_odr <2 x i8> @optix.ptx.ld.shared.v2.b8( <2 x i8> addrspace(3)* %p0 ) #2 {
  %val = load <2 x i8>, <2 x i8> addrspace(3)* %p0, align 2
  ret <2 x i8> %val
}

define linkonce_odr <2 x i8> @optix.ptx.ldu.shared.v2.b8( <2 x i8> addrspace(3)* %p0 ) #2 {
  %val = load <2 x i8>, <2 x i8> addrspace(3)* %p0, align 2
  ret <2 x i8> %val
}

define linkonce_odr <2 x i8> @optix.ptx.ld.volatile.shared.v2.b8( <2 x i8> addrspace(3)* %p0 ) #1 {
  %val = load volatile <2 x i8>, <2 x i8> addrspace(3)* %p0, align 2
  ret <2 x i8> %val
}

define linkonce_odr void @optix.ptx.st.shared.v2.b8( <2 x i8> addrspace(3)* %p0, <2 x i8> %p1) #1 {
  store <2 x i8> %p1, <2 x i8> addrspace(3)* %p0
  ret void
}

define linkonce_odr <2 x i8> @optix.ptx.ld.shared.v2.u8( <2 x i8> addrspace(3)* %p0 ) #2 {
  %val = load <2 x i8>, <2 x i8> addrspace(3)* %p0, align 2
  ret <2 x i8> %val
}

define linkonce_odr <2 x i8> @optix.ptx.ldu.shared.v2.u8( <2 x i8> addrspace(3)* %p0 ) #2 {
  %val = load <2 x i8>, <2 x i8> addrspace(3)* %p0, align 2
  ret <2 x i8> %val
}

define linkonce_odr <2 x i8> @optix.ptx.ld.volatile.shared.v2.u8( <2 x i8> addrspace(3)* %p0 ) #1 {
  %val = load volatile <2 x i8>, <2 x i8> addrspace(3)* %p0, align 2
  ret <2 x i8> %val
}

define linkonce_odr void @optix.ptx.st.shared.v2.u8( <2 x i8> addrspace(3)* %p0, <2 x i8> %p1) #1 {
  store <2 x i8> %p1, <2 x i8> addrspace(3)* %p0
  ret void
}

define linkonce_odr <2 x i8> @optix.ptx.ld.shared.v2.s8( <2 x i8> addrspace(3)* %p0 ) #2 {
  %val = load <2 x i8>, <2 x i8> addrspace(3)* %p0, align 2
  ret <2 x i8> %val
}

define linkonce_odr <2 x i8> @optix.ptx.ldu.shared.v2.s8( <2 x i8> addrspace(3)* %p0 ) #2 {
  %val = load <2 x i8>, <2 x i8> addrspace(3)* %p0, align 2
  ret <2 x i8> %val
}

define linkonce_odr <2 x i8> @optix.ptx.ld.volatile.shared.v2.s8( <2 x i8> addrspace(3)* %p0 ) #1 {
  %val = load volatile <2 x i8>, <2 x i8> addrspace(3)* %p0, align 2
  ret <2 x i8> %val
}

define linkonce_odr void @optix.ptx.st.shared.v2.s8( <2 x i8> addrspace(3)* %p0, <2 x i8> %p1) #1 {
  store <2 x i8> %p1, <2 x i8> addrspace(3)* %p0
  ret void
}

define linkonce_odr <2 x i16> @optix.ptx.ld.shared.v2.b16( <2 x i16> addrspace(3)* %p0 ) #2 {
  %val = load <2 x i16>, <2 x i16> addrspace(3)* %p0, align 4
  ret <2 x i16> %val
}

define linkonce_odr <2 x i16> @optix.ptx.ldu.shared.v2.b16( <2 x i16> addrspace(3)* %p0 ) #2 {
  %val = load <2 x i16>, <2 x i16> addrspace(3)* %p0, align 4
  ret <2 x i16> %val
}

define linkonce_odr <2 x i16> @optix.ptx.ld.volatile.shared.v2.b16( <2 x i16> addrspace(3)* %p0 ) #1 {
  %val = load volatile <2 x i16>, <2 x i16> addrspace(3)* %p0, align 4
  ret <2 x i16> %val
}

define linkonce_odr void @optix.ptx.st.shared.v2.b16( <2 x i16> addrspace(3)* %p0, <2 x i16> %p1) #1 {
  store <2 x i16> %p1, <2 x i16> addrspace(3)* %p0
  ret void
}

define linkonce_odr <2 x i16> @optix.ptx.ld.shared.v2.u16( <2 x i16> addrspace(3)* %p0 ) #2 {
  %val = load <2 x i16>, <2 x i16> addrspace(3)* %p0, align 4
  ret <2 x i16> %val
}

define linkonce_odr <2 x i16> @optix.ptx.ldu.shared.v2.u16( <2 x i16> addrspace(3)* %p0 ) #2 {
  %val = load <2 x i16>, <2 x i16> addrspace(3)* %p0, align 4
  ret <2 x i16> %val
}

define linkonce_odr <2 x i16> @optix.ptx.ld.volatile.shared.v2.u16( <2 x i16> addrspace(3)* %p0 ) #1 {
  %val = load volatile <2 x i16>, <2 x i16> addrspace(3)* %p0, align 4
  ret <2 x i16> %val
}

define linkonce_odr void @optix.ptx.st.shared.v2.u16( <2 x i16> addrspace(3)* %p0, <2 x i16> %p1) #1 {
  store <2 x i16> %p1, <2 x i16> addrspace(3)* %p0
  ret void
}

define linkonce_odr <2 x i16> @optix.ptx.ld.shared.v2.s16( <2 x i16> addrspace(3)* %p0 ) #2 {
  %val = load <2 x i16>, <2 x i16> addrspace(3)* %p0, align 4
  ret <2 x i16> %val
}

define linkonce_odr <2 x i16> @optix.ptx.ldu.shared.v2.s16( <2 x i16> addrspace(3)* %p0 ) #2 {
  %val = load <2 x i16>, <2 x i16> addrspace(3)* %p0, align 4
  ret <2 x i16> %val
}

define linkonce_odr <2 x i16> @optix.ptx.ld.volatile.shared.v2.s16( <2 x i16> addrspace(3)* %p0 ) #1 {
  %val = load volatile <2 x i16>, <2 x i16> addrspace(3)* %p0, align 4
  ret <2 x i16> %val
}

define linkonce_odr void @optix.ptx.st.shared.v2.s16( <2 x i16> addrspace(3)* %p0, <2 x i16> %p1) #1 {
  store <2 x i16> %p1, <2 x i16> addrspace(3)* %p0
  ret void
}

define linkonce_odr <2 x i32> @optix.ptx.ld.shared.v2.b32( <2 x i32> addrspace(3)* %p0 ) #2 {
  %val = load <2 x i32>, <2 x i32> addrspace(3)* %p0, align 8
  ret <2 x i32> %val
}

define linkonce_odr <2 x i32> @optix.ptx.ldu.shared.v2.b32( <2 x i32> addrspace(3)* %p0 ) #2 {
  %val = load <2 x i32>, <2 x i32> addrspace(3)* %p0, align 8
  ret <2 x i32> %val
}

define linkonce_odr <2 x i32> @optix.ptx.ld.volatile.shared.v2.b32( <2 x i32> addrspace(3)* %p0 ) #1 {
  %val = load volatile <2 x i32>, <2 x i32> addrspace(3)* %p0, align 8
  ret <2 x i32> %val
}

define linkonce_odr void @optix.ptx.st.shared.v2.b32( <2 x i32> addrspace(3)* %p0, <2 x i32> %p1) #1 {
  store <2 x i32> %p1, <2 x i32> addrspace(3)* %p0
  ret void
}

define linkonce_odr <2 x i32> @optix.ptx.ld.shared.v2.u32( <2 x i32> addrspace(3)* %p0 ) #2 {
  %val = load <2 x i32>, <2 x i32> addrspace(3)* %p0, align 8
  ret <2 x i32> %val
}

define linkonce_odr <2 x i32> @optix.ptx.ldu.shared.v2.u32( <2 x i32> addrspace(3)* %p0 ) #2 {
  %val = load <2 x i32>, <2 x i32> addrspace(3)* %p0, align 8
  ret <2 x i32> %val
}

define linkonce_odr <2 x i32> @optix.ptx.ld.volatile.shared.v2.u32( <2 x i32> addrspace(3)* %p0 ) #1 {
  %val = load volatile <2 x i32>, <2 x i32> addrspace(3)* %p0, align 8
  ret <2 x i32> %val
}

define linkonce_odr void @optix.ptx.st.shared.v2.u32( <2 x i32> addrspace(3)* %p0, <2 x i32> %p1) #1 {
  store <2 x i32> %p1, <2 x i32> addrspace(3)* %p0
  ret void
}

define linkonce_odr <2 x i32> @optix.ptx.ld.shared.v2.s32( <2 x i32> addrspace(3)* %p0 ) #2 {
  %val = load <2 x i32>, <2 x i32> addrspace(3)* %p0, align 8
  ret <2 x i32> %val
}

define linkonce_odr <2 x i32> @optix.ptx.ldu.shared.v2.s32( <2 x i32> addrspace(3)* %p0 ) #2 {
  %val = load <2 x i32>, <2 x i32> addrspace(3)* %p0, align 8
  ret <2 x i32> %val
}

define linkonce_odr <2 x i32> @optix.ptx.ld.volatile.shared.v2.s32( <2 x i32> addrspace(3)* %p0 ) #1 {
  %val = load volatile <2 x i32>, <2 x i32> addrspace(3)* %p0, align 8
  ret <2 x i32> %val
}

define linkonce_odr void @optix.ptx.st.shared.v2.s32( <2 x i32> addrspace(3)* %p0, <2 x i32> %p1) #1 {
  store <2 x i32> %p1, <2 x i32> addrspace(3)* %p0
  ret void
}

define linkonce_odr <2 x i64> @optix.ptx.ld.shared.v2.b64( <2 x i64> addrspace(3)* %p0 ) #2 {
  %val = load <2 x i64>, <2 x i64> addrspace(3)* %p0, align 16
  ret <2 x i64> %val
}

define linkonce_odr <2 x i64> @optix.ptx.ldu.shared.v2.b64( <2 x i64> addrspace(3)* %p0 ) #2 {
  %val = load <2 x i64>, <2 x i64> addrspace(3)* %p0, align 16
  ret <2 x i64> %val
}

define linkonce_odr <2 x i64> @optix.ptx.ld.volatile.shared.v2.b64( <2 x i64> addrspace(3)* %p0 ) #1 {
  %val = load volatile <2 x i64>, <2 x i64> addrspace(3)* %p0, align 16
  ret <2 x i64> %val
}

define linkonce_odr void @optix.ptx.st.shared.v2.b64( <2 x i64> addrspace(3)* %p0, <2 x i64> %p1) #1 {
  store <2 x i64> %p1, <2 x i64> addrspace(3)* %p0
  ret void
}

define linkonce_odr <2 x i64> @optix.ptx.ld.shared.v2.u64( <2 x i64> addrspace(3)* %p0 ) #2 {
  %val = load <2 x i64>, <2 x i64> addrspace(3)* %p0, align 16
  ret <2 x i64> %val
}

define linkonce_odr <2 x i64> @optix.ptx.ldu.shared.v2.u64( <2 x i64> addrspace(3)* %p0 ) #2 {
  %val = load <2 x i64>, <2 x i64> addrspace(3)* %p0, align 16
  ret <2 x i64> %val
}

define linkonce_odr <2 x i64> @optix.ptx.ld.volatile.shared.v2.u64( <2 x i64> addrspace(3)* %p0 ) #1 {
  %val = load volatile <2 x i64>, <2 x i64> addrspace(3)* %p0, align 16
  ret <2 x i64> %val
}

define linkonce_odr void @optix.ptx.st.shared.v2.u64( <2 x i64> addrspace(3)* %p0, <2 x i64> %p1) #1 {
  store <2 x i64> %p1, <2 x i64> addrspace(3)* %p0
  ret void
}

define linkonce_odr <2 x i64> @optix.ptx.ld.shared.v2.s64( <2 x i64> addrspace(3)* %p0 ) #2 {
  %val = load <2 x i64>, <2 x i64> addrspace(3)* %p0, align 16
  ret <2 x i64> %val
}

define linkonce_odr <2 x i64> @optix.ptx.ldu.shared.v2.s64( <2 x i64> addrspace(3)* %p0 ) #2 {
  %val = load <2 x i64>, <2 x i64> addrspace(3)* %p0, align 16
  ret <2 x i64> %val
}

define linkonce_odr <2 x i64> @optix.ptx.ld.volatile.shared.v2.s64( <2 x i64> addrspace(3)* %p0 ) #1 {
  %val = load volatile <2 x i64>, <2 x i64> addrspace(3)* %p0, align 16
  ret <2 x i64> %val
}

define linkonce_odr void @optix.ptx.st.shared.v2.s64( <2 x i64> addrspace(3)* %p0, <2 x i64> %p1) #1 {
  store <2 x i64> %p1, <2 x i64> addrspace(3)* %p0
  ret void
}

define linkonce_odr <2 x float> @optix.ptx.ld.shared.v2.f32( <2 x float> addrspace(3)* %p0 ) #2 {
  %val = load <2 x float>, <2 x float> addrspace(3)* %p0, align 8
  ret <2 x float> %val
}

define linkonce_odr <2 x float> @optix.ptx.ldu.shared.v2.f32( <2 x float> addrspace(3)* %p0 ) #2 {
  %val = load <2 x float>, <2 x float> addrspace(3)* %p0, align 8
  ret <2 x float> %val
}

define linkonce_odr <2 x float> @optix.ptx.ld.volatile.shared.v2.f32( <2 x float> addrspace(3)* %p0 ) #1 {
  %val = load volatile <2 x float>, <2 x float> addrspace(3)* %p0, align 8
  ret <2 x float> %val
}

define linkonce_odr void @optix.ptx.st.shared.v2.f32( <2 x float> addrspace(3)* %p0, <2 x float> %p1) #1 {
  store <2 x float> %p1, <2 x float> addrspace(3)* %p0
  ret void
}

define linkonce_odr <2 x double> @optix.ptx.ld.shared.v2.f64( <2 x double> addrspace(3)* %p0 ) #2 {
  %val = load <2 x double>, <2 x double> addrspace(3)* %p0, align 16
  ret <2 x double> %val
}

define linkonce_odr <2 x double> @optix.ptx.ldu.shared.v2.f64( <2 x double> addrspace(3)* %p0 ) #2 {
  %val = load <2 x double>, <2 x double> addrspace(3)* %p0, align 16
  ret <2 x double> %val
}

define linkonce_odr <2 x double> @optix.ptx.ld.volatile.shared.v2.f64( <2 x double> addrspace(3)* %p0 ) #1 {
  %val = load volatile <2 x double>, <2 x double> addrspace(3)* %p0, align 16
  ret <2 x double> %val
}

define linkonce_odr void @optix.ptx.st.shared.v2.f64( <2 x double> addrspace(3)* %p0, <2 x double> %p1) #1 {
  store <2 x double> %p1, <2 x double> addrspace(3)* %p0
  ret void
}

define linkonce_odr <4 x i8> @optix.ptx.ld.shared.v4.b8( <4 x i8> addrspace(3)* %p0 ) #2 {
  %val = load <4 x i8>, <4 x i8> addrspace(3)* %p0, align 4
  ret <4 x i8> %val
}

define linkonce_odr <4 x i8> @optix.ptx.ldu.shared.v4.b8( <4 x i8> addrspace(3)* %p0 ) #2 {
  %val = load <4 x i8>, <4 x i8> addrspace(3)* %p0, align 4
  ret <4 x i8> %val
}

define linkonce_odr <4 x i8> @optix.ptx.ld.volatile.shared.v4.b8( <4 x i8> addrspace(3)* %p0 ) #1 {
  %val = load volatile <4 x i8>, <4 x i8> addrspace(3)* %p0, align 4
  ret <4 x i8> %val
}

define linkonce_odr void @optix.ptx.st.shared.v4.b8( <4 x i8> addrspace(3)* %p0, <4 x i8> %p1) #1 {
  store <4 x i8> %p1, <4 x i8> addrspace(3)* %p0
  ret void
}

define linkonce_odr <4 x i8> @optix.ptx.ld.shared.v4.u8( <4 x i8> addrspace(3)* %p0 ) #2 {
  %val = load <4 x i8>, <4 x i8> addrspace(3)* %p0, align 4
  ret <4 x i8> %val
}

define linkonce_odr <4 x i8> @optix.ptx.ldu.shared.v4.u8( <4 x i8> addrspace(3)* %p0 ) #2 {
  %val = load <4 x i8>, <4 x i8> addrspace(3)* %p0, align 4
  ret <4 x i8> %val
}

define linkonce_odr <4 x i8> @optix.ptx.ld.volatile.shared.v4.u8( <4 x i8> addrspace(3)* %p0 ) #1 {
  %val = load volatile <4 x i8>, <4 x i8> addrspace(3)* %p0, align 4
  ret <4 x i8> %val
}

define linkonce_odr void @optix.ptx.st.shared.v4.u8( <4 x i8> addrspace(3)* %p0, <4 x i8> %p1) #1 {
  store <4 x i8> %p1, <4 x i8> addrspace(3)* %p0
  ret void
}

define linkonce_odr <4 x i8> @optix.ptx.ld.shared.v4.s8( <4 x i8> addrspace(3)* %p0 ) #2 {
  %val = load <4 x i8>, <4 x i8> addrspace(3)* %p0, align 4
  ret <4 x i8> %val
}

define linkonce_odr <4 x i8> @optix.ptx.ldu.shared.v4.s8( <4 x i8> addrspace(3)* %p0 ) #2 {
  %val = load <4 x i8>, <4 x i8> addrspace(3)* %p0, align 4
  ret <4 x i8> %val
}

define linkonce_odr <4 x i8> @optix.ptx.ld.volatile.shared.v4.s8( <4 x i8> addrspace(3)* %p0 ) #1 {
  %val = load volatile <4 x i8>, <4 x i8> addrspace(3)* %p0, align 4
  ret <4 x i8> %val
}

define linkonce_odr void @optix.ptx.st.shared.v4.s8( <4 x i8> addrspace(3)* %p0, <4 x i8> %p1) #1 {
  store <4 x i8> %p1, <4 x i8> addrspace(3)* %p0
  ret void
}

define linkonce_odr <4 x i16> @optix.ptx.ld.shared.v4.b16( <4 x i16> addrspace(3)* %p0 ) #2 {
  %val = load <4 x i16>, <4 x i16> addrspace(3)* %p0, align 8
  ret <4 x i16> %val
}

define linkonce_odr <4 x i16> @optix.ptx.ldu.shared.v4.b16( <4 x i16> addrspace(3)* %p0 ) #2 {
  %val = load <4 x i16>, <4 x i16> addrspace(3)* %p0, align 8
  ret <4 x i16> %val
}

define linkonce_odr <4 x i16> @optix.ptx.ld.volatile.shared.v4.b16( <4 x i16> addrspace(3)* %p0 ) #1 {
  %val = load volatile <4 x i16>, <4 x i16> addrspace(3)* %p0, align 8
  ret <4 x i16> %val
}

define linkonce_odr void @optix.ptx.st.shared.v4.b16( <4 x i16> addrspace(3)* %p0, <4 x i16> %p1) #1 {
  store <4 x i16> %p1, <4 x i16> addrspace(3)* %p0
  ret void
}

define linkonce_odr <4 x i16> @optix.ptx.ld.shared.v4.u16( <4 x i16> addrspace(3)* %p0 ) #2 {
  %val = load <4 x i16>, <4 x i16> addrspace(3)* %p0, align 8
  ret <4 x i16> %val
}

define linkonce_odr <4 x i16> @optix.ptx.ldu.shared.v4.u16( <4 x i16> addrspace(3)* %p0 ) #2 {
  %val = load <4 x i16>, <4 x i16> addrspace(3)* %p0, align 8
  ret <4 x i16> %val
}

define linkonce_odr <4 x i16> @optix.ptx.ld.volatile.shared.v4.u16( <4 x i16> addrspace(3)* %p0 ) #1 {
  %val = load volatile <4 x i16>, <4 x i16> addrspace(3)* %p0, align 8
  ret <4 x i16> %val
}

define linkonce_odr void @optix.ptx.st.shared.v4.u16( <4 x i16> addrspace(3)* %p0, <4 x i16> %p1) #1 {
  store <4 x i16> %p1, <4 x i16> addrspace(3)* %p0
  ret void
}

define linkonce_odr <4 x i16> @optix.ptx.ld.shared.v4.s16( <4 x i16> addrspace(3)* %p0 ) #2 {
  %val = load <4 x i16>, <4 x i16> addrspace(3)* %p0, align 8
  ret <4 x i16> %val
}

define linkonce_odr <4 x i16> @optix.ptx.ldu.shared.v4.s16( <4 x i16> addrspace(3)* %p0 ) #2 {
  %val = load <4 x i16>, <4 x i16> addrspace(3)* %p0, align 8
  ret <4 x i16> %val
}

define linkonce_odr <4 x i16> @optix.ptx.ld.volatile.shared.v4.s16( <4 x i16> addrspace(3)* %p0 ) #1 {
  %val = load volatile <4 x i16>, <4 x i16> addrspace(3)* %p0, align 8
  ret <4 x i16> %val
}

define linkonce_odr void @optix.ptx.st.shared.v4.s16( <4 x i16> addrspace(3)* %p0, <4 x i16> %p1) #1 {
  store <4 x i16> %p1, <4 x i16> addrspace(3)* %p0
  ret void
}

define linkonce_odr <4 x i32> @optix.ptx.ld.shared.v4.b32( <4 x i32> addrspace(3)* %p0 ) #2 {
  %val = load <4 x i32>, <4 x i32> addrspace(3)* %p0, align 16
  ret <4 x i32> %val
}

define linkonce_odr <4 x i32> @optix.ptx.ldu.shared.v4.b32( <4 x i32> addrspace(3)* %p0 ) #2 {
  %val = load <4 x i32>, <4 x i32> addrspace(3)* %p0, align 16
  ret <4 x i32> %val
}

define linkonce_odr <4 x i32> @optix.ptx.ld.volatile.shared.v4.b32( <4 x i32> addrspace(3)* %p0 ) #1 {
  %val = load volatile <4 x i32>, <4 x i32> addrspace(3)* %p0, align 16
  ret <4 x i32> %val
}

define linkonce_odr void @optix.ptx.st.shared.v4.b32( <4 x i32> addrspace(3)* %p0, <4 x i32> %p1) #1 {
  store <4 x i32> %p1, <4 x i32> addrspace(3)* %p0
  ret void
}

define linkonce_odr <4 x i32> @optix.ptx.ld.shared.v4.u32( <4 x i32> addrspace(3)* %p0 ) #2 {
  %val = load <4 x i32>, <4 x i32> addrspace(3)* %p0, align 16
  ret <4 x i32> %val
}

define linkonce_odr <4 x i32> @optix.ptx.ldu.shared.v4.u32( <4 x i32> addrspace(3)* %p0 ) #2 {
  %val = load <4 x i32>, <4 x i32> addrspace(3)* %p0, align 16
  ret <4 x i32> %val
}

define linkonce_odr <4 x i32> @optix.ptx.ld.volatile.shared.v4.u32( <4 x i32> addrspace(3)* %p0 ) #1 {
  %val = load volatile <4 x i32>, <4 x i32> addrspace(3)* %p0, align 16
  ret <4 x i32> %val
}

define linkonce_odr void @optix.ptx.st.shared.v4.u32( <4 x i32> addrspace(3)* %p0, <4 x i32> %p1) #1 {
  store <4 x i32> %p1, <4 x i32> addrspace(3)* %p0
  ret void
}

define linkonce_odr <4 x i32> @optix.ptx.ld.shared.v4.s32( <4 x i32> addrspace(3)* %p0 ) #2 {
  %val = load <4 x i32>, <4 x i32> addrspace(3)* %p0, align 16
  ret <4 x i32> %val
}

define linkonce_odr <4 x i32> @optix.ptx.ldu.shared.v4.s32( <4 x i32> addrspace(3)* %p0 ) #2 {
  %val = load <4 x i32>, <4 x i32> addrspace(3)* %p0, align 16
  ret <4 x i32> %val
}

define linkonce_odr <4 x i32> @optix.ptx.ld.volatile.shared.v4.s32( <4 x i32> addrspace(3)* %p0 ) #1 {
  %val = load volatile <4 x i32>, <4 x i32> addrspace(3)* %p0, align 16
  ret <4 x i32> %val
}

define linkonce_odr void @optix.ptx.st.shared.v4.s32( <4 x i32> addrspace(3)* %p0, <4 x i32> %p1) #1 {
  store <4 x i32> %p1, <4 x i32> addrspace(3)* %p0
  ret void
}

define linkonce_odr <4 x i64> @optix.ptx.ld.shared.v4.b64( <4 x i64> addrspace(3)* %p0 ) #2 {
  %val = load <4 x i64>, <4 x i64> addrspace(3)* %p0, align 32
  ret <4 x i64> %val
}

define linkonce_odr <4 x i64> @optix.ptx.ldu.shared.v4.b64( <4 x i64> addrspace(3)* %p0 ) #2 {
  %val = load <4 x i64>, <4 x i64> addrspace(3)* %p0, align 32
  ret <4 x i64> %val
}

define linkonce_odr <4 x i64> @optix.ptx.ld.volatile.shared.v4.b64( <4 x i64> addrspace(3)* %p0 ) #1 {
  %val = load volatile <4 x i64>, <4 x i64> addrspace(3)* %p0, align 32
  ret <4 x i64> %val
}

define linkonce_odr void @optix.ptx.st.shared.v4.b64( <4 x i64> addrspace(3)* %p0, <4 x i64> %p1) #1 {
  store <4 x i64> %p1, <4 x i64> addrspace(3)* %p0
  ret void
}

define linkonce_odr <4 x i64> @optix.ptx.ld.shared.v4.u64( <4 x i64> addrspace(3)* %p0 ) #2 {
  %val = load <4 x i64>, <4 x i64> addrspace(3)* %p0, align 32
  ret <4 x i64> %val
}

define linkonce_odr <4 x i64> @optix.ptx.ldu.shared.v4.u64( <4 x i64> addrspace(3)* %p0 ) #2 {
  %val = load <4 x i64>, <4 x i64> addrspace(3)* %p0, align 32
  ret <4 x i64> %val
}

define linkonce_odr <4 x i64> @optix.ptx.ld.volatile.shared.v4.u64( <4 x i64> addrspace(3)* %p0 ) #1 {
  %val = load volatile <4 x i64>, <4 x i64> addrspace(3)* %p0, align 32
  ret <4 x i64> %val
}

define linkonce_odr void @optix.ptx.st.shared.v4.u64( <4 x i64> addrspace(3)* %p0, <4 x i64> %p1) #1 {
  store <4 x i64> %p1, <4 x i64> addrspace(3)* %p0
  ret void
}

define linkonce_odr <4 x i64> @optix.ptx.ld.shared.v4.s64( <4 x i64> addrspace(3)* %p0 ) #2 {
  %val = load <4 x i64>, <4 x i64> addrspace(3)* %p0, align 32
  ret <4 x i64> %val
}

define linkonce_odr <4 x i64> @optix.ptx.ldu.shared.v4.s64( <4 x i64> addrspace(3)* %p0 ) #2 {
  %val = load <4 x i64>, <4 x i64> addrspace(3)* %p0, align 32
  ret <4 x i64> %val
}

define linkonce_odr <4 x i64> @optix.ptx.ld.volatile.shared.v4.s64( <4 x i64> addrspace(3)* %p0 ) #1 {
  %val = load volatile <4 x i64>, <4 x i64> addrspace(3)* %p0, align 32
  ret <4 x i64> %val
}

define linkonce_odr void @optix.ptx.st.shared.v4.s64( <4 x i64> addrspace(3)* %p0, <4 x i64> %p1) #1 {
  store <4 x i64> %p1, <4 x i64> addrspace(3)* %p0
  ret void
}

define linkonce_odr <4 x float> @optix.ptx.ld.shared.v4.f32( <4 x float> addrspace(3)* %p0 ) #2 {
  %val = load <4 x float>, <4 x float> addrspace(3)* %p0, align 16
  ret <4 x float> %val
}

define linkonce_odr <4 x float> @optix.ptx.ldu.shared.v4.f32( <4 x float> addrspace(3)* %p0 ) #2 {
  %val = load <4 x float>, <4 x float> addrspace(3)* %p0, align 16
  ret <4 x float> %val
}

define linkonce_odr <4 x float> @optix.ptx.ld.volatile.shared.v4.f32( <4 x float> addrspace(3)* %p0 ) #1 {
  %val = load volatile <4 x float>, <4 x float> addrspace(3)* %p0, align 16
  ret <4 x float> %val
}

define linkonce_odr void @optix.ptx.st.shared.v4.f32( <4 x float> addrspace(3)* %p0, <4 x float> %p1) #1 {
  store <4 x float> %p1, <4 x float> addrspace(3)* %p0
  ret void
}

define linkonce_odr <4 x double> @optix.ptx.ld.shared.v4.f64( <4 x double> addrspace(3)* %p0 ) #2 {
  %val = load <4 x double>, <4 x double> addrspace(3)* %p0, align 32
  ret <4 x double> %val
}

define linkonce_odr <4 x double> @optix.ptx.ldu.shared.v4.f64( <4 x double> addrspace(3)* %p0 ) #2 {
  %val = load <4 x double>, <4 x double> addrspace(3)* %p0, align 32
  ret <4 x double> %val
}

define linkonce_odr <4 x double> @optix.ptx.ld.volatile.shared.v4.f64( <4 x double> addrspace(3)* %p0 ) #1 {
  %val = load volatile <4 x double>, <4 x double> addrspace(3)* %p0, align 32
  ret <4 x double> %val
}

define linkonce_odr void @optix.ptx.st.shared.v4.f64( <4 x double> addrspace(3)* %p0, <4 x double> %p1) #1 {
  store <4 x double> %p1, <4 x double> addrspace(3)* %p0
  ret void
}

define linkonce_odr i8 @optix.ptx.ld.const.b8( i8 addrspace(4)* %p0 ) #2 {
  %val = load i8, i8 addrspace(4)* %p0, align 1
  ret i8 %val
}

define linkonce_odr i8 @optix.ptx.ldu.const.b8( i8 addrspace(4)* %p0 ) #2 {
  %val = load i8, i8 addrspace(4)* %p0, align 1
  ret i8 %val
}

define linkonce_odr i8 @optix.ptx.ld.volatile.const.b8( i8 addrspace(4)* %p0 ) #1 {
  %val = load volatile i8, i8 addrspace(4)* %p0, align 1
  ret i8 %val
}

define linkonce_odr void @optix.ptx.st.const.b8( i8 addrspace(4)* %p0, i8 %p1) #1 {
  store i8 %p1, i8 addrspace(4)* %p0
  ret void
}

define linkonce_odr i8 @optix.ptx.ld.const.u8( i8 addrspace(4)* %p0 ) #2 {
  %val = load i8, i8 addrspace(4)* %p0, align 1
  ret i8 %val
}

define linkonce_odr i8 @optix.ptx.ldu.const.u8( i8 addrspace(4)* %p0 ) #2 {
  %val = load i8, i8 addrspace(4)* %p0, align 1
  ret i8 %val
}

define linkonce_odr i8 @optix.ptx.ld.volatile.const.u8( i8 addrspace(4)* %p0 ) #1 {
  %val = load volatile i8, i8 addrspace(4)* %p0, align 1
  ret i8 %val
}

define linkonce_odr void @optix.ptx.st.const.u8( i8 addrspace(4)* %p0, i8 %p1) #1 {
  store i8 %p1, i8 addrspace(4)* %p0
  ret void
}

define linkonce_odr i8 @optix.ptx.ld.const.s8( i8 addrspace(4)* %p0 ) #2 {
  %val = load i8, i8 addrspace(4)* %p0, align 1
  ret i8 %val
}

define linkonce_odr i8 @optix.ptx.ldu.const.s8( i8 addrspace(4)* %p0 ) #2 {
  %val = load i8, i8 addrspace(4)* %p0, align 1
  ret i8 %val
}

define linkonce_odr i8 @optix.ptx.ld.volatile.const.s8( i8 addrspace(4)* %p0 ) #1 {
  %val = load volatile i8, i8 addrspace(4)* %p0, align 1
  ret i8 %val
}

define linkonce_odr void @optix.ptx.st.const.s8( i8 addrspace(4)* %p0, i8 %p1) #1 {
  store i8 %p1, i8 addrspace(4)* %p0
  ret void
}

define linkonce_odr i16 @optix.ptx.ld.const.b16( i16 addrspace(4)* %p0 ) #2 {
  %val = load i16, i16 addrspace(4)* %p0, align 2
  ret i16 %val
}

define linkonce_odr i16 @optix.ptx.ldu.const.b16( i16 addrspace(4)* %p0 ) #2 {
  %val = load i16, i16 addrspace(4)* %p0, align 2
  ret i16 %val
}

define linkonce_odr i16 @optix.ptx.ld.volatile.const.b16( i16 addrspace(4)* %p0 ) #1 {
  %val = load volatile i16, i16 addrspace(4)* %p0, align 2
  ret i16 %val
}

define linkonce_odr void @optix.ptx.st.const.b16( i16 addrspace(4)* %p0, i16 %p1) #1 {
  store i16 %p1, i16 addrspace(4)* %p0
  ret void
}

define linkonce_odr i16 @optix.ptx.ld.const.u16( i16 addrspace(4)* %p0 ) #2 {
  %val = load i16, i16 addrspace(4)* %p0, align 2
  ret i16 %val
}

define linkonce_odr i16 @optix.ptx.ldu.const.u16( i16 addrspace(4)* %p0 ) #2 {
  %val = load i16, i16 addrspace(4)* %p0, align 2
  ret i16 %val
}

define linkonce_odr i16 @optix.ptx.ld.volatile.const.u16( i16 addrspace(4)* %p0 ) #1 {
  %val = load volatile i16, i16 addrspace(4)* %p0, align 2
  ret i16 %val
}

define linkonce_odr void @optix.ptx.st.const.u16( i16 addrspace(4)* %p0, i16 %p1) #1 {
  store i16 %p1, i16 addrspace(4)* %p0
  ret void
}

define linkonce_odr i16 @optix.ptx.ld.const.s16( i16 addrspace(4)* %p0 ) #2 {
  %val = load i16, i16 addrspace(4)* %p0, align 2
  ret i16 %val
}

define linkonce_odr i16 @optix.ptx.ldu.const.s16( i16 addrspace(4)* %p0 ) #2 {
  %val = load i16, i16 addrspace(4)* %p0, align 2
  ret i16 %val
}

define linkonce_odr i16 @optix.ptx.ld.volatile.const.s16( i16 addrspace(4)* %p0 ) #1 {
  %val = load volatile i16, i16 addrspace(4)* %p0, align 2
  ret i16 %val
}

define linkonce_odr void @optix.ptx.st.const.s16( i16 addrspace(4)* %p0, i16 %p1) #1 {
  store i16 %p1, i16 addrspace(4)* %p0
  ret void
}

define linkonce_odr i32 @optix.ptx.ld.const.b32( i32 addrspace(4)* %p0 ) #2 {
  %val = load i32, i32 addrspace(4)* %p0, align 4
  ret i32 %val
}

define linkonce_odr i32 @optix.ptx.ldu.const.b32( i32 addrspace(4)* %p0 ) #2 {
  %val = load i32, i32 addrspace(4)* %p0, align 4
  ret i32 %val
}

define linkonce_odr i32 @optix.ptx.ld.volatile.const.b32( i32 addrspace(4)* %p0 ) #1 {
  %val = load volatile i32, i32 addrspace(4)* %p0, align 4
  ret i32 %val
}

define linkonce_odr void @optix.ptx.st.const.b32( i32 addrspace(4)* %p0, i32 %p1) #1 {
  store i32 %p1, i32 addrspace(4)* %p0
  ret void
}

define linkonce_odr i32 @optix.ptx.ld.const.u32( i32 addrspace(4)* %p0 ) #2 {
  %val = load i32, i32 addrspace(4)* %p0, align 4
  ret i32 %val
}

define linkonce_odr i32 @optix.ptx.ldu.const.u32( i32 addrspace(4)* %p0 ) #2 {
  %val = load i32, i32 addrspace(4)* %p0, align 4
  ret i32 %val
}

define linkonce_odr i32 @optix.ptx.ld.volatile.const.u32( i32 addrspace(4)* %p0 ) #1 {
  %val = load volatile i32, i32 addrspace(4)* %p0, align 4
  ret i32 %val
}

define linkonce_odr void @optix.ptx.st.const.u32( i32 addrspace(4)* %p0, i32 %p1) #1 {
  store i32 %p1, i32 addrspace(4)* %p0
  ret void
}

define linkonce_odr i32 @optix.ptx.ld.const.s32( i32 addrspace(4)* %p0 ) #2 {
  %val = load i32, i32 addrspace(4)* %p0, align 4
  ret i32 %val
}

define linkonce_odr i32 @optix.ptx.ldu.const.s32( i32 addrspace(4)* %p0 ) #2 {
  %val = load i32, i32 addrspace(4)* %p0, align 4
  ret i32 %val
}

define linkonce_odr i32 @optix.ptx.ld.volatile.const.s32( i32 addrspace(4)* %p0 ) #1 {
  %val = load volatile i32, i32 addrspace(4)* %p0, align 4
  ret i32 %val
}

define linkonce_odr void @optix.ptx.st.const.s32( i32 addrspace(4)* %p0, i32 %p1) #1 {
  store i32 %p1, i32 addrspace(4)* %p0
  ret void
}

define linkonce_odr i64 @optix.ptx.ld.const.b64( i64 addrspace(4)* %p0 ) #2 {
  %val = load i64, i64 addrspace(4)* %p0, align 8
  ret i64 %val
}

define linkonce_odr i64 @optix.ptx.ldu.const.b64( i64 addrspace(4)* %p0 ) #2 {
  %val = load i64, i64 addrspace(4)* %p0, align 8
  ret i64 %val
}

define linkonce_odr i64 @optix.ptx.ld.volatile.const.b64( i64 addrspace(4)* %p0 ) #1 {
  %val = load volatile i64, i64 addrspace(4)* %p0, align 8
  ret i64 %val
}

define linkonce_odr void @optix.ptx.st.const.b64( i64 addrspace(4)* %p0, i64 %p1) #1 {
  store i64 %p1, i64 addrspace(4)* %p0
  ret void
}

define linkonce_odr i64 @optix.ptx.ld.const.u64( i64 addrspace(4)* %p0 ) #2 {
  %val = load i64, i64 addrspace(4)* %p0, align 8
  ret i64 %val
}

define linkonce_odr i64 @optix.ptx.ldu.const.u64( i64 addrspace(4)* %p0 ) #2 {
  %val = load i64, i64 addrspace(4)* %p0, align 8
  ret i64 %val
}

define linkonce_odr i64 @optix.ptx.ld.volatile.const.u64( i64 addrspace(4)* %p0 ) #1 {
  %val = load volatile i64, i64 addrspace(4)* %p0, align 8
  ret i64 %val
}

define linkonce_odr void @optix.ptx.st.const.u64( i64 addrspace(4)* %p0, i64 %p1) #1 {
  store i64 %p1, i64 addrspace(4)* %p0
  ret void
}

define linkonce_odr i64 @optix.ptx.ld.const.s64( i64 addrspace(4)* %p0 ) #2 {
  %val = load i64, i64 addrspace(4)* %p0, align 8
  ret i64 %val
}

define linkonce_odr i64 @optix.ptx.ldu.const.s64( i64 addrspace(4)* %p0 ) #2 {
  %val = load i64, i64 addrspace(4)* %p0, align 8
  ret i64 %val
}

define linkonce_odr i64 @optix.ptx.ld.volatile.const.s64( i64 addrspace(4)* %p0 ) #1 {
  %val = load volatile i64, i64 addrspace(4)* %p0, align 8
  ret i64 %val
}

define linkonce_odr void @optix.ptx.st.const.s64( i64 addrspace(4)* %p0, i64 %p1) #1 {
  store i64 %p1, i64 addrspace(4)* %p0
  ret void
}

define linkonce_odr float @optix.ptx.ld.const.f32( float addrspace(4)* %p0 ) #2 {
  %val = load float, float addrspace(4)* %p0, align 4
  ret float %val
}

define linkonce_odr float @optix.ptx.ldu.const.f32( float addrspace(4)* %p0 ) #2 {
  %val = load float, float addrspace(4)* %p0, align 4
  ret float %val
}

define linkonce_odr float @optix.ptx.ld.volatile.const.f32( float addrspace(4)* %p0 ) #1 {
  %val = load volatile float, float addrspace(4)* %p0, align 4
  ret float %val
}

define linkonce_odr void @optix.ptx.st.const.f32( float addrspace(4)* %p0, float %p1) #1 {
  store float %p1, float addrspace(4)* %p0
  ret void
}

define linkonce_odr double @optix.ptx.ld.const.f64( double addrspace(4)* %p0 ) #2 {
  %val = load double, double addrspace(4)* %p0, align 8
  ret double %val
}

define linkonce_odr double @optix.ptx.ldu.const.f64( double addrspace(4)* %p0 ) #2 {
  %val = load double, double addrspace(4)* %p0, align 8
  ret double %val
}

define linkonce_odr double @optix.ptx.ld.volatile.const.f64( double addrspace(4)* %p0 ) #1 {
  %val = load volatile double, double addrspace(4)* %p0, align 8
  ret double %val
}

define linkonce_odr void @optix.ptx.st.const.f64( double addrspace(4)* %p0, double %p1) #1 {
  store double %p1, double addrspace(4)* %p0
  ret void
}

define linkonce_odr <2 x i8> @optix.ptx.ld.const.v2.b8( <2 x i8> addrspace(4)* %p0 ) #2 {
  %val = load <2 x i8>, <2 x i8> addrspace(4)* %p0, align 2
  ret <2 x i8> %val
}

define linkonce_odr <2 x i8> @optix.ptx.ldu.const.v2.b8( <2 x i8> addrspace(4)* %p0 ) #2 {
  %val = load <2 x i8>, <2 x i8> addrspace(4)* %p0, align 2
  ret <2 x i8> %val
}

define linkonce_odr <2 x i8> @optix.ptx.ld.volatile.const.v2.b8( <2 x i8> addrspace(4)* %p0 ) #1 {
  %val = load volatile <2 x i8>, <2 x i8> addrspace(4)* %p0, align 2
  ret <2 x i8> %val
}

define linkonce_odr void @optix.ptx.st.const.v2.b8( <2 x i8> addrspace(4)* %p0, <2 x i8> %p1) #1 {
  store <2 x i8> %p1, <2 x i8> addrspace(4)* %p0
  ret void
}

define linkonce_odr <2 x i8> @optix.ptx.ld.const.v2.u8( <2 x i8> addrspace(4)* %p0 ) #2 {
  %val = load <2 x i8>, <2 x i8> addrspace(4)* %p0, align 2
  ret <2 x i8> %val
}

define linkonce_odr <2 x i8> @optix.ptx.ldu.const.v2.u8( <2 x i8> addrspace(4)* %p0 ) #2 {
  %val = load <2 x i8>, <2 x i8> addrspace(4)* %p0, align 2
  ret <2 x i8> %val
}

define linkonce_odr <2 x i8> @optix.ptx.ld.volatile.const.v2.u8( <2 x i8> addrspace(4)* %p0 ) #1 {
  %val = load volatile <2 x i8>, <2 x i8> addrspace(4)* %p0, align 2
  ret <2 x i8> %val
}

define linkonce_odr void @optix.ptx.st.const.v2.u8( <2 x i8> addrspace(4)* %p0, <2 x i8> %p1) #1 {
  store <2 x i8> %p1, <2 x i8> addrspace(4)* %p0
  ret void
}

define linkonce_odr <2 x i8> @optix.ptx.ld.const.v2.s8( <2 x i8> addrspace(4)* %p0 ) #2 {
  %val = load <2 x i8>, <2 x i8> addrspace(4)* %p0, align 2
  ret <2 x i8> %val
}

define linkonce_odr <2 x i8> @optix.ptx.ldu.const.v2.s8( <2 x i8> addrspace(4)* %p0 ) #2 {
  %val = load <2 x i8>, <2 x i8> addrspace(4)* %p0, align 2
  ret <2 x i8> %val
}

define linkonce_odr <2 x i8> @optix.ptx.ld.volatile.const.v2.s8( <2 x i8> addrspace(4)* %p0 ) #1 {
  %val = load volatile <2 x i8>, <2 x i8> addrspace(4)* %p0, align 2
  ret <2 x i8> %val
}

define linkonce_odr void @optix.ptx.st.const.v2.s8( <2 x i8> addrspace(4)* %p0, <2 x i8> %p1) #1 {
  store <2 x i8> %p1, <2 x i8> addrspace(4)* %p0
  ret void
}

define linkonce_odr <2 x i16> @optix.ptx.ld.const.v2.b16( <2 x i16> addrspace(4)* %p0 ) #2 {
  %val = load <2 x i16>, <2 x i16> addrspace(4)* %p0, align 4
  ret <2 x i16> %val
}

define linkonce_odr <2 x i16> @optix.ptx.ldu.const.v2.b16( <2 x i16> addrspace(4)* %p0 ) #2 {
  %val = load <2 x i16>, <2 x i16> addrspace(4)* %p0, align 4
  ret <2 x i16> %val
}

define linkonce_odr <2 x i16> @optix.ptx.ld.volatile.const.v2.b16( <2 x i16> addrspace(4)* %p0 ) #1 {
  %val = load volatile <2 x i16>, <2 x i16> addrspace(4)* %p0, align 4
  ret <2 x i16> %val
}

define linkonce_odr void @optix.ptx.st.const.v2.b16( <2 x i16> addrspace(4)* %p0, <2 x i16> %p1) #1 {
  store <2 x i16> %p1, <2 x i16> addrspace(4)* %p0
  ret void
}

define linkonce_odr <2 x i16> @optix.ptx.ld.const.v2.u16( <2 x i16> addrspace(4)* %p0 ) #2 {
  %val = load <2 x i16>, <2 x i16> addrspace(4)* %p0, align 4
  ret <2 x i16> %val
}

define linkonce_odr <2 x i16> @optix.ptx.ldu.const.v2.u16( <2 x i16> addrspace(4)* %p0 ) #2 {
  %val = load <2 x i16>, <2 x i16> addrspace(4)* %p0, align 4
  ret <2 x i16> %val
}

define linkonce_odr <2 x i16> @optix.ptx.ld.volatile.const.v2.u16( <2 x i16> addrspace(4)* %p0 ) #1 {
  %val = load volatile <2 x i16>, <2 x i16> addrspace(4)* %p0, align 4
  ret <2 x i16> %val
}

define linkonce_odr void @optix.ptx.st.const.v2.u16( <2 x i16> addrspace(4)* %p0, <2 x i16> %p1) #1 {
  store <2 x i16> %p1, <2 x i16> addrspace(4)* %p0
  ret void
}

define linkonce_odr <2 x i16> @optix.ptx.ld.const.v2.s16( <2 x i16> addrspace(4)* %p0 ) #2 {
  %val = load <2 x i16>, <2 x i16> addrspace(4)* %p0, align 4
  ret <2 x i16> %val
}

define linkonce_odr <2 x i16> @optix.ptx.ldu.const.v2.s16( <2 x i16> addrspace(4)* %p0 ) #2 {
  %val = load <2 x i16>, <2 x i16> addrspace(4)* %p0, align 4
  ret <2 x i16> %val
}

define linkonce_odr <2 x i16> @optix.ptx.ld.volatile.const.v2.s16( <2 x i16> addrspace(4)* %p0 ) #1 {
  %val = load volatile <2 x i16>, <2 x i16> addrspace(4)* %p0, align 4
  ret <2 x i16> %val
}

define linkonce_odr void @optix.ptx.st.const.v2.s16( <2 x i16> addrspace(4)* %p0, <2 x i16> %p1) #1 {
  store <2 x i16> %p1, <2 x i16> addrspace(4)* %p0
  ret void
}

define linkonce_odr <2 x i32> @optix.ptx.ld.const.v2.b32( <2 x i32> addrspace(4)* %p0 ) #2 {
  %val = load <2 x i32>, <2 x i32> addrspace(4)* %p0, align 8
  ret <2 x i32> %val
}

define linkonce_odr <2 x i32> @optix.ptx.ldu.const.v2.b32( <2 x i32> addrspace(4)* %p0 ) #2 {
  %val = load <2 x i32>, <2 x i32> addrspace(4)* %p0, align 8
  ret <2 x i32> %val
}

define linkonce_odr <2 x i32> @optix.ptx.ld.volatile.const.v2.b32( <2 x i32> addrspace(4)* %p0 ) #1 {
  %val = load volatile <2 x i32>, <2 x i32> addrspace(4)* %p0, align 8
  ret <2 x i32> %val
}

define linkonce_odr void @optix.ptx.st.const.v2.b32( <2 x i32> addrspace(4)* %p0, <2 x i32> %p1) #1 {
  store <2 x i32> %p1, <2 x i32> addrspace(4)* %p0
  ret void
}

define linkonce_odr <2 x i32> @optix.ptx.ld.const.v2.u32( <2 x i32> addrspace(4)* %p0 ) #2 {
  %val = load <2 x i32>, <2 x i32> addrspace(4)* %p0, align 8
  ret <2 x i32> %val
}

define linkonce_odr <2 x i32> @optix.ptx.ldu.const.v2.u32( <2 x i32> addrspace(4)* %p0 ) #2 {
  %val = load <2 x i32>, <2 x i32> addrspace(4)* %p0, align 8
  ret <2 x i32> %val
}

define linkonce_odr <2 x i32> @optix.ptx.ld.volatile.const.v2.u32( <2 x i32> addrspace(4)* %p0 ) #1 {
  %val = load volatile <2 x i32>, <2 x i32> addrspace(4)* %p0, align 8
  ret <2 x i32> %val
}

define linkonce_odr void @optix.ptx.st.const.v2.u32( <2 x i32> addrspace(4)* %p0, <2 x i32> %p1) #1 {
  store <2 x i32> %p1, <2 x i32> addrspace(4)* %p0
  ret void
}

define linkonce_odr <2 x i32> @optix.ptx.ld.const.v2.s32( <2 x i32> addrspace(4)* %p0 ) #2 {
  %val = load <2 x i32>, <2 x i32> addrspace(4)* %p0, align 8
  ret <2 x i32> %val
}

define linkonce_odr <2 x i32> @optix.ptx.ldu.const.v2.s32( <2 x i32> addrspace(4)* %p0 ) #2 {
  %val = load <2 x i32>, <2 x i32> addrspace(4)* %p0, align 8
  ret <2 x i32> %val
}

define linkonce_odr <2 x i32> @optix.ptx.ld.volatile.const.v2.s32( <2 x i32> addrspace(4)* %p0 ) #1 {
  %val = load volatile <2 x i32>, <2 x i32> addrspace(4)* %p0, align 8
  ret <2 x i32> %val
}

define linkonce_odr void @optix.ptx.st.const.v2.s32( <2 x i32> addrspace(4)* %p0, <2 x i32> %p1) #1 {
  store <2 x i32> %p1, <2 x i32> addrspace(4)* %p0
  ret void
}

define linkonce_odr <2 x i64> @optix.ptx.ld.const.v2.b64( <2 x i64> addrspace(4)* %p0 ) #2 {
  %val = load <2 x i64>, <2 x i64> addrspace(4)* %p0, align 16
  ret <2 x i64> %val
}

define linkonce_odr <2 x i64> @optix.ptx.ldu.const.v2.b64( <2 x i64> addrspace(4)* %p0 ) #2 {
  %val = load <2 x i64>, <2 x i64> addrspace(4)* %p0, align 16
  ret <2 x i64> %val
}

define linkonce_odr <2 x i64> @optix.ptx.ld.volatile.const.v2.b64( <2 x i64> addrspace(4)* %p0 ) #1 {
  %val = load volatile <2 x i64>, <2 x i64> addrspace(4)* %p0, align 16
  ret <2 x i64> %val
}

define linkonce_odr void @optix.ptx.st.const.v2.b64( <2 x i64> addrspace(4)* %p0, <2 x i64> %p1) #1 {
  store <2 x i64> %p1, <2 x i64> addrspace(4)* %p0
  ret void
}

define linkonce_odr <2 x i64> @optix.ptx.ld.const.v2.u64( <2 x i64> addrspace(4)* %p0 ) #2 {
  %val = load <2 x i64>, <2 x i64> addrspace(4)* %p0, align 16
  ret <2 x i64> %val
}

define linkonce_odr <2 x i64> @optix.ptx.ldu.const.v2.u64( <2 x i64> addrspace(4)* %p0 ) #2 {
  %val = load <2 x i64>, <2 x i64> addrspace(4)* %p0, align 16
  ret <2 x i64> %val
}

define linkonce_odr <2 x i64> @optix.ptx.ld.volatile.const.v2.u64( <2 x i64> addrspace(4)* %p0 ) #1 {
  %val = load volatile <2 x i64>, <2 x i64> addrspace(4)* %p0, align 16
  ret <2 x i64> %val
}

define linkonce_odr void @optix.ptx.st.const.v2.u64( <2 x i64> addrspace(4)* %p0, <2 x i64> %p1) #1 {
  store <2 x i64> %p1, <2 x i64> addrspace(4)* %p0
  ret void
}

define linkonce_odr <2 x i64> @optix.ptx.ld.const.v2.s64( <2 x i64> addrspace(4)* %p0 ) #2 {
  %val = load <2 x i64>, <2 x i64> addrspace(4)* %p0, align 16
  ret <2 x i64> %val
}

define linkonce_odr <2 x i64> @optix.ptx.ldu.const.v2.s64( <2 x i64> addrspace(4)* %p0 ) #2 {
  %val = load <2 x i64>, <2 x i64> addrspace(4)* %p0, align 16
  ret <2 x i64> %val
}

define linkonce_odr <2 x i64> @optix.ptx.ld.volatile.const.v2.s64( <2 x i64> addrspace(4)* %p0 ) #1 {
  %val = load volatile <2 x i64>, <2 x i64> addrspace(4)* %p0, align 16
  ret <2 x i64> %val
}

define linkonce_odr void @optix.ptx.st.const.v2.s64( <2 x i64> addrspace(4)* %p0, <2 x i64> %p1) #1 {
  store <2 x i64> %p1, <2 x i64> addrspace(4)* %p0
  ret void
}

define linkonce_odr <2 x float> @optix.ptx.ld.const.v2.f32( <2 x float> addrspace(4)* %p0 ) #2 {
  %val = load <2 x float>, <2 x float> addrspace(4)* %p0, align 8
  ret <2 x float> %val
}

define linkonce_odr <2 x float> @optix.ptx.ldu.const.v2.f32( <2 x float> addrspace(4)* %p0 ) #2 {
  %val = load <2 x float>, <2 x float> addrspace(4)* %p0, align 8
  ret <2 x float> %val
}

define linkonce_odr <2 x float> @optix.ptx.ld.volatile.const.v2.f32( <2 x float> addrspace(4)* %p0 ) #1 {
  %val = load volatile <2 x float>, <2 x float> addrspace(4)* %p0, align 8
  ret <2 x float> %val
}

define linkonce_odr void @optix.ptx.st.const.v2.f32( <2 x float> addrspace(4)* %p0, <2 x float> %p1) #1 {
  store <2 x float> %p1, <2 x float> addrspace(4)* %p0
  ret void
}

define linkonce_odr <2 x double> @optix.ptx.ld.const.v2.f64( <2 x double> addrspace(4)* %p0 ) #2 {
  %val = load <2 x double>, <2 x double> addrspace(4)* %p0, align 16
  ret <2 x double> %val
}

define linkonce_odr <2 x double> @optix.ptx.ldu.const.v2.f64( <2 x double> addrspace(4)* %p0 ) #2 {
  %val = load <2 x double>, <2 x double> addrspace(4)* %p0, align 16
  ret <2 x double> %val
}

define linkonce_odr <2 x double> @optix.ptx.ld.volatile.const.v2.f64( <2 x double> addrspace(4)* %p0 ) #1 {
  %val = load volatile <2 x double>, <2 x double> addrspace(4)* %p0, align 16
  ret <2 x double> %val
}

define linkonce_odr void @optix.ptx.st.const.v2.f64( <2 x double> addrspace(4)* %p0, <2 x double> %p1) #1 {
  store <2 x double> %p1, <2 x double> addrspace(4)* %p0
  ret void
}

define linkonce_odr <4 x i8> @optix.ptx.ld.const.v4.b8( <4 x i8> addrspace(4)* %p0 ) #2 {
  %val = load <4 x i8>, <4 x i8> addrspace(4)* %p0, align 4
  ret <4 x i8> %val
}

define linkonce_odr <4 x i8> @optix.ptx.ldu.const.v4.b8( <4 x i8> addrspace(4)* %p0 ) #2 {
  %val = load <4 x i8>, <4 x i8> addrspace(4)* %p0, align 4
  ret <4 x i8> %val
}

define linkonce_odr <4 x i8> @optix.ptx.ld.volatile.const.v4.b8( <4 x i8> addrspace(4)* %p0 ) #1 {
  %val = load volatile <4 x i8>, <4 x i8> addrspace(4)* %p0, align 4
  ret <4 x i8> %val
}

define linkonce_odr void @optix.ptx.st.const.v4.b8( <4 x i8> addrspace(4)* %p0, <4 x i8> %p1) #1 {
  store <4 x i8> %p1, <4 x i8> addrspace(4)* %p0
  ret void
}

define linkonce_odr <4 x i8> @optix.ptx.ld.const.v4.u8( <4 x i8> addrspace(4)* %p0 ) #2 {
  %val = load <4 x i8>, <4 x i8> addrspace(4)* %p0, align 4
  ret <4 x i8> %val
}

define linkonce_odr <4 x i8> @optix.ptx.ldu.const.v4.u8( <4 x i8> addrspace(4)* %p0 ) #2 {
  %val = load <4 x i8>, <4 x i8> addrspace(4)* %p0, align 4
  ret <4 x i8> %val
}

define linkonce_odr <4 x i8> @optix.ptx.ld.volatile.const.v4.u8( <4 x i8> addrspace(4)* %p0 ) #1 {
  %val = load volatile <4 x i8>, <4 x i8> addrspace(4)* %p0, align 4
  ret <4 x i8> %val
}

define linkonce_odr void @optix.ptx.st.const.v4.u8( <4 x i8> addrspace(4)* %p0, <4 x i8> %p1) #1 {
  store <4 x i8> %p1, <4 x i8> addrspace(4)* %p0
  ret void
}

define linkonce_odr <4 x i8> @optix.ptx.ld.const.v4.s8( <4 x i8> addrspace(4)* %p0 ) #2 {
  %val = load <4 x i8>, <4 x i8> addrspace(4)* %p0, align 4
  ret <4 x i8> %val
}

define linkonce_odr <4 x i8> @optix.ptx.ldu.const.v4.s8( <4 x i8> addrspace(4)* %p0 ) #2 {
  %val = load <4 x i8>, <4 x i8> addrspace(4)* %p0, align 4
  ret <4 x i8> %val
}

define linkonce_odr <4 x i8> @optix.ptx.ld.volatile.const.v4.s8( <4 x i8> addrspace(4)* %p0 ) #1 {
  %val = load volatile <4 x i8>, <4 x i8> addrspace(4)* %p0, align 4
  ret <4 x i8> %val
}

define linkonce_odr void @optix.ptx.st.const.v4.s8( <4 x i8> addrspace(4)* %p0, <4 x i8> %p1) #1 {
  store <4 x i8> %p1, <4 x i8> addrspace(4)* %p0
  ret void
}

define linkonce_odr <4 x i16> @optix.ptx.ld.const.v4.b16( <4 x i16> addrspace(4)* %p0 ) #2 {
  %val = load <4 x i16>, <4 x i16> addrspace(4)* %p0, align 8
  ret <4 x i16> %val
}

define linkonce_odr <4 x i16> @optix.ptx.ldu.const.v4.b16( <4 x i16> addrspace(4)* %p0 ) #2 {
  %val = load <4 x i16>, <4 x i16> addrspace(4)* %p0, align 8
  ret <4 x i16> %val
}

define linkonce_odr <4 x i16> @optix.ptx.ld.volatile.const.v4.b16( <4 x i16> addrspace(4)* %p0 ) #1 {
  %val = load volatile <4 x i16>, <4 x i16> addrspace(4)* %p0, align 8
  ret <4 x i16> %val
}

define linkonce_odr void @optix.ptx.st.const.v4.b16( <4 x i16> addrspace(4)* %p0, <4 x i16> %p1) #1 {
  store <4 x i16> %p1, <4 x i16> addrspace(4)* %p0
  ret void
}

define linkonce_odr <4 x i16> @optix.ptx.ld.const.v4.u16( <4 x i16> addrspace(4)* %p0 ) #2 {
  %val = load <4 x i16>, <4 x i16> addrspace(4)* %p0, align 8
  ret <4 x i16> %val
}

define linkonce_odr <4 x i16> @optix.ptx.ldu.const.v4.u16( <4 x i16> addrspace(4)* %p0 ) #2 {
  %val = load <4 x i16>, <4 x i16> addrspace(4)* %p0, align 8
  ret <4 x i16> %val
}

define linkonce_odr <4 x i16> @optix.ptx.ld.volatile.const.v4.u16( <4 x i16> addrspace(4)* %p0 ) #1 {
  %val = load volatile <4 x i16>, <4 x i16> addrspace(4)* %p0, align 8
  ret <4 x i16> %val
}

define linkonce_odr void @optix.ptx.st.const.v4.u16( <4 x i16> addrspace(4)* %p0, <4 x i16> %p1) #1 {
  store <4 x i16> %p1, <4 x i16> addrspace(4)* %p0
  ret void
}

define linkonce_odr <4 x i16> @optix.ptx.ld.const.v4.s16( <4 x i16> addrspace(4)* %p0 ) #2 {
  %val = load <4 x i16>, <4 x i16> addrspace(4)* %p0, align 8
  ret <4 x i16> %val
}

define linkonce_odr <4 x i16> @optix.ptx.ldu.const.v4.s16( <4 x i16> addrspace(4)* %p0 ) #2 {
  %val = load <4 x i16>, <4 x i16> addrspace(4)* %p0, align 8
  ret <4 x i16> %val
}

define linkonce_odr <4 x i16> @optix.ptx.ld.volatile.const.v4.s16( <4 x i16> addrspace(4)* %p0 ) #1 {
  %val = load volatile <4 x i16>, <4 x i16> addrspace(4)* %p0, align 8
  ret <4 x i16> %val
}

define linkonce_odr void @optix.ptx.st.const.v4.s16( <4 x i16> addrspace(4)* %p0, <4 x i16> %p1) #1 {
  store <4 x i16> %p1, <4 x i16> addrspace(4)* %p0
  ret void
}

define linkonce_odr <4 x i32> @optix.ptx.ld.const.v4.b32( <4 x i32> addrspace(4)* %p0 ) #2 {
  %val = load <4 x i32>, <4 x i32> addrspace(4)* %p0, align 16
  ret <4 x i32> %val
}

define linkonce_odr <4 x i32> @optix.ptx.ldu.const.v4.b32( <4 x i32> addrspace(4)* %p0 ) #2 {
  %val = load <4 x i32>, <4 x i32> addrspace(4)* %p0, align 16
  ret <4 x i32> %val
}

define linkonce_odr <4 x i32> @optix.ptx.ld.volatile.const.v4.b32( <4 x i32> addrspace(4)* %p0 ) #1 {
  %val = load volatile <4 x i32>, <4 x i32> addrspace(4)* %p0, align 16
  ret <4 x i32> %val
}

define linkonce_odr void @optix.ptx.st.const.v4.b32( <4 x i32> addrspace(4)* %p0, <4 x i32> %p1) #1 {
  store <4 x i32> %p1, <4 x i32> addrspace(4)* %p0
  ret void
}

define linkonce_odr <4 x i32> @optix.ptx.ld.const.v4.u32( <4 x i32> addrspace(4)* %p0 ) #2 {
  %val = load <4 x i32>, <4 x i32> addrspace(4)* %p0, align 16
  ret <4 x i32> %val
}

define linkonce_odr <4 x i32> @optix.ptx.ldu.const.v4.u32( <4 x i32> addrspace(4)* %p0 ) #2 {
  %val = load <4 x i32>, <4 x i32> addrspace(4)* %p0, align 16
  ret <4 x i32> %val
}

define linkonce_odr <4 x i32> @optix.ptx.ld.volatile.const.v4.u32( <4 x i32> addrspace(4)* %p0 ) #1 {
  %val = load volatile <4 x i32>, <4 x i32> addrspace(4)* %p0, align 16
  ret <4 x i32> %val
}

define linkonce_odr void @optix.ptx.st.const.v4.u32( <4 x i32> addrspace(4)* %p0, <4 x i32> %p1) #1 {
  store <4 x i32> %p1, <4 x i32> addrspace(4)* %p0
  ret void
}

define linkonce_odr <4 x i32> @optix.ptx.ld.const.v4.s32( <4 x i32> addrspace(4)* %p0 ) #2 {
  %val = load <4 x i32>, <4 x i32> addrspace(4)* %p0, align 16
  ret <4 x i32> %val
}

define linkonce_odr <4 x i32> @optix.ptx.ldu.const.v4.s32( <4 x i32> addrspace(4)* %p0 ) #2 {
  %val = load <4 x i32>, <4 x i32> addrspace(4)* %p0, align 16
  ret <4 x i32> %val
}

define linkonce_odr <4 x i32> @optix.ptx.ld.volatile.const.v4.s32( <4 x i32> addrspace(4)* %p0 ) #1 {
  %val = load volatile <4 x i32>, <4 x i32> addrspace(4)* %p0, align 16
  ret <4 x i32> %val
}

define linkonce_odr void @optix.ptx.st.const.v4.s32( <4 x i32> addrspace(4)* %p0, <4 x i32> %p1) #1 {
  store <4 x i32> %p1, <4 x i32> addrspace(4)* %p0
  ret void
}

define linkonce_odr <4 x i64> @optix.ptx.ld.const.v4.b64( <4 x i64> addrspace(4)* %p0 ) #2 {
  %val = load <4 x i64>, <4 x i64> addrspace(4)* %p0, align 32
  ret <4 x i64> %val
}

define linkonce_odr <4 x i64> @optix.ptx.ldu.const.v4.b64( <4 x i64> addrspace(4)* %p0 ) #2 {
  %val = load <4 x i64>, <4 x i64> addrspace(4)* %p0, align 32
  ret <4 x i64> %val
}

define linkonce_odr <4 x i64> @optix.ptx.ld.volatile.const.v4.b64( <4 x i64> addrspace(4)* %p0 ) #1 {
  %val = load volatile <4 x i64>, <4 x i64> addrspace(4)* %p0, align 32
  ret <4 x i64> %val
}

define linkonce_odr void @optix.ptx.st.const.v4.b64( <4 x i64> addrspace(4)* %p0, <4 x i64> %p1) #1 {
  store <4 x i64> %p1, <4 x i64> addrspace(4)* %p0
  ret void
}

define linkonce_odr <4 x i64> @optix.ptx.ld.const.v4.u64( <4 x i64> addrspace(4)* %p0 ) #2 {
  %val = load <4 x i64>, <4 x i64> addrspace(4)* %p0, align 32
  ret <4 x i64> %val
}

define linkonce_odr <4 x i64> @optix.ptx.ldu.const.v4.u64( <4 x i64> addrspace(4)* %p0 ) #2 {
  %val = load <4 x i64>, <4 x i64> addrspace(4)* %p0, align 32
  ret <4 x i64> %val
}

define linkonce_odr <4 x i64> @optix.ptx.ld.volatile.const.v4.u64( <4 x i64> addrspace(4)* %p0 ) #1 {
  %val = load volatile <4 x i64>, <4 x i64> addrspace(4)* %p0, align 32
  ret <4 x i64> %val
}

define linkonce_odr void @optix.ptx.st.const.v4.u64( <4 x i64> addrspace(4)* %p0, <4 x i64> %p1) #1 {
  store <4 x i64> %p1, <4 x i64> addrspace(4)* %p0
  ret void
}

define linkonce_odr <4 x i64> @optix.ptx.ld.const.v4.s64( <4 x i64> addrspace(4)* %p0 ) #2 {
  %val = load <4 x i64>, <4 x i64> addrspace(4)* %p0, align 32
  ret <4 x i64> %val
}

define linkonce_odr <4 x i64> @optix.ptx.ldu.const.v4.s64( <4 x i64> addrspace(4)* %p0 ) #2 {
  %val = load <4 x i64>, <4 x i64> addrspace(4)* %p0, align 32
  ret <4 x i64> %val
}

define linkonce_odr <4 x i64> @optix.ptx.ld.volatile.const.v4.s64( <4 x i64> addrspace(4)* %p0 ) #1 {
  %val = load volatile <4 x i64>, <4 x i64> addrspace(4)* %p0, align 32
  ret <4 x i64> %val
}

define linkonce_odr void @optix.ptx.st.const.v4.s64( <4 x i64> addrspace(4)* %p0, <4 x i64> %p1) #1 {
  store <4 x i64> %p1, <4 x i64> addrspace(4)* %p0
  ret void
}

define linkonce_odr <4 x float> @optix.ptx.ld.const.v4.f32( <4 x float> addrspace(4)* %p0 ) #2 {
  %val = load <4 x float>, <4 x float> addrspace(4)* %p0, align 16
  ret <4 x float> %val
}

define linkonce_odr <4 x float> @optix.ptx.ldu.const.v4.f32( <4 x float> addrspace(4)* %p0 ) #2 {
  %val = load <4 x float>, <4 x float> addrspace(4)* %p0, align 16
  ret <4 x float> %val
}

define linkonce_odr <4 x float> @optix.ptx.ld.volatile.const.v4.f32( <4 x float> addrspace(4)* %p0 ) #1 {
  %val = load volatile <4 x float>, <4 x float> addrspace(4)* %p0, align 16
  ret <4 x float> %val
}

define linkonce_odr void @optix.ptx.st.const.v4.f32( <4 x float> addrspace(4)* %p0, <4 x float> %p1) #1 {
  store <4 x float> %p1, <4 x float> addrspace(4)* %p0
  ret void
}

define linkonce_odr <4 x double> @optix.ptx.ld.const.v4.f64( <4 x double> addrspace(4)* %p0 ) #2 {
  %val = load <4 x double>, <4 x double> addrspace(4)* %p0, align 32
  ret <4 x double> %val
}

define linkonce_odr <4 x double> @optix.ptx.ldu.const.v4.f64( <4 x double> addrspace(4)* %p0 ) #2 {
  %val = load <4 x double>, <4 x double> addrspace(4)* %p0, align 32
  ret <4 x double> %val
}

define linkonce_odr <4 x double> @optix.ptx.ld.volatile.const.v4.f64( <4 x double> addrspace(4)* %p0 ) #1 {
  %val = load volatile <4 x double>, <4 x double> addrspace(4)* %p0, align 32
  ret <4 x double> %val
}

define linkonce_odr void @optix.ptx.st.const.v4.f64( <4 x double> addrspace(4)* %p0, <4 x double> %p1) #1 {
  store <4 x double> %p1, <4 x double> addrspace(4)* %p0
  ret void
}

define linkonce_odr i8 @optix.ptx.ld.local.b8( i8 addrspace(5)* %p0 ) #2 {
  %val = load i8, i8 addrspace(5)* %p0, align 1
  ret i8 %val
}

define linkonce_odr i8 @optix.ptx.ldu.local.b8( i8 addrspace(5)* %p0 ) #2 {
  %val = load i8, i8 addrspace(5)* %p0, align 1
  ret i8 %val
}

define linkonce_odr i8 @optix.ptx.ld.volatile.local.b8( i8 addrspace(5)* %p0 ) #1 {
  %val = load volatile i8, i8 addrspace(5)* %p0, align 1
  ret i8 %val
}

define linkonce_odr void @optix.ptx.st.local.b8( i8 addrspace(5)* %p0, i8 %p1) #1 {
  store i8 %p1, i8 addrspace(5)* %p0
  ret void
}

define linkonce_odr i8 @optix.ptx.ld.local.u8( i8 addrspace(5)* %p0 ) #2 {
  %val = load i8, i8 addrspace(5)* %p0, align 1
  ret i8 %val
}

define linkonce_odr i8 @optix.ptx.ldu.local.u8( i8 addrspace(5)* %p0 ) #2 {
  %val = load i8, i8 addrspace(5)* %p0, align 1
  ret i8 %val
}

define linkonce_odr i8 @optix.ptx.ld.volatile.local.u8( i8 addrspace(5)* %p0 ) #1 {
  %val = load volatile i8, i8 addrspace(5)* %p0, align 1
  ret i8 %val
}

define linkonce_odr void @optix.ptx.st.local.u8( i8 addrspace(5)* %p0, i8 %p1) #1 {
  store i8 %p1, i8 addrspace(5)* %p0
  ret void
}

define linkonce_odr i8 @optix.ptx.ld.local.s8( i8 addrspace(5)* %p0 ) #2 {
  %val = load i8, i8 addrspace(5)* %p0, align 1
  ret i8 %val
}

define linkonce_odr i8 @optix.ptx.ldu.local.s8( i8 addrspace(5)* %p0 ) #2 {
  %val = load i8, i8 addrspace(5)* %p0, align 1
  ret i8 %val
}

define linkonce_odr i8 @optix.ptx.ld.volatile.local.s8( i8 addrspace(5)* %p0 ) #1 {
  %val = load volatile i8, i8 addrspace(5)* %p0, align 1
  ret i8 %val
}

define linkonce_odr void @optix.ptx.st.local.s8( i8 addrspace(5)* %p0, i8 %p1) #1 {
  store i8 %p1, i8 addrspace(5)* %p0
  ret void
}

define linkonce_odr i16 @optix.ptx.ld.local.b16( i16 addrspace(5)* %p0 ) #2 {
  %val = load i16, i16 addrspace(5)* %p0, align 2
  ret i16 %val
}

define linkonce_odr i16 @optix.ptx.ldu.local.b16( i16 addrspace(5)* %p0 ) #2 {
  %val = load i16, i16 addrspace(5)* %p0, align 2
  ret i16 %val
}

define linkonce_odr i16 @optix.ptx.ld.volatile.local.b16( i16 addrspace(5)* %p0 ) #1 {
  %val = load volatile i16, i16 addrspace(5)* %p0, align 2
  ret i16 %val
}

define linkonce_odr void @optix.ptx.st.local.b16( i16 addrspace(5)* %p0, i16 %p1) #1 {
  store i16 %p1, i16 addrspace(5)* %p0
  ret void
}

define linkonce_odr i16 @optix.ptx.ld.local.u16( i16 addrspace(5)* %p0 ) #2 {
  %val = load i16, i16 addrspace(5)* %p0, align 2
  ret i16 %val
}

define linkonce_odr i16 @optix.ptx.ldu.local.u16( i16 addrspace(5)* %p0 ) #2 {
  %val = load i16, i16 addrspace(5)* %p0, align 2
  ret i16 %val
}

define linkonce_odr i16 @optix.ptx.ld.volatile.local.u16( i16 addrspace(5)* %p0 ) #1 {
  %val = load volatile i16, i16 addrspace(5)* %p0, align 2
  ret i16 %val
}

define linkonce_odr void @optix.ptx.st.local.u16( i16 addrspace(5)* %p0, i16 %p1) #1 {
  store i16 %p1, i16 addrspace(5)* %p0
  ret void
}

define linkonce_odr i16 @optix.ptx.ld.local.s16( i16 addrspace(5)* %p0 ) #2 {
  %val = load i16, i16 addrspace(5)* %p0, align 2
  ret i16 %val
}

define linkonce_odr i16 @optix.ptx.ldu.local.s16( i16 addrspace(5)* %p0 ) #2 {
  %val = load i16, i16 addrspace(5)* %p0, align 2
  ret i16 %val
}

define linkonce_odr i16 @optix.ptx.ld.volatile.local.s16( i16 addrspace(5)* %p0 ) #1 {
  %val = load volatile i16, i16 addrspace(5)* %p0, align 2
  ret i16 %val
}

define linkonce_odr void @optix.ptx.st.local.s16( i16 addrspace(5)* %p0, i16 %p1) #1 {
  store i16 %p1, i16 addrspace(5)* %p0
  ret void
}

define linkonce_odr i32 @optix.ptx.ld.local.b32( i32 addrspace(5)* %p0 ) #2 {
  %val = load i32, i32 addrspace(5)* %p0, align 4
  ret i32 %val
}

define linkonce_odr i32 @optix.ptx.ldu.local.b32( i32 addrspace(5)* %p0 ) #2 {
  %val = load i32, i32 addrspace(5)* %p0, align 4
  ret i32 %val
}

define linkonce_odr i32 @optix.ptx.ld.volatile.local.b32( i32 addrspace(5)* %p0 ) #1 {
  %val = load volatile i32, i32 addrspace(5)* %p0, align 4
  ret i32 %val
}

define linkonce_odr void @optix.ptx.st.local.b32( i32 addrspace(5)* %p0, i32 %p1) #1 {
  store i32 %p1, i32 addrspace(5)* %p0
  ret void
}

define linkonce_odr i32 @optix.ptx.ld.local.u32( i32 addrspace(5)* %p0 ) #2 {
  %val = load i32, i32 addrspace(5)* %p0, align 4
  ret i32 %val
}

define linkonce_odr i32 @optix.ptx.ldu.local.u32( i32 addrspace(5)* %p0 ) #2 {
  %val = load i32, i32 addrspace(5)* %p0, align 4
  ret i32 %val
}

define linkonce_odr i32 @optix.ptx.ld.volatile.local.u32( i32 addrspace(5)* %p0 ) #1 {
  %val = load volatile i32, i32 addrspace(5)* %p0, align 4
  ret i32 %val
}

define linkonce_odr void @optix.ptx.st.local.u32( i32 addrspace(5)* %p0, i32 %p1) #1 {
  store i32 %p1, i32 addrspace(5)* %p0
  ret void
}

define linkonce_odr i32 @optix.ptx.ld.local.s32( i32 addrspace(5)* %p0 ) #2 {
  %val = load i32, i32 addrspace(5)* %p0, align 4
  ret i32 %val
}

define linkonce_odr i32 @optix.ptx.ldu.local.s32( i32 addrspace(5)* %p0 ) #2 {
  %val = load i32, i32 addrspace(5)* %p0, align 4
  ret i32 %val
}

define linkonce_odr i32 @optix.ptx.ld.volatile.local.s32( i32 addrspace(5)* %p0 ) #1 {
  %val = load volatile i32, i32 addrspace(5)* %p0, align 4
  ret i32 %val
}

define linkonce_odr void @optix.ptx.st.local.s32( i32 addrspace(5)* %p0, i32 %p1) #1 {
  store i32 %p1, i32 addrspace(5)* %p0
  ret void
}

define linkonce_odr i64 @optix.ptx.ld.local.b64( i64 addrspace(5)* %p0 ) #2 {
  %val = load i64, i64 addrspace(5)* %p0, align 8
  ret i64 %val
}

define linkonce_odr i64 @optix.ptx.ldu.local.b64( i64 addrspace(5)* %p0 ) #2 {
  %val = load i64, i64 addrspace(5)* %p0, align 8
  ret i64 %val
}

define linkonce_odr i64 @optix.ptx.ld.volatile.local.b64( i64 addrspace(5)* %p0 ) #1 {
  %val = load volatile i64, i64 addrspace(5)* %p0, align 8
  ret i64 %val
}

define linkonce_odr void @optix.ptx.st.local.b64( i64 addrspace(5)* %p0, i64 %p1) #1 {
  store i64 %p1, i64 addrspace(5)* %p0
  ret void
}

define linkonce_odr i64 @optix.ptx.ld.local.u64( i64 addrspace(5)* %p0 ) #2 {
  %val = load i64, i64 addrspace(5)* %p0, align 8
  ret i64 %val
}

define linkonce_odr i64 @optix.ptx.ldu.local.u64( i64 addrspace(5)* %p0 ) #2 {
  %val = load i64, i64 addrspace(5)* %p0, align 8
  ret i64 %val
}

define linkonce_odr i64 @optix.ptx.ld.volatile.local.u64( i64 addrspace(5)* %p0 ) #1 {
  %val = load volatile i64, i64 addrspace(5)* %p0, align 8
  ret i64 %val
}

define linkonce_odr void @optix.ptx.st.local.u64( i64 addrspace(5)* %p0, i64 %p1) #1 {
  store i64 %p1, i64 addrspace(5)* %p0
  ret void
}

define linkonce_odr i64 @optix.ptx.ld.local.s64( i64 addrspace(5)* %p0 ) #2 {
  %val = load i64, i64 addrspace(5)* %p0, align 8
  ret i64 %val
}

define linkonce_odr i64 @optix.ptx.ldu.local.s64( i64 addrspace(5)* %p0 ) #2 {
  %val = load i64, i64 addrspace(5)* %p0, align 8
  ret i64 %val
}

define linkonce_odr i64 @optix.ptx.ld.volatile.local.s64( i64 addrspace(5)* %p0 ) #1 {
  %val = load volatile i64, i64 addrspace(5)* %p0, align 8
  ret i64 %val
}

define linkonce_odr void @optix.ptx.st.local.s64( i64 addrspace(5)* %p0, i64 %p1) #1 {
  store i64 %p1, i64 addrspace(5)* %p0
  ret void
}

define linkonce_odr float @optix.ptx.ld.local.f32( float addrspace(5)* %p0 ) #2 {
  %val = load float, float addrspace(5)* %p0, align 4
  ret float %val
}

define linkonce_odr float @optix.ptx.ldu.local.f32( float addrspace(5)* %p0 ) #2 {
  %val = load float, float addrspace(5)* %p0, align 4
  ret float %val
}

define linkonce_odr float @optix.ptx.ld.volatile.local.f32( float addrspace(5)* %p0 ) #1 {
  %val = load volatile float, float addrspace(5)* %p0, align 4
  ret float %val
}

define linkonce_odr void @optix.ptx.st.local.f32( float addrspace(5)* %p0, float %p1) #1 {
  store float %p1, float addrspace(5)* %p0
  ret void
}

define linkonce_odr double @optix.ptx.ld.local.f64( double addrspace(5)* %p0 ) #2 {
  %val = load double, double addrspace(5)* %p0, align 8
  ret double %val
}

define linkonce_odr double @optix.ptx.ldu.local.f64( double addrspace(5)* %p0 ) #2 {
  %val = load double, double addrspace(5)* %p0, align 8
  ret double %val
}

define linkonce_odr double @optix.ptx.ld.volatile.local.f64( double addrspace(5)* %p0 ) #1 {
  %val = load volatile double, double addrspace(5)* %p0, align 8
  ret double %val
}

define linkonce_odr void @optix.ptx.st.local.f64( double addrspace(5)* %p0, double %p1) #1 {
  store double %p1, double addrspace(5)* %p0
  ret void
}

define linkonce_odr <2 x i8> @optix.ptx.ld.local.v2.b8( <2 x i8> addrspace(5)* %p0 ) #2 {
  %val = load <2 x i8>, <2 x i8> addrspace(5)* %p0, align 2
  ret <2 x i8> %val
}

define linkonce_odr <2 x i8> @optix.ptx.ldu.local.v2.b8( <2 x i8> addrspace(5)* %p0 ) #2 {
  %val = load <2 x i8>, <2 x i8> addrspace(5)* %p0, align 2
  ret <2 x i8> %val
}

define linkonce_odr <2 x i8> @optix.ptx.ld.volatile.local.v2.b8( <2 x i8> addrspace(5)* %p0 ) #1 {
  %val = load volatile <2 x i8>, <2 x i8> addrspace(5)* %p0, align 2
  ret <2 x i8> %val
}

define linkonce_odr void @optix.ptx.st.local.v2.b8( <2 x i8> addrspace(5)* %p0, <2 x i8> %p1) #1 {
  store <2 x i8> %p1, <2 x i8> addrspace(5)* %p0
  ret void
}

define linkonce_odr <2 x i8> @optix.ptx.ld.local.v2.u8( <2 x i8> addrspace(5)* %p0 ) #2 {
  %val = load <2 x i8>, <2 x i8> addrspace(5)* %p0, align 2
  ret <2 x i8> %val
}

define linkonce_odr <2 x i8> @optix.ptx.ldu.local.v2.u8( <2 x i8> addrspace(5)* %p0 ) #2 {
  %val = load <2 x i8>, <2 x i8> addrspace(5)* %p0, align 2
  ret <2 x i8> %val
}

define linkonce_odr <2 x i8> @optix.ptx.ld.volatile.local.v2.u8( <2 x i8> addrspace(5)* %p0 ) #1 {
  %val = load volatile <2 x i8>, <2 x i8> addrspace(5)* %p0, align 2
  ret <2 x i8> %val
}

define linkonce_odr void @optix.ptx.st.local.v2.u8( <2 x i8> addrspace(5)* %p0, <2 x i8> %p1) #1 {
  store <2 x i8> %p1, <2 x i8> addrspace(5)* %p0
  ret void
}

define linkonce_odr <2 x i8> @optix.ptx.ld.local.v2.s8( <2 x i8> addrspace(5)* %p0 ) #2 {
  %val = load <2 x i8>, <2 x i8> addrspace(5)* %p0, align 2
  ret <2 x i8> %val
}

define linkonce_odr <2 x i8> @optix.ptx.ldu.local.v2.s8( <2 x i8> addrspace(5)* %p0 ) #2 {
  %val = load <2 x i8>, <2 x i8> addrspace(5)* %p0, align 2
  ret <2 x i8> %val
}

define linkonce_odr <2 x i8> @optix.ptx.ld.volatile.local.v2.s8( <2 x i8> addrspace(5)* %p0 ) #1 {
  %val = load volatile <2 x i8>, <2 x i8> addrspace(5)* %p0, align 2
  ret <2 x i8> %val
}

define linkonce_odr void @optix.ptx.st.local.v2.s8( <2 x i8> addrspace(5)* %p0, <2 x i8> %p1) #1 {
  store <2 x i8> %p1, <2 x i8> addrspace(5)* %p0
  ret void
}

define linkonce_odr <2 x i16> @optix.ptx.ld.local.v2.b16( <2 x i16> addrspace(5)* %p0 ) #2 {
  %val = load <2 x i16>, <2 x i16> addrspace(5)* %p0, align 4
  ret <2 x i16> %val
}

define linkonce_odr <2 x i16> @optix.ptx.ldu.local.v2.b16( <2 x i16> addrspace(5)* %p0 ) #2 {
  %val = load <2 x i16>, <2 x i16> addrspace(5)* %p0, align 4
  ret <2 x i16> %val
}

define linkonce_odr <2 x i16> @optix.ptx.ld.volatile.local.v2.b16( <2 x i16> addrspace(5)* %p0 ) #1 {
  %val = load volatile <2 x i16>, <2 x i16> addrspace(5)* %p0, align 4
  ret <2 x i16> %val
}

define linkonce_odr void @optix.ptx.st.local.v2.b16( <2 x i16> addrspace(5)* %p0, <2 x i16> %p1) #1 {
  store <2 x i16> %p1, <2 x i16> addrspace(5)* %p0
  ret void
}

define linkonce_odr <2 x i16> @optix.ptx.ld.local.v2.u16( <2 x i16> addrspace(5)* %p0 ) #2 {
  %val = load <2 x i16>, <2 x i16> addrspace(5)* %p0, align 4
  ret <2 x i16> %val
}

define linkonce_odr <2 x i16> @optix.ptx.ldu.local.v2.u16( <2 x i16> addrspace(5)* %p0 ) #2 {
  %val = load <2 x i16>, <2 x i16> addrspace(5)* %p0, align 4
  ret <2 x i16> %val
}

define linkonce_odr <2 x i16> @optix.ptx.ld.volatile.local.v2.u16( <2 x i16> addrspace(5)* %p0 ) #1 {
  %val = load volatile <2 x i16>, <2 x i16> addrspace(5)* %p0, align 4
  ret <2 x i16> %val
}

define linkonce_odr void @optix.ptx.st.local.v2.u16( <2 x i16> addrspace(5)* %p0, <2 x i16> %p1) #1 {
  store <2 x i16> %p1, <2 x i16> addrspace(5)* %p0
  ret void
}

define linkonce_odr <2 x i16> @optix.ptx.ld.local.v2.s16( <2 x i16> addrspace(5)* %p0 ) #2 {
  %val = load <2 x i16>, <2 x i16> addrspace(5)* %p0, align 4
  ret <2 x i16> %val
}

define linkonce_odr <2 x i16> @optix.ptx.ldu.local.v2.s16( <2 x i16> addrspace(5)* %p0 ) #2 {
  %val = load <2 x i16>, <2 x i16> addrspace(5)* %p0, align 4
  ret <2 x i16> %val
}

define linkonce_odr <2 x i16> @optix.ptx.ld.volatile.local.v2.s16( <2 x i16> addrspace(5)* %p0 ) #1 {
  %val = load volatile <2 x i16>, <2 x i16> addrspace(5)* %p0, align 4
  ret <2 x i16> %val
}

define linkonce_odr void @optix.ptx.st.local.v2.s16( <2 x i16> addrspace(5)* %p0, <2 x i16> %p1) #1 {
  store <2 x i16> %p1, <2 x i16> addrspace(5)* %p0
  ret void
}

define linkonce_odr <2 x i32> @optix.ptx.ld.local.v2.b32( <2 x i32> addrspace(5)* %p0 ) #2 {
  %val = load <2 x i32>, <2 x i32> addrspace(5)* %p0, align 8
  ret <2 x i32> %val
}

define linkonce_odr <2 x i32> @optix.ptx.ldu.local.v2.b32( <2 x i32> addrspace(5)* %p0 ) #2 {
  %val = load <2 x i32>, <2 x i32> addrspace(5)* %p0, align 8
  ret <2 x i32> %val
}

define linkonce_odr <2 x i32> @optix.ptx.ld.volatile.local.v2.b32( <2 x i32> addrspace(5)* %p0 ) #1 {
  %val = load volatile <2 x i32>, <2 x i32> addrspace(5)* %p0, align 8
  ret <2 x i32> %val
}

define linkonce_odr void @optix.ptx.st.local.v2.b32( <2 x i32> addrspace(5)* %p0, <2 x i32> %p1) #1 {
  store <2 x i32> %p1, <2 x i32> addrspace(5)* %p0
  ret void
}

define linkonce_odr <2 x i32> @optix.ptx.ld.local.v2.u32( <2 x i32> addrspace(5)* %p0 ) #2 {
  %val = load <2 x i32>, <2 x i32> addrspace(5)* %p0, align 8
  ret <2 x i32> %val
}

define linkonce_odr <2 x i32> @optix.ptx.ldu.local.v2.u32( <2 x i32> addrspace(5)* %p0 ) #2 {
  %val = load <2 x i32>, <2 x i32> addrspace(5)* %p0, align 8
  ret <2 x i32> %val
}

define linkonce_odr <2 x i32> @optix.ptx.ld.volatile.local.v2.u32( <2 x i32> addrspace(5)* %p0 ) #1 {
  %val = load volatile <2 x i32>, <2 x i32> addrspace(5)* %p0, align 8
  ret <2 x i32> %val
}

define linkonce_odr void @optix.ptx.st.local.v2.u32( <2 x i32> addrspace(5)* %p0, <2 x i32> %p1) #1 {
  store <2 x i32> %p1, <2 x i32> addrspace(5)* %p0
  ret void
}

define linkonce_odr <2 x i32> @optix.ptx.ld.local.v2.s32( <2 x i32> addrspace(5)* %p0 ) #2 {
  %val = load <2 x i32>, <2 x i32> addrspace(5)* %p0, align 8
  ret <2 x i32> %val
}

define linkonce_odr <2 x i32> @optix.ptx.ldu.local.v2.s32( <2 x i32> addrspace(5)* %p0 ) #2 {
  %val = load <2 x i32>, <2 x i32> addrspace(5)* %p0, align 8
  ret <2 x i32> %val
}

define linkonce_odr <2 x i32> @optix.ptx.ld.volatile.local.v2.s32( <2 x i32> addrspace(5)* %p0 ) #1 {
  %val = load volatile <2 x i32>, <2 x i32> addrspace(5)* %p0, align 8
  ret <2 x i32> %val
}

define linkonce_odr void @optix.ptx.st.local.v2.s32( <2 x i32> addrspace(5)* %p0, <2 x i32> %p1) #1 {
  store <2 x i32> %p1, <2 x i32> addrspace(5)* %p0
  ret void
}

define linkonce_odr <2 x i64> @optix.ptx.ld.local.v2.b64( <2 x i64> addrspace(5)* %p0 ) #2 {
  %val = load <2 x i64>, <2 x i64> addrspace(5)* %p0, align 16
  ret <2 x i64> %val
}

define linkonce_odr <2 x i64> @optix.ptx.ldu.local.v2.b64( <2 x i64> addrspace(5)* %p0 ) #2 {
  %val = load <2 x i64>, <2 x i64> addrspace(5)* %p0, align 16
  ret <2 x i64> %val
}

define linkonce_odr <2 x i64> @optix.ptx.ld.volatile.local.v2.b64( <2 x i64> addrspace(5)* %p0 ) #1 {
  %val = load volatile <2 x i64>, <2 x i64> addrspace(5)* %p0, align 16
  ret <2 x i64> %val
}

define linkonce_odr void @optix.ptx.st.local.v2.b64( <2 x i64> addrspace(5)* %p0, <2 x i64> %p1) #1 {
  store <2 x i64> %p1, <2 x i64> addrspace(5)* %p0
  ret void
}

define linkonce_odr <2 x i64> @optix.ptx.ld.local.v2.u64( <2 x i64> addrspace(5)* %p0 ) #2 {
  %val = load <2 x i64>, <2 x i64> addrspace(5)* %p0, align 16
  ret <2 x i64> %val
}

define linkonce_odr <2 x i64> @optix.ptx.ldu.local.v2.u64( <2 x i64> addrspace(5)* %p0 ) #2 {
  %val = load <2 x i64>, <2 x i64> addrspace(5)* %p0, align 16
  ret <2 x i64> %val
}

define linkonce_odr <2 x i64> @optix.ptx.ld.volatile.local.v2.u64( <2 x i64> addrspace(5)* %p0 ) #1 {
  %val = load volatile <2 x i64>, <2 x i64> addrspace(5)* %p0, align 16
  ret <2 x i64> %val
}

define linkonce_odr void @optix.ptx.st.local.v2.u64( <2 x i64> addrspace(5)* %p0, <2 x i64> %p1) #1 {
  store <2 x i64> %p1, <2 x i64> addrspace(5)* %p0
  ret void
}

define linkonce_odr <2 x i64> @optix.ptx.ld.local.v2.s64( <2 x i64> addrspace(5)* %p0 ) #2 {
  %val = load <2 x i64>, <2 x i64> addrspace(5)* %p0, align 16
  ret <2 x i64> %val
}

define linkonce_odr <2 x i64> @optix.ptx.ldu.local.v2.s64( <2 x i64> addrspace(5)* %p0 ) #2 {
  %val = load <2 x i64>, <2 x i64> addrspace(5)* %p0, align 16
  ret <2 x i64> %val
}

define linkonce_odr <2 x i64> @optix.ptx.ld.volatile.local.v2.s64( <2 x i64> addrspace(5)* %p0 ) #1 {
  %val = load volatile <2 x i64>, <2 x i64> addrspace(5)* %p0, align 16
  ret <2 x i64> %val
}

define linkonce_odr void @optix.ptx.st.local.v2.s64( <2 x i64> addrspace(5)* %p0, <2 x i64> %p1) #1 {
  store <2 x i64> %p1, <2 x i64> addrspace(5)* %p0
  ret void
}

define linkonce_odr <2 x float> @optix.ptx.ld.local.v2.f32( <2 x float> addrspace(5)* %p0 ) #2 {
  %val = load <2 x float>, <2 x float> addrspace(5)* %p0, align 8
  ret <2 x float> %val
}

define linkonce_odr <2 x float> @optix.ptx.ldu.local.v2.f32( <2 x float> addrspace(5)* %p0 ) #2 {
  %val = load <2 x float>, <2 x float> addrspace(5)* %p0, align 8
  ret <2 x float> %val
}

define linkonce_odr <2 x float> @optix.ptx.ld.volatile.local.v2.f32( <2 x float> addrspace(5)* %p0 ) #1 {
  %val = load volatile <2 x float>, <2 x float> addrspace(5)* %p0, align 8
  ret <2 x float> %val
}

define linkonce_odr void @optix.ptx.st.local.v2.f32( <2 x float> addrspace(5)* %p0, <2 x float> %p1) #1 {
  store <2 x float> %p1, <2 x float> addrspace(5)* %p0
  ret void
}

define linkonce_odr <2 x double> @optix.ptx.ld.local.v2.f64( <2 x double> addrspace(5)* %p0 ) #2 {
  %val = load <2 x double>, <2 x double> addrspace(5)* %p0, align 16
  ret <2 x double> %val
}

define linkonce_odr <2 x double> @optix.ptx.ldu.local.v2.f64( <2 x double> addrspace(5)* %p0 ) #2 {
  %val = load <2 x double>, <2 x double> addrspace(5)* %p0, align 16
  ret <2 x double> %val
}

define linkonce_odr <2 x double> @optix.ptx.ld.volatile.local.v2.f64( <2 x double> addrspace(5)* %p0 ) #1 {
  %val = load volatile <2 x double>, <2 x double> addrspace(5)* %p0, align 16
  ret <2 x double> %val
}

define linkonce_odr void @optix.ptx.st.local.v2.f64( <2 x double> addrspace(5)* %p0, <2 x double> %p1) #1 {
  store <2 x double> %p1, <2 x double> addrspace(5)* %p0
  ret void
}

define linkonce_odr <4 x i8> @optix.ptx.ld.local.v4.b8( <4 x i8> addrspace(5)* %p0 ) #2 {
  %val = load <4 x i8>, <4 x i8> addrspace(5)* %p0, align 4
  ret <4 x i8> %val
}

define linkonce_odr <4 x i8> @optix.ptx.ldu.local.v4.b8( <4 x i8> addrspace(5)* %p0 ) #2 {
  %val = load <4 x i8>, <4 x i8> addrspace(5)* %p0, align 4
  ret <4 x i8> %val
}

define linkonce_odr <4 x i8> @optix.ptx.ld.volatile.local.v4.b8( <4 x i8> addrspace(5)* %p0 ) #1 {
  %val = load volatile <4 x i8>, <4 x i8> addrspace(5)* %p0, align 4
  ret <4 x i8> %val
}

define linkonce_odr void @optix.ptx.st.local.v4.b8( <4 x i8> addrspace(5)* %p0, <4 x i8> %p1) #1 {
  store <4 x i8> %p1, <4 x i8> addrspace(5)* %p0
  ret void
}

define linkonce_odr <4 x i8> @optix.ptx.ld.local.v4.u8( <4 x i8> addrspace(5)* %p0 ) #2 {
  %val = load <4 x i8>, <4 x i8> addrspace(5)* %p0, align 4
  ret <4 x i8> %val
}

define linkonce_odr <4 x i8> @optix.ptx.ldu.local.v4.u8( <4 x i8> addrspace(5)* %p0 ) #2 {
  %val = load <4 x i8>, <4 x i8> addrspace(5)* %p0, align 4
  ret <4 x i8> %val
}

define linkonce_odr <4 x i8> @optix.ptx.ld.volatile.local.v4.u8( <4 x i8> addrspace(5)* %p0 ) #1 {
  %val = load volatile <4 x i8>, <4 x i8> addrspace(5)* %p0, align 4
  ret <4 x i8> %val
}

define linkonce_odr void @optix.ptx.st.local.v4.u8( <4 x i8> addrspace(5)* %p0, <4 x i8> %p1) #1 {
  store <4 x i8> %p1, <4 x i8> addrspace(5)* %p0
  ret void
}

define linkonce_odr <4 x i8> @optix.ptx.ld.local.v4.s8( <4 x i8> addrspace(5)* %p0 ) #2 {
  %val = load <4 x i8>, <4 x i8> addrspace(5)* %p0, align 4
  ret <4 x i8> %val
}

define linkonce_odr <4 x i8> @optix.ptx.ldu.local.v4.s8( <4 x i8> addrspace(5)* %p0 ) #2 {
  %val = load <4 x i8>, <4 x i8> addrspace(5)* %p0, align 4
  ret <4 x i8> %val
}

define linkonce_odr <4 x i8> @optix.ptx.ld.volatile.local.v4.s8( <4 x i8> addrspace(5)* %p0 ) #1 {
  %val = load volatile <4 x i8>, <4 x i8> addrspace(5)* %p0, align 4
  ret <4 x i8> %val
}

define linkonce_odr void @optix.ptx.st.local.v4.s8( <4 x i8> addrspace(5)* %p0, <4 x i8> %p1) #1 {
  store <4 x i8> %p1, <4 x i8> addrspace(5)* %p0
  ret void
}

define linkonce_odr <4 x i16> @optix.ptx.ld.local.v4.b16( <4 x i16> addrspace(5)* %p0 ) #2 {
  %val = load <4 x i16>, <4 x i16> addrspace(5)* %p0, align 8
  ret <4 x i16> %val
}

define linkonce_odr <4 x i16> @optix.ptx.ldu.local.v4.b16( <4 x i16> addrspace(5)* %p0 ) #2 {
  %val = load <4 x i16>, <4 x i16> addrspace(5)* %p0, align 8
  ret <4 x i16> %val
}

define linkonce_odr <4 x i16> @optix.ptx.ld.volatile.local.v4.b16( <4 x i16> addrspace(5)* %p0 ) #1 {
  %val = load volatile <4 x i16>, <4 x i16> addrspace(5)* %p0, align 8
  ret <4 x i16> %val
}

define linkonce_odr void @optix.ptx.st.local.v4.b16( <4 x i16> addrspace(5)* %p0, <4 x i16> %p1) #1 {
  store <4 x i16> %p1, <4 x i16> addrspace(5)* %p0
  ret void
}

define linkonce_odr <4 x i16> @optix.ptx.ld.local.v4.u16( <4 x i16> addrspace(5)* %p0 ) #2 {
  %val = load <4 x i16>, <4 x i16> addrspace(5)* %p0, align 8
  ret <4 x i16> %val
}

define linkonce_odr <4 x i16> @optix.ptx.ldu.local.v4.u16( <4 x i16> addrspace(5)* %p0 ) #2 {
  %val = load <4 x i16>, <4 x i16> addrspace(5)* %p0, align 8
  ret <4 x i16> %val
}

define linkonce_odr <4 x i16> @optix.ptx.ld.volatile.local.v4.u16( <4 x i16> addrspace(5)* %p0 ) #1 {
  %val = load volatile <4 x i16>, <4 x i16> addrspace(5)* %p0, align 8
  ret <4 x i16> %val
}

define linkonce_odr void @optix.ptx.st.local.v4.u16( <4 x i16> addrspace(5)* %p0, <4 x i16> %p1) #1 {
  store <4 x i16> %p1, <4 x i16> addrspace(5)* %p0
  ret void
}

define linkonce_odr <4 x i16> @optix.ptx.ld.local.v4.s16( <4 x i16> addrspace(5)* %p0 ) #2 {
  %val = load <4 x i16>, <4 x i16> addrspace(5)* %p0, align 8
  ret <4 x i16> %val
}

define linkonce_odr <4 x i16> @optix.ptx.ldu.local.v4.s16( <4 x i16> addrspace(5)* %p0 ) #2 {
  %val = load <4 x i16>, <4 x i16> addrspace(5)* %p0, align 8
  ret <4 x i16> %val
}

define linkonce_odr <4 x i16> @optix.ptx.ld.volatile.local.v4.s16( <4 x i16> addrspace(5)* %p0 ) #1 {
  %val = load volatile <4 x i16>, <4 x i16> addrspace(5)* %p0, align 8
  ret <4 x i16> %val
}

define linkonce_odr void @optix.ptx.st.local.v4.s16( <4 x i16> addrspace(5)* %p0, <4 x i16> %p1) #1 {
  store <4 x i16> %p1, <4 x i16> addrspace(5)* %p0
  ret void
}

define linkonce_odr <4 x i32> @optix.ptx.ld.local.v4.b32( <4 x i32> addrspace(5)* %p0 ) #2 {
  %val = load <4 x i32>, <4 x i32> addrspace(5)* %p0, align 16
  ret <4 x i32> %val
}

define linkonce_odr <4 x i32> @optix.ptx.ldu.local.v4.b32( <4 x i32> addrspace(5)* %p0 ) #2 {
  %val = load <4 x i32>, <4 x i32> addrspace(5)* %p0, align 16
  ret <4 x i32> %val
}

define linkonce_odr <4 x i32> @optix.ptx.ld.volatile.local.v4.b32( <4 x i32> addrspace(5)* %p0 ) #1 {
  %val = load volatile <4 x i32>, <4 x i32> addrspace(5)* %p0, align 16
  ret <4 x i32> %val
}

define linkonce_odr void @optix.ptx.st.local.v4.b32( <4 x i32> addrspace(5)* %p0, <4 x i32> %p1) #1 {
  store <4 x i32> %p1, <4 x i32> addrspace(5)* %p0
  ret void
}

define linkonce_odr <4 x i32> @optix.ptx.ld.local.v4.u32( <4 x i32> addrspace(5)* %p0 ) #2 {
  %val = load <4 x i32>, <4 x i32> addrspace(5)* %p0, align 16
  ret <4 x i32> %val
}

define linkonce_odr <4 x i32> @optix.ptx.ldu.local.v4.u32( <4 x i32> addrspace(5)* %p0 ) #2 {
  %val = load <4 x i32>, <4 x i32> addrspace(5)* %p0, align 16
  ret <4 x i32> %val
}

define linkonce_odr <4 x i32> @optix.ptx.ld.volatile.local.v4.u32( <4 x i32> addrspace(5)* %p0 ) #1 {
  %val = load volatile <4 x i32>, <4 x i32> addrspace(5)* %p0, align 16
  ret <4 x i32> %val
}

define linkonce_odr void @optix.ptx.st.local.v4.u32( <4 x i32> addrspace(5)* %p0, <4 x i32> %p1) #1 {
  store <4 x i32> %p1, <4 x i32> addrspace(5)* %p0
  ret void
}

define linkonce_odr <4 x i32> @optix.ptx.ld.local.v4.s32( <4 x i32> addrspace(5)* %p0 ) #2 {
  %val = load <4 x i32>, <4 x i32> addrspace(5)* %p0, align 16
  ret <4 x i32> %val
}

define linkonce_odr <4 x i32> @optix.ptx.ldu.local.v4.s32( <4 x i32> addrspace(5)* %p0 ) #2 {
  %val = load <4 x i32>, <4 x i32> addrspace(5)* %p0, align 16
  ret <4 x i32> %val
}

define linkonce_odr <4 x i32> @optix.ptx.ld.volatile.local.v4.s32( <4 x i32> addrspace(5)* %p0 ) #1 {
  %val = load volatile <4 x i32>, <4 x i32> addrspace(5)* %p0, align 16
  ret <4 x i32> %val
}

define linkonce_odr void @optix.ptx.st.local.v4.s32( <4 x i32> addrspace(5)* %p0, <4 x i32> %p1) #1 {
  store <4 x i32> %p1, <4 x i32> addrspace(5)* %p0
  ret void
}

define linkonce_odr <4 x i64> @optix.ptx.ld.local.v4.b64( <4 x i64> addrspace(5)* %p0 ) #2 {
  %val = load <4 x i64>, <4 x i64> addrspace(5)* %p0, align 32
  ret <4 x i64> %val
}

define linkonce_odr <4 x i64> @optix.ptx.ldu.local.v4.b64( <4 x i64> addrspace(5)* %p0 ) #2 {
  %val = load <4 x i64>, <4 x i64> addrspace(5)* %p0, align 32
  ret <4 x i64> %val
}

define linkonce_odr <4 x i64> @optix.ptx.ld.volatile.local.v4.b64( <4 x i64> addrspace(5)* %p0 ) #1 {
  %val = load volatile <4 x i64>, <4 x i64> addrspace(5)* %p0, align 32
  ret <4 x i64> %val
}

define linkonce_odr void @optix.ptx.st.local.v4.b64( <4 x i64> addrspace(5)* %p0, <4 x i64> %p1) #1 {
  store <4 x i64> %p1, <4 x i64> addrspace(5)* %p0
  ret void
}

define linkonce_odr <4 x i64> @optix.ptx.ld.local.v4.u64( <4 x i64> addrspace(5)* %p0 ) #2 {
  %val = load <4 x i64>, <4 x i64> addrspace(5)* %p0, align 32
  ret <4 x i64> %val
}

define linkonce_odr <4 x i64> @optix.ptx.ldu.local.v4.u64( <4 x i64> addrspace(5)* %p0 ) #2 {
  %val = load <4 x i64>, <4 x i64> addrspace(5)* %p0, align 32
  ret <4 x i64> %val
}

define linkonce_odr <4 x i64> @optix.ptx.ld.volatile.local.v4.u64( <4 x i64> addrspace(5)* %p0 ) #1 {
  %val = load volatile <4 x i64>, <4 x i64> addrspace(5)* %p0, align 32
  ret <4 x i64> %val
}

define linkonce_odr void @optix.ptx.st.local.v4.u64( <4 x i64> addrspace(5)* %p0, <4 x i64> %p1) #1 {
  store <4 x i64> %p1, <4 x i64> addrspace(5)* %p0
  ret void
}

define linkonce_odr <4 x i64> @optix.ptx.ld.local.v4.s64( <4 x i64> addrspace(5)* %p0 ) #2 {
  %val = load <4 x i64>, <4 x i64> addrspace(5)* %p0, align 32
  ret <4 x i64> %val
}

define linkonce_odr <4 x i64> @optix.ptx.ldu.local.v4.s64( <4 x i64> addrspace(5)* %p0 ) #2 {
  %val = load <4 x i64>, <4 x i64> addrspace(5)* %p0, align 32
  ret <4 x i64> %val
}

define linkonce_odr <4 x i64> @optix.ptx.ld.volatile.local.v4.s64( <4 x i64> addrspace(5)* %p0 ) #1 {
  %val = load volatile <4 x i64>, <4 x i64> addrspace(5)* %p0, align 32
  ret <4 x i64> %val
}

define linkonce_odr void @optix.ptx.st.local.v4.s64( <4 x i64> addrspace(5)* %p0, <4 x i64> %p1) #1 {
  store <4 x i64> %p1, <4 x i64> addrspace(5)* %p0
  ret void
}

define linkonce_odr <4 x float> @optix.ptx.ld.local.v4.f32( <4 x float> addrspace(5)* %p0 ) #2 {
  %val = load <4 x float>, <4 x float> addrspace(5)* %p0, align 16
  ret <4 x float> %val
}

define linkonce_odr <4 x float> @optix.ptx.ldu.local.v4.f32( <4 x float> addrspace(5)* %p0 ) #2 {
  %val = load <4 x float>, <4 x float> addrspace(5)* %p0, align 16
  ret <4 x float> %val
}

define linkonce_odr <4 x float> @optix.ptx.ld.volatile.local.v4.f32( <4 x float> addrspace(5)* %p0 ) #1 {
  %val = load volatile <4 x float>, <4 x float> addrspace(5)* %p0, align 16
  ret <4 x float> %val
}

define linkonce_odr void @optix.ptx.st.local.v4.f32( <4 x float> addrspace(5)* %p0, <4 x float> %p1) #1 {
  store <4 x float> %p1, <4 x float> addrspace(5)* %p0
  ret void
}

define linkonce_odr <4 x double> @optix.ptx.ld.local.v4.f64( <4 x double> addrspace(5)* %p0 ) #2 {
  %val = load <4 x double>, <4 x double> addrspace(5)* %p0, align 32
  ret <4 x double> %val
}

define linkonce_odr <4 x double> @optix.ptx.ldu.local.v4.f64( <4 x double> addrspace(5)* %p0 ) #2 {
  %val = load <4 x double>, <4 x double> addrspace(5)* %p0, align 32
  ret <4 x double> %val
}

define linkonce_odr <4 x double> @optix.ptx.ld.volatile.local.v4.f64( <4 x double> addrspace(5)* %p0 ) #1 {
  %val = load volatile <4 x double>, <4 x double> addrspace(5)* %p0, align 32
  ret <4 x double> %val
}

define linkonce_odr void @optix.ptx.st.local.v4.f64( <4 x double> addrspace(5)* %p0, <4 x double> %p1) #1 {
  store <4 x double> %p1, <4 x double> addrspace(5)* %p0
  ret void
}

; 840 instructions generated
