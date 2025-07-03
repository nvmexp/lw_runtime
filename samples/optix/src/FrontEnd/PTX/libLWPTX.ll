;; Copyright (c) 2019, LWPU CORPORATION.
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
; This file contains wrappers to colwert OptiX PTX intrinsics to inline PTX
; instructions. The LWVM intrinsics these wrappers would need are only supported
; in Direct2IR, so we implement them as inline PTX and only link them in if
; we're using the that backend.
;
; Note that this library is linked into the megakernel, which only supports up
; to PTX ISA 4.1, so any instructions that require newer ISAs should be placed
; in separate libraries.
;-------------------------------------------------------------------------------

attributes #0 = { nounwind readnone alwaysinline } ; Non-volatile functions
attributes #1 = { nounwind alwaysinline }          ; "volatile" functions, which may return different results on different ilwocations
attributes #2 = { nounwind readnone }
attributes #4 = { nounwind }                       ; for texture intrinsics - they MUST not have readnone

define linkonce_odr void @optix.ptx.exit() nounwind alwaysinline noreturn {
    call void asm sideeffect "exit;", ""() 
    ret void
}

define linkonce_odr void @optix.ptx.trap() nounwind alwaysinline noreturn {
    call void asm sideeffect "trap;", ""() 
    ret void
}

declare void @llvm.trap() noreturn nounwind
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
    call void asm sideeffect "pmevent 0;", ""()
    ret void
  pm1:
    call void asm sideeffect "pmevent 1;", ""()
    ret void
  pm2:
    call void asm sideeffect "pmevent 2;", ""()
    ret void
  pm3:
    call void asm sideeffect "pmevent 3;", ""()
    ret void
  pm4:
    call void asm sideeffect "pmevent 4;", ""()
    ret void
  pm5:
    call void asm sideeffect "pmevent 5;", ""()
    ret void
  pm6:
    call void asm sideeffect "pmevent 6;", ""()
    ret void
  pm7:
    call void asm sideeffect "pmevent 7;", ""()
    ret void
  pm8:
    call void asm sideeffect "pmevent 8;", ""()
    ret void
  pm9:
    call void asm sideeffect "pmevent 9;", ""()
    ret void
  pm10:
    call void asm sideeffect "pmevent 10;", ""()
    ret void
  pm11:
    call void asm sideeffect "pmevent 11;", ""()
    ret void
  pm12:
    call void asm sideeffect "pmevent 12;", ""()
    ret void
  pm13:
    call void asm sideeffect "pmevent 13;", ""()
    ret void
  pm14:
    call void asm sideeffect "pmevent 14;", ""()
    ret void
  pm15:
    call void asm sideeffect "pmevent 15;", ""()
    ret void
}

define linkonce_odr void @optix.ptx.pmevent.mask(i32) #1 {
    call void asm sideeffect "pmevent.mask $0;", "r"(i32 %0) 
    ret void
}

define linkonce_odr i32 @optix.ptx.vshl.clamp.u32.u32.u32.b0(i32 %val, i32 %shift) #0 {
 %r = call i32 asm "vshl.u32.u32.u32.clamp $0, $1.b0, $2;", "=r,r,r"(i32 %val, i32 %shift) #0
 ret i32 %r
}

define linkonce_odr i32 @optix.ptx.vshl.clamp.u32.u32.u32.b1(i32 %val, i32 %shift) #0 {
 %r = call i32 asm "vshl.u32.u32.u32.clamp $0, $1.b1, $2;", "=r,r,r"(i32 %val, i32 %shift) #0
 ret i32 %r
}

define linkonce_odr i32 @optix.ptx.vshl.clamp.u32.u32.u32.b2(i32 %val, i32 %shift) #0 {
 %r = call i32 asm "vshl.u32.u32.u32.clamp $0, $1.b2, $2;", "=r,r,r"(i32 %val, i32 %shift) #0
 ret i32 %r
}

define linkonce_odr i32 @optix.ptx.vshl.clamp.u32.u32.u32.b3(i32 %val, i32 %shift) #0 {
 %r = call i32 asm "vshl.u32.u32.u32.clamp $0, $1.b3, $2;", "=r,r,r"(i32 %val, i32 %shift) #0
 ret i32 %r
}

define linkonce_odr i32 @optix.ptx.vshl.wrap.add.u32.u32.u32.b0.b0(i32 %val, i32 %shift, i32 %addend) #0 {
 %r = call i32 asm "vshl.u32.u32.u32.wrap.add $0, $1.b0, $2.b0, $3;", "=r,r,r,r"(i32 %val, i32 %shift, i32 %addend) #0
 ret i32 %r
}

define linkonce_odr i32 @optix.ptx.vshl.wrap.add.u32.u32.u32.b1.b1(i32 %val, i32 %shift, i32 %addend) #0 {
 %r = call i32 asm "vshl.u32.u32.u32.wrap.add $0, $1.b1, $2.b1, $3;", "=r,r,r,r"(i32 %val, i32 %shift, i32 %addend) #0
 ret i32 %r
}

define linkonce_odr i32 @optix.ptx.vshl.wrap.add.u32.u32.u32.b2.b2(i32 %val, i32 %shift, i32 %addend) #0 {
 %r = call i32 asm "vshl.u32.u32.u32.wrap.add $0, $1.b2, $2.b2, $3;", "=r,r,r,r"(i32 %val, i32 %shift, i32 %addend) #0
 ret i32 %r
}

define linkonce_odr i32 @optix.ptx.vshl.wrap.add.u32.u32.u32.b3.b3(i32 %val, i32 %shift, i32 %addend) #0 {
 %r = call i32 asm "vshl.u32.u32.u32.wrap.add $0, $1.b3, $2.b3, $3;", "=r,r,r,r"(i32 %val, i32 %shift, i32 %addend) #0
 ret i32 %r
}

;
; Shared memory intrinsics with i32 shared address
;

define linkonce_odr void @optix.ptx.st.shared.v2.s32(i32 %address, i32 %x, i32 %y) #1 {
  call void asm "st.shared.v2.s32 [$0], {$1, $2};", "r,r,r"(i32 %address, i32 %x, i32 %y) #1
  ret void
}

define linkonce_odr {i32, i32} @optix.ptx.ld.shared.v2.s32(i32 %address) #0 {
  %r = call {i32, i32} asm "ld.shared.v2.s32 {$0, $1}, [$2];", "=r,=r,r"(i32 %address) #0
  ret {i32, i32} %r
}

;
; Wrap LWVM texture intrinsics
; Note some are not implemented as intrinsincs in LWVM yet
;

declare i32 @llvm.lwvm.txq.width( i64 ) #4
define linkonce_odr i32 @optix.lwvm.txq_width(i64 %texref) #0 {
  %w = call i32 @llvm.lwvm.txq.width(i64 %texref) #4
  ret i32 %w
}

declare i32 @llvm.lwvm.txq.height( i64 )
define linkonce_odr i32 @optix.lwvm.txq_height(i64 %texref) #0 {
  %h = call i32 @llvm.lwvm.txq.height(i64 %texref) #4
  ret i32 %h
}

declare i32 @llvm.lwvm.txq.depth( i64 )
define linkonce_odr i32 @optix.lwvm.txq_depth(i64 %texref) #0 {
  %d = call i32 @llvm.lwvm.txq.depth(i64 %texref) #4
  ret i32 %d
}

;
; Special register intrinsics
;

define linkonce_odr i32 @optix.lwvm.read.ptx.sreg.clock() #1 {
    %1 = call i32 asm sideeffect "mov.u32 $0, %clock;", "=r"()
    ret i32 %1
}

define linkonce_odr i32 @optix.lwvm.read.ptx.sreg.clock_hi() #1 {
    %1 = call i32 asm sideeffect "mov.u32 $0, %clock_hi;", "=r"()
    ret i32 %1
}

define linkonce_odr i64 @optix.lwvm.read.ptx.sreg.clock64() #1 {
    %1 = call i64 asm sideeffect "mov.u64 $0, %clock64;", "=l"()
    ret i64 %1
}

define linkonce_odr i32 @optix.lwvm.read.ptx.sreg.laneid() #1 {
    %1 = call i32 asm sideeffect "mov.u32 $0, %laneid;", "=r"()
    ret i32 %1
}

; warpid is volatile
define linkonce_odr i32 @optix.lwvm.read.ptx.sreg.warpid() #1 {
    %1 = call i32 asm sideeffect "mov.u32 $0, %warpid;", "=r"()
    ret i32 %1
}

define linkonce_odr i32 @optix.lwvm.read.ptx.sreg.nwarpid() #1 {
    %1 = call i32 asm sideeffect "mov.u32 $0, %nwarpid;", "=r"()
    ret i32 %1
}

define linkonce_odr i32 @optix.lwvm.read.ptx.sreg.tid.x() #0 {
    %1 = call i32 asm "mov.u32 $0, %tid.x;", "=r"()
    ret i32 %1
}

define linkonce_odr i32 @optix.lwvm.read.ptx.sreg.tid.y() #0 {
    %1 = call i32 asm "mov.u32 $0, %tid.y;", "=r"()
    ret i32 %1
}

define linkonce_odr i32 @optix.lwvm.read.ptx.sreg.tid.z() #0 {
    %1 = call i32 asm "mov.u32 $0, %tid.z;", "=r"()
    ret i32 %1
}

define linkonce_odr <4 x i32> @optix.lwvm.read.ptx.sreg.tid() #0 {
    %1 = call i32 asm "mov.u32 $0, %tid.x;", "=r"()
    %2 = call i32 asm "mov.u32 $0, %tid.y;", "=r"()
    %3 = call i32 asm "mov.u32 $0, %tid.z;", "=r"()
    %4 = insertelement <4 x i32> undef, i32 %1, i32 0
    %5 = insertelement <4 x i32> %4, i32 %2, i32 1
    %6 = insertelement <4 x i32> %5, i32 %3, i32 2
    ret <4 x i32> %6
}

define linkonce_odr i32 @optix.lwvm.read.ptx.sreg.ntid.x() #0 {
    %1 = call i32 asm "mov.u32 $0, %ntid.x;", "=r"()
    ret i32 %1 
}

define linkonce_odr i32 @optix.lwvm.read.ptx.sreg.ntid.y() #0 {
    %1 = call i32 asm "mov.u32 $0, %ntid.y;", "=r"()
    ret i32 %1
}

define linkonce_odr i32 @optix.lwvm.read.ptx.sreg.ntid.z() #0 {
    %1 = call i32 asm "mov.u32 $0, %ntid.z;", "=r"()
    ret i32 %1
}

define linkonce_odr <4 x i32> @optix.lwvm.read.ptx.sreg.ntid() #0 {
    %1 = call i32 asm "mov.u32 $0, %ntid.x;", "=r"()
    %2 = call i32 asm "mov.u32 $0, %ntid.y;", "=r"()
    %3 = call i32 asm "mov.u32 $0, %ntid.z;", "=r"()
    %4 = insertelement <4 x i32> undef, i32 %1, i32 0
    %5 = insertelement <4 x i32> %4, i32 %2, i32 1
    %6 = insertelement <4 x i32> %5, i32 %3, i32 2
    ret <4 x i32> %6
}

define linkonce_odr i32 @optix.lwvm.read.ptx.sreg.ctaid.x() #0 {
    %1 = call i32 asm "mov.u32 $0, %ctaid.x;", "=r"()
    ret i32 %1
}

define linkonce_odr i32 @optix.lwvm.read.ptx.sreg.ctaid.y() #0 {
    %1 = call i32 asm "mov.u32 $0, %ctaid.y;", "=r"()
    ret i32 %1
}

define linkonce_odr i32 @optix.lwvm.read.ptx.sreg.ctaid.z() #0 {
    %1 = call i32 asm "mov.u32 $0, %ctaid.z;", "=r"()
    ret i32 %1
}

define linkonce_odr <4 x i32> @optix.lwvm.read.ptx.sreg.ctaid() #0 {
    %1 = call i32 asm "mov.u32 $0, %ctaid.x;", "=r"()
    %2 = call i32 asm "mov.u32 $0, %ctaid.y;", "=r"()
    %3 = call i32 asm "mov.u32 $0, %ctaid.z;", "=r"()
    %4 = insertelement <4 x i32> undef, i32 %1, i32 0
    %5 = insertelement <4 x i32> %4, i32 %2, i32 1
    %6 = insertelement <4 x i32> %5, i32 %3, i32 2
    ret <4 x i32> %6
}

define linkonce_odr i32 @optix.lwvm.read.ptx.sreg.nctaid.x() #0 {
    %1 = call i32 asm "mov.u32 $0, %nctaid.x;", "=r"()
    ret i32 %1
}

define linkonce_odr i32 @optix.lwvm.read.ptx.sreg.nctaid.y() #0 {
    %1 = call i32 asm "mov.u32 $0, %nctaid.y;", "=r"()
    ret i32 %1
}

define linkonce_odr i32 @optix.lwvm.read.ptx.sreg.nctaid.z() #0 {
    %1 = call i32 asm "mov.u32 $0, %nctaid.z;", "=r"()
    ret i32 %1
}

define linkonce_odr <4 x i32> @optix.lwvm.read.ptx.sreg.nctaid() #0 {
    %1 = call i32 asm "mov.u32 $0, %nctaid.x;", "=r"()
    %2 = call i32 asm "mov.u32 $0, %nctaid.y;", "=r"()
    %3 = call i32 asm "mov.u32 $0, %nctaid.z;", "=r"()
    %4 = insertelement <4 x i32> undef, i32 %1, i32 0
    %5 = insertelement <4 x i32> %4, i32 %2, i32 1
    %6 = insertelement <4 x i32> %5, i32 %3, i32 2
    ret <4 x i32> %6
}

; smid is volatile
define linkonce_odr i32 @optix.lwvm.read.ptx.sreg.smid() #1 {
    %1 = call i32 asm sideeffect "mov.u32 $0, %smid;", "=r"()
    ret i32 %1
}

define linkonce_odr i32 @optix.lwvm.read.ptx.sreg.nsmid() #1 {
    %1 = call i32 asm sideeffect "mov.u32 $0, %nsmid;", "=r"()
    ret i32 %1
}

; gridid is volatile
define linkonce_odr i64 @optix.lwvm.read.ptx.sreg.gridid() #1 {
    %1 = call i64 asm sideeffect "mov.u64 $0, %gridid;", "=l"()
    ret i64 %1
}

define linkonce_odr i32 @optix.lwvm.read.ptx.sreg.lanemask_eq() #1 {
    %1 = call i32 asm sideeffect "mov.u32 $0, %lanemask_eq;", "=r"()
    ret i32 %1
}

define linkonce_odr i32 @optix.lwvm.read.ptx.sreg.lanemask_lt() #1 {
    %1 = call i32 asm sideeffect "mov.u32 $0, %lanemask_lt;", "=r"()
    ret i32 %1
}

define linkonce_odr i32 @optix.lwvm.read.ptx.sreg.lanemask_le() #1 {
    %1 = call i32 asm sideeffect "mov.u32 $0, %lanemask_le;", "=r"()
    ret i32 %1
}

define linkonce_odr i32 @optix.lwvm.read.ptx.sreg.lanemask_gt() #1 {
    %1 = call i32 asm sideeffect "mov.u32 $0, %lanemask_gt;", "=r"()
    ret i32 %1
}

define linkonce_odr i32 @optix.lwvm.read.ptx.sreg.lanemask_ge() #1 {
    %1 = call i32 asm sideeffect "mov.u32 $0, %lanemask_ge;", "=r"()
    ret i32 %1
}

define linkonce_odr i32 @optix.lwvm.read.ptx.sreg.globaltimer_lo() #1 {
    %1 = call i32 asm sideeffect "mov.u32 $0, %globaltimer_lo;", "=r"()
    ret i32 %1
}

define linkonce_odr i32 @optix.lwvm.read.ptx.sreg.globaltimer_hi() #1 {
    %1 = call i32 asm sideeffect "mov.u32 $0, %globaltimer_hi;", "=r"()
    ret i32 %1
}

define linkonce_odr i64 @optix.lwvm.read.ptx.sreg.globaltimer() #1 {
    %1 = call i64 asm sideeffect "mov.u64 $0, %globaltimer;", "=l"()
    ret i64 %1
}

;
; Texture intrinsics.
;

declare {float, float, float, float } @llvm.lwvm.tex.unified.1d.v4f32.f32( i64, float ) #4
define linkonce_odr { float, float, float, float } @optix.lwvm.tex_1d(i64 %texref, float %x) #0 {
  %ffff = call { float, float, float, float } @llvm.lwvm.tex.unified.1d.v4f32.f32 (i64 %texref, float %x ) #4
  ret { float, float, float, float } %ffff
}

declare { float, float, float, float } @llvm.lwvm.tex.unified.2d.v4f32.f32( i64, float, float ) #4
define linkonce_odr { float, float, float, float } @optix.lwvm.tex_2d(i64 %texref, float %x, float %y) #0 {
  %ffff = call { float, float, float, float } @llvm.lwvm.tex.unified.2d.v4f32.f32( i64 %texref, float %x, float %y ) #4
  ret { float, float, float, float } %ffff
}

declare {float, float, float, float } @llvm.lwvm.tex.unified.3d.v4f32.f32( i64, float, float, float ) #4
define linkonce_odr { float, float, float, float } @optix.lwvm.tex_3d(i64 %texref, float %x, float %y, float %z) #0 {
  %ffff = call { float, float, float, float } @llvm.lwvm.tex.unified.3d.v4f32.f32( i64 %texref, float %x, float %y, float %z) #4
  ret { float, float, float, float } %ffff
}

;declare { float, float, float, float } @llvm.lwvm.tex.a2dv4f32(i64 %texref, i32, float, float) #4
define linkonce_odr { float, float, float, float } @optix.lwvm.tex_a2d(i64 %texref, i32 %a, float %x, float %y) #0 {
  %ffff = call { float, float, float, float } asm "tex.a2d.v4.f32.f32 { $0,$1,$2,$3 }, [$4, { $5,$6,$7,$8 } ];", "=f,=f,=f,=f,l,r,f,f,f"(i64 %texref, i32 %a, float %x, float %y, float 0.0)
  ret { float, float, float, float } %ffff
}

;declare { float, float, float, float } @llvm.lwvm.tex.lwbev4f32(i64 %texref, float, float, float) #4
define linkonce_odr { float, float, float, float } @optix.lwvm.tex_lwbe(i64 %texref, float %x, float %y, float %z) #0 {
  %ffff = call { float, float, float, float } asm "tex.lwbe.v4.f32.f32 { $0,$1,$2,$3 }, [$4, { $5,$6,$7,$8 } ];", "=f,=f,=f,=f,l,f,f,f,f"(i64 %texref, float %x, float %y, float %z, float 0.0)
  ret { float, float, float, float } %ffff
}

declare {float, float, float, float } @llvm.lwvm.tex.unified.1d.v4f32.s32( i64, i32 ) #4
define linkonce_odr { float, float, float, float } @optix.lwvm.texfetch_1d(i64 %texref, i32 %x) #0 {
  %ffff = call { float, float, float, float } @llvm.lwvm.tex.unified.1d.v4f32.s32( i64 %texref, i32 %x ) #4
  ret { float, float, float, float } %ffff
}

declare {float, float, float, float } @llvm.lwvm.tex.unified.2d.v4f32.s32( i64, i32, i32 ) #4
define linkonce_odr { float, float, float, float } @optix.lwvm.texfetch_2d(i64 %texref, i32 %x, i32 %y) #0 {
  %ffff = call { float, float, float, float } @llvm.lwvm.tex.unified.2d.v4f32.s32( i64 %texref, i32 %x, i32 %y ) #4
  ret { float, float, float, float } %ffff
}

declare {float, float, float, float } @llvm.lwvm.tex.unified.3d.v4f32.s32( i64, i32, i32, i32 ) #4
define linkonce_odr { float, float, float, float } @optix.lwvm.texfetch_3d(i64 %texref, i32 %x, i32 %y, i32 %z) #0 {
  %ffff = call { float, float, float, float } @llvm.lwvm.tex.unified.3d.v4f32.s32( i64 %texref, i32 %x, i32 %y, i32 %z ) #4
  ret { float, float, float, float } %ffff
}

;declare { float, float, float, float } @llvm.lwvm.tex.a1dv4f32.i32(i64 %texref, i32, i32) #4
define linkonce_odr { float, float, float, float } @optix.lwvm.texfetch_a1d(i64 %texref, i32 %a, i32 %x) #0 {
  %ffff = call { float, float, float, float } asm "tex.a1d.v4.f32.s32 { $0,$1,$2,$3 }, [$4, { $5, $6 } ];", "=f,=f,=f,=f,l,r,r"(i64 %texref, i32 %a, i32 %x)
  ret { float, float, float, float } %ffff
}

;declare { float, float, float, float } @llvm.lwvm.tex.a2dv4f32.i32(i64 %texref, i32, i32, i32) #4
define linkonce_odr { float, float, float, float } @optix.lwvm.texfetch_a2d(i64 %texref, i32 %a, i32 %x, i32 %y) #0 {
  %ffff = call { float, float, float, float } asm "tex.a2d.v4.f32.s32 { $0,$1,$2,$3 }, [$4, { $5,$6,$7,$8 } ];", "=f,=f,=f,=f,l,r,r,r,r"(i64 %texref, i32 %a, i32 %x, i32 %y, i32 0)
  ret { float, float, float, float } %ffff
}


; Not yet implemented
declare { float, float, float, float } @llvm.lwvm.texfetch_2dms(i64 %texref, i32, i32, i32) #4
declare { float, float, float, float } @llvm.lwvm.texfetch_a2dms(i64 %texref, i32, i32, i32, i32) #4

declare {float, float, float, float } @llvm.lwvm.tex.unified.2d.level.v4f32.f32( i64, float, float, float ) #4
define linkonce_odr { float, float, float, float } @optix.lwvm.texlevel_2d(i64 %texref, float %x, float %y, float %lev) #0 {
  %ffff = call { float, float, float, float } @llvm.lwvm.tex.unified.2d.level.v4f32.f32(i64 %texref, float %x, float %y, float %lev) #4
  ret { float, float, float, float } %ffff
}

;declare { float, float, float, float } @llvm.lwvm.texlevel.a2dv4f32(i64 %texref, i32, float, float, float) #4
define linkonce_odr { float, float, float, float } @optix.lwvm.texlevel_a2d(i64 %texref, i32 %a, float %x, float %y, float %lev) #0 {
  %ffff = call { float, float, float, float } asm "tex.level.a2d.v4.f32.f32 { $0,$1,$2,$3 }, [$4, { $5,$6,$7,$8 } ], $9;", "=f,=f,=f,=f,l,r,f,f,f,f"(i64 %texref, i32 %a, float %x, float %y, float 0.0, float %lev)
  ret { float, float, float, float } %ffff
}

;declare { float, float, float, float } @llvm.lwvm.texlevel.lwbev4f32(i64 %texref, float, float, float, float) #4
define linkonce_odr { float, float, float, float } @optix.lwvm.texlevel_lwbe(i64 %texref, float %x, float %y, float %z, float %lev) #0 {
  %ffff = call { float, float, float, float } asm "tex.level.lwbe.v4.f32.f32 { $0,$1,$2,$3 }, [$4, { $5,$6,$7,$8 } ], $9;", "=f,=f,=f,=f,l,f,f,f,f,f"(i64 %texref, float %x, float %y, float %z, float 0.0, float %lev)
  ret { float, float, float, float } %ffff
}

declare {float, float, float, float } @llvm.lwvm.tex.unified.2d.grad.v4f32.f32( i64, float, float, float, float, float, float) #4
define linkonce_odr { float, float, float, float } @optix.lwvm.texgrad_2d(i64 %texref, float %x, float %y, float %dpdx_x, float %dpdx_y, float %dpdy_x, float %dpdy_y) #0 {
  %ffff = call { float, float, float, float } @llvm.lwvm.tex.unified.2d.grad.v4f32.f32(i64 %texref, float %x, float %y, float %dpdx_x, float %dpdx_y, float %dpdy_x, float %dpdy_y) #4
  ret { float, float, float, float } %ffff
}

define linkonce_odr { i32, i32, i32, i32, i1 } @optix.lwvm.texgrad_footprint_2d(i32 %granularity, i64 %texref, float %x, float %y, float %dpdx_x, float %dpdx_y, float %dpdy_x, float %dpdy_y) #0 {
  %iiiib = call { i32, i32, i32, i32, i1 } asm "tex.grad.footprint.2d.v4.b32.f32 { $0,$1,$2,$3 }|$4, [$5, { $6,$7 } ], { $8, $9 }, { $10, $11 }, $12;", "=f,=f,=f,=f,=b,l,f,f,f,f,f,f,r"(i64 %texref, float %x, float %y, float %dpdx_x, float %dpdx_y, float %dpdy_x, float %dpdy_y, i32 %granularity)
  ret { i32, i32, i32, i32, i1 } %iiiib
}

define linkonce_odr { i32, i32, i32, i32, i1 } @optix.lwvm.texgrad_footprint_coarse_2d(i32 %granularity, i64 %texref, float %x, float %y, float %dpdx_x, float %dpdx_y, float %dpdy_x, float %dpdy_y) #0 {
  %iiiib = call { i32, i32, i32, i32, i1 } asm "tex.grad.footprint.coarse.2d.v4.b32.f32 { $0,$1,$2,$3 }|$4, [$5, { $6,$7 } ], { $8, $9 }, { $10, $11 }, $12;", "=f,=f,=f,=f,=b,l,f,f,f,f,f,f,r"(i64 %texref, float %x, float %y, float %dpdx_x, float %dpdx_y, float %dpdy_x, float %dpdy_y, i32 %granularity)
  ret { i32, i32, i32, i32, i1 } %iiiib
}

;declare {float, float, float, float } @llvm.lwvm.texgrad.a2dv4f32.f32( i64, i32, float, float, float, float) #4
define linkonce_odr  { float, float, float, float } @optix.lwvm.texgrad_a2d(i64 %texref, i32 %a, float %x, float %y, float %dpdx_x, float %dpdx_y, float %dpdy_x, float %dpdy_y) #0 {
  %ffff = call { float, float, float, float } asm "tex.grad.a2d.v4.f32.f32 { $0,$1,$2,$3 }, [$4, { $5,$6,$7,$8 } ], {$9,$10}, {$11,$12};", "=f,=f,=f,=f,l,r,f,f,f,f,f,f,f"(i64 %texref, i32 %a, float %x, float %y, float 0.0, float %dpdx_x, float %dpdx_y, float %dpdy_x, float %dpdy_y)
  ret { float, float, float, float } %ffff
}

;declare {float, float, float, float } @llvm.lwvm.texgrad.lwbev4f32.f32( i64, float, float, float, float, float, float, float) #4
define linkonce_odr  { float, float, float, float } @optix.lwvm.texgrad_lwbe(i64 %texref, float %x, float %y, float %z, float %dpdx_x, float %dpdx_y, float %dpdy_x, float %dpdy_y) #0 {
  %ffff = call { float, float, float, float } asm "tex.grad.lwbe.v4.f32.f32 { $0,$1,$2,$3 }, [$4, { $5,$6,$7,$8 } ], {$9,$10}, {$11,$12};", "=f,=f,=f,=f,l,f,f,f,f,f,f,f,f"(i64 %texref, float %x, float %y, float %z, float 0.0, float %dpdx_x, float %dpdx_y, float %dpdy_x, float %dpdy_y)
  ret { float, float, float, float } %ffff
}

;declare {float, float, float, float } @llvm.lwvm.texgrad.alwbev4f32.f32( i64, i32, float, float, float, float, float, float, float) #4
define linkonce_odr  { float, float, float, float } @optix.lwvm.texgrad_alwbe(i64 %texref, i32 %a, float %x, float %y, float %z, float %dpdx_x, float %dpdx_y, float %dpdy_x, float %dpdy_y) #0 {
  %ffff = call { float, float, float, float } asm "tex.grad.alwbe.v4.f32.f32 { $0,$1,$2,$3 }, [$4, { $5,$6,$7,$8 } ], {$9,$10}, {$11,$12};", "=f,=f,=f,=f,l,r,f,f,f,f,f,f,f"(i64 %texref, i32 %a, float %x, float %y, float %z, float %dpdx_x, float %dpdx_y, float %dpdy_x, float %dpdy_y)
  ret { float, float, float, float } %ffff
}
