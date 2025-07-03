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
; This file contains IR wrappers that use activemask to emulate older shfl and
; vote instructions. 

; Note that these new wrappers are only supported on ISAs 6.0 or later. Because
; the megakernel only supports up to ISA 4.1, it should not link these wrappers.
;-------------------------------------------------------------------------------

attributes #0 = { nounwind readnone }

declare i32 @optix.ptx.activemask.b32(  )

; shfl.sync is now the 'better' way. Emulate old shfl with the new
; version. Note: this only works for the semantics promised by ray
; tracing (opportunistic colwergence only)
define linkonce_odr { i32, i1 } @optix.ptx.shfl.up.b32( i32 %p0, i32 %p1, i32 %p2 ) nounwind alwaysinline {
    %am = call i32 @optix.ptx.activemask.b32() #0
    %r0 = call { i32, i1 } asm  " shfl.sync.b32.up $0|$1, $2, $3, $4, $5;", "=r,=b,r,r,r,r"(i32 %p0, i32 %p1, i32 %p2, i32 %am) #0
    ret { i32, i1 } %r0
}

define linkonce_odr { i32, i1 } @optix.ptx.shfl.down.b32( i32 %p0, i32 %p1, i32 %p2 ) nounwind alwaysinline {
    %am = call i32 @optix.ptx.activemask.b32() #0
    %r0 = call { i32, i1 } asm  " shfl.sync.b32.down $0|$1, $2, $3, $4, $5;", "=r,=b,r,r,r,r"(i32 %p0, i32 %p1, i32 %p2, i32 %am) #0
    ret { i32, i1 } %r0
}

define linkonce_odr { i32, i1 } @optix.ptx.shfl.bfly.b32( i32 %p0, i32 %p1, i32 %p2 ) nounwind alwaysinline {
    %am = call i32 @optix.ptx.activemask.b32() #0
    %r0 = call { i32, i1 } asm  " shfl.sync.b32.bfly $0|$1, $2, $3, $4, $5;", "=r,=b,r,r,r,r"(i32 %p0, i32 %p1, i32 %p2, i32 %am) #0
    ret { i32, i1 } %r0
}

define linkonce_odr { i32, i1 } @optix.ptx.shfl.idx.b32( i32 %p0, i32 %p1, i32 %p2 ) nounwind alwaysinline {
    %am = call i32 @optix.ptx.activemask.b32() #0
    %r0 = call { i32, i1 } asm  " shfl.sync.b32.idx $0|$1, $2, $3, $4, $5;", "=r,=b,r,r,r,r"(i32 %p0, i32 %p1, i32 %p2, i32 %am) #0
    ret { i32, i1 } %r0
}

; vote.sync is now the 'better' way. Emulate old vote with the new
; version. Note: this only works for the semantics promised by ray
; tracing (opportunistic colwergence only)
define linkonce i1 @optix.ptx.vote.all.pred( i1 %p0 ) nounwind alwaysinline {
    %am = call i32 @optix.ptx.activemask.b32() #0
    %r0 = call i1 asm sideeffect " vote.sync.all.pred $0, $1, $2;", "=b,b,r"(i1 %p0, i32 %am) 
    ret i1 %r0
}
define linkonce i1 @optix.ptx.vote.any.pred( i1 %p0 ) nounwind alwaysinline {
    %am = call i32 @optix.ptx.activemask.b32() #0
    %r0 = call i1 asm sideeffect " vote.sync.any.pred $0, $1, $2;", "=b,b,r"(i1 %p0, i32 %am) 
    ret i1 %r0
}
define linkonce i1 @optix.ptx.vote.uni.pred( i1 %p0 ) nounwind alwaysinline {
    %am = call i32 @optix.ptx.activemask.b32() #0
    %r0 = call i1 asm sideeffect " vote.sync.uni.pred $0, $1, $2;", "=b,b,r"(i1 %p0, i32 %am) 
    ret i1 %r0
}
define linkonce i32 @optix.ptx.vote.ballot.b32( i1 %p0 ) nounwind alwaysinline {
    %am = call i32 @optix.ptx.activemask.b32() #0
    %r0 = call i32 asm sideeffect " vote.sync.ballot.b32 $0, $1, $2;", "=r,b,r"(i1 %p0, i32 %am) 
    ret i32 %r0
}


