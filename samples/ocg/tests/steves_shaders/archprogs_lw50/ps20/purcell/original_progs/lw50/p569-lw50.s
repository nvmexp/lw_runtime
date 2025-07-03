!!SPA1.0
.CONST_MODE  PAGE
.THREAD_TYPE PIXEL
.MAX_REG     11
.MAX_ATTR    4
# parseasm build date Mar 10 2004 15:40:49
# -profile fp50 -po tbat3 -po lat3 -if ps2x -i allprogs-new32//p569-lw40.s -o allprogs-new32//p569-lw50.s
#vendor LWPU
#version parseasm.0.0
#profile fp50
#program fp30entry
#semantic <null atom>
#semantic <null atom>
#semantic <null atom>
#var float4 o[COLH0] : $vout.O : O[0] : -1 : 0
#var sampler2D  : texunit 2 : -1 : 0
#var float4 f[TEX1] : $vin.F : F[0] : -1 : 0
#var sampler2D  : texunit 1 : -1 : 0
#var float4 f[TEX0] : $vin.F : F[0] : -1 : 0
#var sampler2D  : texunit 0 : -1 : 0
#tram 0 = f[WPOS].w
#tram 1 = f[TEX0].x
#tram 2 = f[TEX0].y
#tram 3 = f[TEX1].x
#tram 4 = f[TEX1].y
BB0:
IPA      R0, 0;
RCP      R2, R0;
IPA      R0, 1, R2;
IPA      R1, 2, R2;
IPA      R5, 3, R2;
IPA      R6, 4, R2;
TEX      R0, 0, 0, 2D;
MOV32    R4, R5;
MOV32    R8, R5;
MOV32    R5, R6;
MOV32    R9, R6;
TEX      R4, 1, 1, 2D;
TEX      R8, 2, 2, 2D;
FMUL32   R7, R0, R4;
FADD32I  R3, -R11, 1.0;
FMUL32   R8, R1, R5;
FMAD     R0, R0, R4, R7;
FMUL32   R4, R2, R6;
FMAD     R1, R1, R5, R8;
FMAD     R2, R2, R6, R4;
END
# 20 instructions, 12 R-regs, 5 interpolants
# 20 inst, (4 mov, 0 mvi, 3 tex, 5 ipa, 1 complex, 7 math)
#    12 64-bit, 7 32-bit, 1 32-bit-const
