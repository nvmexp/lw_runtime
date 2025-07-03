!!SPA1.0
.CONST_MODE  PAGE
.THREAD_TYPE PIXEL
.MAX_REG     15
.MAX_ATTR    15
# parseasm build date Mar 10 2004 15:40:49
# -profile fp50 -po tbat3 -po lat3 -if ps2x -i allprogs-new32//p355-lw40.s -o allprogs-new32//p355-lw50.s
#vendor LWPU
#version parseasm.0.0
#profile fp50
#program fp30entry
#semantic <null atom>
#semantic <null atom>
#var float4 o[COLH2] : $vout.O : O[0] : -1 : 0
#var float4 o[COLH1] : $vout.O : O[0] : -1 : 0
#var float4 o[COLH0] : $vout.O : O[0] : -1 : 0
#var float4 f[COL0] : $vin.F : F[0] : -1 : 0
#var float4 f[TEX1] : $vin.F : F[0] : -1 : 0
#var float4 f[TEX4] : $vin.F : F[0] : -1 : 0
#var float4 f[TEX3] : $vin.F : F[0] : -1 : 0
#var float4 f[TEX2] : $vin.F : F[0] : -1 : 0
#var sampler2D  : texunit 0 : -1 : 0
#var float4 f[TEX0] : $vin.F : F[0] : -1 : 0
#var sampler2D  : texunit 1 : -1 : 0
#tram 0 = f[WPOS].w
#tram 1 = f[COL0].x
#tram 2 = f[TEX0].x
#tram 3 = f[TEX0].y
#tram 4 = f[TEX1].x
#tram 5 = f[TEX1].y
#tram 6 = f[TEX1].z
#tram 7 = f[TEX2].x
#tram 8 = f[TEX2].y
#tram 9 = f[TEX2].z
#tram 10 = f[TEX3].x
#tram 11 = f[TEX3].y
#tram 12 = f[TEX3].z
#tram 13 = f[TEX4].x
#tram 14 = f[TEX4].y
#tram 15 = f[TEX4].z
BB0:
IPA      R0, 0;
RCP      R11, R0;
IPA      R0, 2, R11;
IPA      R1, 3, R11;
MOV32    R4, R0;
MOV32    R5, R1;
TEX      R4, 0, 0, 2D;
TEX      R0, 1, 1, 2D;
IPA      R8, 8, R11;
IPA      R7, 7, R11;
MOV32    <<REG124>>.x, R4;
MOV32    <<REG124>>.y, R5;
MOV32    <<REG124>>.z, R6;
FADD32I  R0, R0, -0.5;
FADD32I  R1, R1, -0.5;
FADD32I  R4, R2, -0.5;
IPA      R2, 9, R11;
FMUL32   R5, R8, R1;
IPA      R8, 11, R11;
IPA      R6, 10, R11;
FMAD     R5, R7, R0, R5;
FMUL32   R8, R8, R1;
IPA      R7, 12, R11;
FMAD     R2, R2, R4, R5;
FMAD     R8, R6, R0, R8;
IPA      R6, 14, R11;
IPA      R5, 13, R11;
FMAD     R7, R7, R4, R8;
FMUL32   R6, R6, R1;
IPA      R1, 15, R11;
FMUL32   R8, R7, R7;
FMAD     R0, R5, R0, R6;
FMAD     R5, R2, R2, R8;
FMAD     R6, R1, R4, R0;
FMAD     R1, R6, R6, R5;
IPA      R0, 4, R11;
RSQ      R8, R1;
IPA      R1, 5, R11;
FMUL32   R5, R2, R8;
FMUL32   R4, R7, R8;
FMUL32   R2, R6, R8;
FMAD     R0, R5, c[0], R0;
MOV32    <<REG122>>.x, R5;
FMAD     R1, R4, c[0], R1;
MOV32    <<REG122>>.y, R4;
MOV32    <<REG124>>.w, R3;
MOV32    <<REG122>>.z, R2;
IPA      R3, 6, R11;
IPA      <<REG122>>.w, 1, R11;
MVI      R4, 0.0;
FMAD     R2, R2, c[0], R3;
MOV32    R3, R4;
END
# 52 instructions, 16 R-regs, 16 interpolants
# 52 inst, (10 mov, 1 mvi, 2 tex, 16 ipa, 2 complex, 21 math)
#    32 64-bit, 17 32-bit, 3 32-bit-const
