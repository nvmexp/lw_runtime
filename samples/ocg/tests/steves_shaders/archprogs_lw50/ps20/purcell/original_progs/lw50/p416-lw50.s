!!SPA1.0
.CONST_MODE  PAGE
.THREAD_TYPE PIXEL
.MAX_REG     11
.MAX_ATTR    10
# parseasm build date Mar 10 2004 15:40:49
# -profile fp50 -po tbat3 -po lat3 -if ps2x -i allprogs-new32//p416-lw40.s -o allprogs-new32//p416-lw50.s
#vendor LWPU
#version parseasm.0.0
#profile fp50
#program fp30entry
#semantic C144q3o24uukb6.C144q3o24uukb6
#semantic <null atom>
#semantic <null atom>
#var float4 o[COLH0] : $vout.O : O[0] : -1 : 0
#var float4 C144q3o24uukb6 :  : c[0] : -1 : 0
#var sampler2D  : texunit 2 : -1 : 0
#var float4 f[TEX2] : $vin.F : F[0] : -1 : 0
#var float4 f[TEX1] : $vin.F : F[0] : -1 : 0
#var float4 f[TEX0] : $vin.F : F[0] : -1 : 0
#var sampler2D  : texunit 0 : -1 : 0
#tram 0 = f[WPOS].w
#tram 1 = f[TEX0].x
#tram 2 = f[TEX0].y
#tram 3 = f[TEX1].x
#tram 4 = f[TEX1].y
#tram 5 = f[TEX1].z
#tram 6 = f[TEX1].w
#tram 7 = f[TEX2].x
#tram 8 = f[TEX2].y
#tram 9 = f[TEX2].z
#tram 10 = f[TEX2].w
BB0:
IPA      R0, 0;
RCP      R5, R0;
IPA      R0, 1, R5;
IPA      R1, 2, R5;
IPA      R2, 6, R5;
IPA      R7, 4, R5;
IPA      R6, 3, R5;
RCP      R4, R2;
IPA      R8, 5, R5;
TEX      R0, 0, 0, 2D;
FMUL32   R7, R4, R7;
FMUL32   R6, R4, R6;
FMUL32   R4, R4, R8;
IPA      R3, 10, R5;
FMUL32   R8, R7, R1;
IPA      R7, 8, R5;
RCP      R3, R3;
FMAD     R8, R6, R0, R8;
IPA      R6, 7, R5;
FMUL32   R7, R3, R7;
FMAD     R4, R4, R2, R8;
FMUL32   R6, R3, R6;
FMUL32   R1, R7, R1;
IPA      R5, 9, R5;
FMAD     R0, R6, R0, R1;
FMUL32   R3, R3, R5;
MVI      R1, 1.0;
FMAD     R5, R3, R2, R0;
MOV32    R3, R1;
TEX      R4, 2, 2, 2D;
FMUL32   R0, R4, c[0];
FMUL32   R1, R5, c[1];
FMUL32   R2, R6, c[2];
END
# 33 instructions, 12 R-regs, 11 interpolants
# 33 inst, (1 mov, 1 mvi, 2 tex, 11 ipa, 3 complex, 15 math)
#    21 64-bit, 12 32-bit, 0 32-bit-const
