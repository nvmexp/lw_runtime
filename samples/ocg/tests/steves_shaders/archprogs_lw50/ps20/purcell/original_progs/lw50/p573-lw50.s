!!SPA1.0
.CONST_MODE  PAGE
.THREAD_TYPE PIXEL
.MAX_REG     15
.MAX_ATTR    25
# parseasm build date Mar 10 2004 15:40:49
# -profile fp50 -po tbat3 -po lat3 -if ps2x -i allprogs-new32//p573-lw40.s -o allprogs-new32//p573-lw50.s
#vendor LWPU
#version parseasm.0.0
#profile fp50
#program fp30entry
#semantic Coit1m35rlpml7.Coit1m35rlpml7
#semantic C144q3o24uukb6.C144q3o24uukb6
#semantic <null atom>
#semantic <null atom>
#semantic <null atom>
#var float4 o[COLH0] : $vout.O : O[0] : -1 : 0
#var float4 Coit1m35rlpml7 :  : c[1] : -1 : 0
#var float4 f[TEX7] : $vin.F : F[0] : -1 : 0
#var float4 f[COL0] : $vin.F : F[0] : -1 : 0
#var float4 f[COL1] : $vin.F : F[0] : -1 : 0
#var float4 C144q3o24uukb6 :  : c[0] : -1 : 0
#var float4 f[TEX0] : $vin.F : F[0] : -1 : 0
#var sampler2D  : texunit 0 : -1 : 0
#var samplerLWBE  : texunit 1 : -1 : 0
#var float4 f[TEX3] : $vin.F : F[0] : -1 : 0
#var float4 f[TEX6] : $vin.F : F[0] : -1 : 0
#var float4 f[TEX5] : $vin.F : F[0] : -1 : 0
#var float4 f[TEX4] : $vin.F : F[0] : -1 : 0
#var float4 f[TEX1] : $vin.F : F[0] : -1 : 0
#var sampler2D  : texunit 3 : -1 : 0
#tram 0 = f[WPOS].w
#tram 1 = f[COL0].x
#tram 2 = f[COL0].y
#tram 3 = f[COL0].z
#tram 4 = f[COL1].x
#tram 5 = f[COL1].y
#tram 6 = f[COL1].z
#tram 7 = f[TEX0].x
#tram 8 = f[TEX0].y
#tram 9 = f[TEX1].x
#tram 10 = f[TEX1].y
#tram 11 = f[TEX3].x
#tram 12 = f[TEX3].y
#tram 13 = f[TEX3].z
#tram 14 = f[TEX4].x
#tram 15 = f[TEX4].y
#tram 16 = f[TEX4].z
#tram 17 = f[TEX5].x
#tram 18 = f[TEX5].y
#tram 19 = f[TEX5].z
#tram 20 = f[TEX6].x
#tram 21 = f[TEX6].y
#tram 22 = f[TEX6].z
#tram 23 = f[TEX7].x
#tram 24 = f[TEX7].y
#tram 25 = f[TEX7].z
BB0:
IPA      R0, 0;
MVI      R7, -1.0;
MVI      R6, -1.0;
RCP      R12, R0;
IPA      R8, 9, R12;
IPA      R9, 10, R12;
IPA      R3, 15, R12;
IPA      R1, 14, R12;
MVI      R5, -1.0;
IPA      R0, 16, R12;
TEX      R8, 3, 3, 2D;
IPA      R4, 18, R12;
IPA      R2, 17, R12;
FMAD     R8, R8, c[0], R7;
FMAD     R9, R9, c[0], R6;
FMAD     R10, R10, c[0], R5;
IPA      R6, 19, R12;
FMUL32   R3, R9, R3;
FMUL32   R4, R9, R4;
IPA      R5, 12, R12;
FMAD     R1, R8, R1, R3;
FMAD     R2, R8, R2, R4;
IPA      R4, 11, R12;
FMAD     R3, R10, R0, R1;
FMAD     R1, R10, R6, R2;
IPA      R0, 21, R12;
IPA      R2, 20, R12;
FMUL32   R6, R1, R5;
FMUL32   R7, R9, R0;
IPA      R0, 22, R12;
FMAD     R6, R3, R4, R6;
FMAD     R13, R8, R2, R7;
IPA      R2, 13, R12;
FMUL32   R7, R1, R1;
FMAD     R0, R10, R0, R13;
FMAD     R7, R3, R3, R7;
FMAD     R6, R0, R2, R6;
FMAD     R7, R0, R0, R7;
FADD32   R6, R6, R6;
FMUL32   R4, R7, R4;
FMUL32   R5, R7, R5;
FMUL32   R2, R7, R2;
FMAD     R4, R6, R3, -R4;
FMAD     R5, R6, R1, -R5;
FMAD     R6, R6, R0, -R2;
IPA      R0, 7, R12;
FMAX     R2, |R4|, |R5|;
IPA      R1, 8, R12;
FMAX     R7, |R6|, R2;
TEX      R0, 0, 0, 2D;
RCP      R7, R7;
FMUL32   R4, R4, R7;
FMUL32   R5, R5, R7;
FMUL32   R6, R6, R7;
TEX      R4, 1, 1, LWBE;
FMUL32   R6, R7, R6;
FMUL32   R5, R7, R5;
FMUL32   R4, R7, R4;
FMUL32   R6, R11, R6;
FMUL32   R5, R11, R5;
FMUL32   R4, R11, R4;
FMUL32   R6, R6, c[2];
FMUL32   R5, R5, c[1];
FMUL32   R4, R4, c[0];
FMUL32I  R11, R6, 16.0;
FMUL32I  R6, R5, 16.0;
FMUL32I  R4, R4, 16.0;
FMUL32I  R5, R10, 0.57735;
FMUL32I  R13, R9, 0.707107;
FMUL32I  R7, R9, -0.707107;
FMAD.SAT R5, R8, c[0], R5;
FMAD     R13, R8, c[0], R13;
FMAD     R7, R8, c[0], R7;
IPA      R9, 6, R12;
FMAD.SAT R8, R10, c[0], R13;
FMAD.SAT R7, R10, c[0], R7;
IPA      R10, 3, R12;
FMUL32   R13, R8, R9;
IPA      R9, 25, R12;
IPA      R14, 5, R12;
FMAD     R13, R5, R10, R13;
IPA      R10, 2, R12;
FMUL32   R14, R8, R14;
FMAD     R9, R7, R9, R13;
IPA      R13, 24, R12;
FMAD     R14, R5, R10, R14;
FMUL32   R9, R9, c[6];
IPA      R10, 4, R12;
FMAD     R13, R7, R13, R14;
FMUL32   R8, R8, R10;
FMUL32   R10, R13, c[5];
FMAD     R2, R2, R9, R11;
FMUL32   R3, R3, c[7];
FMAD     R1, R1, R10, R6;
IPA      R6, 1, R12;
IPA      R9, 23, R12;
FMAD     R5, R5, R6, R8;
FMAD     R5, R7, R9, R5;
FMUL32   R5, R5, c[4];
FMAD     R0, R0, R5, R4;
END
# 100 instructions, 16 R-regs, 26 interpolants
# 100 inst, (0 mov, 3 mvi, 3 tex, 26 ipa, 2 complex, 66 math)
#    66 64-bit, 28 32-bit, 6 32-bit-const
