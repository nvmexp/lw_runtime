!!SPA1.0
.CONST_MODE  PAGE
.THREAD_TYPE PIXEL
.MAX_REG     15
.MAX_ATTR    25
# parseasm build date Feb  3 2004 15:17:32
# -profile fp50 -po tbat3 -po lat3 -if ps2x -i progs/641-lw40.s -o progs/641-lw50.s
#vendor LWPU
#version parseasm.0.0
#profile lwinst
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
FMUL     R3, R9, R3;
FMUL     R4, R9, R4;
IPA      R5, 12, R12;
FMAD     R1, R8, R1, R3;
FMAD     R2, R8, R2, R4;
IPA      R4, 11, R12;
FMAD     R3, R10, R0, R1;
FMAD     R1, R10, R6, R2;
IPA      R0, 21, R12;
IPA      R2, 20, R12;
FMUL     R6, R1, R5;
FMUL     R7, R9, R0;
IPA      R0, 22, R12;
FMAD     R6, R3, R4, R6;
FMAD     R13, R8, R2, R7;
IPA      R2, 13, R12;
FMUL     R7, R1, R1;
FMAD     R0, R10, R0, R13;
FMAD     R7, R3, R3, R7;
FMAD     R6, R0, R2, R6;
FMAD     R7, R0, R0, R7;
FADD     R6, R6, R6;
FMUL     R4, R7, R4;
FMUL     R5, R7, R5;
FMUL     R2, R7, R2;
FMAD     R4, R6, R3, -R4;
FMAD     R5, R6, R1, -R5;
FMAD     R6, R6, R0, -R2;
IPA      R0, 7, R12;
FMAX     R2, |R4|, |R5|;
IPA      R1, 8, R12;
FMAX     R7, |R6|, R2;
TEX      R0, 0, 0, 2D;
RCP      R7, R7;
FMUL     R4, R4, R7;
FMUL     R5, R5, R7;
FMUL     R6, R6, R7;
TEX      R4, 1, 1, LWBE;
FMUL     R6, R7, R6;
FMUL     R5, R7, R5;
FMUL     R4, R7, R4;
FMUL     R6, R11, R6;
FMUL     R5, R11, R5;
FMUL     R4, R11, R4;
FMUL     R6, R6, c[2];
FMUL     R5, R5, c[1];
FMUL     R4, R4, c[0];
FMUL     R6, R6, 16.0;
FMUL     R5, R5, 16.0;
FMUL     R4, R4, 16.0;
FMUL     R7, R10, 0.57735;
FMUL     R11, R9, 0.707107;
FMUL     R9, R9, -0.707107;
FMAD.SAT R7, R8, c[0], R7;
FMAD     R13, R8, c[0], R11;
FMAD     R9, R8, c[0], R9;
IPA      R11, 6, R12;
FMAD.SAT R8, R10, c[0], R13;
FMAD.SAT R9, R10, c[0], R9;
IPA      R10, 3, R12;
FMUL     R14, R8, R11;
IPA      R13, 25, R12;
IPA      R11, 5, R12;
FMAD     R14, R7, R10, R14;
IPA      R10, 2, R12;
FMUL     R11, R8, R11;
FMAD     R14, R9, R13, R14;
IPA      R13, 24, R12;
FMAD     R10, R7, R10, R11;
FMUL     R14, R14, c[6];
IPA      R11, 4, R12;
FMAD     R10, R9, R13, R10;
FMUL     R3, R8, R11;
FMUL     R10, R10, c[5];
IPA      R8, 1, R12;
IPA      R11, 23, R12;
FMAD     R1, R1, R10, R5;
FMAD     R3, R7, R8, R3;
FMAD     R5, R2, R14, R6;
FMUL     R6, R1, 0.59;
FMAD     R3, R9, R11, R3;
MOV      R2, R5;
FMUL     R3, R3, c[4];
FMAD     R0, R0, R3, R4;
FMAD     R3, R0, c[0], R6;
FMAD     R3, R5, c[0], R3;
FMUL     R3, R3, 0.0625;
END
# 104 instructions, 16 R-regs, 26 interpolants
# 104 inst, (1 mov, 3 mvi, 3 tex, 26 ipa, 2 complex, 69 math)
#    65 64-bit, 28 32-bit, 11 32-bit-const
