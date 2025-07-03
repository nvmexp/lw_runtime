!!SPA1.0
.CONST_MODE  PAGE
.THREAD_TYPE PIXEL
.MAX_REG     19
.MAX_ATTR    25
# parseasm build date Feb  3 2004 15:17:32
# -profile fp50 -po tbat3 -po lat3 -if ps2x -i progs/547-lw40.s -o progs/547-lw50.s
#vendor LWPU
#version parseasm.0.0
#profile lwinst
#program fp30entry
#semantic C144q3o24uukb6.C144q3o24uukb6
#semantic Cmur1hd5g8ngo3.Cmur1hd5g8ngo3
#semantic <null atom>
#semantic <null atom>
#semantic <null atom>
#semantic <null atom>
#semantic <null atom>
#semantic <null atom>
#var float4 o[COLH0] : $vout.O : O[0] : -1 : 0
#var float4 f[COL0] : $vin.F : F[0] : -1 : 0
#var float4 C144q3o24uukb6 :  : c[0] : -1 : 0
#var float4 Cmur1hd5g8ngo3 :  : c[4] : -1 : 0
#var float4 f[TEX0] : $vin.F : F[0] : -1 : 0
#var sampler2D  : texunit 0 : -1 : 0
#var sampler2D  : texunit 1 : -1 : 0
#var sampler2D  : texunit 1 : -1 : 0
#var sampler2D  : texunit 1 : -1 : 0
#var samplerLWBE  : texunit 2 : -1 : 0
#var float4 f[TEX3] : $vin.F : F[0] : -1 : 0
#var float4 f[TEX2] : $vin.F : F[0] : -1 : 0
#var float4 f[TEX7] : $vin.F : F[0] : -1 : 0
#var float4 f[TEX6] : $vin.F : F[0] : -1 : 0
#var float4 f[TEX5] : $vin.F : F[0] : -1 : 0
#var float4 f[TEX4] : $vin.F : F[0] : -1 : 0
#var float4 f[TEX1] : $vin.F : F[0] : -1 : 0
#var sampler2D  : texunit 4 : -1 : 0
#tram 0 = f[WPOS].w
#tram 1 = f[COL0].x
#tram 2 = f[COL0].y
#tram 3 = f[COL0].z
#tram 4 = f[TEX0].x
#tram 5 = f[TEX0].y
#tram 6 = f[TEX1].x
#tram 7 = f[TEX1].y
#tram 8 = f[TEX2].x
#tram 9 = f[TEX2].y
#tram 10 = f[TEX2].z
#tram 11 = f[TEX2].w
#tram 12 = f[TEX3].z
#tram 13 = f[TEX3].w
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
RCP      R12, R0;
IPA      R4, 10, R12;
IPA      R5, 11, R12;
IPA      R0, 8, R12;
IPA      R1, 9, R12;
IPA      R8, 6, R12;
IPA      R9, 7, R12;
TEX      R4, 1, 1, 2D;
TEX      R0, 1, 1, 2D;
TEX      R8, 4, 4, 2D;
MVI      R13, -1.0;
FMUL     R4, R7, R4;
FMUL     R5, R7, R5;
FMUL     R6, R7, R6;
FMAD     R7, R8, c[0], R13;
FMUL     R4, R4, 16.0;
FMUL     R5, R5, 16.0;
FMUL     R6, R6, 16.0;
MVI      R13, -1.0;
MVI      R8, -1.0;
FMAD     R9, R9, c[0], R13;
FMAD     R8, R10, c[0], R8;
FMUL     R10, R9, 0.707107;
FMUL     R2, R3, R2;
FMAD     R10, R7, c[0], R10;
FMUL     R1, R3, R1;
FMUL     R3, R3, R0;
FMAD.SAT R10, R8, c[0], R10;
FMUL     R0, R2, 16.0;
FMUL     R1, R1, 16.0;
FMUL     R2, R6, R10;
FMUL     R5, R5, R10;
FMUL     R14, R4, R10;
FMUL     R4, R3, 16.0;
FMUL     R6, R8, 0.57735;
IPA      R3, 18, R12;
IPA      R10, 17, R12;
FMAD.SAT R6, R7, c[0], R6;
FMUL     R13, R9, R3;
IPA      R3, 19, R12;
FMAD     R4, R6, R4, R14;
FMAD     R5, R6, R1, R5;
FMAD     R6, R6, R0, R2;
FMAD     R2, R7, R10, R13;
IPA      R0, 21, R12;
IPA      R1, 20, R12;
FMAD     R15, R8, R3, R2;
FMUL     R2, R9, R0;
IPA      R0, 22, R12;
IPA      R10, 14, R12;
FMAD     R1, R7, R1, R2;
IPA      R13, 15, R12;
IPA      R3, 16, R12;
FMAD     R16, R8, R0, R1;
FMUL     R0, R13, R13;
FMAD     R0, R10, R10, R0;
FMAD     R0, R3, R3, R0;
LG2      R0, R0;
IPA      R1, 24, R12;
FMUL     R0, R0, -0.5;
FMUL     R2, R9, R1;
IPA      R1, 23, R12;
RRO      R14, R0, 2;
IPA      R0, 25, R12;
FMAD     R1, R7, R1, R2;
EX2      R2, R14;
FMAD     R0, R8, R0, R1;
FMUL     R1, R13, R2;
FMUL     R10, R10, R2;
FMUL     R2, R3, R2;
FMUL     R3, R16, R1;
FMUL     R13, R16, R16;
FMAD     R3, R15, R10, R3;
FMAD     R13, R15, R15, R13;
FMAD     R3, R0, R2, R3;
FMAD     R14, R0, R0, R13;
FADD     R13, R3, R3;
FMUL     R10, R10, R14;
FMUL     R1, R1, R14;
FMUL     R2, R2, R14;
FMAD     R10, R13, R15, -R10;
FMAD     R1, R13, R16, -R1;
FMAD     R2, R13, R0, -R2;
FADD     R13, -R3, 1.0;
FMAX     R3, |R10|, |R1|;
FMUL     R0, R13, R13;
FMAX     R15, |R2|, R3;
MOV      R3, c[16];
FMUL     R14, R9, -0.707107;
RCP      R9, R15;
FMUL     R15, R0, R0;
FMAD     R7, R7, c[0], R14;
FMUL     R0, R10, R9;
FMUL     R10, R13, R15;
FMAD.SAT R7, R8, c[0], R7;
FMUL     R1, R1, R9;
FMUL     R2, R2, R9;
FMAD     R8, R10, R3, c[17];
TEX      R0, 2, 2, LWBE;
FMUL     R3, R3, R8;
FMUL     R2, R2, R8;
FMUL     R1, R1, R8;
FMUL     R2, R3, R2;
FMUL     R0, R0, R8;
FMUL     R1, R3, R1;
FMUL     R2, R11, R2;
FMUL     R0, R3, R0;
FMUL     R1, R11, R1;
FMUL     R2, R2, c[2];
FMUL     R0, R11, R0;
FMUL     R3, R1, c[1];
FMUL     R10, R2, 16.0;
FMUL     R1, R0, c[0];
FMUL     R9, R3, 16.0;
IPA      R0, 12, R12;
FMUL     R8, R1, 16.0;
IPA      R1, 13, R12;
TEX      R0, 1, 1, 2D;
FMUL     R2, R3, R2;
FMUL     R1, R3, R1;
FMUL     R0, R3, R0;
FMUL     R2, R2, 16.0;
FMUL     R1, R1, 16.0;
FMUL     R0, R0, 16.0;
FMAD     R6, R7, R2, R6;
FMAD     R5, R7, R1, R5;
FMAD     R4, R7, R0, R4;
IPA      R0, 4, R12;
IPA      R1, 5, R12;
IPA      R7, 1, R12;
IPA      R11, 2, R12;
TEX      R0, 0, 0, 2D;
IPA      R12, 3, R12;
FMUL     R0, R0, R7;
FMUL     R1, R1, R11;
FMUL     R2, R2, R12;
FMAD     R3, R4, R0, R8;
FMAD     R1, R5, R1, R9;
FMAD     R4, R6, R2, R10;
MOV      R0, R3;
FMUL     R5, R1, 0.59;
MOV      R2, R4;
FMAD     R3, R3, c[0], R5;
FMAD     R3, R4, c[0], R3;
FMUL     R3, R3, 0.0625;
END
# 146 instructions, 20 R-regs, 26 interpolants
# 146 inst, (3 mov, 3 mvi, 6 tex, 26 ipa, 4 complex, 104 math)
#    74 64-bit, 50 32-bit, 22 32-bit-const
