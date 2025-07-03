!!SPA1.0
.CONST_MODE  PAGE
.THREAD_TYPE PIXEL
.MAX_REG     15
.MAX_ATTR    16
# parseasm build date Feb 13 2004 14:20:40
# -profile fp50 -po tbat3 -po lat3 -if ps2x -i progs//p557-lw40.s -o progs//p557-lw50.s
#vendor LWPU
#version parseasm.0.0
#profile fp50
#program fp30entry
#semantic C144q3o24uukb6.C144q3o24uukb6
#semantic Cmur1hd5g8ngo3.Cmur1hd5g8ngo3
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
#var float4 f[TEX3] : $vin.F : F[0] : -1 : 0
#var sampler2D  : texunit 1 : -1 : 0
#var sampler2D  : texunit 5 : -1 : 0
#var samplerLWBE  : texunit 2 : -1 : 0
#var float4 f[TEX1] : $vin.F : F[0] : -1 : 0
#var float4 f[TEX4] : $vin.F : F[0] : -1 : 0
#var float4 f[TEX7] : $vin.F : F[0] : -1 : 0
#tram 0 = f[WPOS].w
#tram 1 = f[COL0].x
#tram 2 = f[COL0].y
#tram 3 = f[COL0].z
#tram 4 = f[COL0].w
#tram 5 = f[TEX0].x
#tram 6 = f[TEX0].y
#tram 7 = f[TEX1].z
#tram 8 = f[TEX1].w
#tram 9 = f[TEX3].x
#tram 10 = f[TEX3].y
#tram 11 = f[TEX4].x
#tram 12 = f[TEX4].y
#tram 13 = f[TEX4].z
#tram 14 = f[TEX7].x
#tram 15 = f[TEX7].y
#tram 16 = f[TEX7].z
BB0:
IPA      R0, 0;
RCP      R8, R0;
IPA      R7, 11, R8;
IPA      R6, 12, R8;
IPA      R1, 13, R8;
FMUL     R0, R6, R6;
FMAD     R0, R7, R7, R0;
IPA      R5, 15, R8;
FMAD     R0, R1, R1, R0;
IPA      R4, 14, R8;
LG2      R0, R0;
FMUL32I  R2, R0, -0.5;
IPA      R0, 16, R8;
RRO      R2, R2, 1;
EX2      R3, R2;
FMUL     R2, R5, R5;
FMUL     R7, R7, R3;
FMUL     R6, R6, R3;
FMUL     R1, R1, R3;
FMAD     R3, R4, R4, R2;
FMUL     R2, R5, R6;
FMAD     R3, R0, R0, R3;
FMAD     R2, R4, R7, R2;
FMUL     R7, R3, R7;
FMAD     R2, R0, R1, R2;
FMUL     R6, R3, R6;
FMUL     R3, R3, R1;
FADD     R1, R2, R2;
FMAD     R4, R1, R4, -R7;
FMAD     R5, R1, R5, -R6;
FMAD     R6, R1, R0, -R3;
IPA      R0, 7, R8;
FMAX     R3, |R4|, |R5|;
IPA      R1, 8, R8;
FMAX     R7, |R6|, R3;
FADD32I  R9, -R2, 1.0;
TEX      R0, 5, 5, 2D;
RCP      R11, R7;
FMUL     R7, R9, R9;
MOV      R10, c[16];
FMUL     R4, R4, R11;
FMUL     R5, R5, R11;
FMUL     R6, R6, R11;
FMUL     R11, R7, R7;
TEX      R4, 2, 2, LWBE;
FMUL     R9, R9, R11;
FMAD     R9, R9, R10, c[17];
FMUL     R3, R7, R9;
FMUL     R4, R4, R9;
FMUL     R5, R5, R9;
FMUL     R6, R6, R9;
FMUL     R4, R3, R4;
FMUL     R5, R3, R5;
FMUL     R3, R3, R6;
FMUL     R0, R0, R4;
FMUL     R1, R1, R5;
FMUL     R2, R2, R3;
FMUL     R3, R0, c[0];
FMUL     R1, R1, c[1];
FMUL     R0, R2, c[2];
FMUL32I  R11, R3, 16.0;
FMUL32I  R13, R1, 16.0;
FMUL32I  R9, R0, 16.0;
IPA      R0, 9, R8;
IPA      R1, 10, R8;
IPA      R4, 5, R8;
IPA      R5, 6, R8;
IPA      R10, 1, R8;
TEX      R0, 1, 1, 2D;
TEX      R4, 0, 0, 2D;
IPA      R12, 2, R8;
FMUL     R0, R3, R0;
FMUL     R4, R4, R10;
FMUL     R5, R5, R12;
FMUL32I  R0, R0, 16.0;
FMUL     R1, R3, R1;
FMUL     R2, R3, R2;
FMAD     R0, R0, R4, R11;
FMUL32I  R1, R1, 16.0;
FMUL32I  R2, R2, 16.0;
IPA      R3, 3, R8;
FMAD     R1, R1, R5, R13;
IPA      R4, 4, R8;
FMUL     R5, R6, R3;
FMUL32I  R1, R1, 0.59;
FMUL     R3, R7, R4;
FMAD     R2, R2, R5, R9;
FMAD     R4, R0, c[0], R1;
MVI      R0, 1.0;
MVI      R1, 0.0;
FMAD     R4, R2, c[0], R4;
MVI      R2, 1.0;
FMUL     R3, R3, R4;
FMUL32I  R3, R3, 0.0625;
END
# 94 instructions, 16 R-regs, 17 interpolants
# 94 inst, (1 mov, 3 mvi, 4 tex, 17 ipa, 4 complex, 65 math)
#    84 64-bit, 0 32-bit, 10 32-bit-const
