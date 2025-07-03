!!SPA1.0
.CONST_MODE  PAGE
.THREAD_TYPE PIXEL
.MAX_REG     15
.MAX_ATTR    14
# parseasm build date Feb 13 2004 14:20:40
# -profile fp50 -po tbat3 -po lat3 -if ps2x -i progs//p567-lw40.s -o progs//p567-lw50.s
#vendor LWPU
#version parseasm.0.0
#profile fp50
#program fp30entry
#semantic C144q3o24uukb6.C144q3o24uukb6
#semantic <null atom>
#semantic <null atom>
#semantic <null atom>
#var float4 o[COLH0] : $vout.O : O[0] : -1 : 0
#var float4 C144q3o24uukb6 :  : c[0] : -1 : 0
#var samplerLWBE  : texunit 0 : -1 : 0
#var float4 f[TEX4] : $vin.F : F[0] : -1 : 0
#var float4 f[TEX3] : $vin.F : F[0] : -1 : 0
#var float4 f[TEX2] : $vin.F : F[0] : -1 : 0
#var float4 f[TEX0] : $vin.F : F[0] : -1 : 0
#var sampler2D  : texunit 1 : -1 : 0
#var float4 f[TEX1] : $vin.F : F[0] : -1 : 0
#var samplerLWBE  : texunit 6 : -1 : 0
#tram 0 = f[WPOS].w
#tram 1 = f[TEX0].x
#tram 2 = f[TEX0].y
#tram 3 = f[TEX1].x
#tram 4 = f[TEX1].y
#tram 5 = f[TEX1].z
#tram 6 = f[TEX2].x
#tram 7 = f[TEX2].y
#tram 8 = f[TEX2].z
#tram 9 = f[TEX3].x
#tram 10 = f[TEX3].y
#tram 11 = f[TEX3].z
#tram 12 = f[TEX4].x
#tram 13 = f[TEX4].y
#tram 14 = f[TEX4].z
BB0:
IPA      R0, 0;
RCP      R8, R0;
IPA      R0, 1, R8;
IPA      R1, 2, R8;
IPA      R4, 3, R8;
IPA      R5, 4, R8;
IPA      R6, 5, R8;
TEX      R0, 1, 1, 2D;
FMAX     R7, |R4|, |R5|;
MVI      R9, -1.0;
FMAX     R3, |R6|, R7;
MVI      R7, -1.0;
FMAD     R0, R0, c[0], R9;
RCP      R3, R3;
FMAD     R1, R1, c[0], R7;
IPA      R7, 7, R8;
FMUL     R4, R4, R3;
FMUL     R5, R5, R3;
FMUL     R6, R6, R3;
FMUL     R9, R1, R7;
IPA      R3, 6, R8;
TEX      R4, 6, 6, LWBE;
MVI      R10, -1.0;
FMAD     R9, R0, R3, R9;
IPA      R3, 8, R8;
FMAD     R2, R2, c[0], R10;
IPA      R11, 10, R8;
IPA      R10, 9, R8;
FMAD     R3, R2, R3, R9;
FMUL     R12, R1, R11;
IPA      R11, 11, R8;
IPA      R9, 13, R8;
FMAD     R12, R0, R10, R12;
IPA      R10, 12, R8;
FMUL     R1, R1, R9;
FMAD     R9, R2, R11, R12;
FMAD     R0, R0, R10, R1;
IPA      R1, 14, R8;
FMUL     R8, R9, R9;
MVI      R7, -1.0;
FMAD     R0, R2, R1, R0;
FMAD     R1, R3, R3, R8;
FMAD     R2, R6, c[0], R7;
MVI      R6, -1.0;
FMAD     R7, R0, R0, R1;
MVI      R1, -1.0;
FMAD     R4, R4, c[0], R6;
FMAD     R1, R5, c[0], R1;
FMUL     R6, R4, R7;
FMUL     R5, R9, R1;
FMUL     R1, R1, R7;
FMUL     R7, R2, R7;
FMAD     R4, R3, R4, R5;
FMAD     R2, R0, R2, R4;
FADD     R4, R2, R2;
FMAD     R3, R4, R3, -R6;
FMAD     R1, R4, R9, -R1;
FMAD     R0, R4, R0, -R7;
FMAX     R2, R2, c[0];
FMAX     R4, |R3|, |R1|;
FADD32I  R2, -R2, 1.0;
FMAX     R5, |R0|, R4;
FMUL     R4, R2, R2;
RCP      R6, R5;
FMUL     R7, R4, R4;
FMUL     R4, R3, R6;
FMUL     R5, R1, R6;
FMUL     R6, R0, R6;
FMUL     R2, R2, R7;
MVI      R3, 1.0;
TEX      R4, 0, 0, LWBE;
FMAD     R0, R4, R2, c[0];
FMAD     R1, R5, R2, c[1];
FMAD     R2, R6, R2, c[2];
END
# 74 instructions, 16 R-regs, 15 interpolants
# 74 inst, (0 mov, 7 mvi, 3 tex, 15 ipa, 3 complex, 46 math)
#    73 64-bit, 0 32-bit, 1 32-bit-const
