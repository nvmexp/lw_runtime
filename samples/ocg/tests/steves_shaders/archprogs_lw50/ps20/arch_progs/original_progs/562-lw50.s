!!SPA1.0
.CONST_MODE  PAGE
.THREAD_TYPE PIXEL
.MAX_REG     15
.MAX_ATTR    13
# parseasm build date Feb  3 2004 15:17:32
# -profile fp50 -po tbat3 -po lat3 -if ps2x -i progs/562-lw40.s -o progs/562-lw50.s
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
#var float4 f[COL0] : $vin.F : F[0] : -1 : 0
#var float4 C144q3o24uukb6 :  : c[0] : -1 : 0
#var float4 f[TEX0] : $vin.F : F[0] : -1 : 0
#var sampler2D  : texunit 0 : -1 : 0
#var float4 f[TEX2] : $vin.F : F[0] : -1 : 0
#var sampler2D  : texunit 4 : -1 : 0
#var samplerLWBE  : texunit 1 : -1 : 0
#var float4 f[TEX3] : $vin.F : F[0] : -1 : 0
#var float4 f[TEX6] : $vin.F : F[0] : -1 : 0
#tram 0 = f[WPOS].w
#tram 1 = f[COL0].x
#tram 2 = f[COL0].y
#tram 3 = f[COL0].z
#tram 4 = f[TEX0].x
#tram 5 = f[TEX0].y
#tram 6 = f[TEX2].x
#tram 7 = f[TEX2].y
#tram 8 = f[TEX3].x
#tram 9 = f[TEX3].y
#tram 10 = f[TEX3].z
#tram 11 = f[TEX6].x
#tram 12 = f[TEX6].y
#tram 13 = f[TEX6].z
BB0:
IPA      R0, 0;
RCP      R12, R0;
IPA      R3, 12, R12;
IPA      R4, 9, R12;
IPA      R5, 11, R12;
IPA      R6, 8, R12;
FMUL     R1, R3, R4;
IPA      R0, 13, R12;
IPA      R2, 10, R12;
FMAD     R1, R5, R6, R1;
FMUL     R7, R3, R3;
FMAD     R1, R0, R2, R1;
FMAD     R7, R5, R5, R7;
FADD     R1, R1, R1;
FMAD     R7, R0, R0, R7;
FMUL     R6, R7, R6;
FMUL     R4, R7, R4;
FMUL     R2, R7, R2;
FMAD     R8, R1, R5, -R6;
FMAD     R9, R1, R3, -R4;
FMAD     R10, R1, R0, -R2;
IPA      R4, 6, R12;
FMAX     R1, |R8|, |R9|;
IPA      R5, 7, R12;
IPA      R0, 4, R12;
FMAX     R2, |R10|, R1;
IPA      R1, 5, R12;
TEX      R4, 4, 4, 2D;
RCP      R11, R2;
TEX      R0, 0, 0, 2D;
FMUL     R8, R8, R11;
FMUL     R9, R9, R11;
FMUL     R10, R10, R11;
TEX      R8, 1, 1, LWBE;
FMUL     R7, R11, R8;
FMUL     R8, R11, R9;
FMUL     R9, R11, R10;
FMUL     R4, R4, R7;
FMUL     R5, R5, R8;
FMUL     R6, R6, R9;
FMUL     R5, R5, c[1];
FMUL     R7, R6, c[2];
FMUL     R3, R4, c[0];
FMUL     R6, R5, 16.0;
FMUL     R7, R7, 16.0;
FMUL     R4, R3, 16.0;
IPA      R3, 1, R12;
IPA      R5, 2, R12;
IPA      R8, 3, R12;
FMUL     R3, R3, c[4];
FMUL     R5, R5, c[5];
FMUL     R8, R8, c[6];
FMAD     R3, R0, R3, R4;
FMAD     R1, R1, R5, R6;
FMAD     R4, R2, R8, R7;
MOV      R0, R3;
FMUL     R5, R1, 0.59;
MOV      R2, R4;
FMAD     R3, R3, c[0], R5;
FMAD     R3, R4, c[0], R3;
FMUL     R3, R3, 0.0625;
END
# 61 instructions, 16 R-regs, 14 interpolants
# 61 inst, (2 mov, 0 mvi, 3 tex, 14 ipa, 2 complex, 40 math)
#    33 64-bit, 23 32-bit, 5 32-bit-const
