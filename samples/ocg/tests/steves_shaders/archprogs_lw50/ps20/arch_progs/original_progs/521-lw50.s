!!SPA1.0
.CONST_MODE  PAGE
.THREAD_TYPE PIXEL
.MAX_REG     15
.MAX_ATTR    16
# parseasm build date Feb  3 2004 15:17:32
# -profile fp50 -po tbat3 -po lat3 -if ps2x -i progs/521-lw40.s -o progs/521-lw50.s
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
FMUL     R2, R0, -0.5;
IPA      R0, 16, R8;
RRO      R2, R2, 2;
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
FADD     R9, -R2, 1.0;
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
FMUL     R0, R0, c[0];
FMUL     R1, R1, c[1];
FMUL     R2, R2, c[2];
FMUL     R12, R0, 16.0;
FMUL     R10, R1, 16.0;
FMUL     R9, R2, 16.0;
IPA      R4, 9, R8;
IPA      R5, 10, R8;
IPA      R0, 5, R8;
IPA      R1, 6, R8;
IPA      R11, 1, R8;
TEX      R4, 1, 1, 2D;
TEX      R0, 0, 0, 2D;
IPA      R13, 2, R8;
FMUL     R4, R7, R4;
FMUL     R0, R0, R11;
FMUL     R1, R1, R13;
FMUL     R4, R4, 16.0;
FMUL     R5, R7, R5;
FMUL     R6, R7, R6;
FMAD     R0, R4, R0, R12;
FMUL     R5, R5, 16.0;
FMUL     R4, R6, 16.0;
IPA      R6, 3, R8;
FMAD     R1, R5, R1, R10;
IPA      R5, 4, R8;
FMUL     R2, R2, R6;
FMUL     R3, R3, R5;
FMAD     R2, R4, R2, R9;
END
# 86 instructions, 16 R-regs, 17 interpolants
# 86 inst, (1 mov, 0 mvi, 4 tex, 17 ipa, 4 complex, 60 math)
#    41 64-bit, 37 32-bit, 8 32-bit-const
