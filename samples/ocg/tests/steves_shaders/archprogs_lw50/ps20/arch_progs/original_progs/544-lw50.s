!!SPA1.0
.CONST_MODE  PAGE
.THREAD_TYPE PIXEL
.MAX_REG     15
.MAX_ATTR    13
# parseasm build date Feb  3 2004 15:17:32
# -profile fp50 -po tbat3 -po lat3 -if ps2x -i progs/544-lw40.s -o progs/544-lw50.s
#vendor LWPU
#version parseasm.0.0
#profile lwinst
#program fp30entry
#semantic Cee994ndhnk1rc.Cee994ndhnk1rc
#semantic Coit1m35rlpml7.Coit1m35rlpml7
#semantic C144q3o24uukb6.C144q3o24uukb6
#semantic <null atom>
#semantic <null atom>
#semantic <null atom>
#var float4 o[COLH0] : $vout.O : O[0] : -1 : 0
#var float4 Cee994ndhnk1rc :  : c[5] : -1 : 0
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
IPA      R0, 6, R12;
FMAX     R2, |R8|, |R9|;
IPA      R1, 7, R12;
IPA      R4, 4, R12;
FMAX     R6, |R10|, R2;
IPA      R5, 5, R12;
TEX      R0, 4, 4, 2D;
RCP      R11, R6;
TEX      R4, 0, 0, 2D;
FMUL     R8, R8, R11;
FMUL     R9, R9, R11;
FMUL     R10, R10, R11;
TEX      R8, 1, 1, LWBE;
FMUL     R3, R11, R8;
FMUL     R8, R11, R9;
FMUL     R9, R11, R10;
FMUL     R0, R0, R3;
FMUL     R1, R1, R8;
FMUL     R2, R2, R9;
FMUL     R0, R0, c[0];
FMUL     R1, R1, c[1];
FMUL     R3, R2, c[2];
IPA      R8, 3, R12;
IPA      R2, 2, R12;
FMUL     R8, R8, c[6];
FMUL     R2, R2, c[5];
IPA      R9, 1, R12;
FMUL     R8, R6, R8;
FMUL     R2, R5, R2;
FMUL     R9, R9, c[4];
FMAD     R6, R6, c[22], -R8;
FMAD     R5, R5, c[21], -R2;
FMUL     R9, R4, R9;
FMAD     R6, R7, R6, R8;
FMAD     R2, R7, R5, R2;
FMAD     R4, R4, c[20], -R9;
FMAD     R3, R3, c[0], R6;
FMAD     R1, R1, c[0], R2;
FMAD     R5, R7, R4, R9;
MOV      R2, R3;
FMUL     R4, R1, 0.59;
FMAD     R0, R0, c[0], R5;
FMAD     R4, R0, c[0], R4;
FMAD     R3, R3, c[0], R4;
FMUL     R3, R3, 0.0625;
END
# 66 instructions, 16 R-regs, 14 interpolants
# 66 inst, (1 mov, 0 mvi, 3 tex, 14 ipa, 2 complex, 46 math)
#    39 64-bit, 25 32-bit, 2 32-bit-const
