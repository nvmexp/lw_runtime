!!SPA1.0
.CONST_MODE  PAGE
.THREAD_TYPE PIXEL
.MAX_REG     11
.MAX_ATTR    11
# parseasm build date Feb  3 2004 15:17:32
# -profile fp50 -po tbat3 -po lat3 -if ps2x -i progs/523-lw40.s -o progs/523-lw50.s
#vendor LWPU
#version parseasm.0.0
#profile lwinst
#program fp30entry
#semantic Coit1m35rlpml7.Coit1m35rlpml7
#semantic C144q3o24uukb6.C144q3o24uukb6
#semantic <null atom>
#semantic <null atom>
#var float4 o[COLH0] : $vout.O : O[0] : -1 : 0
#var float4 Coit1m35rlpml7 :  : c[1] : -1 : 0
#var float4 f[COL0] : $vin.F : F[0] : -1 : 0
#var float4 C144q3o24uukb6 :  : c[0] : -1 : 0
#var float4 f[TEX0] : $vin.F : F[0] : -1 : 0
#var sampler2D  : texunit 0 : -1 : 0
#var samplerLWBE  : texunit 1 : -1 : 0
#var float4 f[TEX3] : $vin.F : F[0] : -1 : 0
#var float4 f[TEX6] : $vin.F : F[0] : -1 : 0
#tram 0 = f[WPOS].w
#tram 1 = f[COL0].x
#tram 2 = f[COL0].y
#tram 3 = f[COL0].z
#tram 4 = f[TEX0].x
#tram 5 = f[TEX0].y
#tram 6 = f[TEX3].x
#tram 7 = f[TEX3].y
#tram 8 = f[TEX3].z
#tram 9 = f[TEX6].x
#tram 10 = f[TEX6].y
#tram 11 = f[TEX6].z
BB0:
IPA      R0, 0;
RCP      R8, R0;
IPA      R3, 10, R8;
IPA      R5, 7, R8;
IPA      R4, 9, R8;
IPA      R6, 6, R8;
FMUL     R1, R3, R5;
IPA      R0, 11, R8;
IPA      R2, 8, R8;
FMAD     R1, R4, R6, R1;
FMUL     R7, R3, R3;
FMAD     R1, R0, R2, R1;
FMAD     R7, R4, R4, R7;
FADD     R1, R1, R1;
FMAD     R7, R0, R0, R7;
FMUL     R6, R7, R6;
FMUL     R5, R7, R5;
FMUL     R2, R7, R2;
FMAD     R4, R1, R4, -R6;
FMAD     R5, R1, R3, -R5;
FMAD     R6, R1, R0, -R2;
IPA      R0, 4, R8;
FMAX     R2, |R4|, |R5|;
IPA      R1, 5, R8;
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
FMUL     R7, R6, c[2];
FMUL     R5, R5, c[1];
FMUL     R6, R4, c[0];
FMUL     R4, R7, 16.0;
FMUL     R5, R5, 16.0;
FMUL     R7, R6, 16.0;
IPA      R9, 1, R8;
IPA      R6, 2, R8;
IPA      R10, 3, R8;
FMUL     R9, R9, c[4];
FMUL     R8, R6, c[5];
FMUL     R6, R10, c[6];
FMAD     R0, R0, R9, R7;
FMAD     R1, R1, R8, R5;
FMAD     R2, R2, R6, R4;
FMUL     R3, R3, c[7];
END
# 50 instructions, 12 R-regs, 12 interpolants
# 50 inst, (0 mov, 0 mvi, 2 tex, 12 ipa, 2 complex, 34 math)
#    28 64-bit, 19 32-bit, 3 32-bit-const
