!!SPA1.0
.CONST_MODE  PAGE
.THREAD_TYPE PIXEL
.MAX_REG     7
.MAX_ATTR    6
# parseasm build date Feb 13 2004 14:20:40
# -profile fp50 -po tbat3 -po lat3 -if ps2x -i progs//p545-lw40.s -o progs//p545-lw50.s
#vendor LWPU
#version parseasm.0.0
#profile fp50
#program fp30entry
#semantic Coit1m35rlpml7.Coit1m35rlpml7
#semantic C144q3o24uukb6.C144q3o24uukb6
#semantic <null atom>
#var float4 o[COLH0] : $vout.O : O[0] : -1 : 0
#var float4 Coit1m35rlpml7 :  : c[1] : -1 : 0
#var float4 C144q3o24uukb6 :  : c[0] : -1 : 0
#var samplerLWBE  : texunit 1 : -1 : 0
#var float4 f[TEX3] : $vin.F : F[0] : -1 : 0
#var float4 f[TEX6] : $vin.F : F[0] : -1 : 0
#tram 0 = f[WPOS].w
#tram 1 = f[TEX3].x
#tram 2 = f[TEX3].y
#tram 3 = f[TEX3].z
#tram 4 = f[TEX6].x
#tram 5 = f[TEX6].y
#tram 6 = f[TEX6].z
BB0:
IPA      R0, 0;
RCP      R1, R0;
IPA      R4, 5, R1;
IPA      R5, 2, R1;
IPA      R2, 4, R1;
IPA      R3, 1, R1;
FMUL     R6, R4, R5;
IPA      R0, 6, R1;
IPA      R1, 3, R1;
FMAD     R6, R2, R3, R6;
FMUL     R7, R4, R4;
FMAD     R6, R0, R1, R6;
FMAD     R7, R2, R2, R7;
FADD     R6, R6, R6;
FMAD     R7, R0, R0, R7;
FMUL     R3, R7, R3;
FMUL     R5, R7, R5;
FMUL     R1, R7, R1;
FMAD     R2, R6, R2, -R3;
FMAD     R3, R6, R4, -R5;
FMAD     R4, R6, R0, -R1;
FMAX     R0, |R2|, |R3|;
FMAX     R0, |R4|, R0;
RCP      R5, R0;
FMUL     R0, R2, R5;
FMUL     R1, R3, R5;
FMUL     R2, R4, R5;
TEX      R0, 1, 1, LWBE;
FMUL     R0, R3, R0;
FMUL     R1, R3, R1;
FMUL     R2, R3, R2;
FMUL     R0, R0, c[0];
FMUL     R1, R1, c[1];
FMUL     R2, R2, c[2];
FMUL32I  R0, R0, 16.0;
FMUL32I  R1, R1, 16.0;
FMUL32I  R2, R2, 16.0;
MOV      R3, c[7];
END
# 38 instructions, 8 R-regs, 7 interpolants
# 38 inst, (1 mov, 0 mvi, 1 tex, 7 ipa, 2 complex, 27 math)
#    35 64-bit, 0 32-bit, 3 32-bit-const
