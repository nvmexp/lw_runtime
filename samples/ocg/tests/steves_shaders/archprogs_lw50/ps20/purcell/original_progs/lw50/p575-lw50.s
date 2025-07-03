!!SPA1.0
.CONST_MODE  PAGE
.THREAD_TYPE PIXEL
.MAX_REG     11
.MAX_ATTR    14
# parseasm build date Mar 10 2004 15:40:49
# -profile fp50 -po tbat3 -po lat3 -if ps2x -i allprogs-new32//p575-lw40.s -o allprogs-new32//p575-lw50.s
#vendor LWPU
#version parseasm.0.0
#profile fp50
#program fp30entry
#semantic Coit1m35rlpml7.Coit1m35rlpml7
#semantic C144q3o24uukb6.C144q3o24uukb6
#semantic <null atom>
#semantic <null atom>
#var float4 o[COLH0] : $vout.O : O[0] : -1 : 0
#var float4 Coit1m35rlpml7 :  : c[1] : -1 : 0
#var float4 C144q3o24uukb6 :  : c[0] : -1 : 0
#var samplerLWBE  : texunit 0 : -1 : 0
#var float4 f[TEX4] : $vin.F : F[0] : -1 : 0
#var float4 f[TEX3] : $vin.F : F[0] : -1 : 0
#var float4 f[TEX2] : $vin.F : F[0] : -1 : 0
#var float4 f[TEX1] : $vin.F : F[0] : -1 : 0
#var float4 f[TEX0] : $vin.F : F[0] : -1 : 0
#var sampler2D  : texunit 1 : -1 : 0
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
MVI      R6, -1.0;
MVI      R7, -1.0;
RCP      R4, R0;
IPA      R0, 1, R4;
IPA      R1, 2, R4;
IPA      R10, 7, R4;
IPA      R9, 6, R4;
MVI      R5, -1.0;
IPA      R8, 8, R4;
TEX      R0, 1, 1, 2D;
IPA      R11, 10, R4;
IPA      R3, 9, R4;
FMAD     R0, R0, c[0], R6;
FMAD     R1, R1, c[0], R7;
FMAD     R2, R2, c[0], R5;
IPA      R5, 11, R4;
FMUL32   R7, R1, R10;
FMUL32   R10, R1, R11;
IPA      R6, 3, R4;
FMAD     R7, R0, R9, R7;
FMAD     R9, R0, R3, R10;
IPA      R3, 4, R4;
FMAD     R7, R2, R8, R7;
FMAD     R10, R2, R5, R9;
FMUL32   R8, R3, R3;
IPA      R9, 5, R4;
IPA      R5, 13, R4;
FMAD     R11, R6, R6, R8;
IPA      R8, 12, R4;
FMUL32   R1, R1, R5;
FMAD     R5, R9, R9, R11;
IPA      R4, 14, R4;
FMAD     R0, R0, R8, R1;
RSQ      R8, R5;
FMAD     R0, R2, R4, R0;
FMUL32   R1, R8, R3;
FMUL32   R2, R8, R6;
FMUL32   R3, R8, R9;
FMUL32   R4, R10, R1;
FMUL32   R6, R10, R10;
FMAD     R4, R7, R2, R4;
FMAD     R6, R7, R7, R6;
FMAD     R4, R0, R3, R4;
FMAD     R9, R0, R0, R6;
FADD32   R6, R4, R4;
FMUL32   R2, R2, R9;
FMUL32   R1, R1, R9;
FMUL32   R3, R3, R9;
FMAD     R2, R6, R7, -R2;
FMAD     R1, R6, R10, -R1;
FMAD     R3, R6, R0, -R3;
FMAX     R4, R4, c[0];
FMAX     R0, |R2|, |R1|;
FADD32I  R4, -R4, 1.0;
FMAX     R6, |R3|, R0;
FMUL32   R0, R4, R4;
RCP      R7, R6;
FMUL32   R6, R0, R0;
FMUL32   R0, R2, R7;
FMUL32   R1, R1, R7;
FMUL32   R2, R3, R7;
FMUL32   R4, R4, R6;
FMUL32   R5, R8, R5;
TEX      R0, 0, 0, LWBE;
MOV32    R6, c[6];
FMAD     R3, R5, R6, -c[7];
FMAD     R0, R0, R4, c[0];
FMAD     R1, R1, R4, c[1];
FMAD     R2, R2, R4, c[2];
END
# 70 instructions, 12 R-regs, 15 interpolants
# 70 inst, (1 mov, 3 mvi, 2 tex, 15 ipa, 3 complex, 46 math)
#    48 64-bit, 21 32-bit, 1 32-bit-const
