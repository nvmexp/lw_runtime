!!SPA1.0
.CONST_MODE  PAGE
.THREAD_TYPE PIXEL
.MAX_REG     15
.MAX_ATTR    13
# parseasm build date Mar 10 2004 15:40:49
# -profile fp50 -po tbat3 -po lat3 -if ps2x -i allprogs-new32//p505-lw40.s -o allprogs-new32//p505-lw50.s
#vendor LWPU
#version parseasm.0.0
#profile fp50
#program fp30entry
#semantic C144q3o24uukb6.C144q3o24uukb6
#semantic Cmur1hd5g8ngo3.Cmur1hd5g8ngo3
#semantic <null atom>
#semantic <null atom>
#semantic <null atom>
#var float4 o[COLH0] : $vout.O : O[0] : -1 : 0
#var float4 C144q3o24uukb6 :  : c[0] : -1 : 0
#var float4 f[COL0] : $vin.F : F[0] : -1 : 0
#var float4 Cmur1hd5g8ngo3 :  : c[4] : -1 : 0
#var float4 f[TEX3] : $vin.F : F[0] : -1 : 0
#var sampler2D  : texunit 1 : -1 : 0
#var float4 f[TEX0] : $vin.F : F[0] : -1 : 0
#var sampler2D  : texunit 0 : -1 : 0
#var samplerLWBE  : texunit 2 : -1 : 0
#var float4 f[TEX4] : $vin.F : F[0] : -1 : 0
#var float4 f[TEX7] : $vin.F : F[0] : -1 : 0
#tram 0 = f[WPOS].w
#tram 1 = f[COL0].x
#tram 2 = f[COL0].y
#tram 3 = f[COL0].z
#tram 4 = f[TEX0].x
#tram 5 = f[TEX0].y
#tram 6 = f[TEX3].x
#tram 7 = f[TEX3].y
#tram 8 = f[TEX4].x
#tram 9 = f[TEX4].y
#tram 10 = f[TEX4].z
#tram 11 = f[TEX7].x
#tram 12 = f[TEX7].y
#tram 13 = f[TEX7].z
BB0:
IPA      R0, 0;
RCP      R12, R0;
IPA      R6, 8, R12;
IPA      R5, 9, R12;
IPA      R1, 10, R12;
FMUL32   R0, R5, R5;
FMAD     R0, R6, R6, R0;
IPA      R3, 12, R12;
FMAD     R0, R1, R1, R0;
IPA      R4, 11, R12;
LG2      R0, R0;
FMUL32I  R2, R0, -0.5;
IPA      R0, 13, R12;
RRO      R2, R2, 1;
EX2      R7, R2;
FMUL32   R2, R3, R3;
FMUL32   R6, R6, R7;
FMUL32   R5, R5, R7;
FMUL32   R1, R1, R7;
FMAD     R2, R4, R4, R2;
FMUL32   R7, R3, R5;
FMAD     R2, R0, R0, R2;
FMAD     R7, R4, R6, R7;
FMUL32   R6, R2, R6;
FMAD     R10, R0, R1, R7;
FMUL32   R5, R2, R5;
FMUL32   R2, R2, R1;
FADD32   R1, R10, R10;
FMAD     R4, R1, R4, -R6;
FMAD     R5, R1, R3, -R5;
FMAD     R6, R1, R0, -R2;
IPA      R0, 6, R12;
FMAX     R2, |R4|, |R5|;
IPA      R1, 7, R12;
IPA      R8, 4, R12;
FMAX     R7, |R6|, R2;
IPA      R9, 5, R12;
TEX      R0, 1, 1, 2D;
RCP      R7, R7;
FADD32I  R13, -R10, 1.0;
TEX      R8, 0, 0, 2D;
FMUL32   R4, R4, R7;
FMUL32   R5, R5, R7;
FMUL32   R6, R6, R7;
FMUL32   R14, R13, R13;
MOV32    R15, c[16];
TEX      R4, 2, 2, LWBE;
FMUL32   R14, R14, R14;
FMUL32   R13, R13, R14;
FMAD     R13, R13, R15, c[17];
FMUL32   R7, R7, R13;
FMUL32   R4, R4, R13;
FMUL32   R5, R5, R13;
FMUL32   R6, R6, R13;
FMUL32   R4, R7, R4;
FMUL32   R5, R7, R5;
FMUL32   R6, R7, R6;
FADD32I  R7, -R11, 1.0;
FMUL32   R6, R6, R7;
FMUL32   R5, R5, R7;
FMUL32   R4, R4, R7;
FMUL32   R6, R6, c[2];
FMUL32   R5, R5, c[1];
FMUL32   R4, R4, c[0];
FMUL32I  R11, R6, 16.0;
FMUL32I  R6, R5, 16.0;
FMUL32I  R4, R4, 16.0;
IPA      R7, 3, R12;
IPA      R5, 2, R12;
FMUL32   R2, R3, R2;
FMUL32   R7, R10, R7;
FMUL32   R5, R9, R5;
FMUL32I  R2, R2, 16.0;
FMUL32   R1, R3, R1;
FMUL32   R0, R3, R0;
IPA      R3, 1, R12;
FMAD     R7, R2, R7, R11;
FMUL32I  R1, R1, 16.0;
FMUL32   R3, R8, R3;
FMUL32I  R0, R0, 16.0;
FMAD     R1, R1, R5, R6;
MOV32    R2, R7;
FMAD     R3, R0, R3, R4;
FMUL32I  R4, R1, 0.59;
MOV32    R0, R3;
FMAD     R3, R3, c[0], R4;
FMAD     R3, R7, c[0], R3;
FMUL32I  R3, R3, 0.0625;
END
# 88 instructions, 16 R-regs, 14 interpolants
# 88 inst, (3 mov, 0 mvi, 3 tex, 14 ipa, 4 complex, 64 math)
#    39 64-bit, 38 32-bit, 11 32-bit-const
