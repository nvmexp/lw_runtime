!!SPA1.0
.CONST_MODE  PAGE
.THREAD_TYPE PIXEL
.MAX_REG     15
.MAX_ATTR    12
# parseasm build date Mar 10 2004 15:40:49
# -profile fp50 -po tbat3 -po lat3 -if ps2x -i allprogs-new32//p192-lw40.s -o allprogs-new32//p192-lw50.s
#vendor LWPU
#version parseasm.0.0
#profile fp50
#program fp30entry
#semantic Coit1m35rlpml7.Coit1m35rlpml7
#semantic C144q3o24uukb6.C144q3o24uukb6
#semantic <null atom>
#semantic <null atom>
#var float4 o[COLH0] : $vout.O : O[0] : -1 : 0
#var float4 f[TEX4] : $vin.F : F[0] : -1 : 0
#var float4 f[TEX3] : $vin.F : F[0] : -1 : 0
#var float4 Coit1m35rlpml7 :  : c[1] : -1 : 0
#var float4 C144q3o24uukb6 :  : c[0] : -1 : 0
#var float4 f[TEX1] : $vin.F : F[0] : -1 : 0
#var sampler2D  : texunit 1 : -1 : 0
#var sampler3D  : texunit 0 : -1 : 0
#var float4 f[TEX0] : $vin.F : F[0] : -1 : 0
#tram 0 = f[WPOS].w
#tram 1 = f[TEX0].x
#tram 2 = f[TEX0].y
#tram 3 = f[TEX0].z
#tram 4 = f[TEX1].x
#tram 5 = f[TEX1].y
#tram 6 = f[TEX3].x
#tram 7 = f[TEX3].y
#tram 8 = f[TEX3].z
#tram 9 = f[TEX3].w
#tram 10 = f[TEX4].x
#tram 11 = f[TEX4].y
#tram 12 = f[TEX4].z
BB0:
IPA      R0, 0;
MVI      R4, 0.5;
MVI      R3, 0.5;
RCP      R8, R0;
MVI      R1, 0.5;
IPA      R9, 1, R8;
IPA      R2, 2, R8;
IPA      R0, 3, R8;
FMAD     R4, R9, c[0], R4;
FMAD     R5, R2, c[0], R3;
FMAD     R6, R0, c[0], R1;
IPA      R0, 4, R8;
IPA      R1, 5, R8;
TEX      R4, 0, 0, 3D;
MVI      R2, -1.0;
MVI      R10, 0.25;
MVI      R5, -3.14159;
MVI      R6, -0.001389;
FMAD     R4, R4, c[0], R2;
TEX      R0, 1, 1, 2D;
FMAD     R4, R9, c[3], R4;
FMAD     R4, R4, c[0], R10;
FRC      R4, R4;
FMAD     R4, R4, c[0], R5;
FMUL32   R4, R4, R4;
FMAD     R5, R4, c[0], R6;
FMAD     R6, R4, R5, c[0];
MOV32    R5, -c[6];
FMAD     R6, R4, R6, c[0];
F2F      R5, R5;
FMAD     R4, R4, R6, c[0];
FADD32   R9, R5, c[2];
MOV32    R5, -c[5];
F2F      R4, |R4|;
F2F      R6, R5;
LG2      R4, |R4|;
MOV32    R5, -c[4];
FADD32   R7, R6, c[1];
FMUL32I  R4, R4, 0.5;
F2F      R3, R5;
RRO      R4, R4, 1;
IPA      R5, 6, R8;
FADD32   R3, R3, c[0];
EX2      R10, R4;
IPA      R6, 7, R8;
IPA      R4, 8, R8;
FMAD     R9, R9, R10, c[6];
FMAD     R7, R7, R10, c[5];
FMAD     R11, R3, R10, c[4];
FMUL32   R10, R6, R6;
IPA      R3, 9, R8;
FMUL32   R0, R0, R11;
FMAD     R10, R5, R5, R10;
IPA      R12, 11, R8;
FMUL32I  R0, R0, 1.2;
FMAD     R10, R4, R4, R10;
IPA      R11, 10, R8;
IPA      R8, 12, R8;
FMAD     R3, R3, R3, R10;
RSQ      R3, R3;
FMUL32   R5, R3, R5;
FMUL32   R6, R3, R6;
FMUL32   R3, R3, R4;
FMUL32   R6, R12, R6;
MVI      R4, 1.0;
FMAD     R5, R11, R5, R6;
FMAD     R3, R8, R3, R5;
FMUL32   R1, R1, R7;
FCMP     R3, R3, R3, c[0];
FMUL32I  R1, R1, 1.2;
FMUL32   R2, R2, R9;
LG2      R3, |R3|;
FMUL32I  R2, R2, 1.2;
FMUL32I  R5, R3, 20.0;
MVI      R3, 1.0;
RRO      R5, R5, 1;
EX2      R5, R5;
FMAD     R4, R5, c[0], R4;
FMUL32   R0, R0, R4;
FMUL32   R1, R1, R4;
FMUL32   R2, R2, R4;
END
# 81 instructions, 16 R-regs, 13 interpolants
# 81 inst, (3 mov, 9 mvi, 2 tex, 13 ipa, 6 complex, 48 math)
#    58 64-bit, 18 32-bit, 5 32-bit-const
