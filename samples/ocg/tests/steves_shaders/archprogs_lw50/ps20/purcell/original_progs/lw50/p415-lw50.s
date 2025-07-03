!!SPA1.0
.CONST_MODE  PAGE
.THREAD_TYPE PIXEL
.MAX_REG     15
.MAX_ATTR    18
# parseasm build date Mar 10 2004 15:40:49
# -profile fp50 -po tbat3 -po lat3 -if ps2x -i allprogs-new32//p415-lw40.s -o allprogs-new32//p415-lw50.s
#vendor LWPU
#version parseasm.0.0
#profile fp50
#program fp30entry
#semantic Coit1m35rlpml7.Coit1m35rlpml7
#semantic Cmur1hd5g8ngo3.Cmur1hd5g8ngo3
#semantic <null atom>
#semantic <null atom>
#semantic <null atom>
#var float4 o[COLH0] : $vout.O : O[0] : -1 : 0
#var float4 Coit1m35rlpml7 :  : c[1] : -1 : 0
#var float4 Cmur1hd5g8ngo3 :  : c[4] : -1 : 0
#var sampler2D  : texunit 2 : -1 : 0
#var sampler2D  : texunit 4 : -1 : 0
#var float4 f[TEX7] : $vin.F : F[0] : -1 : 0
#var float4 f[TEX6] : $vin.F : F[0] : -1 : 0
#var float4 f[TEX5] : $vin.F : F[0] : -1 : 0
#var float4 f[TEX4] : $vin.F : F[0] : -1 : 0
#var float4 f[TEX3] : $vin.F : F[0] : -1 : 0
#var float4 f[TEX2] : $vin.F : F[0] : -1 : 0
#var float4 f[TEX1] : $vin.F : F[0] : -1 : 0
#var float4 f[TEX0] : $vin.F : F[0] : -1 : 0
#var sampler2D  : texunit 3 : -1 : 0
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
#tram 11 = f[TEX4].x
#tram 12 = f[TEX4].y
#tram 13 = f[TEX5].x
#tram 14 = f[TEX5].y
#tram 15 = f[TEX6].x
#tram 16 = f[TEX6].y
#tram 17 = f[TEX7].x
#tram 18 = f[TEX7].y
BB0:
IPA      R0, 0;
MVI      R10, -1.0;
MVI      R12, -1.0;
RCP      R8, R0;
IPA      R4, 1, R8;
IPA      R5, 2, R8;
IPA      R1, 10, R8;
IPA      R0, 9, R8;
IPA      R2, 6, R8;
IPA      R3, 8, R8;
TEX      R4, 3, 3, 2D;
RCP      R9, R3;
IPA      R11, 12, R8;
IPA      R3, 11, R8;
FMAD     R10, R4, c[0], R10;
FMAD     R4, R5, c[0], R12;
IPA      R5, 7, R8;
FMUL32   R1, R1, R4;
FMUL32   R11, R11, R4;
FMAD     R0, R0, R10, R1;
FMAD     R1, R3, R10, R11;
FADD32   R0, R0, R2;
FADD32   R1, R1, R5;
IPA      R11, 3, R8;
FMUL32   R0, R0, R9;
FMUL32   R1, R9, R1;
IPA      R12, 4, R8;
IPA      R5, 5, R8;
TEX      R0, 4, 4, 2D;
FMUL32   R13, R12, R12;
FMAD     R3, R11, R11, R13;
FMUL32   R2, R2, c[18];
FMUL32   R1, R1, c[17];
FMUL32   R0, R0, c[16];
FMAD     R3, R5, R5, R3;
LG2      R3, R3;
MVI      R13, -1.0;
FMUL32I  R3, R3, -0.5;
FMAD     R6, R6, c[0], R13;
RRO      R3, R3, 1;
EX2      R3, R3;
FMUL32   R12, R12, R3;
FMUL32   R11, R11, R3;
FMUL32   R3, R5, R3;
FMUL32   R5, R12, R4;
IPA      R13, 16, R8;
IPA      R12, 15, R8;
FMAD     R5, R11, R10, R5;
FMUL32   R11, R13, R4;
IPA      R13, 13, R8;
FMAD     R3, R3, R6, R5;
FMAD     R6, R12, R10, R11;
IPA      R11, 18, R8;
FADD32I  R5, -R3, 1.0;
FADD32   R3, R6, R13;
FMUL32   R6, R11, R4;
FMUL32   R11, R5, R5;
FMUL32   R4, R9, R3;
MOV32    R3, R7;
FMUL32   R11, R11, R11;
IPA      R7, 17, R8;
IPA      R12, 14, R8;
FMUL32   R8, R5, R11;
FMAD     R5, R7, R10, R6;
FADD32   R5, R5, R12;
FMUL32   R5, R9, R5;
TEX      R4, 2, 2, 2D;
FMUL32   R6, R6, c[6];
FMUL32   R5, R5, c[5];
FMUL32   R4, R4, c[4];
FMAD     R2, R2, R8, R6;
FMAD     R1, R1, R8, R5;
FMAD     R0, R0, R8, R4;
END
# 73 instructions, 16 R-regs, 19 interpolants
# 73 inst, (1 mov, 3 mvi, 3 tex, 19 ipa, 4 complex, 43 math)
#    44 64-bit, 27 32-bit, 2 32-bit-const
