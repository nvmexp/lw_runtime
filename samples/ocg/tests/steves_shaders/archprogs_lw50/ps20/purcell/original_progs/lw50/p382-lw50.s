!!SPA1.0
.CONST_MODE  PAGE
.THREAD_TYPE PIXEL
.MAX_REG     15
.MAX_ATTR    25
# parseasm build date Mar 10 2004 15:40:49
# -profile fp50 -po tbat3 -po lat3 -if ps2x -i allprogs-new32//p382-lw40.s -o allprogs-new32//p382-lw50.s
#vendor LWPU
#version parseasm.0.0
#profile fp50
#program fp30entry
#semantic Cmur1hd5g8ngo3.Cmur1hd5g8ngo3
#semantic Clofsu13ov1o1e.Clofsu13ov1o1e
#semantic Coit1m35rlpml7.Coit1m35rlpml7
#semantic C6r0tbn6evb4u3.C6r0tbn6evb4u3
#semantic C144q3o24uukb6.C144q3o24uukb6
#semantic <null atom>
#semantic <null atom>
#semantic <null atom>
#var float4 o[COLH0] : $vout.O : O[0] : -1 : 0
#var float4 Cmur1hd5g8ngo3 :  : c[4] : -1 : 0
#var float4 Clofsu13ov1o1e :  : c[3] : -1 : 0
#var float4 Coit1m35rlpml7 :  : c[1] : -1 : 0
#var float4 C6r0tbn6evb4u3 :  : c[2] : -1 : 0
#var float4 f[TEX7] : $vin.F : F[0] : -1 : 0
#var float4 f[COL0] : $vin.F : F[0] : -1 : 0
#var float4 f[COL1] : $vin.F : F[0] : -1 : 0
#var float4 C144q3o24uukb6 :  : c[0] : -1 : 0
#var float4 f[TEX0] : $vin.F : F[0] : -1 : 0
#var sampler2D  : texunit 0 : -1 : 0
#var samplerLWBE  : texunit 1 : -1 : 0
#var float4 f[TEX3] : $vin.F : F[0] : -1 : 0
#var float4 f[TEX6] : $vin.F : F[0] : -1 : 0
#var float4 f[TEX4] : $vin.F : F[0] : -1 : 0
#var float4 f[TEX5] : $vin.F : F[0] : -1 : 0
#var float4 f[TEX1] : $vin.F : F[0] : -1 : 0
#var sampler2D  : texunit 3 : -1 : 0
#tram 0 = f[WPOS].w
#tram 1 = f[COL0].x
#tram 2 = f[COL0].y
#tram 3 = f[COL0].z
#tram 4 = f[COL1].x
#tram 5 = f[COL1].y
#tram 6 = f[COL1].z
#tram 7 = f[TEX0].x
#tram 8 = f[TEX0].y
#tram 9 = f[TEX1].x
#tram 10 = f[TEX1].y
#tram 11 = f[TEX3].x
#tram 12 = f[TEX3].y
#tram 13 = f[TEX3].z
#tram 14 = f[TEX4].x
#tram 15 = f[TEX4].y
#tram 16 = f[TEX4].z
#tram 17 = f[TEX5].x
#tram 18 = f[TEX5].y
#tram 19 = f[TEX5].z
#tram 20 = f[TEX6].x
#tram 21 = f[TEX6].y
#tram 22 = f[TEX6].z
#tram 23 = f[TEX7].x
#tram 24 = f[TEX7].y
#tram 25 = f[TEX7].z
BB0:
IPA      R0, 0;
MVI      R7, -1.0;
MVI      R6, -1.0;
RCP      R12, R0;
IPA      R8, 9, R12;
IPA      R9, 10, R12;
IPA      R2, 17, R12;
IPA      R1, 14, R12;
MVI      R5, -1.0;
IPA      R0, 20, R12;
TEX      R8, 3, 3, 2D;
IPA      R4, 18, R12;
IPA      R3, 15, R12;
FMAD     R8, R8, c[0], R7;
FMAD     R9, R9, c[0], R6;
FMAD     R10, R10, c[0], R5;
IPA      R5, 21, R12;
FMUL32   R2, R9, R2;
FMUL32   R6, R9, R4;
IPA      R4, 19, R12;
FMAD     R1, R1, R8, R2;
FMAD     R2, R3, R8, R6;
FMUL32   R3, R9, R4;
FMAD     R4, R0, R10, R1;
FMAD     R2, R5, R10, R2;
IPA      R5, 16, R12;
IPA      R0, 22, R12;
FMUL32   R1, R2, R2;
FMAD     R5, R5, R8, R3;
IPA      R3, 12, R12;
FMAD     R1, R4, R4, R1;
FMAD     R0, R0, R10, R5;
FMUL32   R7, R2, R3;
IPA      R6, 11, R12;
FMAD     R5, R0, R0, R1;
IPA      R1, 13, R12;
FMAD     R7, R4, R6, R7;
RCP      R5, R5;
FMAD     R7, R0, R1, R7;
FMUL32   R13, R5, R7;
FMAD     R7, R5, R7, R13;
FMAD     R4, R7, R4, -R6;
FMAD     R5, R7, R2, -R3;
FMAD     R6, R7, R0, -R1;
IPA      R0, 7, R12;
FMAX     R2, |R4|, |R5|;
IPA      R1, 8, R12;
FMAX     R7, |R6|, R2;
TEX      R0, 0, 0, 2D;
RCP      R7, R7;
FMUL32   R4, R4, R7;
FMUL32   R5, R5, R7;
FMUL32   R6, R6, R7;
TEX      R4, 1, 1, LWBE;
FMUL32   R4, R11, R4;
FMUL32   R5, R11, R5;
FMUL32   R4, R4, c[0];
FMUL32   R7, R11, R6;
FMUL32   R6, R5, c[1];
FMAD     R11, R4, R4, -R4;
FMUL32   R5, R7, c[2];
FMAD     R7, R6, R6, -R6;
FMAD     R13, R11, c[8], R4;
FMAD     R4, R5, R5, -R5;
FMAD     R7, R7, c[9], R6;
FMAD     R4, R4, c[10], R5;
FMUL32I  R5, R7, 0.333333;
FMAD     R5, R13, c[0], R5;
FMUL32I  R6, R10, 0.57735;
FMAD     R5, R4, c[0], R5;
FMAD.SAT R6, R8, c[0], R6;
FMUL32I  R11, R9, 0.707107;
FADD32   R4, R4, -R5;
FADD32   R7, R7, -R5;
FADD32   R13, R13, -R5;
FMAD     R4, R4, c[14], R5;
FMAD     R7, R7, c[13], R5;
FMAD     R5, R13, c[12], R5;
FMAD     R13, R8, c[0], R11;
FMUL32I  R11, R9, -0.707107;
IPA      R14, 4, R12;
FMAD.SAT R9, R10, c[0], R13;
FMAD     R8, R8, c[0], R11;
IPA      R11, 1, R12;
FMUL32   R13, R9, R14;
FMAD.SAT R8, R10, c[0], R8;
IPA      R10, 23, R12;
FMAD     R14, R6, R11, R13;
IPA      R11, 5, R12;
IPA      R13, 2, R12;
FMAD     R10, R8, R10, R14;
FMUL32   R14, R9, R11;
IPA      R11, 24, R12;
FMUL32   R10, R10, c[4];
FMAD     R13, R6, R13, R14;
IPA      R15, 6, R12;
IPA      R14, 3, R12;
FMAD     R11, R8, R11, R13;
FMUL32   R9, R9, R15;
IPA      R12, 25, R12;
FMUL32   R11, R11, c[5];
FMAD     R6, R6, R14, R9;
FMAD     R6, R8, R12, R6;
FMUL32   R1, R1, R11;
FMUL32   R0, R0, R10;
FMUL32   R6, R6, c[6];
FMUL32   R1, R1, c[17];
FMUL32   R0, R0, c[16];
FMUL32   R2, R2, R6;
FMAD     R1, R1, c[0], R7;
FMAD     R0, R0, c[0], R5;
FMUL32   R2, R2, c[18];
FMUL32   R3, R3, c[7];
FMAD     R2, R2, c[0], R4;
END
# 114 instructions, 16 R-regs, 26 interpolants
# 114 inst, (0 mov, 3 mvi, 3 tex, 26 ipa, 3 complex, 79 math)
#    79 64-bit, 31 32-bit, 4 32-bit-const
