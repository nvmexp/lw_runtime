!!SPA1.0
.CONST_MODE  PAGE
.THREAD_TYPE PIXEL
.MAX_REG     15
.MAX_ATTR    22
# parseasm build date Mar 10 2004 15:40:49
# -profile fp50 -po tbat3 -po lat3 -if ps2x -i allprogs-new32//p347-lw40.s -o allprogs-new32//p347-lw50.s
#vendor LWPU
#version parseasm.0.0
#profile fp50
#program fp30entry
#semantic Clofsu13ov1o1e.Clofsu13ov1o1e
#semantic C144q3o24uukb6.C144q3o24uukb6
#semantic C6r0tbn6evb4u3.C6r0tbn6evb4u3
#semantic Coit1m35rlpml7.Coit1m35rlpml7
#semantic <null atom>
#semantic <null atom>
#semantic <null atom>
#semantic <null atom>
#semantic <null atom>
#var float4 o[COLH0] : $vout.O : O[0] : -1 : 0
#var float4 Clofsu13ov1o1e :  : c[3] : -1 : 0
#var float4 C144q3o24uukb6 :  : c[0] : -1 : 0
#var float4 C6r0tbn6evb4u3 :  : c[2] : -1 : 0
#var float4 Coit1m35rlpml7 :  : c[1] : -1 : 0
#var sampler2D  : texunit 5 : -1 : 0
#var sampler2D  : texunit 2 : -1 : 0
#var samplerLWBE  : texunit 4 : -1 : 0
#var float4 f[TEX2] : $vin.F : F[0] : -1 : 0
#var float4 f[TEX1] : $vin.F : F[0] : -1 : 0
#var float4 f[TEX4] : $vin.F : F[0] : -1 : 0
#var float4 f[TEX7] : $vin.F : F[0] : -1 : 0
#var float4 f[TEX5] : $vin.F : F[0] : -1 : 0
#var float4 f[TEX6] : $vin.F : F[0] : -1 : 0
#var sampler2D  : texunit 0 : -1 : 0
#var float4 f[TEX0] : $vin.F : F[0] : -1 : 0
#var sampler2D  : texunit 3 : -1 : 0
#tram 0 = f[WPOS].w
#tram 1 = f[TEX0].x
#tram 2 = f[TEX0].y
#tram 3 = f[TEX1].x
#tram 4 = f[TEX1].y
#tram 5 = f[TEX1].z
#tram 6 = f[TEX1].w
#tram 7 = f[TEX2].x
#tram 8 = f[TEX2].y
#tram 9 = f[TEX2].z
#tram 10 = f[TEX2].w
#tram 11 = f[TEX4].x
#tram 12 = f[TEX4].y
#tram 13 = f[TEX4].z
#tram 14 = f[TEX5].x
#tram 15 = f[TEX5].y
#tram 16 = f[TEX5].z
#tram 17 = f[TEX6].x
#tram 18 = f[TEX6].y
#tram 19 = f[TEX6].z
#tram 20 = f[TEX7].x
#tram 21 = f[TEX7].y
#tram 22 = f[TEX7].z
BB0:
IPA      R0, 0;
RCP      R12, R0;
IPA      R8, 1, R12;
IPA      R9, 2, R12;
IPA      R2, 6, R12;
MOV32    R0, R8;
MOV32    R1, R9;
RCP      R2, R2;
IPA      R5, 4, R12;
IPA      R4, 3, R12;
IPA      R3, 5, R12;
FMUL32   R6, R2, R5;
FMUL32   R5, R2, R4;
FMUL32   R4, R2, R3;
IPA      R7, 10, R12;
IPA      R11, 8, R12;
TEX      R0, 0, 0, 2D;
RCP      R7, R7;
IPA      R10, 7, R12;
FMUL32   R3, R7, R11;
FMUL32   R10, R7, R10;
FMUL32   R6, R6, R1;
FMUL32   R3, R3, R1;
IPA      R1, 9, R12;
FMAD     R5, R5, R0, R6;
FMAD     R3, R10, R0, R3;
FMUL32   R1, R7, R1;
FMAD     R4, R4, R2, R5;
MOV32    R0, R8;
FMAD     R5, R1, R2, R3;
MOV32    R1, R9;
TEX      R4, 2, 2, 2D;
TEX      R0, 5, 5, 2D;
FMUL32   R3, R4, c[0];
FMUL32   R5, R5, c[1];
FMUL32   R6, R6, c[2];
FMUL32   R4, R0, R3;
FMUL32   R5, R1, R5;
FMUL32   R6, R2, R6;
MVI      R1, -1.0;
MVI      R2, -1.0;
IPA      R13, 17, R12;
IPA      R7, 14, R12;
MVI      R0, -1.0;
IPA      R3, 20, R12;
TEX      R8, 3, 3, 2D;
IPA      R15, 18, R12;
IPA      R14, 15, R12;
FMAD     R1, R8, c[0], R1;
FMAD     R2, R9, c[0], R2;
FMAD     R0, R10, c[0], R0;
IPA      R8, 21, R12;
FMUL32   R9, R2, R13;
FMUL32   R13, R2, R15;
IPA      R10, 19, R12;
FMAD     R7, R7, R1, R9;
FMAD     R9, R14, R1, R13;
FMUL32   R2, R2, R10;
FMAD     R3, R3, R0, R7;
FMAD     R9, R8, R0, R9;
IPA      R8, 16, R12;
IPA      R7, 22, R12;
FMUL32   R10, R9, R9;
FMAD     R1, R8, R1, R2;
IPA      R2, 12, R12;
FMAD     R8, R3, R3, R10;
FMAD     R0, R7, R0, R1;
FMUL32   R7, R9, R2;
IPA      R1, 11, R12;
FMAD     R8, R0, R0, R8;
IPA      R10, 13, R12;
FMAD     R7, R3, R1, R7;
RCP      R8, R8;
FMAD     R7, R0, R10, R7;
FMUL32   R12, R8, R7;
FMAD     R7, R8, R7, R12;
FMAD     R1, R7, R3, -R1;
FMAD     R2, R7, R9, -R2;
FMAD     R3, R7, R0, -R10;
FMAX     R0, |R1|, |R2|;
FMAX     R0, |R3|, R0;
RCP      R7, R0;
FMUL32   R0, R1, R7;
FMUL32   R1, R2, R7;
FMUL32   R2, R3, R7;
TEX      R0, 4, 4, LWBE;
FMUL32   R0, R11, R0;
FMUL32   R1, R11, R1;
FMUL32   R0, R0, c[4];
FMUL32   R2, R11, R2;
FMUL32   R3, R1, c[5];
FMAD     R1, R0, R0, -R0;
FMUL32   R2, R2, c[6];
FMAD     R7, R3, R3, -R3;
FMAD     R0, R1, c[8], R0;
FMAD     R1, R2, R2, -R2;
FMAD     R3, R7, c[9], R3;
FMAD     R1, R1, c[10], R2;
FMUL32I  R2, R3, 0.333333;
FMAD     R2, R0, c[0], R2;
FMAD     R2, R1, c[0], R2;
FADD32   R0, R0, -R2;
FADD32   R3, R3, -R2;
FADD32   R1, R1, -R2;
FMAD     R0, R0, c[12], R2;
FMAD     R3, R3, c[13], R2;
FMAD     R2, R1, c[14], R2;
FMAD     R0, R4, c[0], R0;
FMAD     R1, R5, c[0], R3;
FMAD     R2, R6, c[0], R2;
MVI      R3, 1.0;
END
# 111 instructions, 16 R-regs, 23 interpolants
# 111 inst, (4 mov, 4 mvi, 5 tex, 23 ipa, 5 complex, 70 math)
#    74 64-bit, 36 32-bit, 1 32-bit-const
