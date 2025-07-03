!!SPA1.0
.CONST_MODE  PAGE
.THREAD_TYPE PIXEL
.MAX_REG     19
.MAX_ATTR    26
# parseasm build date Mar 10 2004 15:40:49
# -profile fp50 -po tbat3 -po lat3 -if ps2x -i allprogs-new32//p373-lw40.s -o allprogs-new32//p373-lw50.s
#vendor LWPU
#version parseasm.0.0
#profile fp50
#program fp30entry
#semantic C85f7kkc64o5fe.C85f7kkc64o5fe
#semantic Cmur1hd5g8ngo3.Cmur1hd5g8ngo3
#semantic Cee994ndhnk1rc.Cee994ndhnk1rc
#semantic Clofsu13ov1o1e.Clofsu13ov1o1e
#semantic C6r0tbn6evb4u3.C6r0tbn6evb4u3
#semantic C144q3o24uukb6.C144q3o24uukb6
#semantic <null atom>
#semantic <null atom>
#semantic <null atom>
#semantic <null atom>
#semantic <null atom>
#semantic <null atom>
#var float4 o[COLH0] : $vout.O : O[0] : -1 : 0
#var float4 C85f7kkc64o5fe :  : c[6] : -1 : 0
#var float4 f[COL0] : $vin.F : F[0] : -1 : 0
#var float4 Cmur1hd5g8ngo3 :  : c[4] : -1 : 0
#var float4 Cee994ndhnk1rc :  : c[5] : -1 : 0
#var float4 Clofsu13ov1o1e :  : c[3] : -1 : 0
#var float4 C6r0tbn6evb4u3 :  : c[2] : -1 : 0
#var float4 C144q3o24uukb6 :  : c[0] : -1 : 0
#var float4 f[TEX0] : $vin.F : F[0] : -1 : 0
#var sampler2D  : texunit 0 : -1 : 0
#var sampler2D  : texunit 1 : -1 : 0
#var sampler2D  : texunit 1 : -1 : 0
#var sampler2D  : texunit 1 : -1 : 0
#var samplerLWBE  : texunit 2 : -1 : 0
#var float4 f[TEX3] : $vin.F : F[0] : -1 : 0
#var float4 f[TEX2] : $vin.F : F[0] : -1 : 0
#var float4 f[TEX4] : $vin.F : F[0] : -1 : 0
#var float4 f[TEX7] : $vin.F : F[0] : -1 : 0
#var float4 f[TEX5] : $vin.F : F[0] : -1 : 0
#var float4 f[TEX6] : $vin.F : F[0] : -1 : 0
#var float4 f[TEX1] : $vin.F : F[0] : -1 : 0
#var sampler2D  : texunit 4 : -1 : 0
#tram 0 = f[WPOS].w
#tram 1 = f[COL0].x
#tram 2 = f[COL0].y
#tram 3 = f[COL0].z
#tram 4 = f[COL0].w
#tram 5 = f[TEX0].x
#tram 6 = f[TEX0].y
#tram 7 = f[TEX1].x
#tram 8 = f[TEX1].y
#tram 9 = f[TEX2].x
#tram 10 = f[TEX2].y
#tram 11 = f[TEX2].z
#tram 12 = f[TEX2].w
#tram 13 = f[TEX3].z
#tram 14 = f[TEX3].w
#tram 15 = f[TEX4].x
#tram 16 = f[TEX4].y
#tram 17 = f[TEX4].z
#tram 18 = f[TEX5].x
#tram 19 = f[TEX5].y
#tram 20 = f[TEX5].z
#tram 21 = f[TEX6].x
#tram 22 = f[TEX6].y
#tram 23 = f[TEX6].z
#tram 24 = f[TEX7].x
#tram 25 = f[TEX7].y
#tram 26 = f[TEX7].z
BB0:
IPA      R0, 0;
RCP      R12, R0;
IPA      R4, 11, R12;
IPA      R5, 12, R12;
IPA      R0, 9, R12;
IPA      R1, 10, R12;
IPA      R8, 7, R12;
IPA      R9, 8, R12;
MVI      R15, -1.0;
MVI      R14, -1.0;
TEX      R4, 1, 1, 2D;
TEX      R0, 1, 1, 2D;
MVI      R13, -1.0;
TEX      R8, 4, 4, 2D;
FMAD     R7, R8, c[0], R15;
FMAD     R14, R9, c[0], R14;
FMAD     R13, R10, c[0], R13;
FMUL32I  R3, R14, 0.707107;
FMUL32I  R8, R13, 0.57735;
FMAD     R3, R7, c[0], R3;
FMAD.SAT R10, R7, c[0], R8;
IPA      R9, 21, R12;
FMAD.SAT R8, R13, c[0], R3;
IPA      R3, 18, R12;
FMUL32   R15, R14, R9;
FMUL32   R4, R4, R8;
FMUL32   R5, R5, R8;
FMUL32   R6, R6, R8;
FMAD     R8, R10, R0, R4;
FMAD     R9, R10, R1, R5;
FMAD     R10, R10, R2, R6;
FMAD     R3, R3, R7, R15;
IPA      R2, 24, R12;
IPA      R1, 22, R12;
IPA      R0, 19, R12;
FMAD     R4, R2, R13, R3;
FMUL32   R3, R14, R1;
IPA      R1, 25, R12;
IPA      R2, 23, R12;
FMAD     R3, R0, R7, R3;
IPA      R0, 20, R12;
FMUL32   R2, R14, R2;
FMAD     R5, R1, R13, R3;
IPA      R1, 26, R12;
FMAD     R2, R0, R7, R2;
FMUL32   R0, R5, R5;
IPA      R16, 16, R12;
FMAD     R3, R1, R13, R2;
FMAD     R0, R4, R4, R0;
FMUL32   R1, R5, R16;
IPA      R15, 15, R12;
FMAD     R0, R3, R3, R0;
IPA      R6, 17, R12;
FMAD     R1, R4, R15, R1;
RCP      R0, R0;
FMAD     R1, R3, R6, R1;
FMUL32   R2, R0, R1;
FMAD     R2, R0, R1, R2;
FMAD     R0, R2, R4, -R15;
FMAD     R1, R2, R5, -R16;
FMAD     R2, R2, R3, -R6;
FMUL32   R17, R16, R16;
FMAX     R18, |R0|, |R1|;
FMAD     R17, R15, R15, R17;
FMAX     R18, |R2|, R18;
FMAD     R17, R6, R6, R17;
RCP      R18, R18;
LG2      R17, R17;
FMUL32   R0, R0, R18;
FMUL32   R1, R1, R18;
FMUL32   R2, R2, R18;
FMUL32I  R17, R17, -0.5;
RRO      R17, R17, 1;
EX2      R17, R17;
FMUL32   R16, R16, R17;
FMUL32   R15, R15, R17;
FMUL32   R6, R6, R17;
FMUL32   R5, R5, R16;
MOV32    R16, c[19];
FMAD     R5, R4, R15, R5;
MOV32    R4, R16;
FMUL32I  R14, R14, -0.707107;
FMAD     R5, R3, R6, R5;
TEX      R0, 2, 2, LWBE;
FMAD     R6, R7, c[0], R14;
FADD32I  R5, -R5, 1.0;
FMAD.SAT R13, R13, c[0], R6;
FMUL32   R3, R5, R5;
FMUL32   R0, R11, R0;
FMUL32   R1, R11, R1;
FMUL32   R3, R3, R3;
FMUL32   R0, R0, c[0];
FMUL32   R1, R1, c[1];
FMUL32   R3, R5, R3;
FMUL32   R5, R11, R2;
FMAD     R2, R0, R0, -R0;
FMAD     R3, R3, c[23], R4;
FMUL32   R4, R5, c[2];
FMAD     R0, R2, c[8], R0;
FMAD     R5, R1, R1, -R1;
FMAD     R2, R4, R4, -R4;
FMAD     R1, R5, c[9], R1;
FMAD     R4, R2, c[10], R4;
FMUL32I  R2, R1, 0.333333;
FMAD     R2, R0, c[0], R2;
FMAD     R2, R4, c[0], R2;
FADD32   R4, R4, -R2;
FADD32   R1, R1, -R2;
FADD32   R0, R0, -R2;
FMAD     R4, R4, c[14], R2;
FMAD     R1, R1, c[13], R2;
FMAD     R0, R0, c[12], R2;
FMUL32   R17, R4, R3;
FMUL32   R14, R1, R3;
FMUL32   R11, R0, R3;
IPA      R4, 13, R12;
IPA      R5, 14, R12;
IPA      R0, 5, R12;
IPA      R1, 6, R12;
IPA      R16, 3, R12;
TEX      R4, 1, 1, 2D;
TEX      R0, 0, 0, 2D;
IPA      R15, 2, R12;
IPA      R7, 1, R12;
FMAD     R4, R13, R4, R8;
FMAD     R5, R13, R5, R9;
FMAD     R6, R13, R6, R10;
FMUL32   R2, R2, R16;
FMUL32   R1, R1, R15;
FMUL32   R0, R0, R7;
FMUL32   R2, R6, R2;
FMUL32   R1, R5, R1;
FMUL32   R0, R4, R0;
FMAD     R2, R2, c[24], R17;
FMAD     R1, R1, c[24], R14;
FMAD     R0, R0, c[24], R11;
IPA      R4, 4, R12;
FMUL32   R3, R3, R4;
END
# 138 instructions, 20 R-regs, 27 interpolants
# 138 inst, (2 mov, 3 mvi, 6 tex, 27 ipa, 5 complex, 95 math)
#    91 64-bit, 41 32-bit, 6 32-bit-const
