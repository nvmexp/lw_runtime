!!SPA1.0
.CONST_MODE  PAGE
.THREAD_TYPE PIXEL
.MAX_REG     15
.MAX_ATTR    14
# parseasm build date Feb 13 2004 14:20:40
# -profile fp50 -po tbat3 -po lat3 -if ps2x -i progs//p555-lw40.s -o progs//p555-lw50.s
#vendor LWPU
#version parseasm.0.0
#profile fp50
#program fp30entry
#semantic C85f7kkc64o5fe.C85f7kkc64o5fe
#semantic Coit1m35rlpml7.Coit1m35rlpml7
#semantic Cmur1hd5g8ngo3.Cmur1hd5g8ngo3
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
#var float4 Coit1m35rlpml7 :  : c[1] : -1 : 0
#var float4 Cmur1hd5g8ngo3 :  : c[4] : -1 : 0
#var float4 Clofsu13ov1o1e :  : c[3] : -1 : 0
#var float4 C6r0tbn6evb4u3 :  : c[2] : -1 : 0
#var float4 C144q3o24uukb6 :  : c[0] : -1 : 0
#var float4 f[TEX1] : $vin.F : F[0] : -1 : 0
#var sampler2D  : texunit 3 : -1 : 0
#var float4 f[TEX0] : $vin.F : F[0] : -1 : 0
#var sampler2D  : texunit 0 : -1 : 0
#var float4 f[TEX2] : $vin.F : F[0] : -1 : 0
#var sampler2D  : texunit 1 : -1 : 0
#var float4 f[TEX3] : $vin.F : F[0] : -1 : 0
#var sampler2D  : texunit 4 : -1 : 0
#var samplerLWBE  : texunit 2 : -1 : 0
#var float4 f[TEX5] : $vin.F : F[0] : -1 : 0
#var float4 f[TEX4] : $vin.F : F[0] : -1 : 0
#var samplerLWBE  : texunit 6 : -1 : 0
#tram 0 = f[WPOS].w
#tram 1 = f[TEX0].x
#tram 2 = f[TEX0].y
#tram 3 = f[TEX1].x
#tram 4 = f[TEX1].y
#tram 5 = f[TEX2].x
#tram 6 = f[TEX2].y
#tram 7 = f[TEX3].x
#tram 8 = f[TEX3].y
#tram 9 = f[TEX4].x
#tram 10 = f[TEX4].y
#tram 11 = f[TEX4].z
#tram 12 = f[TEX5].x
#tram 13 = f[TEX5].y
#tram 14 = f[TEX5].z
BB0:
IPA      R0, 0;
RCP      R12, R0;
IPA      R0, 1, R12;
IPA      R1, 2, R12;
IPA      R4, 3, R12;
IPA      R5, 4, R12;
TEX      R0, 0, 0, 2D;
IPA      R11, 9, R12;
IPA      R10, 10, R12;
IPA      R8, 11, R12;
FMAX     R9, |R11|, |R10|;
TEX      R4, 3, 3, 2D;
FMAX     R9, |R8|, R9;
RCP      R9, R9;
FMUL     R3, R3, R7;
FMUL     R2, R2, R6;
FMUL     R1, R1, R5;
FMUL     R0, R0, R4;
FMUL     R4, R11, R9;
FMUL     R5, R10, R9;
FMUL     R6, R8, R9;
MVI      R9, -1.0;
MVI      R11, -1.0;
TEX      R4, 6, 6, LWBE;
IPA      R10, 13, R12;
IPA      R8, 12, R12;
MVI      R13, -1.0;
IPA      R7, 14, R12;
FMAD     R4, R4, c[0], R9;
FMAD     R5, R5, c[0], R11;
FMAD     R6, R6, c[0], R13;
FMUL     R11, R10, R10;
FMUL     R9, R10, R5;
FMAD     R11, R8, R8, R11;
FMAD     R9, R8, R4, R9;
FMAD     R11, R7, R7, R11;
FMAD     R9, R7, R6, R9;
FMUL     R4, R11, R4;
FMUL     R5, R11, R5;
FMUL     R11, R11, R6;
FADD     R6, R9, R9;
FMAD     R4, R6, R8, -R4;
FMAD     R5, R6, R10, -R5;
FMAD     R6, R6, R7, -R11;
FMAX     R7, |R4|, |R5|;
FADD32I  R9, -R9, 1.0;
FMAX     R8, |R6|, R7;
FMUL     R7, R9, R9;
RCP      R8, R8;
FMUL     R10, R7, R7;
MOV      R7, c[18];
FMUL     R4, R4, R8;
FMUL     R9, R9, R10;
FMUL     R5, R5, R8;
FMUL     R6, R6, R8;
FMAD     R13, R9, R7, c[19];
IPA      R8, 7, R12;
TEX      R4, 2, 2, LWBE;
IPA      R9, 8, R12;
TEX      R8, 4, 4, 2D;
FMUL     R4, R4, R8;
FMUL     R5, R5, R9;
FMUL     R4, R4, c[0];
FMUL     R6, R6, R10;
FMUL     R7, R5, c[1];
FMAD     R5, R4, R4, -R4;
FMUL     R6, R6, c[2];
FMAD     R8, R7, R7, -R7;
FMAD     R4, R5, c[8], R4;
FMAD     R5, R6, R6, -R6;
FMAD     R7, R8, c[9], R7;
FMAD     R6, R5, c[10], R6;
FMUL32I  R5, R7, 0.333333;
FMAD     R5, R4, c[0], R5;
FMAD     R5, R6, c[0], R5;
FADD     R6, R6, -R5;
FADD     R7, R7, -R5;
FADD     R4, R4, -R5;
FMAD     R8, R6, c[14], R5;
FMAD     R6, R7, c[13], R5;
FMAD     R4, R4, c[12], R5;
FMUL     R8, R8, R13;
FMUL     R9, R6, R13;
FMUL     R10, R4, R13;
IPA      R4, 5, R12;
IPA      R5, 6, R12;
MOV      R11, c[26];
TEX      R4, 1, 1, 2D;
MOV      R12, c[26];
FMUL     R4, R4, c[4];
FMUL     R5, R5, c[5];
FMUL     R6, R6, c[6];
FMUL     R4, R4, c[24];
FMUL     R7, R5, c[24];
FMUL     R5, R6, c[24];
FMAD     R4, R11, R4, c[25];
FMAD     R7, R12, R7, c[25];
MOV      R6, c[26];
FMAD     R0, R0, R4, R10;
FMAD     R1, R1, R7, R9;
FMAD     R4, R6, R5, c[25];
FMAD     R2, R2, R4, R8;
END
# 102 instructions, 16 R-regs, 15 interpolants
# 102 inst, (4 mov, 3 mvi, 6 tex, 15 ipa, 3 complex, 71 math)
#    100 64-bit, 0 32-bit, 2 32-bit-const
