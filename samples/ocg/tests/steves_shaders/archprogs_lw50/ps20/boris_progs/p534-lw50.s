!!SPA1.0
.CONST_MODE  PAGE
.THREAD_TYPE PIXEL
.MAX_REG     27
.MAX_ATTR    17
# parseasm build date Feb 13 2004 14:20:40
# -profile fp50 -po tbat3 -po lat3 -if ps2x -i progs//p534-lw40.s -o progs//p534-lw50.s
#vendor LWPU
#version parseasm.0.0
#profile fp50
#program fp30entry
#semantic Coit1m35rlpml7.Coit1m35rlpml7
#semantic Clofsu13ov1o1e.Clofsu13ov1o1e
#semantic C6r0tbn6evb4u3.C6r0tbn6evb4u3
#semantic C144q3o24uukb6.C144q3o24uukb6
#semantic <null atom>
#semantic <null atom>
#semantic <null atom>
#semantic <null atom>
#semantic <null atom>
#semantic <null atom>
#semantic Cee994ndhnk1rc.Cee994ndhnk1rc
#semantic <null atom>
#var float4 o[COLH0] : $vout.O : O[0] : -1 : 0
#var float4 Coit1m35rlpml7 :  : c[1] : -1 : 0
#var float4 Clofsu13ov1o1e :  : c[3] : -1 : 0
#var float4 C6r0tbn6evb4u3 :  : c[2] : -1 : 0
#var float4 C144q3o24uukb6 :  : c[0] : -1 : 0
#var sampler2D  : texunit 2 : -1 : 0
#var sampler2D  : texunit 2 : -1 : 0
#var sampler2D  : texunit 2 : -1 : 0
#var sampler2D  : texunit 2 : -1 : 0
#var sampler2D  : texunit 5 : -1 : 0
#var samplerLWBE  : texunit 4 : -1 : 0
#var float4 Cee994ndhnk1rc :  : c[5] : -1 : 0
#var float4 f[TEX5] : $vin.F : F[0] : -1 : 0
#var float4 f[TEX1] : $vin.F : F[0] : -1 : 0
#var float4 f[TEX4] : $vin.F : F[0] : -1 : 0
#var float4 f[TEX2] : $vin.F : F[0] : -1 : 0
#var float4 f[TEX3] : $vin.F : F[0] : -1 : 0
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
#tram 11 = f[TEX3].z
#tram 12 = f[TEX4].x
#tram 13 = f[TEX4].y
#tram 14 = f[TEX4].z
#tram 15 = f[TEX5].x
#tram 16 = f[TEX5].y
#tram 17 = f[TEX5].z
BB0:
IPA      R0, 0;
MVI      R3, -1.0;
RCP      R2, R0;
IPA      R24, 1, R2;
IPA      R25, 2, R2;
IPA      R0, 17, R2;
MOV      R20, R24;
MOV      R21, R25;
RCP      R0, R0;
IPA      R1, 15, R2;
MVI      R4, -1.0;
TEX      R20, 3, 3, 2D;
FMUL     R1, R0, R1;
IPA      R5, 16, R2;
FMUL     R0, R0, R5;
FMAD     R3, R20, c[0], R3;
FMAD     R6, R21, c[0], R4;
FMUL     R5, R23, c[20];
IPA      R8, 9, R2;
IPA      R7, 6, R2;
FMAD     R4, R3, R5, R1;
FMAD     R5, R6, R5, R0;
FMUL     R8, R6, R8;
FADD32I  R0, R4, 0.001953;
FADD32I  R1, R5, 0.001953;
FMAD     R9, R7, R3, R8;
MVI      R8, -1.0;
IPA      R7, 12, R2;
IPA      R10, 10, R2;
FMAD     R14, R22, c[0], R8;
IPA      R8, 7, R2;
FMUL     R10, R6, R10;
FMAD     R9, R7, R14, R9;
IPA      R7, 13, R2;
FMAD     R8, R8, R3, R10;
IPA      R11, 4, R2;
IPA      R10, 3, R2;
FMAD     R7, R7, R14, R8;
IPA      R13, 11, R2;
IPA      R8, 8, R2;
FMUL     R12, R7, R11;
FMUL     R13, R6, R13;
IPA      R6, 14, R2;
FMAD     R12, R9, R10, R12;
FMAD     R3, R8, R3, R13;
IPA      R8, 5, R2;
FMUL     R13, R7, R7;
FMAD     R6, R6, R14, R3;
TEX      R0, 2, 2, 2D;
FMAD     R13, R9, R9, R13;
FMAD     R12, R6, R8, R12;
FMAD     R13, R6, R6, R13;
FADD     R3, R12, R12;
FMUL     R10, R13, R10;
FMUL     R12, R13, R11;
FMUL     R8, R13, R8;
FMAD     R11, R3, R9, -R10;
FMAD     R10, R3, R7, -R12;
FMAD     R6, R3, R6, -R8;
FADD32I  R12, R4, -0.000977;
FMAX     R3, |R11|, |R10|;
FADD32I  R13, R5, 0.001953;
FADD32I  R8, R4, 0.001953;
FMAX     R3, |R6|, R3;
FADD32I  R4, R4, -0.000977;
FADD32I  R9, R5, -0.000977;
RCP      R7, R3;
FADD32I  R5, R5, -0.000977;
MOV      R3, R23;
FMUL     R16, R11, R7;
FMUL     R17, R10, R7;
FMUL     R18, R6, R7;
TEX      R4, 2, 2, 2D;
TEX      R8, 2, 2, 2D;
TEX      R12, 2, 2, 2D;
FMUL32I  R7, R8, 0.222222;
FMUL32I  R8, R9, 0.222222;
FMAD     R4, R4, c[0], R7;
FMUL32I  R7, R10, 0.222222;
FMAD     R5, R5, c[0], R8;
FMAD     R4, R12, c[0], R4;
FMAD     R6, R6, c[0], R7;
FMAD     R5, R13, c[0], R5;
FMAD     R0, R0, c[0], R4;
FMAD     R4, R14, c[0], R6;
FMAD     R1, R1, c[0], R5;
TEX      R16, 4, 4, LWBE;
FMAD     R2, R2, c[0], R4;
FMUL     R4, R23, R16;
FMUL     R5, R23, R17;
FMUL     R4, R4, c[0];
FMUL     R7, R23, R18;
FMUL     R6, R5, c[1];
FMAD     R8, R4, R4, -R4;
FMUL     R5, R7, c[2];
FMAD     R7, R6, R6, -R6;
FMAD     R8, R8, c[8], R4;
FMAD     R4, R5, R5, -R5;
FMAD     R10, R7, c[9], R6;
FMAD     R9, R4, c[10], R5;
FMUL32I  R4, R10, 0.333333;
FMAD     R6, R8, c[0], R4;
MOV      R4, R24;
MOV      R5, R25;
FMAD     R11, R9, c[0], R6;
TEX      R4, 5, 5, 2D;
FADD     R12, R9, -R11;
FADD     R9, R10, -R11;
FMAD     R7, R12, c[14], R11;
FMAD     R9, R9, c[13], R11;
FADD     R10, R8, -R11;
MOV      R8, c[4];
MOV      R12, c[5];
FMAD     R10, R10, c[12], R11;
FADD     R8, R8, c[4];
FADD     R11, R12, c[5];
MOV      R12, c[6];
FMUL     R4, R4, R8;
FMUL     R5, R5, R11;
FADD     R8, R12, c[6];
FMAD     R0, R0, R4, R10;
FMAD     R1, R1, R5, R9;
FMUL     R4, R6, R8;
FMAD     R2, R2, R4, R7;
END
# 124 instructions, 28 R-regs, 18 interpolants
# 124 inst, (8 mov, 3 mvi, 7 tex, 18 ipa, 3 complex, 85 math)
#    112 64-bit, 0 32-bit, 12 32-bit-const
