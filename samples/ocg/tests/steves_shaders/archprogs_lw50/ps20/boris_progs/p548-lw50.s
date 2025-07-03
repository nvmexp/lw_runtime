!!SPA1.0
.CONST_MODE  PAGE
.THREAD_TYPE PIXEL
.MAX_REG     15
.MAX_ATTR    13
# parseasm build date Feb 13 2004 14:20:40
# -profile fp50 -po tbat3 -po lat3 -if ps2x -i progs//p548-lw40.s -o progs//p548-lw50.s
#vendor LWPU
#version parseasm.0.0
#profile fp50
#program fp30entry
#semantic <null atom>
#semantic <null atom>
#semantic <null atom>
#semantic <null atom>
#semantic <null atom>
#var float4 o[COLH0] : $vout.O : O[0] : -1 : 0
#var float4 f[COL0] : $vin.F : F[0] : -1 : 0
#var float4 f[TEX0] : $vin.F : F[0] : -1 : 0
#var sampler2D  : texunit 0 : -1 : 0
#var sampler2D  : texunit 1 : -1 : 0
#var float4 f[TEX1] : $vin.F : F[0] : -1 : 0
#var sampler2D  : texunit 4 : -1 : 0
#var sampler2D  : texunit 1 : -1 : 0
#var float4 f[TEX3] : $vin.F : F[0] : -1 : 0
#var float4 f[TEX2] : $vin.F : F[0] : -1 : 0
#var sampler2D  : texunit 1 : -1 : 0
#tram 0 = f[WPOS].w
#tram 1 = f[COL0].x
#tram 2 = f[COL0].y
#tram 3 = f[COL0].z
#tram 4 = f[TEX0].x
#tram 5 = f[TEX0].y
#tram 6 = f[TEX1].x
#tram 7 = f[TEX1].y
#tram 8 = f[TEX2].x
#tram 9 = f[TEX2].y
#tram 10 = f[TEX2].z
#tram 11 = f[TEX2].w
#tram 12 = f[TEX3].z
#tram 13 = f[TEX3].w
BB0:
IPA      R0, 0;
RCP      R12, R0;
IPA      R4, 10, R12;
IPA      R5, 11, R12;
IPA      R0, 8, R12;
IPA      R1, 9, R12;
IPA      R8, 6, R12;
IPA      R9, 7, R12;
TEX      R4, 1, 1, 2D;
TEX      R0, 1, 1, 2D;
TEX      R8, 4, 4, 2D;
MVI      R13, -1.0;
FMUL     R4, R7, R4;
FMUL     R5, R7, R5;
FMUL     R6, R7, R6;
FMAD     R7, R8, c[0], R13;
FMUL32I  R4, R4, 16.0;
FMUL32I  R5, R5, 16.0;
FMUL32I  R11, R6, 16.0;
MVI      R8, -1.0;
MVI      R6, -1.0;
FMAD     R8, R9, c[0], R8;
FMAD     R6, R10, c[0], R6;
FMUL32I  R10, R8, 0.707107;
FMUL     R9, R3, R2;
FMAD     R2, R7, c[0], R10;
FMUL     R1, R3, R1;
FMUL     R0, R3, R0;
FMAD.SAT R3, R6, c[0], R2;
FMUL32I  R2, R9, 16.0;
FMUL32I  R9, R1, 16.0;
FMUL     R10, R11, R3;
FMUL     R5, R5, R3;
FMUL     R3, R4, R3;
FMUL32I  R4, R0, 16.0;
FMUL32I  R8, R8, -0.707107;
FMUL32I  R1, R6, 0.57735;
IPA      R0, 12, R12;
FMAD     R8, R7, c[0], R8;
FMAD.SAT R7, R7, c[0], R1;
IPA      R1, 13, R12;
FMAD.SAT R6, R6, c[0], R8;
FMAD     R4, R7, R4, R3;
FMAD     R5, R7, R9, R5;
FMAD     R7, R7, R2, R10;
TEX      R0, 1, 1, 2D;
FMUL     R2, R3, R2;
FMUL     R1, R3, R1;
FMUL     R0, R3, R0;
FMUL32I  R2, R2, 16.0;
FMUL32I  R1, R1, 16.0;
FMUL32I  R0, R0, 16.0;
FMAD     R7, R6, R2, R7;
FMAD     R5, R6, R1, R5;
FMAD     R4, R6, R0, R4;
IPA      R0, 4, R12;
IPA      R1, 5, R12;
IPA      R6, 1, R12;
IPA      R8, 2, R12;
TEX      R0, 0, 0, 2D;
IPA      R9, 3, R12;
FMUL     R0, R0, R6;
FMUL     R1, R1, R8;
FMUL     R2, R2, R9;
FMUL     R3, R4, R0;
FMUL     R1, R5, R1;
FMUL     R4, R7, R2;
MOV      R0, R3;
FMUL32I  R5, R1, 0.59;
MOV      R2, R4;
FMAD     R3, R3, c[0], R5;
FMAD     R3, R4, c[0], R3;
FMUL32I  R3, R3, 0.0625;
END
# 73 instructions, 16 R-regs, 14 interpolants
# 73 inst, (2 mov, 3 mvi, 5 tex, 14 ipa, 1 complex, 48 math)
#    59 64-bit, 0 32-bit, 14 32-bit-const
