!!SPA1.0
.CONST_MODE  PAGE
.THREAD_TYPE PIXEL
.MAX_REG     15
.MAX_ATTR    17
# parseasm build date Feb  3 2004 15:17:32
# -profile fp50 -po tbat3 -po lat3 -if ps2x -i progs/568-lw40.s -o progs/568-lw50.s
#vendor LWPU
#version parseasm.0.0
#profile lwinst
#program fp30entry
#semantic <null atom>
#semantic <null atom>
#semantic <null atom>
#semantic <null atom>
#semantic <null atom>
#semantic <null atom>
#semantic <null atom>
#var float4 o[COLH0] : $vout.O : O[0] : -1 : 0
#var float4 f[COL0] : $vin.F : F[0] : -1 : 0
#var sampler2D  : texunit 0 : -1 : 0
#var sampler2D  : texunit 6 : -1 : 0
#var sampler2D  : texunit 1 : -1 : 0
#var sampler2D  : texunit 1 : -1 : 0
#var float4 f[TEX1] : $vin.F : F[0] : -1 : 0
#var sampler2D  : texunit 4 : -1 : 0
#var sampler2D  : texunit 1 : -1 : 0
#var sampler2D  : texunit 1 : -1 : 0
#var float4 f[TEX0] : $vin.F : F[0] : -1 : 0
#var float4 f[TEX2] : $vin.F : F[0] : -1 : 0
#var float4 f[TEX3] : $vin.F : F[0] : -1 : 0
#tram 0 = f[WPOS].w
#tram 1 = f[COL0].x
#tram 2 = f[COL0].y
#tram 3 = f[COL0].z
#tram 4 = f[TEX0].x
#tram 5 = f[TEX0].y
#tram 6 = f[TEX0].z
#tram 7 = f[TEX0].w
#tram 8 = f[TEX1].x
#tram 9 = f[TEX1].y
#tram 10 = f[TEX2].x
#tram 11 = f[TEX2].y
#tram 12 = f[TEX2].z
#tram 13 = f[TEX2].w
#tram 14 = f[TEX3].x
#tram 15 = f[TEX3].y
#tram 16 = f[TEX3].z
#tram 17 = f[TEX3].w
BB0:
IPA      R0, 0;
RCP      R12, R0;
IPA      R4, 12, R12;
IPA      R5, 13, R12;
IPA      R0, 10, R12;
IPA      R1, 11, R12;
IPA      R8, 8, R12;
IPA      R9, 9, R12;
TEX      R4, 1, 1, 2D;
TEX      R0, 1, 1, 2D;
TEX      R8, 4, 4, 2D;
MVI      R13, -1.0;
FMUL     R4, R7, R4;
FMUL     R11, R7, R5;
FMUL     R6, R7, R6;
FMAD     R5, R8, c[0], R13;
FMUL     R4, R4, 16.0;
FMUL     R11, R11, 16.0;
FMUL     R8, R6, 16.0;
MVI      R7, -1.0;
MVI      R6, -1.0;
FMAD     R9, R9, c[0], R7;
FMAD     R7, R10, c[0], R6;
FMUL     R10, R9, 0.707107;
FMUL     R6, R3, R2;
FMAD     R2, R5, c[0], R10;
FMUL     R10, R3, R1;
FMUL     R6, R6, 16.0;
FMAD.SAT R1, R7, c[0], R2;
FMUL     R2, R10, 16.0;
FMUL     R0, R3, R0;
FMUL     R8, R8, R1;
FMUL     R10, R11, R1;
FMUL     R11, R4, R1;
FMUL     R1, R0, 16.0;
FMUL     R0, R9, -0.707107;
FMUL     R9, R7, 0.57735;
IPA      R4, 6, R12;
FMAD     R0, R5, c[0], R0;
FMAD.SAT R9, R5, c[0], R9;
IPA      R5, 7, R12;
FMAD.SAT R0, R7, c[0], R0;
FMAD     R1, R9, R1, R11;
FMAD     R2, R9, R2, R10;
FMAD     R13, R9, R6, R8;
IPA      R8, 14, R12;
IPA      R9, 15, R12;
TEX      R4, 6, 6, 2D;
TEX      R8, 1, 1, 2D;
FMUL     R7, R11, R8;
FMUL     R8, R11, R9;
FMUL     R9, R11, R10;
FMUL     R7, R7, 16.0;
FMUL     R10, R8, 16.0;
FMUL     R11, R9, 16.0;
FMUL     R8, R4, R7;
FMUL     R9, R5, R10;
FMUL     R10, R6, R11;
IPA      R4, 16, R12;
IPA      R5, 17, R12;
TEX      R4, 1, 1, 2D;
FMUL     R6, R7, R6;
FMUL     R5, R7, R5;
FMUL     R4, R7, R4;
FMUL     R6, R6, 16.0;
FMUL     R5, R5, 16.0;
FMUL     R4, R4, 16.0;
FMAD     R11, R0, R6, R13;
FMAD     R2, R0, R5, R2;
FMAD     R0, R0, R4, R1;
IPA      R4, 4, R12;
IPA      R5, 5, R12;
IPA      R1, 1, R12;
IPA      R13, 2, R12;
TEX      R4, 0, 0, 2D;
FMUL     R1, R4, R1;
FMUL     R4, R5, R13;
IPA      R5, 3, R12;
FMAD     R0, R0, R1, -R8;
FMAD     R1, R2, R4, -R9;
FMUL     R4, R6, R5;
FMAD     R2, R3, R0, R8;
FMAD     R1, R3, R1, R9;
FMAD     R5, R11, R4, -R10;
MOV      R0, R2;
FMUL     R4, R1, 0.59;
FMAD     R3, R3, R5, R10;
FMAD     R4, R2, c[0], R4;
MOV      R2, R3;
FMAD     R3, R3, c[0], R4;
FMUL     R3, R3, 0.0625;
END
# 91 instructions, 16 R-regs, 18 interpolants
# 91 inst, (2 mov, 3 mvi, 7 tex, 18 ipa, 1 complex, 60 math)
#    48 64-bit, 23 32-bit, 20 32-bit-const
