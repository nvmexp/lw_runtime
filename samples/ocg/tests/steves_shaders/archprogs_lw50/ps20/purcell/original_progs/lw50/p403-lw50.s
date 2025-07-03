!!SPA1.0
.CONST_MODE  PAGE
.THREAD_TYPE PIXEL
.MAX_REG     15
.MAX_ATTR    14
# parseasm build date Mar 10 2004 15:40:49
# -profile fp50 -po tbat3 -po lat3 -if ps2x -i allprogs-new32//p403-lw40.s -o allprogs-new32//p403-lw50.s
#vendor LWPU
#version parseasm.0.0
#profile fp50
#program fp30entry
#semantic C85f7kkc64o5fe.C85f7kkc64o5fe
#semantic <null atom>
#semantic <null atom>
#semantic <null atom>
#semantic <null atom>
#semantic <null atom>
#var float4 o[COLH0] : $vout.O : O[0] : -1 : 0
#var float4 C85f7kkc64o5fe :  : c[6] : -1 : 0
#var float4 f[COL0] : $vin.F : F[0] : -1 : 0
#var float4 f[TEX0] : $vin.F : F[0] : -1 : 0
#var sampler2D  : texunit 0 : -1 : 0
#var sampler2D  : texunit 1 : -1 : 0
#var sampler2D  : texunit 1 : -1 : 0
#var float4 f[TEX1] : $vin.F : F[0] : -1 : 0
#var sampler2D  : texunit 4 : -1 : 0
#var sampler2D  : texunit 1 : -1 : 0
#var float4 f[TEX3] : $vin.F : F[0] : -1 : 0
#var float4 f[TEX2] : $vin.F : F[0] : -1 : 0
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
BB0:
IPA      R0, 0;
RCP      R12, R0;
IPA      R4, 11, R12;
IPA      R5, 12, R12;
IPA      R0, 9, R12;
IPA      R1, 10, R12;
IPA      R8, 7, R12;
IPA      R9, 8, R12;
MVI      R14, -1.0;
MVI      R15, -1.0;
TEX      R4, 1, 1, 2D;
TEX      R0, 1, 1, 2D;
MVI      R13, -1.0;
TEX      R8, 4, 4, 2D;
FMAD     R7, R8, c[0], R14;
FMAD     R8, R9, c[0], R15;
FMAD     R10, R10, c[0], R13;
FMUL32I  R9, R8, 0.707107;
FMUL32I  R3, R10, 0.57735;
FMAD     R9, R7, c[0], R9;
FMAD.SAT R3, R7, c[0], R3;
FMUL32I  R8, R8, -0.707107;
FMAD.SAT R9, R10, c[0], R9;
FMAD     R7, R7, c[0], R8;
FMUL32   R6, R6, R9;
FMUL32   R5, R5, R9;
FMUL32   R4, R4, R9;
FMAD     R8, R3, R2, R6;
FMAD     R9, R3, R1, R5;
FMAD     R11, R3, R0, R4;
FMAD.SAT R10, R10, c[0], R7;
IPA      R4, 13, R12;
IPA      R5, 14, R12;
IPA      R0, 5, R12;
IPA      R1, 6, R12;
IPA      R13, 1, R12;
TEX      R4, 1, 1, 2D;
TEX      R0, 0, 0, 2D;
IPA      R7, 2, R12;
FMAD     R6, R10, R6, R8;
FMAD     R5, R10, R5, R9;
FMAD     R4, R10, R4, R11;
FMUL32   R0, R0, R13;
FMUL32   R1, R1, R7;
IPA      R7, 3, R12;
FMUL32   R0, R4, R0;
FMUL32   R1, R5, R1;
FMUL32   R2, R2, R7;
FMUL32   R0, R0, c[24];
FMUL32   R1, R1, c[24];
FMUL32   R2, R6, R2;
IPA      R4, 4, R12;
FMUL32   R2, R2, c[24];
FMUL32   R3, R3, R4;
END
# 54 instructions, 16 R-regs, 15 interpolants
# 54 inst, (0 mov, 3 mvi, 5 tex, 15 ipa, 1 complex, 30 math)
#    38 64-bit, 13 32-bit, 3 32-bit-const
