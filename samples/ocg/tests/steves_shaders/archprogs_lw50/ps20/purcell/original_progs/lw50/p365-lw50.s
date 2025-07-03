!!SPA1.0
.CONST_MODE  PAGE
.THREAD_TYPE PIXEL
.MAX_REG     15
.MAX_ATTR    30
# parseasm build date Mar 10 2004 15:40:49
# -profile fp50 -po tbat3 -po lat3 -if ps2x -i allprogs-new32//p365-lw40.s -o allprogs-new32//p365-lw50.s
#vendor LWPU
#version parseasm.0.0
#profile fp50
#program fp30entry
#semantic <null atom>
#semantic <null atom>
#semantic <null atom>
#semantic <null atom>
#semantic <null atom>
#semantic C144q3o24uukb6.C144q3o24uukb6
#semantic Coit1m35rlpml7.Coit1m35rlpml7
#semantic <null atom>
#semantic <null atom>
#semantic <null atom>
#semantic <null atom>
#semantic <null atom>
#semantic <null atom>
#semantic <null atom>
#semantic <null atom>
#semantic <null atom>
#semantic <null atom>
#var float4 o[COLH0] : $vout.O : O[0] : -1 : 0
#var sampler2D  : texunit 0 : -1 : 0
#var sampler2D  : texunit 0 : -1 : 0
#var sampler2D  : texunit 0 : -1 : 0
#var sampler2D  : texunit 0 : -1 : 0
#var sampler2D  : texunit 0 : -1 : 0
#var float4 f[TEX7] : $vin.F : F[0] : -1 : 0
#var float4 f[TEX6] : $vin.F : F[0] : -1 : 0
#var float4 C144q3o24uukb6 :  : c[0] : -1 : 0
#var float4 Coit1m35rlpml7 :  : c[1] : -1 : 0
#var float4 f[TEX5] : $vin.F : F[0] : -1 : 0
#var sampler2D  : texunit 0 : -1 : 0
#var sampler2D  : texunit 0 : -1 : 0
#var sampler2D  : texunit 0 : -1 : 0
#var sampler2D  : texunit 0 : -1 : 0
#var sampler2D  : texunit 0 : -1 : 0
#var sampler2D  : texunit 0 : -1 : 0
#var sampler2D  : texunit 0 : -1 : 0
#var sampler2D  : texunit 0 : -1 : 0
#var sampler2D  : texunit 0 : -1 : 0
#var float4 f[TEX0] : $vin.F : F[0] : -1 : 0
#var sampler2D  : texunit 0 : -1 : 0
#var float4 f[TEX4] : $vin.F : F[0] : -1 : 0
#var float4 f[TEX3] : $vin.F : F[0] : -1 : 0
#var float4 f[TEX2] : $vin.F : F[0] : -1 : 0
#var float4 f[TEX1] : $vin.F : F[0] : -1 : 0
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
#tram 11 = f[TEX3].x
#tram 12 = f[TEX3].y
#tram 13 = f[TEX3].z
#tram 14 = f[TEX3].w
#tram 15 = f[TEX4].x
#tram 16 = f[TEX4].y
#tram 17 = f[TEX4].z
#tram 18 = f[TEX4].w
#tram 19 = f[TEX5].x
#tram 20 = f[TEX5].y
#tram 21 = f[TEX5].z
#tram 22 = f[TEX5].w
#tram 23 = f[TEX6].x
#tram 24 = f[TEX6].y
#tram 25 = f[TEX6].z
#tram 26 = f[TEX6].w
#tram 27 = f[TEX7].x
#tram 28 = f[TEX7].y
#tram 29 = f[TEX7].z
#tram 30 = f[TEX7].w
BB0:
IPA      R0, 0;
RCP      R12, R0;
IPA      R4, 3, R12;
IPA      R5, 4, R12;
IPA      R8, 1, R12;
IPA      R9, 2, R12;
IPA      R0, 6, R12;
IPA      R1, 5, R12;
TEX      R4, 0, 0, 2D;
TEX      R8, 0, 0, 2D;
TEX      R0, 0, 0, 2D;
FMUL32   R8, R8, c[7];
FMUL32   R9, R9, c[7];
FMUL32   R10, R10, c[7];
FMAD     R8, R4, c[0], R8;
FMAD     R5, R5, c[0], R9;
FMAD     R6, R6, c[0], R10;
FMUL32   R9, R11, c[7];
IPA      R4, 7, R12;
FMAD     R7, R7, c[0], R9;
FMAD     R8, R0, c[0], R8;
FMAD     R9, R1, c[0], R5;
FMAD     R10, R2, c[0], R6;
FMAD     R11, R3, c[0], R7;
IPA      R5, 8, R12;
IPA      R0, 10, R12;
IPA      R1, 9, R12;
TEX      R4, 0, 0, 2D;
TEX      R0, 0, 0, 2D;
FMAD     R7, R7, c[1], R11;
FMAD     R6, R6, c[1], R10;
FMAD     R5, R5, c[1], R9;
FMAD     R4, R4, c[1], R8;
FMAD     R8, R2, c[1], R6;
FMAD     R9, R1, c[1], R5;
FMAD     R10, R0, c[1], R4;
FMAD     R11, R3, c[1], R7;
IPA      R4, 11, R12;
IPA      R5, 12, R12;
IPA      R0, 14, R12;
IPA      R1, 13, R12;
TEX      R4, 0, 0, 2D;
TEX      R0, 0, 0, 2D;
FMAD     R7, R7, c[2], R11;
FMAD     R6, R6, c[2], R8;
FMAD     R5, R5, c[2], R9;
FMAD     R4, R4, c[2], R10;
FMAD     R8, R2, c[2], R6;
FMAD     R9, R1, c[2], R5;
FMAD     R10, R0, c[2], R4;
FMAD     R11, R3, c[2], R7;
IPA      R4, 15, R12;
IPA      R5, 16, R12;
IPA      R0, 18, R12;
IPA      R1, 17, R12;
TEX      R4, 0, 0, 2D;
TEX      R0, 0, 0, 2D;
FMAD     R7, R7, c[3], R11;
FMAD     R6, R6, c[3], R8;
FMAD     R5, R5, c[3], R9;
FMAD     R4, R4, c[3], R10;
FMAD     R8, R2, c[3], R6;
FMAD     R9, R1, c[3], R5;
FMAD     R10, R0, c[3], R4;
FMAD     R11, R3, c[3], R7;
IPA      R4, 19, R12;
IPA      R5, 20, R12;
IPA      R0, 22, R12;
IPA      R1, 21, R12;
TEX      R4, 0, 0, 2D;
TEX      R0, 0, 0, 2D;
FMAD     R7, R7, c[4], R11;
FMAD     R6, R6, c[4], R8;
FMAD     R5, R5, c[4], R9;
FMAD     R4, R4, c[4], R10;
FMAD     R8, R2, c[4], R6;
FMAD     R9, R1, c[4], R5;
FMAD     R10, R0, c[4], R4;
FMAD     R11, R3, c[4], R7;
IPA      R4, 23, R12;
IPA      R5, 24, R12;
IPA      R0, 26, R12;
IPA      R1, 25, R12;
TEX      R4, 0, 0, 2D;
TEX      R0, 0, 0, 2D;
FMAD     R7, R7, c[5], R11;
FMAD     R6, R6, c[5], R8;
FMAD     R5, R5, c[5], R9;
FMAD     R4, R4, c[5], R10;
FMAD     R8, R2, c[5], R6;
FMAD     R9, R1, c[5], R5;
FMAD     R10, R0, c[5], R4;
FMAD     R11, R3, c[5], R7;
IPA      R4, 27, R12;
IPA      R5, 28, R12;
IPA      R0, 30, R12;
IPA      R1, 29, R12;
TEX      R4, 0, 0, 2D;
TEX      R0, 0, 0, 2D;
FMAD     R7, R7, c[6], R11;
FMAD     R6, R6, c[6], R8;
FMAD     R5, R5, c[6], R9;
FMAD     R4, R4, c[6], R10;
FMAD     R2, R2, c[6], R6;
FMAD     R1, R1, c[6], R5;
FMAD     R0, R0, c[6], R4;
FMAD     R3, R3, c[6], R7;
END
# 107 instructions, 16 R-regs, 31 interpolants
# 107 inst, (0 mov, 0 mvi, 15 tex, 31 ipa, 1 complex, 60 math)
#    103 64-bit, 4 32-bit, 0 32-bit-const
