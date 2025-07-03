!!SPA1.0
.CONST_MODE  PAGE
.THREAD_TYPE PIXEL
.MAX_REG     15
.MAX_ATTR    16
# parseasm build date Mar 10 2004 15:40:49
# -profile fp50 -po tbat3 -po lat3 -if ps2x -i allprogs-new32//p180-lw40.s -o allprogs-new32//p180-lw50.s
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
#var float4 f[COL1] : $vin.F : F[0] : -1 : 0
#var float4 f[TEX4] : $vin.F : F[0] : -1 : 0
#var sampler2D  : texunit 4 : -1 : 0
#var float4 f[TEX3] : $vin.F : F[0] : -1 : 0
#var sampler2D  : texunit 3 : -1 : 0
#var float4 f[TEX2] : $vin.F : F[0] : -1 : 0
#var sampler2D  : texunit 2 : -1 : 0
#var float4 f[TEX1] : $vin.F : F[0] : -1 : 0
#var sampler2D  : texunit 1 : -1 : 0
#var float4 f[TEX0] : $vin.F : F[0] : -1 : 0
#var sampler2D  : texunit 0 : -1 : 0
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
#tram 11 = f[TEX2].x
#tram 12 = f[TEX2].y
#tram 13 = f[TEX3].x
#tram 14 = f[TEX3].y
#tram 15 = f[TEX4].x
#tram 16 = f[TEX4].y
BB0:
IPA      R0, 0;
RCP      R12, R0;
IPA      R0, 13, R12;
IPA      R1, 14, R12;
IPA      R4, 15, R12;
IPA      R5, 16, R12;
IPA      R8, 9, R12;
IPA      R9, 10, R12;
MVI      R14, -1.0;
MVI      R11, -1.0;
MVI      R13, -1.0;
TEX      R0, 3, 3, 2D;
MVI      R10, -1.0;
TEX      R4, 4, 4, 2D;
MVI      R3, -1.0;
FMAD     R7, R2, c[0], R14;
FMAD     R2, R0, c[0], R11;
FMAD     R1, R1, c[0], R13;
FMAD     R6, R6, c[0], R10;
FMAD     R3, R4, c[0], R3;
MVI      R0, -1.0;
TEX      R8, 1, 1, 2D;
FMAD     R4, R5, c[0], R0;
IPA      R0, 11, R12;
FMUL32   R4, R1, R4;
IPA      R1, 12, R12;
FMAD     R4, R2, R3, R4;
TEX      R0, 2, 2, 2D;
IPA      R5, 3, R12;
FMAD.SAT R4, R7, R6, R4;
IPA      R7, 6, R12;
FMUL32   R3, R4, R4;
IPA      R4, 2, R12;
IPA      R6, 5, R12;
FMUL32   R8, R3, R8;
FMUL32   R9, R3, R9;
FMUL32   R3, R3, R10;
FMUL32   R0, R8, R0;
FMUL32   R8, R9, R1;
FMUL32   R3, R3, R2;
IPA      R1, 1, R12;
FMAD     R2, R8, R4, R6;
FMAD     R3, R3, R5, R7;
IPA      R6, 4, R12;
IPA      R4, 7, R12;
IPA      R5, 8, R12;
FMAD     R0, R0, R1, R6;
TEX      R4, 0, 0, 2D;
FMUL32   R0, R0, R4;
FMUL32   R1, R2, R5;
FMUL32   R2, R3, R6;
FMUL32I  R0, R0, 2.0;
FMUL32I  R1, R1, 2.0;
FMUL32I  R2, R2, 2.0;
MOV32.SAT R0, R0;
MOV32.SAT R1, R1;
MOV32.SAT R2, R2;
F2F.SAT  R0, R0;
F2F.SAT  R1, R1;
F2F.SAT  R2, R2;
MOV32.SAT R3, R7;
F2F.SAT  R3, R3;
END
# 62 instructions, 16 R-regs, 17 interpolants
# 62 inst, (4 mov, 6 mvi, 5 tex, 17 ipa, 1 complex, 29 math)
#    44 64-bit, 15 32-bit, 3 32-bit-const
