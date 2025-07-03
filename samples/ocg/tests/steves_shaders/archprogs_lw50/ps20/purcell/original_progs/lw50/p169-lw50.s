!!SPA1.0
.CONST_MODE  PAGE
.THREAD_TYPE PIXEL
.MAX_REG     15
.MAX_ATTR    16
# parseasm build date Mar 10 2004 15:40:49
# -profile fp50 -po tbat3 -po lat3 -if ps2x -i allprogs-new32//p169-lw40.s -o allprogs-new32//p169-lw50.s
#vendor LWPU
#version parseasm.0.0
#profile fp50
#program fp30entry
#semantic <null atom>
#semantic <null atom>
#semantic <null atom>
#semantic <null atom>
#semantic <null atom>
#semantic <null atom>
#var float4 o[COLH0] : $vout.O : O[0] : -1 : 0
#var float4 f[COL0] : $vin.F : F[0] : -1 : 0
#var float4 f[TEX5] : $vin.F : F[0] : -1 : 0
#var sampler2D  : texunit 5 : -1 : 0
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
#tram 4 = f[COL0].w
#tram 5 = f[TEX0].x
#tram 6 = f[TEX0].y
#tram 7 = f[TEX1].x
#tram 8 = f[TEX1].y
#tram 9 = f[TEX2].x
#tram 10 = f[TEX2].y
#tram 11 = f[TEX3].x
#tram 12 = f[TEX3].y
#tram 13 = f[TEX4].x
#tram 14 = f[TEX4].y
#tram 15 = f[TEX5].x
#tram 16 = f[TEX5].y
BB0:
IPA      R0, 0;
RCP      R12, R0;
IPA      R8, 11, R12;
IPA      R9, 12, R12;
IPA      R0, 13, R12;
IPA      R1, 14, R12;
IPA      R4, 5, R12;
IPA      R5, 6, R12;
MVI      R14, -1.0;
MVI      R13, -1.0;
MVI      R7, -1.0;
TEX      R8, 3, 3, 2D;
MVI      R6, -1.0;
TEX      R0, 4, 4, 2D;
MVI      R11, -1.0;
FMAD     R10, R10, c[0], R14;
FMAD     R8, R8, c[0], R13;
FMAD     R3, R9, c[0], R7;
FMAD     R9, R2, c[0], R6;
FMAD     R11, R0, c[0], R11;
MVI      R2, -1.0;
TEX      R4, 0, 0, 2D;
IPA      R0, 15, R12;
FMAD     R2, R1, c[0], R2;
IPA      R1, 16, R12;
FMUL32   R13, R3, R2;
TEX      R0, 5, 5, 2D;
FMAD     R11, R8, R11, R13;
IPA      R8, 9, R12;
FMAD.SAT R3, R10, R9, R11;
IPA      R9, 10, R12;
FMUL32   R3, R3, R3;
TEX      R8, 2, 2, 2D;
FMUL32   R3, R3, R7;
IPA      R7, 3, R12;
FMUL32   R0, R3, R0;
FMUL32   R4, R4, R8;
FMUL32   R5, R5, R9;
FMUL32   R6, R6, R10;
FMUL32   R1, R3, R1;
FMUL32   R8, R3, R2;
IPA      R3, 2, R12;
IPA      R2, 1, R12;
FMAD     R6, R8, R7, R6;
FMAD     R5, R1, R3, R5;
FMAD     R4, R0, R2, R4;
IPA      R0, 7, R12;
IPA      R1, 8, R12;
TEX      R0, 1, 1, 2D;
FMUL32   R0, R4, R0;
FMUL32   R1, R5, R1;
FMUL32   R2, R6, R2;
FMUL32I  R0, R0, 2.0;
FMUL32I  R1, R1, 2.0;
FMUL32I  R2, R2, 2.0;
MOV32.SAT R0, R0;
MOV32.SAT R1, R1;
MOV32.SAT R2, R2;
F2F.SAT  R0, R0;
F2F.SAT  R1, R1;
F2F.SAT  R2, R2;
IPA      R3, 4, R12;
MOV32.SAT R3, R3;
F2F.SAT  R3, R3;
END
# 64 instructions, 16 R-regs, 17 interpolants
# 64 inst, (4 mov, 6 mvi, 6 tex, 17 ipa, 1 complex, 30 math)
#    45 64-bit, 16 32-bit, 3 32-bit-const
