!!SPA1.0
.CONST_MODE  PAGE
.THREAD_TYPE PIXEL
.MAX_REG     15
.MAX_ATTR    14
# parseasm build date Mar 10 2004 15:40:49
# -profile fp50 -po tbat3 -po lat3 -if ps2x -i allprogs-new32//p160-lw40.s -o allprogs-new32//p160-lw50.s
#vendor LWPU
#version parseasm.0.0
#profile fp50
#program fp30entry
#semantic <null atom>
#semantic <null atom>
#var float4 o[COLH0] : $vout.O : O[0] : -1 : 0
#var float4 f[COL0] : $vin.F : F[0] : -1 : 0
#var float4 f[TEX0] : $vin.F : F[0] : -1 : 0
#var sampler2D  : texunit 0 : -1 : 0
#var float4 f[TEX1] : $vin.F : F[0] : -1 : 0
#var float4 f[TEX4] : $vin.F : F[0] : -1 : 0
#var float4 f[TEX2] : $vin.F : F[0] : -1 : 0
#var sampler2D  : texunit 2 : -1 : 0
#tram 0 = f[WPOS].w
#tram 1 = f[COL0].x
#tram 2 = f[COL0].y
#tram 3 = f[COL0].z
#tram 4 = f[COL0].w
#tram 5 = f[TEX0].x
#tram 6 = f[TEX0].y
#tram 7 = f[TEX1].x
#tram 8 = f[TEX1].y
#tram 9 = f[TEX1].z
#tram 10 = f[TEX2].x
#tram 11 = f[TEX2].y
#tram 12 = f[TEX4].x
#tram 13 = f[TEX4].y
#tram 14 = f[TEX4].z
BB0:
IPA      R0, 0;
RCP      R8, R0;
MVI      R11, -1.0;
IPA      R0, 10, R8;
IPA      R1, 11, R8;
IPA      R4, 5, R8;
IPA      R5, 6, R8;
MVI      R13, -1.0;
IPA      R14, 13, R8;
IPA      R12, 12, R8;
MVI      R9, -1.0;
TEX      R0, 2, 2, 2D;
IPA      R10, 14, R8;
TEX      R4, 0, 0, 2D;
FMAD     R0, R0, c[0], R11;
FMAD     R1, R1, c[0], R13;
FMAD     R3, R2, c[0], R9;
IPA      R2, 7, R8;
FMUL32   R11, R1, R14;
IPA      R9, 8, R8;
IPA      R1, 9, R8;
FMAD     R0, R0, R12, R11;
FMUL32   R9, R9, R9;
IPA      R11, 1, R8;
FMAD.SAT R0, R3, R10, R0;
FMAD     R2, R2, R2, R9;
FMUL32   R3, R0, R4;
FMAD.SAT R1, R1, R1, R2;
FMUL32   R4, R0, R5;
IPA      R5, 2, R8;
FADD32I  R2, -R1, 1.0;
FMUL32   R1, R2, R3;
FMUL32   R3, R2, R4;
FMUL32   R1, R1, R11;
FMUL32   R3, R3, R5;
FMUL32   R4, R0, R6;
FMUL32I  R1, R1, 2.0;
FMUL32I  R5, R3, 2.0;
FMUL32   R3, R2, R4;
MOV32.SAT R1, R1;
MOV32.SAT R4, R5;
FMUL32   R5, R0, R7;
F2F.SAT  R0, R1;
F2F.SAT  R1, R4;
FMUL32   R4, R2, R5;
IPA      R2, 3, R8;
IPA      R5, 4, R8;
FMUL32   R2, R3, R2;
FMUL32   R3, R4, R5;
FMUL32I  R2, R2, 2.0;
FMUL32I  R3, R3, 2.0;
MOV32.SAT R2, R2;
MOV32.SAT R3, R3;
F2F.SAT  R2, R2;
F2F.SAT  R3, R3;
END
# 55 instructions, 16 R-regs, 15 interpolants
# 55 inst, (4 mov, 3 mvi, 2 tex, 15 ipa, 1 complex, 30 math)
#    32 64-bit, 18 32-bit, 5 32-bit-const
