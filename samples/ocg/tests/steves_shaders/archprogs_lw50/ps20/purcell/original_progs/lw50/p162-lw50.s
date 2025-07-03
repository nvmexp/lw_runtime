!!SPA1.0
.CONST_MODE  PAGE
.THREAD_TYPE PIXEL
.MAX_REG     15
.MAX_ATTR    16
# parseasm build date Mar 10 2004 15:40:49
# -profile fp50 -po tbat3 -po lat3 -if ps2x -i allprogs-new32//p162-lw40.s -o allprogs-new32//p162-lw50.s
#vendor LWPU
#version parseasm.0.0
#profile fp50
#program fp30entry
#semantic <null atom>
#semantic <null atom>
#semantic <null atom>
#semantic C76pv1sbdfq7lf.C76pv1sbdfq7lf
#semantic <null atom>
#semantic <null atom>
#var float4 o[COLH0] : $vout.O : O[0] : -1 : 0
#var float4 f[COL0] : $vin.F : F[0] : -1 : 0
#var sampler2D  : texunit 5 : -1 : 0
#var sampler2D  : texunit 1 : -1 : 0
#var float4 f[TEX0] : $vin.F : F[0] : -1 : 0
#var sampler2D  : texunit 0 : -1 : 0
#var float4 f[TEX1] : $vin.F : F[0] : -1 : 0
#var float4 C76pv1sbdfq7lf :  : c[320] : -1 : 0
#var float4 f[TEX5] : $vin.F : F[0] : -1 : 0
#var sampler2D  : texunit 3 : -1 : 0
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
#tram 15 = f[TEX5].x
#tram 16 = f[TEX5].y
BB0:
IPA      R0, 0;
RCP      R12, R0;
IPA      R0, 15, R12;
IPA      R1, 16, R12;
IPA      R4, 10, R12;
IPA      R5, 11, R12;
MVI      R10, -1.0;
MVI      R9, -1.0;
TEX      R0, 3, 3, 2D;
MVI      R8, -1.0;
TEX      R4, 2, 2, 2D;
IPA      R3, 13, R12;
IPA      R7, 12, R12;
FMAD     R5, R5, c[0], R10;
FMAD     R4, R4, c[0], R9;
FMAD     R6, R6, c[0], R8;
FMUL32   R1, R5, R1;
FMUL32   R5, R5, R3;
IPA      R3, 14, R12;
FMAD     R0, R4, R0, R1;
FMAD     R5, R4, R7, R5;
IPA      R4, 5, R12;
FMAD.SAT R1, R6, R2, R0;
FMAD.SAT R15, R6, R3, R5;
MOV32    R8, R4;
IPA      R5, 6, R12;
MOV32    R0, R15;
MOV32    R9, R5;
TEX      R0, 5, 5, 2D;
TEX      R8, 0, 0, 2D;
TEX      R4, 1, 1, 2D;
IPA      R13, 7, R12;
IPA      R14, 8, R12;
FMUL32   R11, R15, R11;
FMUL32   R10, R15, R10;
FMUL32   R9, R15, R9;
FMUL32   R8, R15, R8;
FMUL32   R15, R14, R14;
IPA      R14, 9, R12;
FMAD     R0, R0, R4, R8;
FMAD     R4, R13, R13, R15;
IPA      R8, 1, R12;
FMAD     R1, R1, R5, R9;
FMAD.SAT R4, R14, R14, R4;
IPA      R5, 2, R12;
FADD32I  R4, -R4, 1.0;
FMUL32   R0, R4, R0;
FMUL32   R1, R4, R1;
FMUL32   R0, R0, R8;
FMAD     R2, R2, R6, R10;
FMUL32   R1, R1, R5;
FMUL32I  R0, R0, 2.0;
FMUL32   R2, R4, R2;
FMUL32I  R1, R1, 2.0;
MOV32.SAT R0, R0;
IPA      R5, 3, R12;
FMAD     R3, R3, R7, R11;
F2F.SAT  R0, R0;
FMUL32   R2, R2, R5;
FMUL32   R3, R4, R3;
MOV32.SAT R1, R1;
FMUL32I  R2, R2, 2.0;
IPA      R4, 4, R12;
F2F.SAT  R1, R1;
MOV32.SAT R2, R2;
FMUL32   R3, R3, R4;
F2F.SAT  R2, R2;
FMUL32I  R3, R3, 2.0;
MOV32.SAT R3, R3;
F2F.SAT  R3, R3;
END
# 70 instructions, 16 R-regs, 17 interpolants
# 70 inst, (7 mov, 3 mvi, 5 tex, 17 ipa, 1 complex, 37 math)
#    43 64-bit, 22 32-bit, 5 32-bit-const
