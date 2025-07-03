!!SPA1.0
.CONST_MODE  PAGE
.THREAD_TYPE PIXEL
.MAX_REG     15
.MAX_ATTR    19
# parseasm build date Mar 10 2004 15:40:49
# -profile fp50 -po tbat3 -po lat3 -if ps2x -i allprogs-new32//p158-lw40.s -o allprogs-new32//p158-lw50.s
#vendor LWPU
#version parseasm.0.0
#profile fp50
#program fp30entry
#semantic <null atom>
#semantic <null atom>
#semantic <null atom>
#semantic <null atom>
#semantic C76pv1sbdfq7lf.C76pv1sbdfq7lf
#semantic <null atom>
#semantic <null atom>
#var float4 o[COLH0] : $vout.O : O[0] : -1 : 0
#var float4 f[COL0] : $vin.F : F[0] : -1 : 0
#var float4 f[TEX3] : $vin.F : F[0] : -1 : 0
#var sampler2D  : texunit 4 : -1 : 0
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
#tram 12 = f[TEX3].x
#tram 13 = f[TEX3].y
#tram 14 = f[TEX3].w
#tram 15 = f[TEX4].x
#tram 16 = f[TEX4].y
#tram 17 = f[TEX4].z
#tram 18 = f[TEX5].x
#tram 19 = f[TEX5].y
BB0:
IPA      R0, 0;
RCP      R12, R0;
IPA      R0, 18, R12;
IPA      R1, 19, R12;
IPA      R4, 10, R12;
IPA      R5, 11, R12;
MVI      R10, -1.0;
MVI      R9, -1.0;
TEX      R0, 3, 3, 2D;
MVI      R8, -1.0;
TEX      R4, 2, 2, 2D;
IPA      R3, 16, R12;
IPA      R7, 15, R12;
FMAD     R5, R5, c[0], R10;
FMAD     R4, R4, c[0], R9;
FMAD     R6, R6, c[0], R8;
FMUL32   R1, R5, R1;
FMUL32   R5, R5, R3;
IPA      R3, 17, R12;
FMAD     R0, R4, R0, R1;
FMAD     R5, R4, R7, R5;
IPA      R4, 5, R12;
FMAD.SAT R1, R6, R2, R0;
FMAD.SAT R13, R6, R3, R5;
MOV32    R8, R4;
IPA      R5, 6, R12;
MOV32    R0, R13;
MOV32    R9, R5;
TEX      R0, 5, 5, 2D;
TEX      R8, 0, 0, 2D;
TEX      R4, 1, 1, 2D;
FMUL32   R8, R13, R8;
FMUL32   R9, R13, R9;
FMUL32   R10, R13, R10;
FMAD     R0, R0, R4, R8;
FMAD     R1, R1, R5, R9;
FMAD     R2, R2, R6, R10;
FMUL32   R5, R13, R11;
IPA      R4, 14;
IPA      R6, 7, R12;
FMAD     R8, R3, R7, R5;
RCP      R5, R4;
IPA      R3, 8, R12;
IPA      R9, 9, R12;
IPA      R4, 12, R5;
IPA      R5, 13, R5;
FMUL32   R7, R3, R3;
FSET     R3, R9, c[0], GE;
MOV32    R11, c[1280];
FMAD     R10, R6, R6, R7;
TEX      R4, 4, 4, 2D;
FCMP     R3, R3, R11, c[1281];
FMAD.SAT R9, R9, R9, R10;
IPA      R10, 1, R12;
FADD32I  R9, -R9, 1.0;
FMUL32   R4, R4, R9;
FMUL32   R5, R5, R9;
FMUL32   R0, R4, R0;
FMUL32   R1, R5, R1;
IPA      R4, 2, R12;
FMUL32   R0, R0, R3;
FMUL32   R1, R1, R3;
FMUL32   R0, R0, R10;
FMUL32   R1, R1, R4;
FMUL32   R4, R6, R9;
FMUL32I  R0, R0, 2.0;
FMUL32I  R1, R1, 2.0;
FMUL32   R2, R4, R2;
MOV32.SAT R0, R0;
MOV32.SAT R1, R1;
FMUL32   R2, R2, R3;
FMUL32   R4, R7, R9;
F2F.SAT  R0, R0;
F2F.SAT  R1, R1;
FMUL32   R5, R4, R8;
IPA      R4, 3, R12;
IPA      R6, 4, R12;
FMUL32   R3, R5, R3;
FMUL32   R2, R2, R4;
FMUL32   R3, R3, R6;
FMUL32I  R2, R2, 2.0;
FMUL32I  R3, R3, 2.0;
MOV32.SAT R2, R2;
MOV32.SAT R3, R3;
F2F.SAT  R2, R2;
F2F.SAT  R3, R3;
END
# 86 instructions, 16 R-regs, 20 interpolants
# 86 inst, (8 mov, 3 mvi, 6 tex, 20 ipa, 2 complex, 47 math)
#    50 64-bit, 31 32-bit, 5 32-bit-const
