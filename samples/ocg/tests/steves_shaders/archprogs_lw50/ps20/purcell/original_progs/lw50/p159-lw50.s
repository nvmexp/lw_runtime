!!SPA1.0
.CONST_MODE  PAGE
.THREAD_TYPE PIXEL
.MAX_REG     15
.MAX_ATTR    17
# parseasm build date Mar 10 2004 15:40:49
# -profile fp50 -po tbat3 -po lat3 -if ps2x -i allprogs-new32//p159-lw40.s -o allprogs-new32//p159-lw50.s
#vendor LWPU
#version parseasm.0.0
#profile fp50
#program fp30entry
#semantic C76pv1sbdfq7lf.C76pv1sbdfq7lf
#semantic <null atom>
#semantic <null atom>
#semantic <null atom>
#var float4 o[COLH0] : $vout.O : O[0] : -1 : 0
#var float4 f[COL0] : $vin.F : F[0] : -1 : 0
#var float4 C76pv1sbdfq7lf :  : c[320] : -1 : 0
#var float4 f[TEX3] : $vin.F : F[0] : -1 : 0
#var sampler2D  : texunit 4 : -1 : 0
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
#tram 12 = f[TEX3].x
#tram 13 = f[TEX3].y
#tram 14 = f[TEX3].w
#tram 15 = f[TEX4].x
#tram 16 = f[TEX4].y
#tram 17 = f[TEX4].z
BB0:
IPA      R1, 0;
IPA      R0, 14;
RCP      R12, R1;
RCP      R1, R0;
IPA      R8, 10, R12;
IPA      R9, 11, R12;
IPA      R4, 5, R12;
IPA      R5, 6, R12;
IPA      R0, 12, R1;
IPA      R1, 13, R1;
MVI      R3, -1.0;
MVI      R15, -1.0;
IPA      R14, 16, R12;
IPA      R13, 15, R12;
TEX      R8, 2, 2, 2D;
MVI      R2, -1.0;
TEX      R4, 0, 0, 2D;
IPA      R11, 17, R12;
FMAD     R8, R8, c[0], R3;
FMAD     R9, R9, c[0], R15;
FMAD     R10, R10, c[0], R2;
TEX      R0, 4, 4, 2D;
FMUL32   R14, R9, R14;
IPA      R9, 7, R12;
FMAD     R14, R8, R13, R14;
IPA      R13, 8, R12;
IPA      R8, 9, R12;
FMAD.SAT R10, R10, R11, R14;
FMUL32   R11, R13, R13;
FMUL32   R7, R10, R7;
FMUL32   R6, R10, R6;
FMUL32   R5, R10, R5;
FMUL32   R4, R10, R4;
FMAD     R9, R9, R9, R11;
FSET     R10, R8, c[0], GE;
MOV32    R11, c[1280];
FMAD.SAT R8, R8, R8, R9;
FCMP     R9, R10, R11, c[1281];
FADD32I  R8, -R8, 1.0;
IPA      R10, 1, R12;
FMUL32   R3, R3, R8;
FMUL32   R2, R2, R8;
FMUL32   R1, R1, R8;
FMUL32   R0, R0, R8;
FMUL32   R1, R1, R5;
FMUL32   R0, R0, R4;
IPA      R4, 2, R12;
FMUL32   R1, R1, R9;
FMUL32   R0, R0, R9;
FMUL32   R1, R1, R4;
FMUL32   R0, R0, R10;
FMUL32   R2, R2, R6;
FMUL32I  R1, R1, 2.0;
FMUL32I  R0, R0, 2.0;
FMUL32   R2, R2, R9;
MOV32.SAT R1, R1;
MOV32.SAT R0, R0;
FMUL32   R3, R3, R7;
F2F.SAT  R1, R1;
F2F.SAT  R0, R0;
FMUL32   R4, R3, R9;
IPA      R3, 3, R12;
IPA      R5, 4, R12;
FMUL32   R2, R2, R3;
FMUL32   R3, R4, R5;
FMUL32I  R2, R2, 2.0;
FMUL32I  R3, R3, 2.0;
MOV32.SAT R2, R2;
MOV32.SAT R3, R3;
F2F.SAT  R2, R2;
F2F.SAT  R3, R3;
END
# 71 instructions, 16 R-regs, 18 interpolants
# 71 inst, (5 mov, 3 mvi, 3 tex, 18 ipa, 2 complex, 40 math)
#    39 64-bit, 27 32-bit, 5 32-bit-const
