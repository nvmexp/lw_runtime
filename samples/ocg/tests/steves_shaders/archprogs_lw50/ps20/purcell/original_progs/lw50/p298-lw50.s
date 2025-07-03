!!SPA1.0
.CONST_MODE  PAGE
.THREAD_TYPE PIXEL
.MAX_REG     15
.MAX_ATTR    11
# parseasm build date Mar 10 2004 15:40:49
# -profile fp50 -po tbat3 -po lat3 -if ps2x -i allprogs-new32//p298-lw40.s -o allprogs-new32//p298-lw50.s
#vendor LWPU
#version parseasm.0.0
#profile fp50
#program fp30entry
#semantic C76pv1sbdfq7lf.C76pv1sbdfq7lf
#semantic <null atom>
#semantic <null atom>
#var float4 o[COLH0] : $vout.O : O[0] : -1 : 0
#var float4 f[COL0] : $vin.F : F[0] : -1 : 0
#var float4 C76pv1sbdfq7lf :  : c[320] : -1 : 0
#var float4 f[COL1] : $vin.F : F[0] : -1 : 0
#var float4 f[TEX1] : $vin.F : F[0] : -1 : 0
#var sampler2D  : texunit 1 : -1 : 0
#var float4 f[TEX0] : $vin.F : F[0] : -1 : 0
#var sampler2D  : texunit 0 : -1 : 0
#tram 0 = f[WPOS].w
#tram 1 = f[COL0].x
#tram 2 = f[COL0].y
#tram 3 = f[COL0].z
#tram 4 = f[COL0].w
#tram 5 = f[COL1].x
#tram 6 = f[COL1].y
#tram 7 = f[COL1].z
#tram 8 = f[TEX0].x
#tram 9 = f[TEX0].y
#tram 10 = f[TEX1].x
#tram 11 = f[TEX1].y
BB0:
IPA      R0, 0;
RCP      R8, R0;
MVI      R3, -1.0;
IPA      R4, 8, R8;
IPA      R5, 9, R8;
IPA      R0, 10, R8;
IPA      R1, 11, R8;
MVI      R10, -1.0;
IPA      R6, 6, R8;
MVI      R7, -1.0;
IPA      R2, 5, R8;
MVI      R9, -1.0;
FMAD     R11, R6, c[0], R7;
TEX      R4, 0, 0, 2D;
FMAD     R9, R2, c[0], R9;
MVI      R2, -1.0;
IPA      R7, 7, R8;
MVI      R12, -1.0;
FMAD     R3, R4, c[0], R3;
FMAD     R5, R5, c[0], R10;
FMAD     R4, R6, c[0], R2;
FMAD     R6, R7, c[0], R12;
FMUL32   R2, R5, R11;
IPA      R7, 3, R8;
IPA      R5, 2, R8;
FMAD     R10, R3, R9, R2;
IPA      R9, 1, R8;
TEX      R0, 1, 1, 2D;
FMAD     R4, R4, R6, R10;
FADD32   R10, R4, c[1282];
FADD32   R6, R4, c[1281];
FADD32   R4, R4, c[1280];
FMUL32   R7, R7, R10;
FMUL32   R5, R5, R6;
FMUL32   R4, R9, R4;
FMUL32   R2, R7, R2;
FMUL32   R1, R5, R1;
FMUL32   R0, R4, R0;
MOV32.SAT R2, R2;
MOV32.SAT R1, R1;
MOV32.SAT R0, R0;
F2F.SAT  R2, R2;
F2F.SAT  R1, R1;
F2F.SAT  R0, R0;
IPA      R4, 4, R8;
FMUL32   R3, R3, R4;
MOV32.SAT R3, R3;
F2F.SAT  R3, R3;
END
# 48 instructions, 16 R-regs, 12 interpolants
# 48 inst, (4 mov, 6 mvi, 2 tex, 12 ipa, 1 complex, 23 math)
#    33 64-bit, 15 32-bit, 0 32-bit-const
