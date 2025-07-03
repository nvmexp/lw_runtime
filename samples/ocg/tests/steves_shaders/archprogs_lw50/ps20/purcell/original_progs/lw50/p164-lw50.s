!!SPA1.0
.CONST_MODE  PAGE
.THREAD_TYPE PIXEL
.MAX_REG     15
.MAX_ATTR    11
# parseasm build date Mar 10 2004 15:40:49
# -profile fp50 -po tbat3 -po lat3 -if ps2x -i allprogs-new32//p164-lw40.s -o allprogs-new32//p164-lw50.s
#vendor LWPU
#version parseasm.0.0
#profile fp50
#program fp30entry
#semantic <null atom>
#semantic <null atom>
#semantic <null atom>
#semantic <null atom>
#var float4 o[COLH0] : $vout.O : O[0] : -1 : 0
#var float4 f[COL0] : $vin.F : F[0] : -1 : 0
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
#tram 4 = f[TEX0].x
#tram 5 = f[TEX0].y
#tram 6 = f[TEX1].x
#tram 7 = f[TEX1].y
#tram 8 = f[TEX2].x
#tram 9 = f[TEX2].y
#tram 10 = f[TEX3].x
#tram 11 = f[TEX3].y
BB0:
IPA      R0, 0;
RCP      R8, R0;
MVI      R9, -1.0;
IPA      R0, 6, R8;
IPA      R1, 7, R8;
IPA      R4, 10, R8;
IPA      R5, 11, R8;
MVI      R13, -1.0;
MVI      R12, -1.0;
MVI      R10, -1.0;
TEX      R0, 1, 1, 2D;
MVI      R11, -1.0;
TEX      R4, 3, 3, 2D;
MVI      R3, -1.0;
FMAD     R9, R2, c[0], R9;
FMAD     R2, R0, c[0], R13;
FMAD     R7, R1, c[0], R12;
FMAD     R10, R6, c[0], R10;
FMAD     R6, R4, c[0], R11;
FMAD     R3, R5, c[0], R3;
IPA      R0, 4, R8;
IPA      R1, 5, R8;
FMUL32   R3, R7, R3;
IPA      R4, 8, R8;
IPA      R5, 9, R8;
FMAD     R11, R2, R6, R3;
TEX      R0, 0, 0, 2D;
TEX      R4, 2, 2, 2D;
FMAD.SAT R9, R9, R10, R11;
FMUL32   R3, R9, R9;
MVI      R4, -1.0;
FMUL32   R3, R3, R3;
FMAD     R4, R7, c[0], R4;
IPA      R5, 1, R8;
FMUL32   R3, R3, R3;
MOV32.SAT R4, R4;
FMUL32   R3, R3, R3;
F2F.SAT  R4, R4;
FADD32.SAT R4, R4, R4;
IPA      R6, 2, R8;
F2F.SAT.M4 R4, R4;
FMUL32   R0, R4, R0;
FMUL32   R1, R4, R1;
FMUL32   R0, R5, R0;
FMUL32   R1, R6, R1;
FMUL32   R2, R4, R2;
FMUL32   R0, R3, R0;
FMUL32   R1, R3, R1;
IPA      R5, 3, R8;
MOV32.SAT R0, R0;
MOV32.SAT R1, R1;
FMUL32   R2, R5, R2;
F2F.SAT  R0, R0;
F2F.SAT  R1, R1;
FMUL32   R2, R3, R2;
MOV32.SAT R3, R4;
MOV32.SAT R2, R2;
F2F.SAT  R3, R3;
F2F.SAT  R2, R2;
END
# 59 instructions, 16 R-regs, 12 interpolants
# 59 inst, (5 mov, 7 mvi, 4 tex, 12 ipa, 1 complex, 30 math)
#    39 64-bit, 20 32-bit, 0 32-bit-const
