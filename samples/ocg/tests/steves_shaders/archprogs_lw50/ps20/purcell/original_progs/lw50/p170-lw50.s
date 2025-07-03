!!SPA1.0
.CONST_MODE  PAGE
.THREAD_TYPE PIXEL
.MAX_REG     19
.MAX_ATTR    12
# parseasm build date Mar 10 2004 15:40:49
# -profile fp50 -po tbat3 -po lat3 -if ps2x -i allprogs-new32//p170-lw40.s -o allprogs-new32//p170-lw50.s
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
#tram 4 = f[COL0].w
#tram 5 = f[TEX0].x
#tram 6 = f[TEX0].y
#tram 7 = f[TEX1].x
#tram 8 = f[TEX1].y
#tram 9 = f[TEX2].x
#tram 10 = f[TEX2].y
#tram 11 = f[TEX3].x
#tram 12 = f[TEX3].y
BB0:
IPA      R0, 0;
RCP      R12, R0;
IPA      R0, 5, R12;
IPA      R1, 6, R12;
IPA      R4, 7, R12;
IPA      R5, 8, R12;
IPA      R8, 9, R12;
IPA      R9, 10, R12;
IPA      R13, 1, R12;
TEX      R0, 0, 0, 2D;
IPA      R14, 2, R12;
TEX      R4, 1, 1, 2D;
IPA      R15, 3, R12;
TEX      R8, 2, 2, 2D;
IPA      R16, 4, R12;
FMUL32   R0, R0, R4;
FMUL32   R1, R1, R5;
FMUL32   R2, R2, R6;
FMUL32   R0, R0, R13;
FMUL32   R1, R1, R14;
FMUL32   R2, R2, R15;
FMUL32   R3, R3, R7;
IPA      R4, 11, R12;
FMUL32   R3, R3, R16;
FMUL32   R2, R2, R10;
FMUL32   R1, R1, R9;
FMUL32   R3, R3, R11;
FMUL32   R0, R0, R8;
IPA      R5, 12, R12;
TEX      R4, 3, 3, 2D;
FMUL32   R0, R0, R4;
FMUL32   R1, R1, R5;
FMUL32   R2, R2, R6;
F2F.M2   R0, R0;
F2F.M2   R1, R1;
FMUL32   R3, R3, R7;
MOV32.SAT R0, R0;
MOV32.SAT R1, R1;
F2F.M2   R2, R2;
F2F.SAT  R0, R0;
F2F.SAT  R1, R1;
MOV32.SAT R2, R2;
F2F.M2   R3, R3;
F2F.SAT  R2, R2;
MOV32.SAT R3, R3;
F2F.SAT  R3, R3;
END
# 46 instructions, 20 R-regs, 13 interpolants
# 46 inst, (4 mov, 0 mvi, 4 tex, 13 ipa, 1 complex, 24 math)
#    26 64-bit, 20 32-bit, 0 32-bit-const
