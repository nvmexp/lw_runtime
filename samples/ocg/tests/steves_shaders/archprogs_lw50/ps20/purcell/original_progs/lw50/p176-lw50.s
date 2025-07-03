!!SPA1.0
.CONST_MODE  PAGE
.THREAD_TYPE PIXEL
.MAX_REG     11
.MAX_ATTR    12
# parseasm build date Mar 10 2004 15:40:49
# -profile fp50 -po tbat3 -po lat3 -if ps2x -i allprogs-new32//p176-lw40.s -o allprogs-new32//p176-lw50.s
#vendor LWPU
#version parseasm.0.0
#profile fp50
#program fp30entry
#semantic <null atom>
#semantic <null atom>
#var float4 o[COLH0] : $vout.O : O[0] : -1 : 0
#var float4 f[COL0] : $vin.F : F[0] : -1 : 0
#var sampler2D  : texunit 3 : -1 : 0
#var float4 f[TEX3] : $vin.F : F[0] : -1 : 0
#var float4 f[TEX2] : $vin.F : F[0] : -1 : 0
#var float4 f[TEX1] : $vin.F : F[0] : -1 : 0
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
#tram 9 = f[TEX1].z
#tram 10 = f[TEX2].x
#tram 11 = f[TEX2].y
#tram 12 = f[TEX2].z
BB0:
IPA      R0, 0;
MVI      R7, -1.0;
MVI      R9, -1.0;
RCP      R4, R0;
IPA      R0, 5, R4;
IPA      R1, 6, R4;
IPA      R10, 8, R4;
IPA      R8, 7, R4;
MVI      R5, -1.0;
IPA      R6, 9, R4;
TEX      R0, 0, 0, 2D;
IPA      R11, 11, R4;
IPA      R3, 10, R4;
FMAD     R0, R0, c[0], R7;
FMAD     R1, R1, c[0], R9;
FMAD     R2, R2, c[0], R5;
IPA      R5, 12, R4;
FMUL32   R9, R1, R10;
FMUL32   R1, R1, R11;
IPA      R7, 1, R4;
FMAD     R9, R0, R8, R9;
FMAD     R1, R0, R3, R1;
IPA      R8, 2, R4;
FMAD     R0, R2, R6, R9;
FMAD     R1, R2, R5, R1;
TEX      R0, 3, 3, 2D;
IPA      R5, 3, R4;
FMUL32   R0, R0, R7;
FMUL32   R1, R1, R8;
FMUL32   R2, R2, R5;
MOV32.SAT R0, R0;
MOV32.SAT R1, R1;
MOV32.SAT R2, R2;
F2F.SAT  R0, R0;
F2F.SAT  R1, R1;
F2F.SAT  R2, R2;
IPA      R4, 4, R4;
FMUL32   R3, R3, R4;
MOV32.SAT R3, R3;
F2F.SAT  R3, R3;
END
# 40 instructions, 12 R-regs, 13 interpolants
# 40 inst, (4 mov, 3 mvi, 2 tex, 13 ipa, 1 complex, 17 math)
#    30 64-bit, 10 32-bit, 0 32-bit-const
