!!SPA1.0
.CONST_MODE  PAGE
.THREAD_TYPE PIXEL
.MAX_REG     11
.MAX_ATTR    11
# parseasm build date Mar 10 2004 15:40:49
# -profile fp50 -po tbat3 -po lat3 -if ps2x -i allprogs-new32//p161-lw40.s -o allprogs-new32//p161-lw50.s
#vendor LWPU
#version parseasm.0.0
#profile fp50
#program fp30entry
#semantic <null atom>
#semantic <null atom>
#semantic <null atom>
#var float4 o[COLH0] : $vout.O : O[0] : -1 : 0
#var float4 f[TEX5] : $vin.F : F[0] : -1 : 0
#var sampler2D  : texunit 2 : -1 : 0
#var float4 f[TEX4] : $vin.F : F[0] : -1 : 0
#var float4 f[TEX3] : $vin.F : F[0] : -1 : 0
#var float4 f[TEX2] : $vin.F : F[0] : -1 : 0
#var float4 f[TEX1] : $vin.F : F[0] : -1 : 0
#var sampler2D  : texunit 1 : -1 : 0
#var float4 f[TEX0] : $vin.F : F[0] : -1 : 0
#var sampler2D  : texunit 0 : -1 : 0
#tram 0 = f[WPOS].w
#tram 1 = f[TEX0].x
#tram 2 = f[TEX0].y
#tram 3 = f[TEX1].x
#tram 4 = f[TEX1].y
#tram 5 = f[TEX2].x
#tram 6 = f[TEX2].y
#tram 7 = f[TEX2].z
#tram 8 = f[TEX3].x
#tram 9 = f[TEX3].y
#tram 10 = f[TEX3].z
#tram 11 = f[TEX5].z
BB0:
IPA      R0, 0;
MVI      R7, -1.0;
MVI      R9, -1.0;
RCP      R8, R0;
IPA      R0, 3, R8;
IPA      R1, 4, R8;
IPA      R10, 6, R8;
IPA      R5, 5, R8;
MVI      R6, -1.0;
IPA      R4, 7, R8;
TEX      R0, 1, 1, 2D;
IPA      R11, 9, R8;
IPA      R3, 8, R8;
FMAD     R7, R0, c[0], R7;
FMAD     R0, R1, c[0], R9;
FMAD     R2, R2, c[0], R6;
IPA      R6, 10, R8;
FMUL32   R1, R0, R10;
FMUL32   R9, R0, R11;
IPA      R0, 1, R8;
FMAD     R5, R7, R5, R1;
FMAD     R3, R7, R3, R9;
IPA      R1, 2, R8;
FMAD     R4, R2, R4, R5;
FMAD     R5, R2, R6, R3;
TEX      R0, 0, 0, 2D;
TEX      R4, 2, 2, 2D;
FMUL32   R0, R0, R4;
FMUL32   R1, R1, R5;
FMUL32   R2, R2, R6;
MOV32.SAT R0, R0;
MOV32.SAT R1, R1;
MOV32.SAT R2, R2;
F2F.SAT  R0, R0;
F2F.SAT  R1, R1;
F2F.SAT  R2, R2;
IPA      R3, 11, R8;
MOV32.SAT R3, R3;
F2F.SAT  R3, R3;
END
# 39 instructions, 12 R-regs, 12 interpolants
# 39 inst, (4 mov, 3 mvi, 3 tex, 12 ipa, 1 complex, 16 math)
#    30 64-bit, 9 32-bit, 0 32-bit-const
