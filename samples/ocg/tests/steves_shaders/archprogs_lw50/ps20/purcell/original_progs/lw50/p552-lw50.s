!!SPA1.0
.CONST_MODE  PAGE
.THREAD_TYPE PIXEL
.MAX_REG     11
.MAX_ATTR    8
# parseasm build date Mar 10 2004 15:40:49
# -profile fp50 -po tbat3 -po lat3 -if ps2x -i allprogs-new32//p552-lw40.s -o allprogs-new32//p552-lw50.s
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
#var float4 f[TEX3] : $vin.F : F[0] : -1 : 0
#var sampler2D  : texunit 1 : -1 : 0
#tram 0 = f[WPOS].w
#tram 1 = f[COL0].x
#tram 2 = f[COL0].y
#tram 3 = f[COL0].z
#tram 4 = f[COL0].w
#tram 5 = f[TEX0].x
#tram 6 = f[TEX0].y
#tram 7 = f[TEX3].x
#tram 8 = f[TEX3].y
BB0:
IPA      R0, 0;
RCP      R8, R0;
IPA      R0, 7, R8;
IPA      R1, 8, R8;
IPA      R4, 5, R8;
IPA      R5, 6, R8;
TEX      R0, 1, 1, 2D;
TEX      R4, 0, 0, 2D;
IPA      R9, 4, R8;
FMUL32   R2, R3, R2;
FMUL32   R1, R3, R1;
FMUL32   R0, R3, R0;
FMUL32   R3, R7, R9;
FMUL32I  R2, R2, 16.0;
FMUL32I  R1, R1, 16.0;
FMUL32I  R7, R0, 16.0;
IPA      R0, 3, R8;
IPA      R9, 2, R8;
IPA      R8, 1, R8;
FMUL32   R0, R6, R0;
FMUL32   R5, R5, R9;
FMUL32   R6, R4, R8;
FMUL32   R4, R2, R0;
FMUL32   R1, R1, R5;
FMUL32   R2, R7, R6;
MVI      R0, 1.0;
FMUL32I  R6, R1, 0.59;
MVI      R1, 0.0;
MVI      R5, 1.0;
FMAD     R6, R2, c[0], R6;
MOV32    R2, R5;
FMAD     R4, R4, c[0], R6;
FMUL32   R3, R3, R4;
FMUL32I  R3, R3, 0.0625;
END
# 34 instructions, 12 R-regs, 9 interpolants
# 34 inst, (1 mov, 3 mvi, 2 tex, 9 ipa, 1 complex, 18 math)
#    17 64-bit, 12 32-bit, 5 32-bit-const
