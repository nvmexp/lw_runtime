!!SPA1.0
.CONST_MODE  PAGE
.THREAD_TYPE PIXEL
.MAX_REG     11
.MAX_ATTR    8
# parseasm build date Feb 13 2004 14:20:40
# -profile fp50 -po tbat3 -po lat3 -if ps2x -i progs//p551-lw40.s -o progs//p551-lw50.s
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
IPA      R4, 7, R8;
IPA      R5, 8, R8;
IPA      R0, 5, R8;
IPA      R1, 6, R8;
TEX      R4, 1, 1, 2D;
TEX      R0, 0, 0, 2D;
IPA      R9, 1, R8;
FMUL     R6, R7, R6;
FMUL     R5, R7, R5;
FMUL     R4, R7, R4;
FMUL     R0, R0, R9;
FMUL32I  R6, R6, 16.0;
FMUL32I  R5, R5, 16.0;
FMUL32I  R9, R4, 16.0;
IPA      R4, 2, R8;
IPA      R7, 3, R8;
FMUL     R0, R9, R0;
FMUL     R1, R1, R4;
FMUL     R2, R2, R7;
IPA      R4, 4, R8;
FMUL     R1, R5, R1;
FMUL     R2, R6, R2;
FMUL     R3, R3, R4;
END
# 25 instructions, 12 R-regs, 9 interpolants
# 25 inst, (0 mov, 0 mvi, 2 tex, 9 ipa, 1 complex, 13 math)
#    22 64-bit, 0 32-bit, 3 32-bit-const
