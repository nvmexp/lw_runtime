!!SPA1.0
.CONST_MODE  PAGE
.THREAD_TYPE PIXEL
.MAX_REG     7
.MAX_ATTR    3
# parseasm build date Feb 13 2004 14:20:40
# -profile fp50 -po tbat3 -po lat3 -if ps2x -i progs//p286-lw40.s -o progs//p286-lw50.s
#vendor LWPU
#version parseasm.0.0
#profile fp50
#program fp30entry
#semantic <null atom>
#var float4 o[COLH0] : $vout.O : O[0] : -1 : 0
#var float4 f[COL0] : $vin.F : F[0] : -1 : 0
#var float4 f[TEX0] : $vin.F : F[0] : -1 : 0
#var sampler2D  : texunit 0 : -1 : 0
#tram 0 = f[WPOS].w
#tram 1 = f[COL0].w
#tram 2 = f[TEX0].x
#tram 3 = f[TEX0].y
BB0:
IPA      R2, 0;
MVI      R0, 1.0;
MVI      R1, 1.0;
RCP      R3, R2;
MVI      R2, 1.0;
IPA      R4, 2, R3;
IPA      R5, 3, R3;
IPA      R3, 1, R3;
TEX      R4, 0, 0, 2D;
FMUL     R3, R3, R7;
END
# 10 instructions, 8 R-regs, 4 interpolants
# 10 inst, (0 mov, 3 mvi, 1 tex, 4 ipa, 1 complex, 1 math)
#    10 64-bit, 0 32-bit, 0 32-bit-const
