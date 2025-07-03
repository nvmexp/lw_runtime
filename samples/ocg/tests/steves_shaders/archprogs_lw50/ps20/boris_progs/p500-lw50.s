!!SPA1.0
.CONST_MODE  PAGE
.THREAD_TYPE PIXEL
.MAX_REG     3
.MAX_ATTR    2
# parseasm build date Feb 13 2004 14:20:40
# -profile fp50 -po tbat3 -po lat3 -if ps2x -i progs//p500-lw40.s -o progs//p500-lw50.s
#vendor LWPU
#version parseasm.0.0
#profile fp50
#program fp30entry
#semantic C144q3o24uukb6.C144q3o24uukb6
#semantic <null atom>
#var float4 o[COLH0] : $vout.O : O[0] : -1 : 0
#var float4 C144q3o24uukb6 :  : c[0] : -1 : 0
#var float4 f[TEX0] : $vin.F : F[0] : -1 : 0
#var sampler2D  : texunit 0 : -1 : 0
#tram 0 = f[WPOS].w
#tram 1 = f[TEX0].x
#tram 2 = f[TEX0].y
BB0:
IPA      R0, 0;
RCP      R1, R0;
IPA      R0, 1, R1;
IPA      R1, 2, R1;
TEX      R0, 0, 0, 2D;
FMUL     R3, R3, c[3];
END
# 6 instructions, 4 R-regs, 3 interpolants
# 6 inst, (0 mov, 0 mvi, 1 tex, 3 ipa, 1 complex, 1 math)
#    6 64-bit, 0 32-bit, 0 32-bit-const
