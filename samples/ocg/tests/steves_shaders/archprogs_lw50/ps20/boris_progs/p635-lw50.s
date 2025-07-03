!!SPA1.0
.CONST_MODE  PAGE
.THREAD_TYPE PIXEL
.MAX_REG     7
.MAX_ATTR    5
# parseasm build date Feb 13 2004 14:20:40
# -profile fp50 -po tbat3 -po lat3 -if ps2x -i progs//p635-lw40.s -o progs//p635-lw50.s
#vendor LWPU
#version parseasm.0.0
#profile fp50
#program fp30entry
#semantic Coit1m35rlpml7.Coit1m35rlpml7
#semantic <null atom>
#var float4 o[COLH0] : $vout.O : O[0] : -1 : 0
#var float4 Coit1m35rlpml7 :  : c[1] : -1 : 0
#var float4 f[COL0] : $vin.F : F[0] : -1 : 0
#var float4 f[TEX0] : $vin.F : F[0] : -1 : 0
#var sampler2D  : texunit 0 : -1 : 0
#tram 0 = f[WPOS].w
#tram 1 = f[COL0].x
#tram 2 = f[COL0].y
#tram 3 = f[COL0].z
#tram 4 = f[TEX0].x
#tram 5 = f[TEX0].y
BB0:
IPA      R0, 0;
RCP      R2, R0;
IPA      R0, 4, R2;
IPA      R1, 5, R2;
IPA      R4, 1, R2;
IPA      R3, 2, R2;
IPA      R2, 3, R2;
FMUL     R6, R4, c[4];
FMUL     R5, R3, c[5];
FMUL     R4, R2, c[6];
TEX      R0, 0, 0, 2D;
FMUL     R0, R0, R6;
FMUL     R1, R1, R5;
FMUL     R2, R2, R4;
FMUL     R3, R3, c[7];
END
# 15 instructions, 8 R-regs, 6 interpolants
# 15 inst, (0 mov, 0 mvi, 1 tex, 6 ipa, 1 complex, 7 math)
#    15 64-bit, 0 32-bit, 0 32-bit-const
