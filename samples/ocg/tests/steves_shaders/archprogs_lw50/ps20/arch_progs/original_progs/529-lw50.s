!!SPA1.0
.CONST_MODE  PAGE
.THREAD_TYPE PIXEL
.MAX_REG     7
.MAX_ATTR    6
# parseasm build date Feb  3 2004 15:17:32
# -profile fp50 -po tbat3 -po lat3 -if ps2x -i progs/529-lw40.s -o progs/529-lw50.s
#vendor LWPU
#version parseasm.0.0
#profile lwinst
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
#tram 4 = f[COL0].w
#tram 5 = f[TEX0].x
#tram 6 = f[TEX0].y
BB0:
IPA      R0, 0;
RCP      R4, R0;
IPA      R0, 5, R4;
IPA      R1, 6, R4;
IPA      R5, 1, R4;
IPA      R6, 2, R4;
TEX      R0, 0, 0, 2D;
IPA      R7, 3, R4;
FMUL     R0, R0, R5;
FMUL     R1, R1, R6;
FMUL     R2, R2, R7;
FMUL     R0, R0, c[4];
FMUL     R1, R1, c[5];
FMUL     R5, R2, c[6];
FMUL     R2, R3, c[7];
IPA      R3, 4, R4;
FMUL     R4, R1, 0.59;
FMUL     R3, R2, R3;
FMAD     R4, R0, c[0], R4;
MOV      R2, R5;
FMAD     R4, R5, c[0], R4;
FMUL     R3, R3, R4;
FMUL     R3, R3, 0.0625;
END
# 23 instructions, 8 R-regs, 7 interpolants
# 23 inst, (1 mov, 0 mvi, 1 tex, 7 ipa, 1 complex, 13 math)
#    11 64-bit, 10 32-bit, 2 32-bit-const
