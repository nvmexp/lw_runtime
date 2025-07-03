!!SPA1.0
.CONST_MODE  PAGE
.THREAD_TYPE PIXEL
.MAX_REG     7
.MAX_ATTR    5
# parseasm build date Feb  3 2004 15:17:32
# -profile fp50 -po tbat3 -po lat3 -if ps2x -i progs/633-lw40.s -o progs/633-lw50.s
#vendor LWPU
#version parseasm.0.0
#profile lwinst
#program fp30entry
#semantic Cee994ndhnk1rc.Cee994ndhnk1rc
#semantic Coit1m35rlpml7.Coit1m35rlpml7
#semantic <null atom>
#var float4 o[COLH0] : $vout.O : O[0] : -1 : 0
#var float4 Cee994ndhnk1rc :  : c[5] : -1 : 0
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
RCP      R6, R0;
IPA      R0, 4, R6;
IPA      R1, 5, R6;
IPA      R4, 1, R6;
IPA      R5, 2, R6;
TEX      R0, 0, 0, 2D;
FMUL     R4, R4, c[4];
FMUL     R5, R5, c[5];
FMUL     R4, R0, R4;
FMUL     R5, R1, R5;
IPA      R6, 3, R6;
FMAD     R0, R0, c[20], -R4;
FMAD     R1, R1, c[21], -R5;
FMUL     R6, R6, c[6];
FMAD     R4, R3, R0, R4;
FMAD     R1, R3, R1, R5;
FMUL     R5, R2, R6;
MOV      R0, R4;
FMUL     R6, R1, 0.59;
FMAD     R2, R2, c[22], -R5;
FMAD     R4, R4, c[0], R6;
FMAD     R2, R3, R2, R5;
FMAD     R3, R2, c[0], R4;
FMUL     R3, R3, 0.0625;
END
# 25 instructions, 8 R-regs, 6 interpolants
# 25 inst, (1 mov, 0 mvi, 1 tex, 6 ipa, 1 complex, 16 math)
#    16 64-bit, 7 32-bit, 2 32-bit-const
