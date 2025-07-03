!!SPA1.0
.CONST_MODE  PAGE
.THREAD_TYPE PIXEL
.MAX_REG     15
.MAX_ATTR    7
# parseasm build date Feb  3 2004 15:17:32
# -profile fp50 -po tbat3 -po lat3 -if ps2x -i progs/541-lw40.s -o progs/541-lw50.s
#vendor LWPU
#version parseasm.0.0
#profile lwinst
#program fp30entry
#semantic Cg86p5d28dfggc.Cg86p5d28dfggc
#semantic <null atom>
#semantic <null atom>
#var float4 o[COLH0] : $vout.O : O[0] : -1 : 0
#var float4 f[COL0] : $vin.F : F[0] : -1 : 0
#var float4 Cg86p5d28dfggc :  : c[7] : -1 : 0
#var float4 f[TEX0] : $vin.F : F[0] : -1 : 0
#var sampler2D  : texunit 0 : -1 : 0
#var float4 f[TEX3] : $vin.F : F[0] : -1 : 0
#var sampler2D  : texunit 1 : -1 : 0
#tram 0 = f[WPOS].w
#tram 1 = f[COL0].x
#tram 2 = f[COL0].y
#tram 3 = f[COL0].z
#tram 4 = f[TEX0].x
#tram 5 = f[TEX0].y
#tram 6 = f[TEX3].x
#tram 7 = f[TEX3].y
BB0:
IPA      R0, 0;
RCP      R2, R0;
IPA      R0, 4, R2;
IPA      R1, 5, R2;
IPA      R4, 6, R2;
IPA      R5, 7, R2;
IPA      R11, 3, R2;
IPA      R9, 2, R2;
IPA      R8, 1, R2;
TEX      R0, 0, 0, 2D;
MVI      R12, -16.0;
TEX      R4, 1, 1, 2D;
MVI      R10, -16.0;
FMUL     R2, R2, R11;
FMUL     R1, R1, R9;
FMUL     R0, R0, R8;
FMUL     R6, R7, R6;
FMUL     R5, R7, R5;
FMUL     R4, R7, R4;
FMAD     R7, R6, R12, c[30];
FMAD     R9, R5, R10, c[29];
MVI      R8, -16.0;
FMUL     R9, R3, R9;
FMAD     R8, R4, R8, c[28];
FMUL     R7, R3, R7;
FMAD     R5, R5, c[0], R9;
FMUL     R3, R3, R8;
FMAD     R6, R6, c[0], R7;
FMUL     R1, R1, R5;
FMAD     R4, R4, c[0], R3;
FMUL     R3, R2, R6;
FMUL     R5, R1, 0.59;
FMUL     R0, R0, R4;
MOV      R2, R3;
FMAD     R4, R0, c[0], R5;
FMAD     R3, R3, c[0], R4;
FMUL     R3, R3, 0.0625;
END
# 37 instructions, 16 R-regs, 8 interpolants
# 37 inst, (1 mov, 3 mvi, 2 tex, 8 ipa, 1 complex, 22 math)
#    19 64-bit, 13 32-bit, 5 32-bit-const
