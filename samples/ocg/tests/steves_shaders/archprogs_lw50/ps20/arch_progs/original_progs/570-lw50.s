!!SPA1.0
.CONST_MODE  PAGE
.THREAD_TYPE PIXEL
.MAX_REG     11
.MAX_ATTR    4
# parseasm build date Feb  3 2004 15:17:32
# -profile fp50 -po tbat3 -po lat3 -if ps2x -i progs/570-lw40.s -o progs/570-lw50.s
#vendor LWPU
#version parseasm.0.0
#profile lwinst
#program fp30entry
#semantic <null atom>
#semantic <null atom>
#semantic <null atom>
#var float4 o[COLH0] : $vout.O : O[0] : -1 : 0
#var sampler2D  : texunit 2 : -1 : 0
#var float4 f[TEX1] : $vin.F : F[0] : -1 : 0
#var sampler2D  : texunit 1 : -1 : 0
#var float4 f[TEX0] : $vin.F : F[0] : -1 : 0
#var sampler2D  : texunit 0 : -1 : 0
#tram 0 = f[WPOS].w
#tram 1 = f[TEX0].x
#tram 2 = f[TEX0].y
#tram 3 = f[TEX1].x
#tram 4 = f[TEX1].y
BB0:
IPA      R0, 0;
RCP      R0, R0;
IPA      R8, 1, R0;
IPA      R9, 2, R0;
IPA      R1, 3, R0;
IPA      R2, 4, R0;
TEX      R8, 0, 0, 2D;
MOV      R0, R1;
MOV      R4, R1;
MOV      R1, R2;
MOV      R5, R2;
FMUL     R9, R9, 0.33333;
TEX      R0, 1, 1, 2D;
TEX      R4, 2, 2, 2D;
FMAD     R8, R8, c[0], R9;
FMAD     R4, R10, c[0], R8;
MVI      R0, 0.0;
FADD     R6, -R7, 1.0;
FMUL     R5, R3, R4;
MVI      R1, 0.0;
MVI      R2, 1.0;
FMAD     R3, R3, R4, R5;
FMUL     R3, R3, R6;
END
# 23 instructions, 12 R-regs, 5 interpolants
# 23 inst, (4 mov, 3 mvi, 3 tex, 5 ipa, 1 complex, 7 math)
#    12 64-bit, 6 32-bit, 5 32-bit-const
