!!SPA1.0
.CONST_MODE  PAGE
.THREAD_TYPE PIXEL
.MAX_REG     15
.MAX_ATTR    9
# parseasm build date Feb 13 2004 14:20:40
# -profile fp50 -po tbat3 -po lat3 -if ps2x -i progs//p312-lw40.s -o progs//p312-lw50.s
#vendor LWPU
#version parseasm.0.0
#profile fp50
#program fp30entry
#semantic <null atom>
#semantic <null atom>
#semantic <null atom>
#var float4 o[COLH0] : $vout.O : O[0] : -1 : 0
#var float4 f[COL0] : $vin.F : F[0] : -1 : 0
#var float4 f[TEX2] : $vin.F : F[0] : -1 : 0
#var sampler2D  : texunit 2 : -1 : 0
#var float4 f[TEX1] : $vin.F : F[0] : -1 : 0
#var sampler2D  : texunit 1 : -1 : 0
#var float4 f[TEX0] : $vin.F : F[0] : -1 : 0
#var sampler2D  : texunit 0 : -1 : 0
#tram 0 = f[WPOS].w
#tram 1 = f[COL0].x
#tram 2 = f[COL0].y
#tram 3 = f[COL0].z
#tram 4 = f[TEX0].x
#tram 5 = f[TEX0].y
#tram 6 = f[TEX1].x
#tram 7 = f[TEX1].y
#tram 8 = f[TEX2].x
#tram 9 = f[TEX2].y
BB0:
IPA      R0, 0;
RCP      R12, R0;
IPA      R8, 6, R12;
IPA      R9, 7, R12;
IPA      R0, 8, R12;
IPA      R1, 9, R12;
IPA      R4, 4, R12;
IPA      R5, 5, R12;
IPA      R14, 3, R12;
TEX      R8, 1, 1, 2D;
TEX      R0, 2, 2, 2D;
TEX      R4, 0, 0, 2D;
IPA      R13, 2, R12;
FMAD     R6, -R11, R6, R6;
FMAD     R5, -R11, R5, R5;
FMAD     R4, -R11, R4, R4;
FMAD     R6, R11, R10, R6;
FMAD     R5, R11, R9, R5;
FMAD     R4, R11, R8, R4;
FMUL     R6, R14, R6;
FMUL     R5, R13, R5;
F2F.M2   R6, R6;
F2F.M2   R5, R5;
IPA      R3, 1, R12;
FADD     R2, R6, R2;
FADD     R1, R5, R1;
FMUL     R4, R3, R4;
MOV      R3, R7;
F2F.M2   R4, R4;
FADD     R0, R4, R0;
END
# 30 instructions, 16 R-regs, 10 interpolants
# 30 inst, (1 mov, 0 mvi, 3 tex, 10 ipa, 1 complex, 15 math)
#    30 64-bit, 0 32-bit, 0 32-bit-const
