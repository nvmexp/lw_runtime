!!SPA1.0
.CONST_MODE  PAGE
.THREAD_TYPE PIXEL
.MAX_REG     15
.MAX_ATTR    6
# parseasm build date Feb  3 2004 15:17:32
# -profile fp50 -po tbat3 -po lat3 -if ps2x -i progs/533-lw40.s -o progs/533-lw50.s
#vendor LWPU
#version parseasm.0.0
#profile lwinst
#program fp30entry
#semantic <null atom>
#semantic <null atom>
#semantic <null atom>
#semantic <null atom>
#var float4 o[COLH0] : $vout.O : O[0] : -1 : 0
#var sampler2D  : texunit 2 : -1 : 0
#var float4 f[TEX1] : $vin.F : F[0] : -1 : 0
#var sampler2D  : texunit 1 : -1 : 0
#var float4 f[TEX0] : $vin.F : F[0] : -1 : 0
#var sampler2D  : texunit 0 : -1 : 0
#var float4 f[TEX2] : $vin.F : F[0] : -1 : 0
#var sampler2D  : texunit 3 : -1 : 0
#tram 0 = f[WPOS].w
#tram 1 = f[TEX0].x
#tram 2 = f[TEX0].y
#tram 3 = f[TEX1].x
#tram 4 = f[TEX1].y
#tram 5 = f[TEX2].x
#tram 6 = f[TEX2].y
BB0:
IPA      R0, 0;
RCP      R0, R0;
IPA      R8, 1, R0;
IPA      R9, 2, R0;
IPA      R4, 3, R0;
IPA      R5, 4, R0;
IPA      R13, 5, R0;
IPA      R12, 6, R0;
TEX      R8, 0, 0, 2D;
MOV      R0, R13;
MOV      R1, R12;
TEX      R4, 1, 1, 2D;
TEX      R0, 3, 3, 2D;
FMUL     R11, R9, 0.33333;
FMAD     R11, R8, c[0], R11;
FADD     R7, R8, -R4;
FMAD     R0, R10, c[0], R11;
FADD     R1, R9, -R5;
FADD     R2, R10, -R6;
FMAD     R7, R3, R7, R4;
FMAD     R1, R3, R1, R5;
FMUL     R5, R5, 0.33333;
FMAD     R2, R3, R2, R6;
MOV      R8, R13;
FMAD     R4, R4, c[0], R5;
MOV      R9, R12;
FMAD     R4, R6, c[0], R4;
TEX      R8, 2, 2, 2D;
FADD     R0, R0, -R4;
FMAD     R3, R3, R0, R4;
FADD     R0, R8, R8;
FADD     R4, R9, R9;
FADD     R5, R10, R10;
FMUL     R0, R7, R0;
FMUL     R1, R1, R4;
FMUL     R2, R2, R5;
FADD     R4, R11, R11;
FMUL     R3, R3, R4;
END
# 38 instructions, 16 R-regs, 7 interpolants
# 38 inst, (4 mov, 0 mvi, 4 tex, 7 ipa, 1 complex, 22 math)
#    20 64-bit, 16 32-bit, 2 32-bit-const
