!!SPA1.0
.CONST_MODE  PAGE
.THREAD_TYPE PIXEL
.MAX_REG     35
.MAX_ATTR    26
# parseasm build date Mar 10 2004 15:40:49
# -profile fp50 -po tbat3 -po lat3 -if ps2x -i allprogs-new32//p519-lw40.s -o allprogs-new32//p519-lw50.s
#vendor LWPU
#version parseasm.0.0
#profile fp50
#program fp30entry
#semantic <null atom>
#semantic <null atom>
#semantic <null atom>
#semantic <null atom>
#semantic <null atom>
#semantic <null atom>
#semantic <null atom>
#semantic <null atom>
#semantic <null atom>
#semantic <null atom>
#semantic <null atom>
#semantic <null atom>
#semantic <null atom>
#var float4 o[COLH0] : $vout.O : O[0] : -1 : 0
#var sampler2D  : texunit 0 : -1 : 0
#var sampler2D  : texunit 0 : -1 : 0
#var sampler2D  : texunit 0 : -1 : 0
#var sampler2D  : texunit 0 : -1 : 0
#var sampler2D  : texunit 0 : -1 : 0
#var sampler2D  : texunit 0 : -1 : 0
#var sampler2D  : texunit 0 : -1 : 0
#var sampler2D  : texunit 0 : -1 : 0
#var sampler2D  : texunit 0 : -1 : 0
#var sampler2D  : texunit 0 : -1 : 0
#var float4 f[TEX7] : $vin.F : F[0] : -1 : 0
#var float4 f[TEX6] : $vin.F : F[0] : -1 : 0
#var float4 f[TEX5] : $vin.F : F[0] : -1 : 0
#var float4 f[TEX4] : $vin.F : F[0] : -1 : 0
#var float4 f[TEX3] : $vin.F : F[0] : -1 : 0
#var float4 f[TEX0] : $vin.F : F[0] : -1 : 0
#var sampler2D  : texunit 0 : -1 : 0
#var float4 f[TEX2] : $vin.F : F[0] : -1 : 0
#var sampler2D  : texunit 0 : -1 : 0
#var float4 f[TEX1] : $vin.F : F[0] : -1 : 0
#var sampler2D  : texunit 0 : -1 : 0
#tram 0 = f[WPOS].w
#tram 1 = f[TEX0].x
#tram 2 = f[TEX0].y
#tram 3 = f[TEX1].x
#tram 4 = f[TEX1].y
#tram 5 = f[TEX2].x
#tram 6 = f[TEX2].y
#tram 7 = f[TEX3].x
#tram 8 = f[TEX3].y
#tram 9 = f[TEX3].z
#tram 10 = f[TEX3].w
#tram 11 = f[TEX4].x
#tram 12 = f[TEX4].y
#tram 13 = f[TEX4].z
#tram 14 = f[TEX4].w
#tram 15 = f[TEX5].x
#tram 16 = f[TEX5].y
#tram 17 = f[TEX5].z
#tram 18 = f[TEX5].w
#tram 19 = f[TEX6].x
#tram 20 = f[TEX6].y
#tram 21 = f[TEX6].z
#tram 22 = f[TEX6].w
#tram 23 = f[TEX7].x
#tram 24 = f[TEX7].y
#tram 25 = f[TEX7].z
#tram 26 = f[TEX7].w
BB0:
IPA      R0, 0;
RCP      R18, R0;
IPA      R8, 3, R18;
IPA      R9, 4, R18;
IPA      R4, 5, R18;
IPA      R5, 6, R18;
IPA      R0, 1, R18;
IPA      R1, 2, R18;
TEX      R8, 0, 0, 2D;
TEX      R4, 0, 0, 2D;
TEX      R0, 0, 0, 2D;
FADD32   R4, R8, R4;
FADD32   R5, R9, R5;
FADD32   R6, R10, R6;
FADD32   R7, R11, R7;
FMUL32I  R4, R4, 0.2185;
FMUL32I  R5, R5, 0.2185;
FMUL32I  R6, R6, 0.2185;
FMUL32I  R7, R7, 0.2185;
IPA      R8, 26, R18;
IPA      R9, 25, R18;
FMAD     R19, R3, c[0], R7;
FMAD     R23, R2, c[0], R6;
FMAD     R24, R1, c[0], R5;
FMAD     R27, R0, c[0], R4;
IPA      R12, 23, R18;
IPA      R13, 24, R18;
TEX      R8, 0, 0, 2D;
IPA      R4, 22, R18;
IPA      R5, 21, R18;
IPA      R0, 19, R18;
IPA      R1, 20, R18;
TEX      R12, 0, 0, 2D;
TEX      R4, 0, 0, 2D;
TEX      R0, 0, 0, 2D;
FADD32   R16, R11, R15;
FADD32   R17, R10, R14;
FADD32   R20, R9, R13;
FADD32   R21, R8, R12;
FADD32   R22, R4, R0;
FADD32   R25, R5, R1;
FADD32   R26, R6, R2;
FADD32   R28, R7, R3;
IPA      R12, 18, R18;
IPA      R13, 17, R18;
IPA      R8, 15, R18;
IPA      R9, 16, R18;
TEX      R12, 0, 0, 2D;
IPA      R4, 14, R18;
IPA      R5, 13, R18;
IPA      R0, 11, R18;
IPA      R1, 12, R18;
TEX      R8, 0, 0, 2D;
TEX      R4, 0, 0, 2D;
TEX      R0, 0, 0, 2D;
FADD32   R11, R15, R11;
FADD32   R10, R14, R10;
FADD32   R9, R13, R9;
FADD32   R8, R12, R8;
FADD32   R12, R4, R0;
FADD32   R14, R5, R1;
FADD32   R15, R6, R2;
FADD32   R13, R7, R3;
IPA      R0, 10, R18;
IPA      R1, 9, R18;
IPA      R4, 7, R18;
IPA      R5, 8, R18;
TEX      R0, 0, 0, 2D;
TEX      R4, 0, 0, 2D;
FADD32   R0, R0, R4;
FADD32   R1, R1, R5;
FADD32   R2, R2, R6;
FMAD     R0, R0, c[0], R27;
FMAD     R1, R1, c[0], R24;
FMAD     R2, R2, c[0], R23;
FADD32   R4, R3, R7;
FMAD     R3, R14, c[0], R1;
FMAD     R1, R15, c[0], R2;
FMAD     R2, R4, c[0], R19;
FMAD     R4, R12, c[0], R0;
FMAD     R0, R9, c[0], R3;
FMAD     R3, R13, c[0], R2;
FMAD     R2, R8, c[0], R4;
FMAD     R1, R10, c[0], R1;
FMAD     R3, R11, c[0], R3;
FMAD     R0, R25, c[0], R0;
FMAD     R4, R26, c[0], R1;
FMAD     R3, R28, c[0], R3;
FMAD     R5, R22, c[0], R2;
FMAD     R1, R20, c[0], R0;
FMAD     R2, R17, c[0], R4;
FMAD     R0, R21, c[0], R5;
FMAD     R3, R16, c[0], R3;
END
# 93 instructions, 36 R-regs, 27 interpolants
# 93 inst, (0 mov, 0 mvi, 13 tex, 27 ipa, 1 complex, 52 math)
#    65 64-bit, 24 32-bit, 4 32-bit-const
