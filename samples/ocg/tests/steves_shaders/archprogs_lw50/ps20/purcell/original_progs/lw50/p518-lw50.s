!!SPA1.0
.CONST_MODE  PAGE
.THREAD_TYPE PIXEL
.MAX_REG     15
.MAX_ATTR    8
# parseasm build date Mar 10 2004 15:40:49
# -profile fp50 -po tbat3 -po lat3 -if ps2x -i allprogs-new32//p518-lw40.s -o allprogs-new32//p518-lw50.s
#vendor LWPU
#version parseasm.0.0
#profile fp50
#program fp30entry
#semantic <null atom>
#semantic <null atom>
#semantic <null atom>
#semantic <null atom>
#var float4 o[COLH0] : $vout.O : O[0] : -1 : 0
#var float4 f[TEX3] : $vin.F : F[0] : -1 : 0
#var sampler2D  : texunit 0 : -1 : 0
#var float4 f[TEX2] : $vin.F : F[0] : -1 : 0
#var sampler2D  : texunit 0 : -1 : 0
#var float4 f[TEX1] : $vin.F : F[0] : -1 : 0
#var sampler2D  : texunit 0 : -1 : 0
#var float4 f[TEX0] : $vin.F : F[0] : -1 : 0
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
BB0:
IPA      R0, 0;
RCP      R12, R0;
IPA      R4, 1, R12;
IPA      R5, 2, R12;
IPA      R8, 3, R12;
IPA      R9, 4, R12;
IPA      R0, 5, R12;
IPA      R1, 6, R12;
TEX      R4, 0, 0, 2D;
TEX      R8, 0, 0, 2D;
TEX      R0, 0, 0, 2D;
FADD32I.SAT R7, R7, -0.062745;
FADD32   R4, R4, R8;
FADD32   R5, R5, R9;
FADD32   R6, R6, R10;
FADD32   R4, R0, R4;
FADD32   R5, R1, R5;
FADD32   R6, R2, R6;
FADD32I.SAT R1, R11, -0.062745;
FADD32I.SAT R8, R3, -0.062745;
IPA      R0, 7, R12;
FMUL32I  R2, R1, 0.25;
IPA      R1, 8, R12;
FMAD     R7, R7, c[0], R2;
TEX      R0, 0, 0, 2D;
FMAD     R7, R8, c[0], R7;
FADD32   R0, R0, R4;
FADD32   R1, R1, R5;
FADD32   R2, R2, R6;
FMUL32I  R0, R0, 0.25;
FMUL32I  R1, R1, 0.25;
FMUL32I  R2, R2, 0.25;
FADD32I.SAT R3, R3, -0.062745;
FMAD     R3, R3, c[0], R7;
END
# 34 instructions, 16 R-regs, 9 interpolants
# 34 inst, (0 mov, 0 mvi, 4 tex, 9 ipa, 1 complex, 20 math)
#    17 64-bit, 9 32-bit, 8 32-bit-const
