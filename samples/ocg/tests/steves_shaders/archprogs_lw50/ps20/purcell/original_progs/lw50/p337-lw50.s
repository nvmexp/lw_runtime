!!SPA1.0
.CONST_MODE  PAGE
.THREAD_TYPE PIXEL
.MAX_REG     11
.MAX_ATTR    6
# parseasm build date Mar 10 2004 15:40:49
# -profile fp50 -po tbat3 -po lat3 -if ps2x -i allprogs-new32//p337-lw40.s -o allprogs-new32//p337-lw50.s
#vendor LWPU
#version parseasm.0.0
#profile fp50
#program fp30entry
#semantic <null atom>
#semantic <null atom>
#semantic <null atom>
#var float4 o[COLH0] : $vout.O : O[0] : -1 : 0
#var float4 f[TEX2] : $vin.F : F[0] : -1 : 0
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
#tram 5 = f[TEX2].x
#tram 6 = f[TEX2].y
BB0:
IPA      R0, 0;
RCP      R1, R0;
IPA      R4, 5, R1;
IPA      R5, 6, R1;
IPA      R8, 3, R1;
IPA      R9, 4, R1;
IPA      R0, 1, R1;
IPA      R1, 2, R1;
TEX      R4, 2, 2, 2D;
TEX      R8, 1, 1, 2D;
TEX      R0, 0, 0, 2D;
FMAD     R4, -R7, R11, R11;
FMAD     R5, -R7, R10, R10;
FMAD     R6, -R7, R9, R9;
FMAD     R8, -R7, R8, R8;
FMAD     R2, R7, R2, R5;
FMAD     R1, R7, R1, R6;
FMAD     R0, R7, R0, R8;
FMAD     R3, R7, R3, R4;
MOV32.SAT R1, R1;
MOV32.SAT R0, R0;
MOV32.SAT R2, R2;
F2F.SAT  R1, R1;
F2F.SAT  R0, R0;
F2F.SAT  R2, R2;
MOV32.SAT R3, R3;
F2F.SAT  R3, R3;
END
# 27 instructions, 12 R-regs, 7 interpolants
# 27 inst, (4 mov, 0 mvi, 3 tex, 7 ipa, 1 complex, 12 math)
#    23 64-bit, 4 32-bit, 0 32-bit-const
