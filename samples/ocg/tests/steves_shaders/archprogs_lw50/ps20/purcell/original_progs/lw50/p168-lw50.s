!!SPA1.0
.CONST_MODE  PAGE
.THREAD_TYPE PIXEL
.MAX_REG     11
.MAX_ATTR    6
# parseasm build date Mar 10 2004 15:40:49
# -profile fp50 -po tbat3 -po lat3 -if ps2x -i allprogs-new32//p168-lw40.s -o allprogs-new32//p168-lw50.s
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
IPA      R4, 3, R1;
IPA      R5, 4, R1;
IPA      R8, 5, R1;
IPA      R9, 6, R1;
IPA      R0, 1, R1;
IPA      R1, 2, R1;
TEX      R4, 1, 1, 2D;
TEX      R8, 2, 2, 2D;
TEX      R0, 0, 0, 2D;
FMUL32   R2, R2, R6;
FMUL32   R1, R1, R5;
FMUL32   R0, R0, R4;
F2F.M2   R2, R2;
F2F.M2   R1, R1;
F2F.M2   R0, R0;
FMUL32   R2, R2, R10;
FMUL32   R1, R1, R9;
FMUL32   R0, R0, R8;
MOV32.SAT R2, R2;
MOV32.SAT R1, R1;
MOV32.SAT R0, R0;
F2F.SAT  R2, R2;
F2F.SAT  R1, R1;
F2F.SAT  R0, R0;
MOV32.SAT R3, R3;
F2F.SAT  R3, R3;
END
# 28 instructions, 12 R-regs, 7 interpolants
# 28 inst, (4 mov, 0 mvi, 3 tex, 7 ipa, 1 complex, 13 math)
#    18 64-bit, 10 32-bit, 0 32-bit-const
