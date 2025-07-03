!!SPA1.0
.CONST_MODE  PAGE
.THREAD_TYPE PIXEL
.MAX_REG     7
.MAX_ATTR    3
# parseasm build date Mar 10 2004 15:40:49
# -profile fp50 -po tbat3 -po lat3 -if ps2x -i allprogs-new32//p286-lw40.s -o allprogs-new32//p286-lw50.s
#vendor LWPU
#version parseasm.0.0
#profile fp50
#program fp30entry
#semantic <null atom>
#var float4 o[COLH0] : $vout.O : O[0] : -1 : 0
#var float4 f[COL0] : $vin.F : F[0] : -1 : 0
#var float4 f[TEX0] : $vin.F : F[0] : -1 : 0
#var sampler2D  : texunit 0 : -1 : 0
#tram 0 = f[WPOS].w
#tram 1 = f[COL0].w
#tram 2 = f[TEX0].x
#tram 3 = f[TEX0].y
BB0:
IPA      R0, 0;
RCP      R2, R0;
IPA      R0, 2, R2;
IPA      R1, 3, R2;
IPA      R5, 1, R2;
MVI      R4, 1.0;
TEX      R0, 0, 0, 2D;
F2F.SAT  R4, R4;
MOV32    R0, R4;
FMUL32   R3, R5, R3;
MOV32    R1, R4;
MOV32    R2, R4;
MOV32.SAT R3, R3;
F2F.SAT  R3, R3;
END
# 14 instructions, 8 R-regs, 4 interpolants
# 14 inst, (4 mov, 1 mvi, 1 tex, 4 ipa, 1 complex, 3 math)
#    9 64-bit, 5 32-bit, 0 32-bit-const
