!!SPA1.0
.CONST_MODE  PAGE
.THREAD_TYPE PIXEL
.MAX_REG     7
.MAX_ATTR    4
# parseasm build date Mar 10 2004 15:40:49
# -profile fp50 -po tbat3 -po lat3 -if ps2x -i allprogs-new32//p167-lw40.s -o allprogs-new32//p167-lw50.s
#vendor LWPU
#version parseasm.0.0
#profile fp50
#program fp30entry
#semantic <null atom>
#semantic <null atom>
#var float4 o[COLH0] : $vout.O : O[0] : -1 : 0
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
RCP      R1, R0;
IPA      R4, 3, R1;
IPA      R5, 4, R1;
IPA      R0, 1, R1;
IPA      R1, 2, R1;
TEX      R4, 1, 1, 2D;
TEX      R0, 0, 0, 2D;
FMUL32   R0, R0, R4;
FMUL32   R1, R1, R5;
FMUL32   R2, R2, R6;
MOV32.SAT R0, R0;
MOV32.SAT R1, R1;
MOV32.SAT R2, R2;
F2F.SAT  R0, R0;
F2F.SAT  R1, R1;
F2F.SAT  R2, R2;
MOV32.SAT R3, R3;
F2F.SAT  R3, R3;
END
# 19 instructions, 8 R-regs, 5 interpolants
# 19 inst, (4 mov, 0 mvi, 2 tex, 5 ipa, 1 complex, 7 math)
#    12 64-bit, 7 32-bit, 0 32-bit-const
