!!SPA1.0
.CONST_MODE  PAGE
.THREAD_TYPE PIXEL
.MAX_REG     19
.MAX_ATTR    2
# parseasm build date Mar 10 2004 15:40:49
# -profile fp50 -po tbat3 -po lat3 -if ps2x -i allprogs-new32//p369-lw40.s -o allprogs-new32//p369-lw50.s
#vendor LWPU
#version parseasm.0.0
#profile fp50
#program fp30entry
#semantic <null atom>
#var float4 o[COLH3] : $vout.O : O[0] : -1 : 0
#var float4 o[COLH2] : $vout.O : O[0] : -1 : 0
#var float4 o[COLH1] : $vout.O : O[0] : -1 : 0
#var float4 o[COLH0] : $vout.O : O[0] : -1 : 0
#var float4 f[TEX2] : $vin.F : F[0] : -1 : 0
#var sampler2D  : texunit 0 : -1 : 0
#tram 0 = f[WPOS].w
#tram 1 = f[TEX2].x
#tram 2 = f[TEX2].y
BB0:
IPA      R0, 0;
MVI      R2, -1.0;
MVI      R1, -1.0;
RCP      R0, R0;
IPA      R4, 1, R0;
IPA      R5, 2, R0;
MVI      R0, -1.0;
TEX      R4, 0, 0, 2D;
FMAD     R4, R4, c[0], R2;
FMAD     R2, R5, c[0], R1;
FMAD     R0, R6, c[0], R0;
MVI      R5, 0.5;
FMUL32   R1, R2, R2;
MVI      R3, 0.5;
FMAD     R6, R4, R4, R1;
MVI      R1, 0.5;
FMAD     R6, R0, R0, R6;
MVI      R7, 1.0;
RSQ      R6, R6;
MOV32    <<REG126>>.w, R7;
MVI      <<REG124>>.x, 0.21952;
FMUL32   R4, R4, R6;
FMUL32   R2, R2, R6;
FMUL32   R0, R0, R6;
FMAD     <<REG126>>.x, R4, c[0], R5;
FMAD     <<REG126>>.y, R2, c[0], R3;
FMAD     <<REG126>>.z, R0, c[0], R1;
MVI      R2, 0.21952;
MVI      R1, 0.21952;
MVI      R0, 0.058594;
MOV32    <<REG124>>.y, R2;
MOV32    <<REG124>>.z, R1;
MOV32    <<REG124>>.w, R0;
MVI      <<REG122>>.x, 0.501961;
MVI      R2, 0.501961;
MVI      R1, 0.501961;
MVI      R0, 1.0;
MOV32    <<REG122>>.y, R2;
MOV32    <<REG122>>.z, R1;
MOV32    <<REG122>>.w, R0;
MVI      R0, 0.0;
MVI      R1, 0.0;
MVI      R2, 0.0;
MVI      R3, 1.0;
END
# 44 instructions, 20 R-regs, 3 interpolants
# 44 inst, (7 mov, 19 mvi, 1 tex, 3 ipa, 2 complex, 12 math)
#    33 64-bit, 11 32-bit, 0 32-bit-const
