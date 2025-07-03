!!SPA1.0
.CONST_MODE  PAGE
.THREAD_TYPE PIXEL
.MAX_REG     19
.MAX_ATTR    2
# parseasm build date Mar 10 2004 15:40:49
# -profile fp50 -po tbat3 -po lat3 -if ps2x -i allprogs-new32//p370-lw40.s -o allprogs-new32//p370-lw50.s
#vendor LWPU
#version parseasm.0.0
#profile fp50
#program fp30entry
#semantic <null atom>
#semantic <null atom>
#var float4 o[COLH3] : $vout.O : O[0] : -1 : 0
#var float4 o[COLH2] : $vout.O : O[0] : -1 : 0
#var float4 o[COLH1] : $vout.O : O[0] : -1 : 0
#var float4 o[COLH0] : $vout.O : O[0] : -1 : 0
#var sampler2D  : texunit 1 : -1 : 0
#var float4 f[TEX2] : $vin.F : F[0] : -1 : 0
#var sampler2D  : texunit 0 : -1 : 0
#tram 0 = f[WPOS].w
#tram 1 = f[TEX2].x
#tram 2 = f[TEX2].y
BB0:
IPA      R0, 0;
MVI      R5, -1.0;
RCP      R0, R0;
MVI      R7, -1.0;
IPA      R9, 1, R0;
IPA      R6, 2, R0;
MOV32    R0, R9;
MOV32    R1, R6;
MVI      R4, -1.0;
TEX      R0, 1, 1, 2D;
FMAD     R5, R0, c[0], R5;
FMAD     R7, R1, c[0], R7;
FMAD     R10, R2, c[0], R4;
MVI      R11, 0.5;
FMUL32   R0, R7, R7;
MVI      R8, 0.5;
FMAD     R0, R5, R5, R0;
MVI      R4, 0.5;
FMAD     R2, R10, R10, R0;
MOV32    R0, R9;
MOV32    R1, R6;
RSQ      R6, R2;
TEX      R0, 0, 0, 2D;
FMUL32   R5, R5, R6;
FMUL32   R7, R7, R6;
FMUL32   R6, R10, R6;
FMAD     <<REG126>>.x, R5, c[0], R11;
FMAD     <<REG126>>.y, R7, c[0], R8;
FMAD     <<REG126>>.z, R6, c[0], R4;
MOV32    <<REG124>>.x, R3;
MOV32    <<REG124>>.y, R3;
MOV32    <<REG124>>.z, R3;
FCMP     R0, R0, R0, c[0];
FCMP     R3, R1, R1, c[0];
LG2      R1, |R0|;
LG2      R0, |R3|;
FCMP     R3, R2, R2, c[0];
FMUL32I  R2, R1, 0.454545;
FMUL32I  R1, R0, 0.454545;
LG2      R0, |R3|;
RRO      R2, R2, 1;
RRO      R1, R1, 1;
FMUL32I  R0, R0, 0.454545;
EX2      <<REG122>>.x, R2;
EX2      <<REG122>>.y, R1;
RRO      R2, R0, 1;
MVI      R1, 1.0;
MVI      R0, 0.058594;
EX2      <<REG122>>.z, R2;
MOV32    <<REG126>>.w, R1;
MOV32    <<REG124>>.w, R0;
MVI      R2, 1.0;
MVI      R0, 0.0;
MVI      R1, 0.0;
MOV32    <<REG122>>.w, R2;
MVI      R2, 0.0;
MVI      R3, 1.0;
END
# 57 instructions, 20 R-regs, 3 interpolants
# 57 inst, (10 mov, 13 mvi, 2 tex, 3 ipa, 8 complex, 21 math)
#    40 64-bit, 14 32-bit, 3 32-bit-const
