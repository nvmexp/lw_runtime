!!SPA1.0
.CONST_MODE  PAGE
.THREAD_TYPE PIXEL
.MAX_REG     15
.MAX_ATTR    9
# parseasm build date Mar 10 2004 15:40:49
# -profile fp50 -po tbat3 -po lat3 -if ps2x -i allprogs-new32//p356-lw40.s -o allprogs-new32//p356-lw50.s
#vendor LWPU
#version parseasm.0.0
#profile fp50
#program fp30entry
#semantic <null atom>
#var float4 o[COLH2] : $vout.O : O[0] : -1 : 0
#var float4 o[COLH1] : $vout.O : O[0] : -1 : 0
#var float4 o[COLH0] : $vout.O : O[0] : -1 : 0
#var float4 f[COL0] : $vin.F : F[0] : -1 : 0
#var float4 f[TEX2] : $vin.F : F[0] : -1 : 0
#var float4 f[TEX1] : $vin.F : F[0] : -1 : 0
#var float4 f[TEX0] : $vin.F : F[0] : -1 : 0
#var sampler2D  : texunit 0 : -1 : 0
#tram 0 = f[WPOS].w
#tram 1 = f[COL0].x
#tram 2 = f[TEX0].x
#tram 3 = f[TEX0].y
#tram 4 = f[TEX1].x
#tram 5 = f[TEX1].y
#tram 6 = f[TEX1].z
#tram 7 = f[TEX2].x
#tram 8 = f[TEX2].y
#tram 9 = f[TEX2].z
BB0:
IPA      R0, 0;
RCP      R2, R0;
IPA      R4, 2, R2;
IPA      R5, 3, R2;
IPA      R8, 7, R2;
IPA      R3, 8, R2;
IPA      R1, 9, R2;
TEX      R4, 0, 0, 2D;
FMUL32   R0, R3, R3;
FMAD     R0, R8, R8, R0;
MOV32    <<REG124>>.x, R4;
MOV32    <<REG124>>.y, R5;
MOV32    <<REG124>>.z, R6;
FMAD     R4, R1, R1, R0;
MVI      R0, 0.0;
IPA      <<REG122>>.w, 1, R2;
RSQ      R4, R4;
MOV32    <<REG124>>.w, R0;
IPA      R0, 4, R2;
FMUL32   <<REG122>>.x, R4, R8;
FMUL32   <<REG122>>.y, R4, R3;
FMUL32   <<REG122>>.z, R4, R1;
IPA      R1, 5, R2;
IPA      R2, 6, R2;
MVI      R3, 0.0;
END
# 25 instructions, 16 R-regs, 10 interpolants
# 25 inst, (4 mov, 2 mvi, 1 tex, 10 ipa, 2 complex, 6 math)
#    17 64-bit, 8 32-bit, 0 32-bit-const
