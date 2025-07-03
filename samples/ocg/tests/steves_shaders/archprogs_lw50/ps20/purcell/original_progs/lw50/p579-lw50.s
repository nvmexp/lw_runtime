!!SPA1.0
.CONST_MODE  PAGE
.THREAD_TYPE PIXEL
.MAX_REG     7
.MAX_ATTR    2
# parseasm build date Mar 10 2004 15:40:49
# -profile fp50 -po tbat3 -po lat3 -if ps2x -i allprogs-new32//p579-lw40.s -o allprogs-new32//p579-lw50.s
#vendor LWPU
#version parseasm.0.0
#profile fp50
#program fp30entry
#semantic Coit1m35rlpml7.Coit1m35rlpml7
#semantic <null atom>
#var float4 o[COLH0] : $vout.O : O[0] : -1 : 0
#var float4 Coit1m35rlpml7 :  : c[1] : -1 : 0
#var float4 f[TEX0] : $vin.F : F[0] : -1 : 0
#var sampler2D  : texunit 0 : -1 : 0
#tram 0 = f[WPOS].w
#tram 1 = f[TEX0].x
#tram 2 = f[TEX0].y
BB0:
IPA      R0, 0;
RCP      R1, R0;
IPA      R0, 1, R1;
IPA      R1, 2, R1;
TEX      R0, 0, 0, 2D;
FMUL32   R3, R0, c[4];
FMUL32   R1, R1, c[5];
FMUL32   R4, R2, c[6];
MOV32    R0, R3;
FMUL32I  R5, R1, 0.59;
MOV32    R2, R4;
FMAD     R3, R3, c[0], R5;
FMAD     R3, R4, c[0], R3;
FMUL32I  R3, R3, 0.0625;
END
# 14 instructions, 8 R-regs, 3 interpolants
# 14 inst, (2 mov, 0 mvi, 1 tex, 3 ipa, 1 complex, 7 math)
#    7 64-bit, 5 32-bit, 2 32-bit-const
