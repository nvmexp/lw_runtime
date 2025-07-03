!!SPA1.0
.CONST_MODE  PAGE
.THREAD_TYPE PIXEL
.MAX_REG     7
.MAX_ATTR    5
# parseasm build date Mar 10 2004 15:40:49
# -profile fp50 -po tbat3 -po lat3 -if ps2x -i allprogs-new32//p509-lw40.s -o allprogs-new32//p509-lw50.s
#vendor LWPU
#version parseasm.0.0
#profile fp50
#program fp30entry
#semantic Coit1m35rlpml7.Coit1m35rlpml7
#semantic <null atom>
#var float4 o[COLH0] : $vout.O : O[0] : -1 : 0
#var float4 Coit1m35rlpml7 :  : c[1] : -1 : 0
#var float4 f[COL0] : $vin.F : F[0] : -1 : 0
#var float4 f[TEX0] : $vin.F : F[0] : -1 : 0
#var sampler2D  : texunit 0 : -1 : 0
#tram 0 = f[WPOS].w
#tram 1 = f[COL0].x
#tram 2 = f[COL0].y
#tram 3 = f[COL0].z
#tram 4 = f[TEX0].x
#tram 5 = f[TEX0].y
BB0:
IPA      R0, 0;
RCP      R4, R0;
IPA      R0, 4, R4;
IPA      R1, 5, R4;
IPA      R5, 1, R4;
IPA      R6, 2, R4;
TEX      R0, 0, 0, 2D;
FMUL32   R5, R5, c[4];
FMUL32   R6, R6, c[5];
IPA      R4, 3, R4;
FMUL32   R3, R0, R5;
FMUL32   R1, R1, R6;
FMUL32   R4, R4, c[6];
MOV32    R0, R3;
FMUL32I  R5, R1, 0.59;
FMUL32   R4, R2, R4;
FMAD     R3, R3, c[0], R5;
MOV32    R2, R4;
FMAD     R3, R4, c[0], R3;
FMUL32I  R3, R3, 0.0625;
END
# 20 instructions, 8 R-regs, 6 interpolants
# 20 inst, (2 mov, 0 mvi, 1 tex, 6 ipa, 1 complex, 10 math)
#    10 64-bit, 8 32-bit, 2 32-bit-const
