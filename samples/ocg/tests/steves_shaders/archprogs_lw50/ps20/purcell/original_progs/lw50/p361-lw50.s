!!SPA1.0
.CONST_MODE  PAGE
.THREAD_TYPE PIXEL
.MAX_REG     11
.MAX_ATTR    2
# parseasm build date Mar 10 2004 15:40:49
# -profile fp50 -po tbat3 -po lat3 -if ps2x -i allprogs-new32//p361-lw40.s -o allprogs-new32//p361-lw50.s
#vendor LWPU
#version parseasm.0.0
#profile fp50
#program fp30entry
#semantic Coit1m35rlpml7.Coit1m35rlpml7
#semantic <null atom>
#semantic C144q3o24uukb6.C144q3o24uukb6
#semantic <null atom>
#semantic <null atom>
#var float4 o[COLH0] : $vout.O : O[0] : -1 : 0
#var float4 Coit1m35rlpml7 :  : c[1] : -1 : 0
#var sampler2D  : texunit 2 : -1 : 0
#var float4 C144q3o24uukb6 :  : c[0] : -1 : 0
#var sampler2D  : texunit 1 : -1 : 0
#var float4 f[TEX0] : $vin.F : F[0] : -1 : 0
#var sampler2D  : texunit 0 : -1 : 0
#tram 0 = f[WPOS].w
#tram 1 = f[TEX0].x
#tram 2 = f[TEX0].y
BB0:
IPA      R0, 0;
RCP      R0, R0;
IPA      R1, 1, R0;
IPA      R2, 2, R0;
MOV32    R0, R1;
MOV32    R4, R1;
MOV32    R1, R2;
MOV32    R5, R2;
TEX      R0, 0, 0, 2D;
TEX      R4, 1, 1, 2D;
FMUL32   R3, R1, R1;
FMAD     R3, R0, R0, R3;
FMAD     R3, R2, R2, R3;
RSQ      R3, R3;
FMAD     R0, R0, -R3, -c[0];
FMAD     R1, R1, -R3, -c[1];
FMAD     R2, R2, -R3, -c[2];
FMUL32   R3, R1, R1;
FMAD     R3, R0, R0, R3;
FMAD     R7, R2, R2, R3;
FMUL32   R3, R5, -c[1];
RSQ      R7, R7;
FMAD     R3, R4, -c[0], R3;
FMUL32   R2, R2, R7;
FMUL32   R8, R0, R7;
FMUL32   R1, R1, R7;
FMAD     R0, R6, -c[2], R3;
FMUL32   R1, R1, R5;
FMAD     R1, R8, R4, R1;
FMAD     R1, R2, R6, R1;
TEX      R0, 2, 2, 2D;
FMUL32   R0, R0, c[4];
FMUL32   R1, R1, c[5];
FMUL32   R2, R2, c[6];
FMUL32   R3, R3, c[7];
END
# 35 instructions, 12 R-regs, 3 interpolants
# 35 inst, (4 mov, 0 mvi, 3 tex, 3 ipa, 3 complex, 22 math)
#    20 64-bit, 15 32-bit, 0 32-bit-const
