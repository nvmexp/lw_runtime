!!SPA1.0
.CONST_MODE  PAGE
.THREAD_TYPE PIXEL
.MAX_REG     15
.MAX_ATTR    2
# parseasm build date Mar 10 2004 15:40:49
# -profile fp50 -po tbat3 -po lat3 -if ps2x -i allprogs-new32//p366-lw40.s -o allprogs-new32//p366-lw50.s
#vendor LWPU
#version parseasm.0.0
#profile fp50
#program fp30entry
#semantic Coit1m35rlpml7.Coit1m35rlpml7
#semantic C144q3o24uukb6.C144q3o24uukb6
#semantic <null atom>
#semantic <null atom>
#semantic <null atom>
#semantic <null atom>
#var float4 o[COLH0] : $vout.O : O[0] : -1 : 0
#var float4 Coit1m35rlpml7 :  : c[1] : -1 : 0
#var float4 C144q3o24uukb6 :  : c[0] : -1 : 0
#var sampler2D  : texunit 3 : -1 : 0
#var sampler2D  : texunit 1 : -1 : 0
#var sampler2D  : texunit 0 : -1 : 0
#var float4 f[TEX0] : $vin.F : F[0] : -1 : 0
#var sampler2D  : texunit 2 : -1 : 0
#tram 0 = f[WPOS].w
#tram 1 = f[TEX0].x
#tram 2 = f[TEX0].y
BB0:
IPA      R0, 0;
RCP      R0, R0;
IPA      R12, 1, R0;
IPA      R13, 2, R0;
MOV32    R4, R12;
MOV32    R5, R13;
MOV32    R8, R12;
MOV32    R9, R13;
MOV32    R0, R12;
MOV32    R1, R13;
TEX      R4, 0, 0, 2D;
TEX      R8, 2, 2, 2D;
TEX      R0, 1, 1, 2D;
FMUL32   R4, R11, c[3];
FMUL32   R5, R10, c[2];
FMUL32   R6, R9, c[1];
FMUL32   R8, R8, c[0];
FMAD     R9, R7, c[6], R5;
FMAD     R6, R7, c[5], R6;
FMAD     R8, R7, c[4], R8;
FMAD     R5, R7, c[7], R4;
FMUL32   R2, R2, R9;
FMUL32   R1, R1, R6;
FMUL32   R0, R0, R8;
FMAD     R4, R3, R5, R2;
FMAD     R6, R3, R5, R1;
FMAD     R8, R3, R5, R0;
MOV32    R0, R12;
MOV32    R1, R13;
FMUL32   R7, R3, R5;
F2F      R10, -R4;
F2F      R9, -R6;
FMAD     R5, R3, R5, R7;
F2F      R11, -R8;
TEX      R0, 3, 3, 2D;
FADD32   R0, R0, R11;
FADD32   R1, R1, R9;
FADD32   R2, R2, R10;
FMAD     R0, R0, c[0], R8;
FMAD     R1, R1, c[0], R6;
FMAD     R2, R2, c[0], R4;
FMAD     R3, R7, c[0], R3;
FMAD     R3, R3, c[0], R5;
END
# 43 instructions, 16 R-regs, 3 interpolants
# 43 inst, (8 mov, 0 mvi, 4 tex, 3 ipa, 1 complex, 27 math)
#    24 64-bit, 19 32-bit, 0 32-bit-const
