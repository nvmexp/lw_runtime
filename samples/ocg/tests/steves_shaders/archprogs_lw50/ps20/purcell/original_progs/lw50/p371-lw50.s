!!SPA1.0
.CONST_MODE  PAGE
.THREAD_TYPE PIXEL
.MAX_REG     23
.MAX_ATTR    12
# parseasm build date Mar 10 2004 15:40:49
# -profile fp50 -po tbat3 -po lat3 -if ps2x -i allprogs-new32//p371-lw40.s -o allprogs-new32//p371-lw50.s
#vendor LWPU
#version parseasm.0.0
#profile fp50
#program fp30entry
#semantic Coit1m35rlpml7.Coit1m35rlpml7
#semantic <null atom>
#semantic <null atom>
#semantic <null atom>
#semantic <null atom>
#semantic C144q3o24uukb6.C144q3o24uukb6
#var float4 o[COLH0] : $vout.O : O[0] : -1 : 0
#var float4 Coit1m35rlpml7 :  : c[1] : -1 : 0
#var float4 f[TEX3] : $vin.F : F[0] : -1 : 0
#var float4 f[TEX4] : $vin.F : F[0] : -1 : 0
#var float4 f[TEX2] : $vin.F : F[0] : -1 : 0
#var samplerLWBE  : texunit 4 : -1 : 0
#var sampler2D  : texunit 0 : -1 : 0
#var sampler2D  : texunit 1 : -1 : 0
#var sampler2D  : texunit 7 : -1 : 0
#var float4 C144q3o24uukb6 :  : c[0] : -1 : 0
#var float4 f[TEX1] : $vin.F : F[0] : -1 : 0
#tram 0 = f[WPOS].w
#tram 1 = f[TEX1].x
#tram 2 = f[TEX1].y
#tram 3 = f[TEX1].w
#tram 4 = f[TEX2].x
#tram 5 = f[TEX2].y
#tram 6 = f[TEX2].z
#tram 7 = f[TEX3].x
#tram 8 = f[TEX3].y
#tram 9 = f[TEX3].z
#tram 10 = f[TEX4].x
#tram 11 = f[TEX4].y
#tram 12 = f[TEX4].z
BB0:
IPA      R1, 0;
MOV32    R0, c[0];
RCP      R12, R1;
MOV32    R2, R0;
IPA      R0, 3, R12;
IPA      R1, 1, R12;
IPA      R3, 2, R12;
RCP      R0, R0;
MOV32    R4, c[1];
FMAD     R1, R0, R1, c[0];
FMAD     R0, R0, R3, c[0];
MOV32    R3, R4;
FMAD     R1, R1, c[0], R2;
FMAD     R2, R0, c[0], R3;
MOV32    R4, R1;
MOV32    R0, R1;
MOV32    R8, R1;
MOV32    R5, R2;
MOV32    R1, R2;
MOV32    R9, R2;
MVI      R15, -1.0;
MVI      R10, -1.0;
TEX      R4, 1, 1, 2D;
IPA      R13, 7, R12;
IPA      R14, 8, R12;
TEX      R0, 7, 7, 2D;
FMUL32   R11, R14, R14;
FMUL32I  R7, R7, 256.0;
FMAD     R11, R13, R13, R11;
IPA      R3, 9, R12;
FMAD     R0, R0, c[0], R15;
FMAD     R1, R1, c[0], R10;
FMAD     R15, R3, R3, R11;
MVI      R16, -1.0;
TEX      R8, 0, 0, 2D;
RSQ      R15, R15;
FMAD     R2, R2, c[0], R16;
FMUL32   R14, R15, R14;
FMUL32   R13, R15, R13;
FMUL32   R3, R15, R3;
FMUL32   R15, R1, R14;
IPA      R18, 10, R12;
IPA      R19, 11, R12;
FMAD     R15, R0, R13, R15;
IPA      R17, 12, R12;
FMUL32   R16, R19, R19;
FMAD     R15, R2, R3, R15;
FMAD     R20, R18, R18, R16;
FMUL32I  R16, R15, -2.0;
FMAD     R20, R17, R17, R20;
FMAD     R0, R0, R16, R13;
FMAD     R1, R1, R16, R14;
FMAD     R2, R2, R16, R3;
RSQ      R3, R20;
FMUL32   R13, R3, R19;
FMUL32   R14, R3, R18;
FMUL32   R3, R3, R17;
FMUL32   R1, R1, -R13;
FMAD     R0, R0, -R14, R1;
FMAD     R0, R2, -R3, R0;
FCMP     R0, R0, R0, c[0];
FCMP     R1, R15, R15, c[0];
LG2      R0, |R0|;
FMUL32   R0, R0, R7;
RRO      R0, R0, 1;
EX2      R0, R0;
FMUL32   R2, R4, R0;
FMUL32   R3, R5, R0;
FMUL32   R0, R6, R0;
FMAD     R2, R8, R1, R2;
FMAD     R3, R9, R1, R3;
FMAD     R0, R10, R1, R0;
FMUL32   R4, R2, c[4];
FMUL32   R5, R3, c[5];
FMUL32   R6, R0, c[6];
IPA      R0, 4, R12;
IPA      R1, 5, R12;
IPA      R2, 6, R12;
FMAX     R3, |R0|, |R1|;
FMUL32   R7, R1, R1;
FMAX     R3, |R2|, R3;
FMAD     R7, R0, R0, R7;
RCP      R3, R3;
FMAD     R7, R2, R2, R7;
FMUL32   R0, R0, R3;
FMUL32   R1, R1, R3;
FMUL32   R2, R2, R3;
RSQ      R8, R7;
TEX      R0, 4, 4, LWBE;
FMAD     R8, R8, R7, c[0];
FADD32   R0, R0, -R8;
FADD32   R1, R1, -R8;
FADD32   R2, R2, -R8;
FADD32   R3, R3, -R8;
FSET     R0, R0, c[0], LTU;
FSET     R1, R1, c[0], LTU;
FSET     R2, R2, c[0], LTU;
FSET     R3, R3, c[0], LTU;
FMUL32I  R1, R1, 0.25;
FADD32I  R7, -R7, 1.0;
FMAD     R0, R0, c[0], R1;
FCMP     R1, R7, R7, c[0];
FMAD     R0, R2, c[0], R0;
FMUL32   R1, R1, R1;
FMAD     R0, R3, c[0], R0;
FADD32I  R0, -R0, 1.0;
FMUL32   R2, R4, R0;
FMUL32   R3, R5, R0;
FMUL32   R2, R2, R1;
FMUL32   R3, R3, R1;
LG2      R2, |R2|;
FMUL32   R0, R6, R0;
LG2      R3, |R3|;
FMUL32I  R2, R2, 0.454545;
FMUL32   R0, R0, R1;
FMUL32I  R1, R3, 0.454545;
RRO      R2, R2, 1;
LG2      R3, |R0|;
RRO      R1, R1, 1;
EX2      R0, R2;
FMUL32I  R2, R3, 0.454545;
EX2      R1, R1;
MVI      R3, 1.0;
RRO      R2, R2, 1;
EX2      R2, R2;
END
# 125 instructions, 24 R-regs, 13 interpolants
# 125 inst, (10 mov, 4 mvi, 4 tex, 13 ipa, 14 complex, 80 math)
#    75 64-bit, 42 32-bit, 8 32-bit-const
