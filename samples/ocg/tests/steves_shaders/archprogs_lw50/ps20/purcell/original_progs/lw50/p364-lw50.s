!!SPA1.0
.CONST_MODE  PAGE
.THREAD_TYPE PIXEL
.MAX_REG     27
.MAX_ATTR    8
# parseasm build date Mar 10 2004 15:40:49
# -profile fp50 -po tbat3 -po lat3 -if ps2x -i allprogs-new32//p364-lw40.s -o allprogs-new32//p364-lw50.s
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
#semantic <null atom>
#semantic <null atom>
#semantic <null atom>
#semantic <null atom>
#semantic <null atom>
#semantic <null atom>
#semantic <null atom>
#semantic <null atom>
#var float4 o[COLH0] : $vout.O : O[0] : -1 : 0
#var float4 Coit1m35rlpml7 :  : c[1] : -1 : 0
#var float4 C144q3o24uukb6 :  : c[0] : -1 : 0
#var sampler2D  : texunit 1 : -1 : 0
#var sampler2D  : texunit 1 : -1 : 0
#var sampler2D  : texunit 1 : -1 : 0
#var sampler2D  : texunit 1 : -1 : 0
#var sampler2D  : texunit 0 : -1 : 0
#var sampler2D  : texunit 0 : -1 : 0
#var sampler2D  : texunit 0 : -1 : 0
#var sampler2D  : texunit 0 : -1 : 0
#var float4 f[TEX3] : $vin.F : F[0] : -1 : 0
#var sampler2D  : texunit 2 : -1 : 0
#var float4 f[TEX2] : $vin.F : F[0] : -1 : 0
#var sampler2D  : texunit 2 : -1 : 0
#var float4 f[TEX1] : $vin.F : F[0] : -1 : 0
#var sampler2D  : texunit 2 : -1 : 0
#var float4 f[TEX0] : $vin.F : F[0] : -1 : 0
#var sampler2D  : texunit 2 : -1 : 0
#tram 0 = f[WPOS].w
#tram 1 = f[TEX0].x
#tram 2 = f[TEX0].y
#tram 3 = f[TEX1].x
#tram 4 = f[TEX1].y
#tram 5 = f[TEX2].x
#tram 6 = f[TEX2].y
#tram 7 = f[TEX3].x
#tram 8 = f[TEX3].y
BB0:
IPA      R0, 0;
RCP      R20, R0;
IPA      R26, 1, R20;
IPA      R25, 2, R20;
IPA      R19, 3, R20;
MOV32    R4, R26;
MOV32    R5, R25;
MOV32    R8, R19;
IPA      R18, 4, R20;
IPA      R15, 5, R20;
IPA      R14, 6, R20;
MOV32    R9, R18;
MOV32    R0, R15;
MOV32    R1, R14;
TEX      R4, 1, 1, 2D;
TEX      R8, 1, 1, 2D;
TEX      R0, 1, 1, 2D;
FADD32   R6, R6, R10;
FADD32   R5, R5, R9;
FADD32   R7, R7, R11;
FADD32   R4, R4, R8;
FADD32   R21, R1, R5;
FADD32   R22, R3, R7;
FADD32   R23, R0, R4;
FADD32   R24, R2, R6;
MOV32    R0, R26;
MOV32    R1, R25;
MOV32    R16, R19;
MOV32    R17, R18;
MOV32    R12, R15;
MOV32    R13, R14;
TEX      R0, 0, 0, 2D;
MOV32    R4, R26;
MOV32    R5, R25;
MOV32    R8, R19;
MOV32    R9, R18;
TEX      R16, 0, 0, 2D;
MOV32    R0, R15;
MOV32    R1, R14;
TEX      R12, 0, 0, 2D;
TEX      R4, 2, 2, 2D;
FADD32   R2, R3, R19;
TEX      R8, 2, 2, 2D;
FADD32   R12, R15, R2;
TEX      R0, 2, 2, 2D;
FADD32   R4, R4, R8;
FADD32   R7, R7, R11;
FADD32   R5, R5, R9;
FADD32   R6, R6, R10;
FADD32   R8, R3, R7;
FADD32   R9, R1, R5;
FADD32   R10, R2, R6;
FADD32   R14, R0, R4;
IPA      R11, 7, R20;
IPA      R13, 8, R20;
MOV32    R0, R11;
MOV32    R1, R13;
MOV32    R4, R11;
MOV32    R5, R13;
TEX      R0, 0, 0, 2D;
TEX      R4, 2, 2, 2D;
FADD32   R0, R3, R12;
FADD32   R1, R4, R14;
FADD32   R2, R7, R8;
FADD32   R3, R5, R9;
FADD32   R4, R6, R10;
FMUL32   R1, R1, c[0];
FMUL32   R2, R2, c[3];
FMUL32   R3, R3, c[1];
FMUL32   R6, R4, c[2];
FMAD     R4, R0, c[7], R2;
FMAD     R5, R0, c[5], R3;
FMAD     R6, R0, c[6], R6;
FMAD     R7, R0, c[4], R1;
MOV32    R0, R11;
MOV32    R1, R13;
TEX      R0, 1, 1, 2D;
FADD32   R2, R2, R24;
FADD32   R1, R1, R21;
FADD32   R3, R3, R22;
FADD32   R0, R0, R23;
FMUL32   R1, R1, R5;
FMUL32   R2, R2, R6;
FMUL32   R0, R0, R7;
FMAD     R1, R3, R4, R1;
FMAD     R2, R3, R4, R2;
FMAD     R0, R3, R4, R0;
FADD32I  R1, R1, -1.0;
FADD32I  R2, R2, -1.0;
FADD32I  R0, R0, -1.0;
FMUL32   R5, R3, R4;
FMAD     R3, R3, R4, R5;
FADD32I  R3, R3, -1.0;
END
# 93 instructions, 28 R-regs, 9 interpolants
# 93 inst, (24 mov, 0 mvi, 12 tex, 9 ipa, 1 complex, 47 math)
#    30 64-bit, 59 32-bit, 4 32-bit-const
