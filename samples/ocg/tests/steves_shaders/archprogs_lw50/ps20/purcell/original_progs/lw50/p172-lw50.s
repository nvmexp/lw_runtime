!!SPA1.0
.CONST_MODE  PAGE
.THREAD_TYPE PIXEL
.MAX_REG     19
.MAX_ATTR    12
# parseasm build date Mar 10 2004 15:40:49
# -profile fp50 -po tbat3 -po lat3 -if ps2x -i allprogs-new32//p172-lw40.s -o allprogs-new32//p172-lw50.s
#vendor LWPU
#version parseasm.0.0
#profile fp50
#program fp30entry
#semantic <null atom>
#semantic <null atom>
#semantic <null atom>
#semantic <null atom>
#semantic C6r0tbn6evb4u3.C6r0tbn6evb4u3
#semantic C144q3o24uukb6.C144q3o24uukb6
#semantic Coit1m35rlpml7.Coit1m35rlpml7
#semantic <null atom>
#semantic <null atom>
#semantic <null atom>
#var float4 o[COLH0] : $vout.O : O[0] : -1 : 0
#var float4 f[TEX4] : $vin.F : F[0] : -1 : 0
#var sampler2D  : texunit 5 : -1 : 0
#var sampler2D  : texunit 2 : -1 : 0
#var sampler2D  : texunit 1 : -1 : 0
#var samplerLWBE  : texunit 4 : -1 : 0
#var float4 f[TEX2] : $vin.F : F[0] : -1 : 0
#var float4 C6r0tbn6evb4u3 :  : c[2] : -1 : 0
#var float4 C144q3o24uukb6 :  : c[0] : -1 : 0
#var float4 f[TEX1] : $vin.F : F[0] : -1 : 0
#var float4 Coit1m35rlpml7 :  : c[1] : -1 : 0
#var sampler2D  : texunit 0 : -1 : 0
#var float4 f[TEX3] : $vin.F : F[0] : -1 : 0
#var sampler2D  : texunit 3 : -1 : 0
#var float4 f[TEX0] : $vin.F : F[0] : -1 : 0
#var sampler2D  : texunit 3 : -1 : 0
#tram 0 = f[WPOS].w
#tram 1 = f[TEX0].x
#tram 2 = f[TEX0].y
#tram 3 = f[TEX1].x
#tram 4 = f[TEX1].y
#tram 5 = f[TEX1].w
#tram 6 = f[TEX2].x
#tram 7 = f[TEX2].y
#tram 8 = f[TEX2].z
#tram 9 = f[TEX3].x
#tram 10 = f[TEX3].y
#tram 11 = f[TEX4].x
#tram 12 = f[TEX4].y
BB0:
IPA      R0, 0;
RCP      R12, R0;
IPA      R14, 1, R12;
IPA      R13, 2, R12;
IPA      R8, 9, R12;
MOV32    R4, R14;
MOV32    R5, R13;
IPA      R9, 10, R12;
IPA      R0, 11, R12;
IPA      R1, 12, R12;
MOV32    R2, c[4];
MOV32    R16, c[4];
MOV32    R15, c[4];
TEX      R4, 3, 3, 2D;
MOV32    R3, c[4];
TEX      R8, 3, 3, 2D;
MOV32    R7, c[4];
FMAD     R2, R6, R2, c[5];
FMAD     R5, R5, R16, c[5];
FMAD     R4, R4, R15, c[5];
FMAD     R3, R10, R3, c[5];
FMAD     R9, R9, R7, c[5];
MOV32    R7, c[4];
IPA      R6, 7, R12;
FADD32   R5, R5, R9;
FMAD     R7, R8, R7, c[5];
FADD32   R2, R2, R3;
FMUL32   R5, R5, c[8];
FADD32   R3, R4, R7;
FMUL32   R4, R2, c[8];
FMUL32   R10, R5, R6;
FMUL32   R8, R3, c[8];
IPA      R9, 6, R12;
IPA      R7, 8, R12;
TEX      R0, 5, 5, 2D;
FMAD     R10, R8, R9, R10;
FMAD     R10, R4, R7, R10;
FADD32   R11, R10, R10;
FMAD     R8, R11, R8, -R9;
FMAD     R5, R11, R5, -R6;
FMAD     R6, R11, R4, -R7;
FADD32   R4, -R10, c[7];
FMAX     R7, |R8|, |R5|;
LG2      R4, |R4|;
FMAX     R7, |R6|, R7;
FMUL32   R4, R4, c[6];
RCP      R7, R7;
RRO      R4, R4, 1;
FMUL32   R8, R8, R7;
FMUL32   R9, R5, R7;
FMUL32   R10, R6, R7;
EX2      R6, R4;
MOV32    R4, R14;
MOV32    R5, R13;
FMUL32   R16, R6, R3;
TEX      R8, 4, 4, LWBE;
MOV32    R13, c[4];
IPA      R7, 5, R12;
IPA      R6, 3, R12;
RCP      R14, R7;
MOV32    R15, c[4];
IPA      R17, 4, R12;
FMUL32   R12, R14, R6;
TEX      R4, 0, 0, 2D;
FMUL32   R14, R14, R17;
FMAD     R4, R4, R13, c[5];
FMAD     R5, R5, R15, c[5];
FMAD     R4, R4, c[0], R12;
FMAD     R5, R5, c[1], R14;
MOV32    R12, R4;
MOV32    R13, R5;
TEX      R12, 1, 1, 2D;
TEX      R4, 2, 2, 2D;
FADD32   R8, R8, R12;
FADD32   R9, R9, R13;
FADD32   R10, R10, R14;
FADD32   R11, R11, R15;
FADD32   R9, R9, -R5;
FADD32   R10, R10, -R6;
FADD32   R11, R11, -R7;
FMAD     R5, R16, R9, R5;
FMAD     R6, R16, R10, R6;
FMAD     R7, R16, R11, R7;
FMUL32   R1, R5, R1;
FMUL32   R2, R6, R2;
FMUL32   R3, R7, R3;
FADD32   R5, R8, -R4;
FMAD     R4, R16, R5, R4;
FMUL32   R0, R4, R0;
END
# 89 instructions, 20 R-regs, 13 interpolants
# 89 inst, (14 mov, 0 mvi, 7 tex, 13 ipa, 5 complex, 50 math)
#    47 64-bit, 42 32-bit, 0 32-bit-const
