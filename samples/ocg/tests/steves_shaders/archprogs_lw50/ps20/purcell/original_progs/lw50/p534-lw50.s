!!SPA1.0
.CONST_MODE  PAGE
.THREAD_TYPE PIXEL
.MAX_REG     27
.MAX_ATTR    17
# parseasm build date Mar 10 2004 15:40:49
# -profile fp50 -po tbat3 -po lat3 -if ps2x -i allprogs-new32//p534-lw40.s -o allprogs-new32//p534-lw50.s
#vendor LWPU
#version parseasm.0.0
#profile fp50
#program fp30entry
#semantic Coit1m35rlpml7.Coit1m35rlpml7
#semantic Clofsu13ov1o1e.Clofsu13ov1o1e
#semantic C6r0tbn6evb4u3.C6r0tbn6evb4u3
#semantic C144q3o24uukb6.C144q3o24uukb6
#semantic <null atom>
#semantic <null atom>
#semantic <null atom>
#semantic <null atom>
#semantic <null atom>
#semantic <null atom>
#semantic Cee994ndhnk1rc.Cee994ndhnk1rc
#semantic <null atom>
#var float4 o[COLH0] : $vout.O : O[0] : -1 : 0
#var float4 Coit1m35rlpml7 :  : c[1] : -1 : 0
#var float4 Clofsu13ov1o1e :  : c[3] : -1 : 0
#var float4 C6r0tbn6evb4u3 :  : c[2] : -1 : 0
#var float4 C144q3o24uukb6 :  : c[0] : -1 : 0
#var sampler2D  : texunit 2 : -1 : 0
#var sampler2D  : texunit 2 : -1 : 0
#var sampler2D  : texunit 2 : -1 : 0
#var sampler2D  : texunit 2 : -1 : 0
#var sampler2D  : texunit 5 : -1 : 0
#var samplerLWBE  : texunit 4 : -1 : 0
#var float4 Cee994ndhnk1rc :  : c[5] : -1 : 0
#var float4 f[TEX5] : $vin.F : F[0] : -1 : 0
#var float4 f[TEX1] : $vin.F : F[0] : -1 : 0
#var float4 f[TEX4] : $vin.F : F[0] : -1 : 0
#var float4 f[TEX2] : $vin.F : F[0] : -1 : 0
#var float4 f[TEX3] : $vin.F : F[0] : -1 : 0
#var float4 f[TEX0] : $vin.F : F[0] : -1 : 0
#var sampler2D  : texunit 3 : -1 : 0
#tram 0 = f[WPOS].w
#tram 1 = f[TEX0].x
#tram 2 = f[TEX0].y
#tram 3 = f[TEX1].x
#tram 4 = f[TEX1].y
#tram 5 = f[TEX1].z
#tram 6 = f[TEX2].x
#tram 7 = f[TEX2].y
#tram 8 = f[TEX2].z
#tram 9 = f[TEX3].x
#tram 10 = f[TEX3].y
#tram 11 = f[TEX3].z
#tram 12 = f[TEX4].x
#tram 13 = f[TEX4].y
#tram 14 = f[TEX4].z
#tram 15 = f[TEX5].x
#tram 16 = f[TEX5].y
#tram 17 = f[TEX5].z
BB0:
IPA      R0, 0;
MVI      R3, -1.0;
RCP      R2, R0;
IPA      R24, 1, R2;
IPA      R25, 2, R2;
IPA      R0, 17, R2;
MOV32    R20, R24;
MOV32    R21, R25;
RCP      R0, R0;
IPA      R1, 15, R2;
TEX      R20, 3, 3, 2D;
MVI      R5, -1.0;
FMUL32   R1, R0, R1;
IPA      R4, 16, R2;
FMAD     R3, R20, c[0], R3;
FMAD     R12, R21, c[0], R5;
FMUL32   R4, R0, R4;
FMUL32   R0, R23, c[20];
FMAD     R5, R3, R0, R1;
FMAD     R4, R12, R0, R4;
IPA      R0, 9, R2;
FADD32I  R7, R5, 0.001953;
FADD32I  R8, R4, 0.001953;
FMUL32   R11, R12, R0;
MOV32    R0, R7;
MOV32    R1, R8;
IPA      R9, 6, R2;
MVI      R10, -1.0;
IPA      R6, 12, R2;
FMAD     R9, R9, R3, R11;
FMAD     R13, R22, c[0], R10;
IPA      R11, 10, R2;
IPA      R10, 7, R2;
FMAD     R9, R6, R13, R9;
FMUL32   R14, R12, R11;
IPA      R6, 13, R2;
IPA      R11, 4, R2;
FMAD     R14, R10, R3, R14;
IPA      R10, 3, R2;
IPA      R15, 11, R2;
FMAD     R6, R6, R13, R14;
IPA      R14, 8, R2;
FMUL32   R15, R12, R15;
FMUL32   R16, R6, R11;
IPA      R12, 14, R2;
FMAD     R3, R14, R3, R15;
FMAD     R14, R9, R10, R16;
IPA      R15, 5, R2;
FMAD     R12, R12, R13, R3;
FMUL32   R13, R6, R6;
TEX      R0, 2, 2, 2D;
FMAD     R14, R12, R15, R14;
FMAD     R13, R9, R9, R13;
FADD32   R3, R14, R14;
FMAD     R13, R12, R12, R13;
FMUL32   R10, R13, R10;
FMUL32   R14, R13, R11;
FMUL32   R13, R13, R15;
FMAD     R11, R3, R9, -R10;
FMAD     R10, R3, R6, -R14;
FMAD     R6, R3, R12, -R13;
FADD32I  R3, R5, -0.000977;
MOV32    R5, R8;
MOV32    R8, R7;
FADD32I  R7, R4, -0.000977;
FMAX     R9, |R11|, |R10|;
MOV32    R4, R3;
MOV32    R12, R3;
FMAX     R3, |R6|, R9;
MOV32    R9, R7;
MOV32    R13, R7;
RCP      R7, R3;
MOV32    R3, R23;
TEX      R12, 2, 2, 2D;
FMUL32   R16, R11, R7;
FMUL32   R17, R10, R7;
FMUL32   R18, R6, R7;
TEX      R8, 2, 2, 2D;
TEX      R4, 2, 2, 2D;
FMUL32I  R7, R8, 0.222222;
FMUL32I  R8, R9, 0.222222;
FMUL32I  R9, R10, 0.222222;
FMAD     R7, R12, c[0], R7;
FMAD     R8, R13, c[0], R8;
FMAD     R9, R14, c[0], R9;
FMAD     R4, R4, c[0], R7;
FMAD     R5, R5, c[0], R8;
FMAD     R6, R6, c[0], R9;
FMAD     R0, R0, c[0], R4;
FMAD     R1, R1, c[0], R5;
FMAD     R2, R2, c[0], R6;
TEX      R16, 4, 4, LWBE;
FMUL32   R4, R23, R16;
FMUL32   R5, R23, R17;
FMUL32   R4, R4, c[0];
FMUL32   R7, R23, R18;
FMUL32   R6, R5, c[1];
FMAD     R8, R4, R4, -R4;
FMUL32   R5, R7, c[2];
FMAD     R7, R6, R6, -R6;
FMAD     R8, R8, c[8], R4;
FMAD     R4, R5, R5, -R5;
FMAD     R10, R7, c[9], R6;
FMAD     R9, R4, c[10], R5;
FMUL32I  R4, R10, 0.333333;
FMAD     R6, R8, c[0], R4;
MOV32    R4, R24;
MOV32    R5, R25;
FMAD     R11, R9, c[0], R6;
TEX      R4, 5, 5, 2D;
FADD32   R12, R9, -R11;
FADD32   R9, R10, -R11;
FMAD     R7, R12, c[14], R11;
FMAD     R9, R9, c[13], R11;
FADD32   R10, R8, -R11;
MOV32    R8, c[4];
MOV32    R12, c[5];
FMAD     R10, R10, c[12], R11;
FADD32   R8, R8, c[4];
FADD32   R11, R12, c[5];
MOV32    R12, c[6];
FMUL32   R4, R4, R8;
FMUL32   R5, R5, R11;
FADD32   R8, R12, c[6];
FMAD     R0, R0, R4, R10;
FMAD     R1, R1, R5, R9;
FMUL32   R4, R6, R8;
FMAD     R2, R2, R4, R7;
END
# 128 instructions, 28 R-regs, 18 interpolants
# 128 inst, (16 mov, 3 mvi, 7 tex, 18 ipa, 3 complex, 81 math)
#    74 64-bit, 46 32-bit, 8 32-bit-const
