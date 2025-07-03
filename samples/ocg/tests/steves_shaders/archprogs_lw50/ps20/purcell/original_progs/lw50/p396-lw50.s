!!SPA1.0
.CONST_MODE  PAGE
.THREAD_TYPE PIXEL
.MAX_REG     39
.MAX_ATTR    22
# parseasm build date Mar 10 2004 15:40:49
# -profile fp50 -po tbat3 -po lat3 -if ps2x -i allprogs-new32//p396-lw40.s -o allprogs-new32//p396-lw50.s
#vendor LWPU
#version parseasm.0.0
#profile fp50
#program fp30entry
#semantic Clofsu13ov1o1e.Clofsu13ov1o1e
#semantic C6r0tbn6evb4u3.C6r0tbn6evb4u3
#semantic C144q3o24uukb6.C144q3o24uukb6
#semantic Coit1m35rlpml7.Coit1m35rlpml7
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
#semantic <null atom>
#var float4 o[COLH0] : $vout.O : O[0] : -1 : 0
#var float4 Clofsu13ov1o1e :  : c[3] : -1 : 0
#var float4 C6r0tbn6evb4u3 :  : c[2] : -1 : 0
#var float4 C144q3o24uukb6 :  : c[0] : -1 : 0
#var float4 Coit1m35rlpml7 :  : c[1] : -1 : 0
#var samplerLWBE  : texunit 4 : -1 : 0
#var float4 f[TEX4] : $vin.F : F[0] : -1 : 0
#var float4 f[TEX7] : $vin.F : F[0] : -1 : 0
#var float4 f[TEX5] : $vin.F : F[0] : -1 : 0
#var float4 f[TEX6] : $vin.F : F[0] : -1 : 0
#var sampler2D  : texunit 5 : -1 : 0
#var sampler2D  : texunit 2 : -1 : 0
#var sampler2D  : texunit 2 : -1 : 0
#var sampler2D  : texunit 2 : -1 : 0
#var sampler2D  : texunit 2 : -1 : 0
#var sampler2D  : texunit 2 : -1 : 0
#var sampler2D  : texunit 2 : -1 : 0
#var sampler2D  : texunit 2 : -1 : 0
#var sampler2D  : texunit 2 : -1 : 0
#var sampler2D  : texunit 2 : -1 : 0
#var float4 f[TEX2] : $vin.F : F[0] : -1 : 0
#var float4 f[TEX1] : $vin.F : F[0] : -1 : 0
#var sampler2D  : texunit 3 : -1 : 0
#var float4 f[TEX0] : $vin.F : F[0] : -1 : 0
#var sampler2D  : texunit 0 : -1 : 0
#tram 0 = f[WPOS].w
#tram 1 = f[TEX0].x
#tram 2 = f[TEX0].y
#tram 3 = f[TEX1].x
#tram 4 = f[TEX1].y
#tram 5 = f[TEX1].z
#tram 6 = f[TEX1].w
#tram 7 = f[TEX2].x
#tram 8 = f[TEX2].y
#tram 9 = f[TEX2].z
#tram 10 = f[TEX2].w
#tram 11 = f[TEX4].x
#tram 12 = f[TEX4].y
#tram 13 = f[TEX4].z
#tram 14 = f[TEX5].x
#tram 15 = f[TEX5].y
#tram 16 = f[TEX5].z
#tram 17 = f[TEX6].x
#tram 18 = f[TEX6].y
#tram 19 = f[TEX6].z
#tram 20 = f[TEX7].x
#tram 21 = f[TEX7].y
#tram 22 = f[TEX7].z
BB0:
IPA      R0, 0;
RCP      R36, R0;
IPA      R37, 1, R36;
IPA      R38, 2, R36;
IPA      R0, 6, R36;
MOV32    R4, R37;
MOV32    R5, R38;
RCP      R0, R0;
IPA      R3, 4, R36;
IPA      R2, 3, R36;
IPA      R1, 5, R36;
FMUL32   R3, R0, R3;
FMUL32   R2, R0, R2;
FMUL32   R1, R0, R1;
IPA      R0, 10, R36;
TEX      R4, 0, 0, 2D;
IPA      R8, 8, R36;
RCP      R0, R0;
IPA      R7, 7, R36;
FMUL32   R8, R0, R8;
FMUL32   R3, R3, R5;
FMUL32   R7, R0, R7;
FMUL32   R5, R8, R5;
FMAD     R2, R2, R4, R3;
IPA      R3, 9, R36;
FMAD     R4, R7, R4, R5;
FMAD     R2, R1, R6, R2;
FMUL32   R0, R0, R3;
FADD32I  R1, R2, 0.001953;
FMAD     R4, R0, R6, R4;
MOV32    R0, R1;
FADD32I  R5, R4, 0.001953;
MOV32    R32, R1;
MOV32    R24, R1;
MOV32    R1, R5;
MOV32    R33, R4;
FADD32I  R3, R4, -0.001953;
MOV32    R28, R2;
MOV32    R29, R5;
MOV32    R25, R3;
MOV32    R20, R2;
MOV32    R17, R5;
MOV32    R21, R4;
MOV32    R9, R4;
MOV32    R12, R2;
FADD32I  R2, R2, -0.001953;
MOV32    R13, R3;
MOV32    R5, R3;
MOV32    R16, R2;
MOV32    R8, R2;
MOV32    R4, R2;
TEX      R0, 2, 2, 2D;
TEX      R8, 2, 2, 2D;
TEX      R4, 2, 2, 2D;
TEX      R16, 2, 2, 2D;
TEX      R12, 2, 2, 2D;
FADD32   R3, R6, R10;
FADD32   R5, R5, R9;
FADD32   R4, R4, R8;
TEX      R20, 2, 2, 2D;
FADD32   R4, R16, R4;
FADD32   R5, R17, R5;
FADD32   R6, R18, R3;
FADD32   R3, R12, R4;
FADD32   R4, R13, R5;
FADD32   R5, R14, R6;
TEX      R28, 2, 2, 2D;
TEX      R24, 2, 2, 2D;
FADD32   R3, R20, R3;
FADD32   R4, R21, R4;
FADD32   R5, R22, R5;
TEX      R32, 2, 2, 2D;
FADD32   R5, R30, R5;
FADD32   R4, R29, R4;
FADD32   R3, R28, R3;
FADD32   R5, R26, R5;
FADD32   R4, R25, R4;
FADD32   R3, R24, R3;
FADD32   R5, R34, R5;
FADD32   R4, R33, R4;
FADD32   R3, R32, R3;
FADD32   R2, R2, R5;
FADD32   R1, R1, R4;
FADD32   R0, R0, R3;
FMUL32   R2, R2, c[2];
FMUL32   R1, R1, c[1];
FMUL32   R0, R0, c[0];
FMUL32I  R12, R2, 0.111111;
FMUL32I  R9, R1, 0.111111;
FMUL32I  R8, R0, 0.111111;
MOV32    R0, R37;
MOV32    R4, R37;
MOV32    R1, R38;
MOV32    R5, R38;
MVI      R10, -1.0;
TEX      R0, 5, 5, 2D;
MVI      R11, -1.0;
TEX      R4, 3, 3, 2D;
IPA      R3, 17, R36;
FMUL32   R2, R2, R12;
FMUL32   R1, R1, R9;
FMUL32   R0, R0, R8;
FMAD     R4, R4, c[0], R10;
FMAD     R8, R5, c[0], R11;
IPA      R9, 14, R36;
MVI      R5, -1.0;
FMUL32   R10, R8, R3;
IPA      R3, 20, R36;
FMAD     R5, R6, c[0], R5;
FMAD     R9, R9, R4, R10;
IPA      R10, 18, R36;
IPA      R6, 15, R36;
FMAD     R9, R3, R5, R9;
FMUL32   R11, R8, R10;
IPA      R10, 21, R36;
IPA      R3, 19, R36;
FMAD     R11, R6, R4, R11;
IPA      R6, 16, R36;
FMUL32   R3, R8, R3;
FMAD     R10, R10, R5, R11;
IPA      R8, 22, R36;
FMAD     R4, R6, R4, R3;
FMUL32   R6, R10, R10;
IPA      R3, 12, R36;
FMAD     R4, R8, R5, R4;
FMAD     R5, R9, R9, R6;
FMUL32   R8, R10, R3;
IPA      R6, 11, R36;
FMAD     R5, R4, R4, R5;
IPA      R11, 13, R36;
FMAD     R8, R9, R6, R8;
RCP      R5, R5;
FMAD     R8, R4, R11, R8;
FMUL32   R12, R5, R8;
FMAD     R5, R5, R8, R12;
FMAD     R6, R5, R9, -R6;
FMAD     R3, R5, R10, -R3;
FMAD     R4, R5, R4, -R11;
FMAX     R5, |R6|, |R3|;
FMAX     R5, |R4|, R5;
RCP      R5, R5;
FMUL32   R8, R6, R5;
FMUL32   R9, R3, R5;
FMUL32   R10, R4, R5;
TEX      R8, 4, 4, LWBE;
FMUL32   R3, R7, R8;
FMUL32   R4, R7, R9;
FMUL32   R3, R3, c[4];
FMUL32   R5, R7, R10;
FMUL32   R6, R4, c[5];
FMAD     R4, R3, R3, -R3;
FMUL32   R5, R5, c[6];
FMAD     R7, R6, R6, -R6;
FMAD     R3, R4, c[8], R3;
FMAD     R4, R5, R5, -R5;
FMAD     R6, R7, c[9], R6;
FMAD     R4, R4, c[10], R5;
FMUL32I  R5, R6, 0.333333;
FMAD     R5, R3, c[0], R5;
FMAD     R5, R4, c[0], R5;
FADD32   R3, R3, -R5;
FADD32   R6, R6, -R5;
FADD32   R4, R4, -R5;
FMAD     R3, R3, c[12], R5;
FMAD     R6, R6, c[13], R5;
FMAD     R4, R4, c[14], R5;
FMAD     R0, R0, c[0], R3;
FMAD     R1, R1, c[0], R6;
FMAD     R2, R2, c[0], R4;
MVI      R3, 1.0;
END
# 170 instructions, 40 R-regs, 23 interpolants
# 170 inst, (24 mov, 4 mvi, 13 tex, 23 ipa, 5 complex, 101 math)
#    82 64-bit, 80 32-bit, 8 32-bit-const
