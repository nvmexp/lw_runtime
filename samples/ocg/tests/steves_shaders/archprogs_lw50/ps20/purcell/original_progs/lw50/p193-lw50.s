!!SPA1.0
.CONST_MODE  PAGE
.THREAD_TYPE PIXEL
.MAX_REG     15
.MAX_ATTR    13
# parseasm build date Mar 10 2004 15:40:49
# -profile fp50 -po tbat3 -po lat3 -if ps2x -i allprogs-new32//p193-lw40.s -o allprogs-new32//p193-lw50.s
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
#semantic Clofsu13ov1o1e.Clofsu13ov1o1e
#semantic C6r0tbn6evb4u3.C6r0tbn6evb4u3
#var float4 o[COLH0] : $vout.O : O[0] : -1 : 0
#var float4 f[TEX3] : $vin.F : F[0] : -1 : 0
#var float4 f[TEX4] : $vin.F : F[0] : -1 : 0
#var float4 Coit1m35rlpml7 :  : c[1] : -1 : 0
#var float4 C144q3o24uukb6 :  : c[0] : -1 : 0
#var float4 f[TEX1] : $vin.F : F[0] : -1 : 0
#var sampler2D  : texunit 1 : -1 : 0
#var sampler3D  : texunit 0 : -1 : 0
#var sampler3D  : texunit 0 : -1 : 0
#var sampler3D  : texunit 0 : -1 : 0
#var sampler3D  : texunit 0 : -1 : 0
#var float4 Clofsu13ov1o1e :  : c[3] : -1 : 0
#var float4 C6r0tbn6evb4u3 :  : c[2] : -1 : 0
#var float4 f[TEX0] : $vin.F : F[0] : -1 : 0
#tram 0 = f[WPOS].w
#tram 1 = f[TEX0].x
#tram 2 = f[TEX0].y
#tram 3 = f[TEX0].z
#tram 4 = f[TEX1].x
#tram 5 = f[TEX1].y
#tram 6 = f[TEX3].x
#tram 7 = f[TEX3].y
#tram 8 = f[TEX3].z
#tram 9 = f[TEX3].w
#tram 10 = f[TEX4].x
#tram 11 = f[TEX4].y
#tram 12 = f[TEX4].z
#tram 13 = f[TEX4].w
BB0:
IPA      R0, 0;
RCP      R12, R0;
IPA      R13, 1, R12;
IPA      R1, 2, R12;
IPA      R14, 3, R12;
FMUL32   R4, R13, c[10];
FMUL32   R2, R1, c[10];
FMUL32   R3, R14, c[10];
FMUL32I  R0, R4, 4.0;
FMUL32   R10, R1, c[13];
FMUL32I  R1, R2, 4.0;
MOV32    R5, R2;
FMUL32I  R2, R3, 4.0;
MOV32    R6, R3;
MOV32    R8, R10;
TEX      R0, 0, 0, 3D;
TEX      R4, 0, 0, 3D;
MOV32    R9, R10;
MVI      R1, -1.0;
FMAD     R0, R0, c[0], R4;
TEX      R8, 0, 0, 3D;
FMAD     R2, R0, c[9], R13;
FMAD     R0, R0, c[9], R14;
FMAD     R4, R8, c[0], R1;
MVI      R3, 2.0;
FMAD     R1, R4, c[12], R0;
FMAD     R0, R4, c[12], R2;
FMUL32   R2, R0, R0;
FMAD     R4, R1, R1, R2;
RSQ      R5, R4;
FMUL32   R0, R0, c[15];
FMUL32   R2, R1, c[15];
FMUL32   R4, R5, R4;
MVI      R1, 0.0;
FMUL32   R5, R4, c[8];
MVI      R4, -1.0;
FMAD     R6, R5, c[0], R3;
RCP      R7, R5;
TEX      R0, 0, 0, 3D;
MVI      R8, 0.45;
FMUL32.SAT R6, R6, R7;
F2F.SAT  R1, R6;
FMAD     R0, R0, c[0], R4;
FMUL32   R1, R1, c[14];
MVI      R2, 0.25;
FMAD     R0, R0, R1, R5;
FRC      R0, R0;
FMAD     R4, R0, c[0], R8;
FMAD     R1, R0, c[0], R2;
RCP      R2, R0;
MOV32    R0, -c[2];
FMUL32   R1, R1, R2;
F2F      R3, R0;
FMAD     R0, R4, R2, c[0];
FMUL32   R2, R4, R2;
FMIN     R1, R1, c[0];
FADD32   R3, R3, c[6];
FCMP     R2, R0, c[0], R2;
MOV32    R4, -c[1];
MOV32    R0, -c[0];
FADD32   R2, -R1, R2;
F2F      R1, R4;
F2F      R0, R0;
FMAD     R13, R3, R2, c[2];
FADD32   R3, R1, c[5];
FADD32   R1, R0, c[4];
IPA      R0, 4, R12;
FMAD     R11, R3, R2, c[1];
FMAD     R4, R1, R2, c[0];
IPA      R1, 5, R12;
IPA      R7, 10, R12;
IPA      R8, 11, R12;
IPA      R6, 12, R12;
IPA      R5, 13, R12;
FMUL32   R9, R8, R8;
TEX      R0, 1, 1, 2D;
IPA      R10, 6, R12;
FMAD     R9, R7, R7, R9;
IPA      R3, 7, R12;
FMAD     R9, R6, R6, R9;
FMUL32   R0, R0, R4;
FMUL32   R4, R3, R3;
FMAD     R5, R5, R5, R9;
IPA      R9, 8, R12;
FMAD     R4, R10, R10, R4;
RSQ      R5, R5;
IPA      R12, 9, R12;
FMAD     R4, R9, R9, R4;
FMUL32   R8, R5, R8;
FMUL32   R7, R5, R7;
FMUL32   R5, R5, R6;
FMAD     R4, R12, R12, R4;
RSQ      R4, R4;
FMUL32   R3, R4, R3;
FMUL32   R6, R4, R10;
FMUL32   R4, R4, R9;
FMUL32   R3, R8, R3;
FMUL32   R1, R1, R11;
MVI      R8, 1.0;
FMAD     R3, R7, R6, R3;
FMUL32   R2, R2, R13;
MVI      R6, 0.0;
FMAD     R4, R5, R4, R3;
MOV32    R3, R6;
FCMP     R4, R4, R4, c[0];
FMUL32   R4, R4, R4;
FMAD     R4, R4, c[0], R8;
FMUL32   R0, R0, R4;
FMUL32   R1, R1, R4;
FMUL32   R2, R2, R4;
END
# 110 instructions, 16 R-regs, 14 interpolants
# 110 inst, (8 mov, 8 mvi, 5 tex, 14 ipa, 6 complex, 69 math)
#    66 64-bit, 41 32-bit, 3 32-bit-const
