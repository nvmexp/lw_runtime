!!SPA1.0
.CONST_MODE  PAGE
.THREAD_TYPE PIXEL
.MAX_REG     15
.MAX_ATTR    13
# parseasm build date Mar 10 2004 15:40:49
# -profile fp50 -po tbat3 -po lat3 -if ps2x -i allprogs-new32//p408-lw40.s -o allprogs-new32//p408-lw50.s
#vendor LWPU
#version parseasm.0.0
#profile fp50
#program fp30entry
#semantic Clofsu13ov1o1e.Clofsu13ov1o1e
#semantic Cee994ndhnk1rc.Cee994ndhnk1rc
#semantic C6r0tbn6evb4u3.C6r0tbn6evb4u3
#semantic Cmur1hd5g8ngo3.Cmur1hd5g8ngo3
#semantic C144q3o24uukb6.C144q3o24uukb6
#semantic Coit1m35rlpml7.Coit1m35rlpml7
#semantic <null atom>
#semantic <null atom>
#semantic <null atom>
#var float4 o[COLH0] : $vout.O : O[0] : -1 : 0
#var float4 Clofsu13ov1o1e :  : c[3] : -1 : 0
#var float4 Cee994ndhnk1rc :  : c[5] : -1 : 0
#var float4 C6r0tbn6evb4u3 :  : c[2] : -1 : 0
#var float4 Cmur1hd5g8ngo3 :  : c[4] : -1 : 0
#var float4 C144q3o24uukb6 :  : c[0] : -1 : 0
#var float4 Coit1m35rlpml7 :  : c[1] : -1 : 0
#var float4 f[COL0] : $vin.F : F[0] : -1 : 0
#var float4 f[TEX0] : $vin.F : F[0] : -1 : 0
#var sampler2D  : texunit 0 : -1 : 0
#var float4 f[TEX2] : $vin.F : F[0] : -1 : 0
#var sampler2D  : texunit 4 : -1 : 0
#var samplerLWBE  : texunit 1 : -1 : 0
#var float4 f[TEX3] : $vin.F : F[0] : -1 : 0
#var float4 f[TEX6] : $vin.F : F[0] : -1 : 0
#tram 0 = f[WPOS].w
#tram 1 = f[COL0].x
#tram 2 = f[COL0].y
#tram 3 = f[COL0].z
#tram 4 = f[TEX0].x
#tram 5 = f[TEX0].y
#tram 6 = f[TEX2].x
#tram 7 = f[TEX2].y
#tram 8 = f[TEX3].x
#tram 9 = f[TEX3].y
#tram 10 = f[TEX3].z
#tram 11 = f[TEX6].x
#tram 12 = f[TEX6].y
#tram 13 = f[TEX6].z
BB0:
IPA      R0, 0;
RCP      R12, R0;
IPA      R1, 12, R12;
IPA      R4, 9, R12;
IPA      R0, 11, R12;
IPA      R6, 8, R12;
FMUL32   R5, R1, R4;
IPA      R2, 13, R12;
IPA      R3, 10, R12;
FMAD     R7, R0, R6, R5;
FMUL32   R5, R1, R1;
FMAD     R7, R2, R3, R7;
FMAD     R5, R0, R0, R5;
FMAD     R5, R2, R2, R5;
RCP      R5, R5;
FMUL32   R8, R5, R7;
FMAD     R5, R5, R7, R8;
FMAD     R0, R5, R0, -R6;
FMAD     R1, R5, R1, -R4;
FMAD     R2, R5, R2, -R3;
IPA      R8, 6, R12;
FMAX     R3, |R0|, |R1|;
IPA      R9, 7, R12;
IPA      R4, 4, R12;
FMAX     R3, |R2|, R3;
IPA      R5, 5, R12;
TEX      R8, 4, 4, 2D;
RCP      R3, R3;
TEX      R4, 0, 0, 2D;
FMUL32   R0, R0, R3;
FMUL32   R1, R1, R3;
FMUL32   R2, R2, R3;
TEX      R0, 1, 1, LWBE;
FMUL32   R0, R0, R8;
FMUL32   R1, R1, R9;
FMUL32   R0, R0, c[0];
FMUL32   R2, R2, R10;
FMUL32   R1, R1, c[1];
FMAD     R8, R0, R0, -R0;
FMUL32   R2, R2, c[2];
FMAD     R3, R1, R1, -R1;
FMAD     R8, R8, c[8], R0;
FMAD     R0, R2, R2, -R2;
FMAD     R1, R3, c[9], R1;
FMAD     R2, R0, c[10], R2;
FMUL32I  R0, R1, 0.333333;
FMAD     R0, R8, c[0], R0;
IPA      R3, 3, R12;
FMAD     R0, R2, c[0], R0;
FMUL32   R3, R3, c[6];
FADD32   R2, R2, -R0;
FADD32   R1, R1, -R0;
FADD32   R8, R8, -R0;
FMAD     R2, R2, c[14], R0;
FMAD     R1, R1, c[13], R0;
FMAD     R0, R8, c[12], R0;
FMUL32   R8, R6, R3;
IPA      R3, 2, R12;
FMUL32   R9, R8, c[18];
FMUL32   R3, R3, c[5];
FMAD     R8, R8, c[18], R9;
FMUL32   R3, R5, R3;
IPA      R9, 1, R12;
FMAD     R10, R6, c[22], -R8;
FMUL32   R6, R3, c[17];
FMUL32   R9, R9, c[4];
FMAD     R8, R7, R10, R8;
FMAD     R3, R3, c[17], R6;
FMUL32   R6, R4, R9;
FMAD     R5, R5, c[21], -R3;
FADD32   R2, R2, R8;
FMUL32   R8, R6, c[16];
FMAD     R5, R7, R5, R3;
MOV32    R3, c[7];
FMAD     R6, R6, c[16], R8;
FADD32   R1, R1, R5;
FMAD     R4, R4, c[20], -R6;
FMAD     R4, R7, R4, R6;
FADD32   R0, R0, R4;
END
# 79 instructions, 16 R-regs, 14 interpolants
# 79 inst, (1 mov, 0 mvi, 3 tex, 14 ipa, 3 complex, 58 math)
#    50 64-bit, 28 32-bit, 1 32-bit-const
