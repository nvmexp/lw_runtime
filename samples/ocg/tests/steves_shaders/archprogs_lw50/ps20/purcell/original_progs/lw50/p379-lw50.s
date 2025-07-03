!!SPA1.0
.CONST_MODE  PAGE
.THREAD_TYPE PIXEL
.MAX_REG     11
.MAX_ATTR    11
# parseasm build date Mar 10 2004 15:40:49
# -profile fp50 -po tbat3 -po lat3 -if ps2x -i allprogs-new32//p379-lw40.s -o allprogs-new32//p379-lw50.s
#vendor LWPU
#version parseasm.0.0
#profile fp50
#program fp30entry
#semantic Cmur1hd5g8ngo3.Cmur1hd5g8ngo3
#semantic Clofsu13ov1o1e.Clofsu13ov1o1e
#semantic Coit1m35rlpml7.Coit1m35rlpml7
#semantic C6r0tbn6evb4u3.C6r0tbn6evb4u3
#semantic C144q3o24uukb6.C144q3o24uukb6
#semantic <null atom>
#semantic <null atom>
#var float4 o[COLH0] : $vout.O : O[0] : -1 : 0
#var float4 Cmur1hd5g8ngo3 :  : c[4] : -1 : 0
#var float4 Clofsu13ov1o1e :  : c[3] : -1 : 0
#var float4 Coit1m35rlpml7 :  : c[1] : -1 : 0
#var float4 f[COL0] : $vin.F : F[0] : -1 : 0
#var float4 C6r0tbn6evb4u3 :  : c[2] : -1 : 0
#var float4 C144q3o24uukb6 :  : c[0] : -1 : 0
#var float4 f[TEX0] : $vin.F : F[0] : -1 : 0
#var sampler2D  : texunit 0 : -1 : 0
#var samplerLWBE  : texunit 1 : -1 : 0
#var float4 f[TEX3] : $vin.F : F[0] : -1 : 0
#var float4 f[TEX6] : $vin.F : F[0] : -1 : 0
#tram 0 = f[WPOS].w
#tram 1 = f[COL0].x
#tram 2 = f[COL0].y
#tram 3 = f[COL0].z
#tram 4 = f[TEX0].x
#tram 5 = f[TEX0].y
#tram 6 = f[TEX3].x
#tram 7 = f[TEX3].y
#tram 8 = f[TEX3].z
#tram 9 = f[TEX6].x
#tram 10 = f[TEX6].y
#tram 11 = f[TEX6].z
BB0:
IPA      R0, 0;
RCP      R3, R0;
IPA      R1, 10, R3;
IPA      R5, 7, R3;
IPA      R0, 9, R3;
IPA      R7, 6, R3;
FMUL32   R6, R1, R5;
IPA      R2, 11, R3;
IPA      R4, 8, R3;
FMAD     R8, R0, R7, R6;
FMUL32   R6, R1, R1;
FMAD     R8, R2, R4, R8;
FMAD     R6, R0, R0, R6;
FMAD     R6, R2, R2, R6;
RCP      R6, R6;
FMUL32   R9, R6, R8;
FMAD     R6, R6, R8, R9;
FMAD     R0, R6, R0, -R7;
FMAD     R1, R6, R1, -R5;
FMAD     R2, R6, R2, -R4;
IPA      R4, 4, R3;
FMAX     R6, |R0|, |R1|;
IPA      R5, 5, R3;
FMAX     R9, |R2|, R6;
TEX      R4, 0, 0, 2D;
IPA      R8, 3, R3;
RCP      R9, R9;
FMUL32   R8, R8, c[6];
FMUL32   R0, R0, R9;
FMUL32   R1, R1, R9;
FMUL32   R2, R2, R9;
FMUL32   R6, R6, R8;
IPA      R8, 2, R3;
IPA      R3, 1, R3;
FMUL32   R6, R6, c[18];
FMUL32   R8, R8, c[5];
FMUL32   R3, R3, c[4];
FADD32I  R7, -R7, 1.0;
FMUL32   R5, R5, R8;
FMUL32   R4, R4, R3;
TEX      R0, 1, 1, LWBE;
FMUL32   R5, R5, c[17];
FMUL32   R4, R4, c[16];
FMUL32   R0, R0, R7;
FMUL32   R1, R1, R7;
FMUL32   R0, R0, c[0];
FMUL32   R2, R2, R7;
FMUL32   R3, R1, c[1];
FMAD     R1, R0, R0, -R0;
FMUL32   R2, R2, c[2];
FMAD     R7, R3, R3, -R3;
FMAD     R0, R1, c[8], R0;
FMAD     R1, R2, R2, -R2;
FMAD     R3, R7, c[9], R3;
FMAD     R1, R1, c[10], R2;
FMUL32I  R2, R3, 0.333333;
FMAD     R2, R0, c[0], R2;
FMAD     R2, R1, c[0], R2;
FADD32   R0, R0, -R2;
FADD32   R3, R3, -R2;
FADD32   R1, R1, -R2;
FMAD     R0, R0, c[12], R2;
FMAD     R3, R3, c[13], R2;
FMAD     R2, R1, c[14], R2;
FMAD     R0, R4, c[0], R0;
FMAD     R1, R5, c[0], R3;
FMAD     R2, R6, c[0], R2;
MOV32    R3, c[7];
END
# 68 instructions, 12 R-regs, 12 interpolants
# 68 inst, (1 mov, 0 mvi, 2 tex, 12 ipa, 3 complex, 50 math)
#    41 64-bit, 25 32-bit, 2 32-bit-const
