!!SPA1.0
.CONST_MODE  PAGE
.THREAD_TYPE PIXEL
.MAX_REG     11
.MAX_ATTR    11
# parseasm build date Mar 10 2004 15:40:49
# -profile fp50 -po tbat3 -po lat3 -if ps2x -i allprogs-new32//p393-lw40.s -o allprogs-new32//p393-lw50.s
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
RCP      R8, R0;
IPA      R2, 10, R8;
IPA      R3, 7, R8;
IPA      R4, 9, R8;
IPA      R6, 6, R8;
FMUL32   R5, R2, R3;
IPA      R0, 11, R8;
IPA      R1, 8, R8;
FMAD     R7, R4, R6, R5;
FMUL32   R5, R2, R2;
FMAD     R7, R0, R1, R7;
FMAD     R5, R4, R4, R5;
FMAD     R5, R0, R0, R5;
RCP      R5, R5;
FMUL32   R9, R5, R7;
FMAD     R7, R5, R7, R9;
FMAD     R4, R7, R4, -R6;
FMAD     R5, R7, R2, -R3;
FMAD     R6, R7, R0, -R1;
IPA      R0, 4, R8;
FMAX     R2, |R4|, |R5|;
IPA      R1, 5, R8;
FMAX     R7, |R6|, R2;
TEX      R0, 0, 0, 2D;
RCP      R7, R7;
FMUL32   R4, R4, R7;
FMUL32   R5, R5, R7;
FMUL32   R6, R6, R7;
TEX      R4, 1, 1, LWBE;
FMUL32   R4, R4, c[0];
FMUL32   R5, R5, c[1];
FMUL32   R7, R6, c[2];
FMAD     R6, R4, R4, -R4;
FMAD     R10, R5, R5, -R5;
FMAD     R9, R7, R7, -R7;
FMAD     R4, R6, c[8], R4;
FMAD     R5, R10, c[9], R5;
FMAD     R9, R9, c[10], R7;
FMUL32I  R6, R5, 0.333333;
FMAD     R7, R4, c[0], R6;
IPA      R6, 1, R8;
FMAD     R7, R9, c[0], R7;
FMUL32   R6, R6, c[4];
FADD32   R9, R9, -R7;
FADD32   R5, R5, -R7;
FMUL32   R0, R0, R6;
FMAD     R6, R9, c[14], R7;
FMAD     R5, R5, c[13], R7;
FADD32   R4, R4, -R7;
FMUL32   R0, R0, c[16];
IPA      R9, 2, R8;
FMAD     R4, R4, c[12], R7;
IPA      R7, 3, R8;
FMUL32   R8, R9, c[5];
FMAD     R0, R0, c[0], R4;
FMUL32   R4, R7, c[6];
FMUL32   R1, R1, R8;
FMUL32   R3, R3, c[7];
FMUL32   R2, R2, R4;
FMUL32   R1, R1, c[17];
FMUL32   R2, R2, c[18];
FMAD     R1, R1, c[0], R5;
FMAD     R2, R2, c[0], R6;
END
# 64 instructions, 12 R-regs, 12 interpolants
# 64 inst, (0 mov, 0 mvi, 2 tex, 12 ipa, 3 complex, 47 math)
#    41 64-bit, 22 32-bit, 1 32-bit-const
