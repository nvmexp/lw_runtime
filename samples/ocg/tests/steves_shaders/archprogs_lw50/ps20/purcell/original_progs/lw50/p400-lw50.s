!!SPA1.0
.CONST_MODE  PAGE
.THREAD_TYPE PIXEL
.MAX_REG     11
.MAX_ATTR    6
# parseasm build date Mar 10 2004 15:40:49
# -profile fp50 -po tbat3 -po lat3 -if ps2x -i allprogs-new32//p400-lw40.s -o allprogs-new32//p400-lw50.s
#vendor LWPU
#version parseasm.0.0
#profile fp50
#program fp30entry
#semantic Coit1m35rlpml7.Coit1m35rlpml7
#semantic Clofsu13ov1o1e.Clofsu13ov1o1e
#semantic C6r0tbn6evb4u3.C6r0tbn6evb4u3
#semantic C144q3o24uukb6.C144q3o24uukb6
#semantic <null atom>
#var float4 o[COLH0] : $vout.O : O[0] : -1 : 0
#var float4 Coit1m35rlpml7 :  : c[1] : -1 : 0
#var float4 Clofsu13ov1o1e :  : c[3] : -1 : 0
#var float4 C6r0tbn6evb4u3 :  : c[2] : -1 : 0
#var float4 C144q3o24uukb6 :  : c[0] : -1 : 0
#var samplerLWBE  : texunit 1 : -1 : 0
#var float4 f[TEX3] : $vin.F : F[0] : -1 : 0
#var float4 f[TEX6] : $vin.F : F[0] : -1 : 0
#tram 0 = f[WPOS].w
#tram 1 = f[TEX3].x
#tram 2 = f[TEX3].y
#tram 3 = f[TEX3].z
#tram 4 = f[TEX6].x
#tram 5 = f[TEX6].y
#tram 6 = f[TEX6].z
BB0:
IPA      R0, 0;
RCP      R4, R0;
IPA      R2, 5, R4;
IPA      R6, 2, R4;
IPA      R1, 4, R4;
IPA      R5, 1, R4;
FMUL32   R3, R2, R6;
IPA      R0, 6, R4;
IPA      R4, 3, R4;
FMAD     R7, R1, R5, R3;
FMUL32   R3, R2, R2;
FMAD     R7, R0, R4, R7;
FMAD     R3, R1, R1, R3;
FMAD     R3, R0, R0, R3;
RCP      R3, R3;
FMUL32   R8, R3, R7;
FMAD     R3, R3, R7, R8;
FMAD     R1, R3, R1, -R5;
FMAD     R2, R3, R2, -R6;
FMAD     R3, R3, R0, -R4;
FMAX     R0, |R1|, |R2|;
FMAX     R0, |R3|, R0;
RCP      R4, R0;
FMUL32   R0, R1, R4;
FMUL32   R1, R2, R4;
FMUL32   R2, R3, R4;
TEX      R0, 1, 1, LWBE;
FMUL32   R0, R0, c[0];
FMUL32   R1, R1, c[1];
FMUL32   R2, R2, c[2];
FMAD     R5, R0, R0, -R0;
FMAD     R4, R1, R1, -R1;
FMAD     R3, R2, R2, -R2;
FMAD     R0, R5, c[8], R0;
FMAD     R1, R4, c[9], R1;
FMAD     R4, R3, c[10], R2;
FMUL32I  R2, R1, 0.333333;
FMAD     R2, R0, c[0], R2;
MOV32    R3, c[7];
FMAD     R2, R4, c[0], R2;
FADD32   R0, R0, -R2;
FADD32   R1, R1, -R2;
FADD32   R4, R4, -R2;
FMAD     R0, R0, c[12], R2;
FMAD     R1, R1, c[13], R2;
FMAD     R2, R4, c[14], R2;
END
# 46 instructions, 12 R-regs, 7 interpolants
# 46 inst, (1 mov, 0 mvi, 1 tex, 7 ipa, 3 complex, 34 math)
#    32 64-bit, 13 32-bit, 1 32-bit-const
