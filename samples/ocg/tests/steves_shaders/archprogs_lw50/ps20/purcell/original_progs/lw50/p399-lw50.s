!!SPA1.0
.CONST_MODE  PAGE
.THREAD_TYPE PIXEL
.MAX_REG     11
.MAX_ATTR    8
# parseasm build date Mar 10 2004 15:40:49
# -profile fp50 -po tbat3 -po lat3 -if ps2x -i allprogs-new32//p399-lw40.s -o allprogs-new32//p399-lw50.s
#vendor LWPU
#version parseasm.0.0
#profile fp50
#program fp30entry
#semantic Clofsu13ov1o1e.Clofsu13ov1o1e
#semantic Coit1m35rlpml7.Coit1m35rlpml7
#semantic C6r0tbn6evb4u3.C6r0tbn6evb4u3
#semantic C144q3o24uukb6.C144q3o24uukb6
#semantic <null atom>
#semantic <null atom>
#var float4 o[COLH0] : $vout.O : O[0] : -1 : 0
#var float4 Clofsu13ov1o1e :  : c[3] : -1 : 0
#var float4 Coit1m35rlpml7 :  : c[1] : -1 : 0
#var float4 C6r0tbn6evb4u3 :  : c[2] : -1 : 0
#var float4 C144q3o24uukb6 :  : c[0] : -1 : 0
#var float4 f[TEX2] : $vin.F : F[0] : -1 : 0
#var sampler2D  : texunit 4 : -1 : 0
#var samplerLWBE  : texunit 1 : -1 : 0
#var float4 f[TEX3] : $vin.F : F[0] : -1 : 0
#var float4 f[TEX6] : $vin.F : F[0] : -1 : 0
#tram 0 = f[WPOS].w
#tram 1 = f[TEX2].x
#tram 2 = f[TEX2].y
#tram 3 = f[TEX3].x
#tram 4 = f[TEX3].y
#tram 5 = f[TEX3].z
#tram 6 = f[TEX6].x
#tram 7 = f[TEX6].y
#tram 8 = f[TEX6].z
BB0:
IPA      R0, 0;
RCP      R0, R0;
IPA      R3, 7, R0;
IPA      R7, 4, R0;
IPA      R2, 6, R0;
IPA      R6, 3, R0;
FMUL32   R4, R3, R7;
IPA      R1, 8, R0;
IPA      R5, 5, R0;
FMAD     R8, R2, R6, R4;
FMUL32   R4, R3, R3;
FMAD     R8, R1, R5, R8;
FMAD     R4, R2, R2, R4;
FMAD     R4, R1, R1, R4;
RCP      R4, R4;
FMUL32   R9, R4, R8;
FMAD     R4, R4, R8, R9;
FMAD     R2, R4, R2, -R6;
FMAD     R3, R4, R3, -R7;
FMAD     R8, R4, R1, -R5;
IPA      R4, 1, R0;
IPA      R5, 2, R0;
FMAX     R0, |R2|, |R3|;
TEX      R4, 4, 4, 2D;
FMAX     R0, |R8|, R0;
RCP      R9, R0;
FMUL32   R0, R2, R9;
FMUL32   R1, R3, R9;
FMUL32   R2, R8, R9;
TEX      R0, 1, 1, LWBE;
FMUL32   R0, R0, R4;
FMUL32   R1, R1, R5;
FMUL32   R0, R0, c[0];
FMUL32   R3, R2, R6;
FMUL32   R2, R1, c[1];
FMAD     R4, R0, R0, -R0;
FMUL32   R1, R3, c[2];
FMAD     R3, R2, R2, -R2;
FMAD     R0, R4, c[8], R0;
FMAD     R4, R1, R1, -R1;
FMAD     R2, R3, c[9], R2;
FMAD     R1, R4, c[10], R1;
FMUL32I  R3, R2, 0.333333;
FMAD     R4, R0, c[0], R3;
FMUL32   R3, R7, c[7];
FMAD     R4, R1, c[0], R4;
FADD32   R0, R0, -R4;
FADD32   R2, R2, -R4;
FADD32   R5, R1, -R4;
FMAD     R0, R0, c[12], R4;
FMAD     R1, R2, c[13], R4;
FMAD     R2, R5, c[14], R4;
END
# 52 instructions, 12 R-regs, 9 interpolants
# 52 inst, (0 mov, 0 mvi, 2 tex, 9 ipa, 3 complex, 38 math)
#    35 64-bit, 16 32-bit, 1 32-bit-const
