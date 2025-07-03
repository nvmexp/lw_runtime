!!SPA1.0
.CONST_MODE  PAGE
.THREAD_TYPE PIXEL
.MAX_REG     11
.MAX_ATTR    7
# parseasm build date Mar 10 2004 15:40:49
# -profile fp50 -po tbat3 -po lat3 -if ps2x -i allprogs-new32//p200-lw40.s -o allprogs-new32//p200-lw50.s
#vendor LWPU
#version parseasm.0.0
#profile fp50
#program fp30entry
#semantic Coit1m35rlpml7.Coit1m35rlpml7
#semantic C6r0tbn6evb4u3.C6r0tbn6evb4u3
#semantic C144q3o24uukb6.C144q3o24uukb6
#semantic <null atom>
#semantic <null atom>
#var float4 o[COLH0] : $vout.O : O[0] : -1 : 0
#var float4 Coit1m35rlpml7 :  : c[1] : -1 : 0
#var float4 C6r0tbn6evb4u3 :  : c[2] : -1 : 0
#var float4 f[TEX2] : $vin.F : F[0] : -1 : 0
#var float4 C144q3o24uukb6 :  : c[0] : -1 : 0
#var float4 f[TEX1] : $vin.F : F[0] : -1 : 0
#var sampler2D  : texunit 1 : -1 : 0
#var float4 f[TEX0] : $vin.F : F[0] : -1 : 0
#var sampler2D  : texunit 0 : -1 : 0
#tram 0 = f[WPOS].w
#tram 1 = f[TEX0].x
#tram 2 = f[TEX0].y
#tram 3 = f[TEX1].x
#tram 4 = f[TEX1].y
#tram 5 = f[TEX2].x
#tram 6 = f[TEX2].y
#tram 7 = f[TEX2].z
BB0:
IPA      R0, 0;
RCP      R7, R0;
IPA      R0, 3, R7;
IPA      R1, 4, R7;
IPA      R4, 1, R7;
IPA      R5, 2, R7;
IPA      R8, 5, R7;
IPA      R6, 6, R7;
TEX      R0, 1, 1, 2D;
FADD32   R9, -R8, c[0];
FADD32   R6, -R6, c[1];
IPA      R8, 7, R7;
FMUL32   R10, R6, R6;
TEX      R4, 0, 0, 2D;
FADD32   R8, -R8, c[2];
FMAD     R9, R9, R9, R10;
FMAD     R8, R8, R8, R9;
FADD32   R8, R8, c[9];
RCP      R8, R8;
FMAD     R7, R8, c[7], R7;
FMAD     R6, R8, c[6], R6;
FMAD     R5, R8, c[5], R5;
FMAD     R4, R8, c[4], R4;
FADD32   R2, R2, -R6;
FADD32   R1, R1, -R5;
FADD32   R0, R0, -R4;
FMAD     R2, R3, R2, R6;
FMAD     R1, R3, R1, R5;
FMAD     R0, R3, R0, R4;
FADD32   R4, R3, -R7;
FMAD     R3, R3, R4, R7;
END
# 31 instructions, 12 R-regs, 8 interpolants
# 31 inst, (0 mov, 0 mvi, 2 tex, 8 ipa, 2 complex, 19 math)
#    22 64-bit, 9 32-bit, 0 32-bit-const
