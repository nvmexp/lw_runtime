!!SPA1.0
.CONST_MODE  PAGE
.THREAD_TYPE PIXEL
.MAX_REG     19
.MAX_ATTR    16
# parseasm build date Mar 10 2004 15:40:49
# -profile fp50 -po tbat3 -po lat3 -if ps2x -i allprogs-new32//p391-lw40.s -o allprogs-new32//p391-lw50.s
#vendor LWPU
#version parseasm.0.0
#profile fp50
#program fp30entry
#semantic C85f7kkc64o5fe.C85f7kkc64o5fe
#semantic Cmur1hd5g8ngo3.Cmur1hd5g8ngo3
#semantic Cee994ndhnk1rc.Cee994ndhnk1rc
#semantic Clofsu13ov1o1e.Clofsu13ov1o1e
#semantic C6r0tbn6evb4u3.C6r0tbn6evb4u3
#semantic C144q3o24uukb6.C144q3o24uukb6
#semantic <null atom>
#semantic <null atom>
#semantic <null atom>
#semantic <null atom>
#var float4 o[COLH0] : $vout.O : O[0] : -1 : 0
#var float4 C85f7kkc64o5fe :  : c[6] : -1 : 0
#var float4 f[COL0] : $vin.F : F[0] : -1 : 0
#var float4 Cmur1hd5g8ngo3 :  : c[4] : -1 : 0
#var float4 Cee994ndhnk1rc :  : c[5] : -1 : 0
#var float4 Clofsu13ov1o1e :  : c[3] : -1 : 0
#var float4 C6r0tbn6evb4u3 :  : c[2] : -1 : 0
#var float4 C144q3o24uukb6 :  : c[0] : -1 : 0
#var float4 f[TEX0] : $vin.F : F[0] : -1 : 0
#var sampler2D  : texunit 0 : -1 : 0
#var float4 f[TEX2] : $vin.F : F[0] : -1 : 0
#var sampler2D  : texunit 1 : -1 : 0
#var float4 f[TEX3] : $vin.F : F[0] : -1 : 0
#var sampler2D  : texunit 5 : -1 : 0
#var samplerLWBE  : texunit 2 : -1 : 0
#var float4 f[TEX4] : $vin.F : F[0] : -1 : 0
#var float4 f[TEX7] : $vin.F : F[0] : -1 : 0
#tram 0 = f[WPOS].w
#tram 1 = f[COL0].x
#tram 2 = f[COL0].y
#tram 3 = f[COL0].z
#tram 4 = f[COL0].w
#tram 5 = f[TEX0].x
#tram 6 = f[TEX0].y
#tram 7 = f[TEX2].x
#tram 8 = f[TEX2].y
#tram 9 = f[TEX3].x
#tram 10 = f[TEX3].y
#tram 11 = f[TEX4].x
#tram 12 = f[TEX4].y
#tram 13 = f[TEX4].z
#tram 14 = f[TEX7].x
#tram 15 = f[TEX7].y
#tram 16 = f[TEX7].z
BB0:
IPA      R0, 0;
RCP      R9, R0;
IPA      R0, 7, R9;
IPA      R1, 8, R9;
IPA      R4, 5, R9;
IPA      R5, 6, R9;
IPA      R11, 3, R9;
IPA      R10, 2, R9;
TEX      R0, 1, 1, 2D;
IPA      R8, 1, R9;
TEX      R4, 0, 0, 2D;
IPA      R3, 4, R9;
IPA      R12, 15, R9;
IPA      R13, 12, R9;
FMUL32   R6, R6, R11;
FMUL32   R5, R5, R10;
FMUL32   R4, R4, R8;
FMUL32   R2, R2, R6;
FMUL32   R1, R1, R5;
FMUL32   R0, R0, R4;
FMUL32   R3, R7, R3;
FMUL32   R4, R12, R13;
IPA      R15, 14, R9;
IPA      R17, 11, R9;
IPA      R14, 16, R9;
IPA      R16, 13, R9;
FMAD     R5, R15, R17, R4;
FMUL32   R4, R12, R12;
FMAD     R5, R14, R16, R5;
FMAD     R4, R15, R15, R4;
FMAD     R4, R14, R14, R4;
RCP      R4, R4;
FMUL32   R6, R4, R5;
FMAD     R6, R4, R5, R6;
FMAD     R4, R6, R15, -R17;
FMAD     R5, R6, R12, -R13;
FMAD     R6, R6, R14, -R16;
IPA      R8, 9, R9;
IPA      R9, 10, R9;
FMAX     R7, |R4|, |R5|;
TEX      R8, 5, 5, 2D;
FMAX     R7, |R6|, R7;
RCP      R7, R7;
FMUL32   R4, R4, R7;
FMUL32   R5, R5, R7;
FMUL32   R6, R6, R7;
TEX      R4, 2, 2, LWBE;
FMUL32   R4, R4, R8;
FMUL32   R5, R5, R9;
FMUL32   R4, R4, c[0];
FMUL32   R6, R6, R10;
FMUL32   R7, R5, c[1];
FMAD     R5, R4, R4, -R4;
FMUL32   R6, R6, c[2];
FMAD     R8, R7, R7, -R7;
FMAD     R4, R5, c[8], R4;
FMAD     R5, R6, R6, -R6;
FMAD     R7, R8, c[9], R7;
FMUL32   R8, R13, R13;
FMAD     R5, R5, c[10], R6;
FMUL32I  R6, R7, 0.333333;
FMAD     R8, R17, R17, R8;
FMAD     R6, R4, c[0], R6;
FMAD     R8, R16, R16, R8;
FMAD     R6, R5, c[0], R6;
LG2      R8, R8;
FADD32   R4, R4, -R6;
FMUL32I  R8, R8, -0.5;
FMAD     R4, R4, c[12], R6;
RRO      R8, R8, 1;
EX2      R8, R8;
FMUL32   R9, R17, R8;
FMUL32   R10, R13, R8;
FMUL32   R8, R16, R8;
MOV32    R11, c[19];
FMUL32   R10, R12, R10;
FMAD     R9, R15, R9, R10;
FMAD     R8, R14, R8, R9;
FADD32   R9, R7, -R6;
FADD32I  R7, -R8, 1.0;
FMAD     R9, R9, c[13], R6;
FMUL32   R8, R7, R7;
FADD32   R5, R5, -R6;
FMUL32   R8, R8, R8;
FMAD     R5, R5, c[14], R6;
FMUL32   R6, R7, R8;
FMAD     R6, R6, c[23], R11;
FMUL32   R4, R4, R6;
FMUL32   R7, R9, R6;
FMUL32   R5, R5, R6;
FMAD     R0, R0, c[24], R4;
FMAD     R1, R1, c[24], R7;
FMAD     R2, R2, c[24], R5;
END
# 93 instructions, 20 R-regs, 17 interpolants
# 93 inst, (1 mov, 0 mvi, 4 tex, 17 ipa, 5 complex, 66 math)
#    56 64-bit, 34 32-bit, 3 32-bit-const
