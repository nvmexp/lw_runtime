!!SPA1.0
.CONST_MODE  PAGE
.THREAD_TYPE PIXEL
.MAX_REG     19
.MAX_ATTR    14
# parseasm build date Mar 10 2004 15:40:49
# -profile fp50 -po tbat3 -po lat3 -if ps2x -i allprogs-new32//p406-lw40.s -o allprogs-new32//p406-lw50.s
#vendor LWPU
#version parseasm.0.0
#profile fp50
#program fp30entry
#semantic C85f7kkc64o5fe.C85f7kkc64o5fe
#semantic Clofsu13ov1o1e.Clofsu13ov1o1e
#semantic C6r0tbn6evb4u3.C6r0tbn6evb4u3
#semantic C144q3o24uukb6.C144q3o24uukb6
#semantic <null atom>
#semantic <null atom>
#semantic <null atom>
#semantic Cmur1hd5g8ngo3.Cmur1hd5g8ngo3
#semantic Cee994ndhnk1rc.Cee994ndhnk1rc
#var float4 o[COLH0] : $vout.O : O[0] : -1 : 0
#var float4 C85f7kkc64o5fe :  : c[6] : -1 : 0
#var float4 Clofsu13ov1o1e :  : c[3] : -1 : 0
#var float4 f[COL0] : $vin.F : F[0] : -1 : 0
#var float4 C6r0tbn6evb4u3 :  : c[2] : -1 : 0
#var float4 C144q3o24uukb6 :  : c[0] : -1 : 0
#var float4 f[TEX2] : $vin.F : F[0] : -1 : 0
#var sampler2D  : texunit 1 : -1 : 0
#var float4 f[TEX0] : $vin.F : F[0] : -1 : 0
#var sampler2D  : texunit 0 : -1 : 0
#var samplerLWBE  : texunit 2 : -1 : 0
#var float4 Cmur1hd5g8ngo3 :  : c[4] : -1 : 0
#var float4 Cee994ndhnk1rc :  : c[5] : -1 : 0
#var float4 f[TEX7] : $vin.F : F[0] : -1 : 0
#var float4 f[TEX4] : $vin.F : F[0] : -1 : 0
#tram 0 = f[WPOS].w
#tram 1 = f[COL0].x
#tram 2 = f[COL0].y
#tram 3 = f[COL0].z
#tram 4 = f[COL0].w
#tram 5 = f[TEX0].x
#tram 6 = f[TEX0].y
#tram 7 = f[TEX2].x
#tram 8 = f[TEX2].y
#tram 9 = f[TEX4].x
#tram 10 = f[TEX4].y
#tram 11 = f[TEX4].z
#tram 12 = f[TEX7].x
#tram 13 = f[TEX7].y
#tram 14 = f[TEX7].z
BB0:
IPA      R0, 0;
RCP      R12, R0;
IPA      R18, 13, R12;
IPA      R16, 10, R12;
IPA      R17, 12, R12;
IPA      R15, 9, R12;
FMUL32   R0, R18, R16;
IPA      R13, 14, R12;
IPA      R14, 11, R12;
FMAD     R1, R17, R15, R0;
FMUL32   R0, R18, R18;
FMAD     R1, R13, R14, R1;
FMAD     R0, R17, R17, R0;
FMAD     R0, R13, R13, R0;
RCP      R0, R0;
FMUL32   R2, R0, R1;
FMAD     R0, R0, R1, R2;
FMAD     R4, R0, R17, -R15;
FMAD     R5, R0, R18, -R16;
FMAD     R6, R0, R13, -R14;
IPA      R0, 7, R12;
FMAX     R2, |R4|, |R5|;
IPA      R1, 8, R12;
IPA      R8, 5, R12;
FMAX     R7, |R6|, R2;
IPA      R9, 6, R12;
TEX      R0, 1, 1, 2D;
RCP      R7, R7;
TEX      R8, 0, 0, 2D;
FMUL32   R4, R4, R7;
FMUL32   R5, R5, R7;
FMUL32   R6, R6, R7;
TEX      R4, 2, 2, LWBE;
FADD32I  R11, -R11, 1.0;
FMUL32   R4, R4, R11;
FMUL32   R5, R5, R11;
FMUL32   R4, R4, c[0];
FMUL32   R7, R6, R11;
FMUL32   R6, R5, c[1];
FMAD     R11, R4, R4, -R4;
FMUL32   R5, R7, c[2];
FMAD     R7, R6, R6, -R6;
FMAD     R4, R11, c[8], R4;
FMAD     R11, R5, R5, -R5;
FMAD     R7, R7, c[9], R6;
FMUL32   R6, R16, R16;
FMAD     R5, R11, c[10], R5;
FMUL32I  R11, R7, 0.333333;
FMAD     R6, R15, R15, R6;
FMAD     R11, R4, c[0], R11;
FMAD     R6, R14, R14, R6;
FMAD     R11, R5, c[0], R11;
LG2      R6, R6;
FADD32   R5, R5, -R11;
FMUL32I  R6, R6, -0.5;
FMAD     R5, R5, c[14], R11;
RRO      R6, R6, 1;
EX2      R6, R6;
FMUL32   R15, R15, R6;
FMUL32   R16, R16, R6;
FMUL32   R6, R14, R6;
MOV32    R14, c[19];
FMUL32   R16, R18, R16;
FMAD     R15, R17, R15, R16;
FADD32   R7, R7, -R11;
FMAD     R6, R13, R6, R15;
FMAD     R7, R7, c[13], R11;
FADD32   R13, R4, -R11;
FADD32I  R4, -R6, 1.0;
FMAD     R6, R13, c[12], R11;
FMUL32   R11, R4, R4;
IPA      R3, 1, R12;
IPA      R13, 2, R12;
FMUL32   R11, R11, R11;
FMUL32   R3, R8, R3;
FMUL32   R8, R9, R13;
FMUL32   R4, R4, R11;
IPA      R9, 3, R12;
FMUL32   R1, R1, R8;
FMAD     R4, R4, c[23], R14;
FMUL32   R8, R10, R9;
FMUL32   R0, R0, R3;
FMUL32   R3, R4, R5;
FMUL32   R2, R2, R8;
FMUL32   R5, R4, R7;
FMUL32   R4, R4, R6;
FMAD     R2, R2, c[24], R3;
FMAD     R1, R1, c[24], R5;
FMAD     R0, R0, c[24], R4;
IPA      R3, 4, R12;
END
# 90 instructions, 20 R-regs, 15 interpolants
# 90 inst, (1 mov, 0 mvi, 3 tex, 15 ipa, 5 complex, 66 math)
#    53 64-bit, 33 32-bit, 4 32-bit-const
