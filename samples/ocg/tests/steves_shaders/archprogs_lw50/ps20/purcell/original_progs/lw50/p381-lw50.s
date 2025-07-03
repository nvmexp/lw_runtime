!!SPA1.0
.CONST_MODE  PAGE
.THREAD_TYPE PIXEL
.MAX_REG     15
.MAX_ATTR    13
# parseasm build date Mar 10 2004 15:40:49
# -profile fp50 -po tbat3 -po lat3 -if ps2x -i allprogs-new32//p381-lw40.s -o allprogs-new32//p381-lw50.s
#vendor LWPU
#version parseasm.0.0
#profile fp50
#program fp30entry
#semantic Cmur1hd5g8ngo3.Cmur1hd5g8ngo3
#semantic Coit1m35rlpml7.Coit1m35rlpml7
#semantic <null atom>
#semantic <null atom>
#var float4 o[COLH0] : $vout.O : O[0] : -1 : 0
#var float4 Cmur1hd5g8ngo3 :  : c[4] : -1 : 0
#var float4 Coit1m35rlpml7 :  : c[1] : -1 : 0
#var float4 f[TEX7] : $vin.F : F[0] : -1 : 0
#var float4 f[COL0] : $vin.F : F[0] : -1 : 0
#var float4 f[COL1] : $vin.F : F[0] : -1 : 0
#var float4 f[TEX0] : $vin.F : F[0] : -1 : 0
#var sampler2D  : texunit 0 : -1 : 0
#var float4 f[TEX1] : $vin.F : F[0] : -1 : 0
#var sampler2D  : texunit 3 : -1 : 0
#tram 0 = f[WPOS].w
#tram 1 = f[COL0].x
#tram 2 = f[COL0].y
#tram 3 = f[COL0].z
#tram 4 = f[COL1].x
#tram 5 = f[COL1].y
#tram 6 = f[COL1].z
#tram 7 = f[TEX0].x
#tram 8 = f[TEX0].y
#tram 9 = f[TEX1].x
#tram 10 = f[TEX1].y
#tram 11 = f[TEX7].x
#tram 12 = f[TEX7].y
#tram 13 = f[TEX7].z
BB0:
IPA      R0, 0;
RCP      R8, R0;
MVI      R9, -1.0;
IPA      R4, 9, R8;
IPA      R5, 10, R8;
IPA      R0, 7, R8;
IPA      R1, 8, R8;
MVI      R11, -1.0;
MVI      R10, -1.0;
TEX      R4, 3, 3, 2D;
TEX      R0, 0, 0, 2D;
FMAD     R9, R4, c[0], R9;
FMAD     R4, R6, c[0], R11;
FMAD     R10, R5, c[0], R10;
IPA      R7, 6, R8;
FMUL32I  R5, R4, 0.57735;
FMUL32I  R6, R10, 0.707107;
FMUL32I  R10, R10, -0.707107;
FMAD.SAT R5, R9, c[0], R5;
FMAD     R6, R9, c[0], R6;
FMAD     R9, R9, c[0], R10;
IPA      R10, 3, R8;
FMAD.SAT R6, R4, c[0], R6;
FMAD.SAT R4, R4, c[0], R9;
IPA      R11, 13, R8;
FMUL32   R12, R6, R7;
IPA      R7, 5, R8;
IPA      R9, 2, R8;
FMAD     R12, R5, R10, R12;
FMUL32   R10, R6, R7;
IPA      R7, 12, R8;
FMAD     R11, R4, R11, R12;
FMAD     R10, R5, R9, R10;
IPA      R9, 4, R8;
FMUL32   R11, R11, c[6];
FMAD     R7, R4, R7, R10;
FMUL32   R6, R6, R9;
FMUL32   R9, R7, c[5];
IPA      R7, 1, R8;
IPA      R8, 11, R8;
FMUL32   R1, R1, R9;
FMAD     R5, R5, R7, R6;
FMUL32   R2, R2, R11;
FMUL32   R6, R1, c[17];
FMAD     R4, R4, R8, R5;
FMUL32   R3, R3, c[7];
FMAD     R1, R1, c[17], R6;
FMUL32   R4, R4, c[4];
FMUL32   R5, R2, c[18];
FMUL32   R0, R0, R4;
FMAD     R2, R2, c[18], R5;
FMUL32   R4, R0, c[16];
FMAD     R0, R0, c[16], R4;
END
# 53 instructions, 16 R-regs, 14 interpolants
# 53 inst, (0 mov, 3 mvi, 2 tex, 14 ipa, 1 complex, 33 math)
#    37 64-bit, 13 32-bit, 3 32-bit-const
