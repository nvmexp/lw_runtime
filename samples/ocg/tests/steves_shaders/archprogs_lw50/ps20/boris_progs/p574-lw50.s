!!SPA1.0
.CONST_MODE  PAGE
.THREAD_TYPE PIXEL
.MAX_REG     15
.MAX_ATTR    10
# parseasm build date Feb 13 2004 14:20:40
# -profile fp50 -po tbat3 -po lat3 -if ps2x -i progs//p574-lw40.s -o progs//p574-lw50.s
#vendor LWPU
#version parseasm.0.0
#profile fp50
#program fp30entry
#semantic Cmur1hd5g8ngo3.Cmur1hd5g8ngo3
#semantic Coit1m35rlpml7.Coit1m35rlpml7
#semantic <null atom>
#semantic <null atom>
#semantic <null atom>
#semantic Cee994ndhnk1rc.Cee994ndhnk1rc
#semantic <null atom>
#var float4 o[COLH0] : $vout.O : O[0] : -1 : 0
#var float4 Cmur1hd5g8ngo3 :  : c[4] : -1 : 0
#var float4 Coit1m35rlpml7 :  : c[1] : -1 : 0
#var float4 f[TEX1] : $vin.F : F[0] : -1 : 0
#var samplerLWBE  : texunit 6 : -1 : 0
#var sampler2D  : texunit 2 : -1 : 0
#var sampler2D  : texunit 4 : -1 : 0
#var float4 Cee994ndhnk1rc :  : c[5] : -1 : 0
#var float4 f[TEX2] : $vin.F : F[0] : -1 : 0
#var float4 f[TEX3] : $vin.F : F[0] : -1 : 0
#var float4 f[TEX0] : $vin.F : F[0] : -1 : 0
#var sampler2D  : texunit 3 : -1 : 0
#tram 0 = f[WPOS].w
#tram 1 = f[TEX0].x
#tram 2 = f[TEX0].y
#tram 3 = f[TEX1].x
#tram 4 = f[TEX1].y
#tram 5 = f[TEX1].z
#tram 6 = f[TEX2].x
#tram 7 = f[TEX2].y
#tram 8 = f[TEX2].z
#tram 9 = f[TEX2].w
#tram 10 = f[TEX3].x
BB0:
IPA      R0, 0;
RCP      R8, R0;
IPA      R0, 3, R8;
IPA      R1, 4, R8;
IPA      R2, 5, R8;
IPA      R4, 1, R8;
FMAX     R3, |R0|, |R1|;
IPA      R5, 2, R8;
FMAX     R3, |R2|, R3;
MVI      R9, -1.0;
TEX      R4, 3, 3, 2D;
RCP      R3, R3;
MVI      R10, -1.0;
MVI      R12, -1.0;
FMUL     R0, R0, R3;
FMUL     R1, R1, R3;
FMUL     R2, R2, R3;
MVI      R13, -1.0;
TEX      R0, 6, 6, LWBE;
MVI      R11, -1.0;
FMAD     R5, R5, c[0], R13;
FMAD     R4, R4, c[0], R11;
MVI      R3, -1.0;
FMAD     R2, R2, c[0], R9;
FMAD     R0, R0, c[0], R10;
FMAD     R1, R1, c[0], R12;
FMAD     R6, R6, c[0], R3;
FMUL     R3, R7, R4;
FMUL     R1, R1, R5;
IPA      R9, 10, R8;
IPA      R10, 9, R8;
FMAD     R4, R0, R4, R1;
RCP      R0, R9;
FMUL     R1, R7, R5;
FMAD.SAT R2, R2, R6, R4;
FMUL     R4, R0, R10;
IPA      R5, 8, R8;
FADD32I  R2, -R2, 1.0;
FMAD     R4, R3, c[23], R4;
FMUL     R5, R0, R5;
FMUL     R6, R2, R2;
IPA      R7, 6, R8;
FMAD     R5, R1, c[22], R5;
IPA      R8, 7, R8;
FMUL     R7, R0, R7;
FMUL     R6, R6, R6;
FMUL     R9, R0, R8;
FMAD     R0, R3, c[20], R7;
FMUL     R8, R2, R6;
FMAD     R1, R1, c[21], R9;
TEX      R4, 2, 2, 2D;
TEX      R0, 4, 4, 2D;
FMUL     R4, R4, c[4];
FMUL     R5, R5, c[5];
FMUL     R6, R6, c[6];
FMUL     R7, R7, c[7];
FMAD     R1, R1, c[17], -R5;
FMAD     R2, R2, c[18], -R6;
FMAD     R3, R3, c[19], -R7;
FMAD     R1, R8, R1, R5;
FMAD     R2, R8, R2, R6;
FMAD     R3, R8, R3, R7;
FMAD     R0, R0, c[16], -R4;
FMAD     R0, R8, R0, R4;
END
# 64 instructions, 16 R-regs, 11 interpolants
# 64 inst, (0 mov, 6 mvi, 4 tex, 11 ipa, 3 complex, 40 math)
#    63 64-bit, 0 32-bit, 1 32-bit-const
