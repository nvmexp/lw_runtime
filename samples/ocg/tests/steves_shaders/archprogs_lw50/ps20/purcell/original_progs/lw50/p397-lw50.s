!!SPA1.0
.CONST_MODE  PAGE
.THREAD_TYPE PIXEL
.MAX_REG     11
.MAX_ATTR    5
# parseasm build date Mar 10 2004 15:40:49
# -profile fp50 -po tbat3 -po lat3 -if ps2x -i allprogs-new32//p397-lw40.s -o allprogs-new32//p397-lw50.s
#vendor LWPU
#version parseasm.0.0
#profile fp50
#program fp30entry
#semantic Cee994ndhnk1rc.Cee994ndhnk1rc
#semantic Cmur1hd5g8ngo3.Cmur1hd5g8ngo3
#semantic Coit1m35rlpml7.Coit1m35rlpml7
#semantic <null atom>
#var float4 o[COLH0] : $vout.O : O[0] : -1 : 0
#var float4 Cee994ndhnk1rc :  : c[5] : -1 : 0
#var float4 Cmur1hd5g8ngo3 :  : c[4] : -1 : 0
#var float4 Coit1m35rlpml7 :  : c[1] : -1 : 0
#var float4 f[COL0] : $vin.F : F[0] : -1 : 0
#var float4 f[TEX0] : $vin.F : F[0] : -1 : 0
#var sampler2D  : texunit 0 : -1 : 0
#tram 0 = f[WPOS].w
#tram 1 = f[COL0].x
#tram 2 = f[COL0].y
#tram 3 = f[COL0].z
#tram 4 = f[TEX0].x
#tram 5 = f[TEX0].y
BB0:
IPA      R0, 0;
RCP      R3, R0;
IPA      R4, 4, R3;
IPA      R5, 5, R3;
IPA      R0, 1, R3;
TEX      R4, 0, 0, 2D;
FMUL32   R0, R0, c[4];
IPA      R1, 2, R3;
FMUL32   R1, R1, c[5];
FMUL32   R0, R4, R0;
FMUL32   R2, R5, R1;
FMUL32   R1, R0, c[16];
IPA      R8, 3, R3;
FMUL32   R3, R2, c[17];
FMAD     R0, R0, c[16], R1;
FMUL32   R1, R8, c[6];
FMAD     R2, R2, c[17], R3;
FMAD     R3, R4, c[20], -R0;
FMUL32   R4, R6, R1;
FMAD     R1, R5, c[21], -R2;
FMAD     R0, R7, R3, R0;
FMUL32   R5, R4, c[18];
FMAD     R1, R7, R1, R2;
MOV32    R3, c[7];
FMAD     R2, R4, c[18], R5;
FMAD     R4, R6, c[22], -R2;
FMAD     R2, R7, R4, R2;
END
# 27 instructions, 12 R-regs, 6 interpolants
# 27 inst, (1 mov, 0 mvi, 1 tex, 6 ipa, 1 complex, 18 math)
#    17 64-bit, 10 32-bit, 0 32-bit-const
