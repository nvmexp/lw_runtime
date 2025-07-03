!!SPA1.0
.CONST_MODE  PAGE
.THREAD_TYPE PIXEL
.MAX_REG     11
.MAX_ATTR    8
# parseasm build date Mar 10 2004 15:40:49
# -profile fp50 -po tbat3 -po lat3 -if ps2x -i allprogs-new32//p412-lw40.s -o allprogs-new32//p412-lw50.s
#vendor LWPU
#version parseasm.0.0
#profile fp50
#program fp30entry
#semantic Cg86p5d28dfggc.Cg86p5d28dfggc
#semantic C85f7kkc64o5fe.C85f7kkc64o5fe
#semantic <null atom>
#semantic <null atom>
#var float4 o[COLH0] : $vout.O : O[0] : -1 : 0
#var float4 Cg86p5d28dfggc :  : c[7] : -1 : 0
#var float4 C85f7kkc64o5fe :  : c[6] : -1 : 0
#var float4 f[COL0] : $vin.F : F[0] : -1 : 0
#var float4 f[TEX0] : $vin.F : F[0] : -1 : 0
#var sampler2D  : texunit 0 : -1 : 0
#var float4 f[TEX2] : $vin.F : F[0] : -1 : 0
#var sampler2D  : texunit 1 : -1 : 0
#tram 0 = f[WPOS].w
#tram 1 = f[COL0].x
#tram 2 = f[COL0].y
#tram 3 = f[COL0].z
#tram 4 = f[COL0].w
#tram 5 = f[TEX0].x
#tram 6 = f[TEX0].y
#tram 7 = f[TEX2].x
#tram 8 = f[TEX2].y
BB0:
IPA      R0, 0;
RCP      R8, R0;
IPA      R4, 7, R8;
IPA      R5, 8, R8;
IPA      R0, 5, R8;
IPA      R1, 6, R8;
IPA      R11, 3, R8;
IPA      R10, 2, R8;
IPA      R9, 1, R8;
TEX      R4, 1, 1, 2D;
TEX      R0, 0, 0, 2D;
FMUL32   R2, R2, R11;
FMUL32   R1, R1, R10;
FMUL32   R0, R0, R9;
FMUL32   R5, R5, R1;
FMUL32   R4, R4, R0;
FMUL32   R6, R6, R2;
FMUL32   R5, R5, c[24];
FMUL32   R4, R4, c[24];
FMUL32   R6, R6, c[24];
FMAD     R1, R1, c[29], -R5;
FMAD     R0, R0, c[28], -R4;
FMAD     R2, R2, c[30], -R6;
FMAD     R1, R3, R1, R5;
FMAD     R0, R3, R0, R4;
FMAD     R2, R3, R2, R6;
IPA      R3, 4, R8;
END
# 27 instructions, 12 R-regs, 9 interpolants
# 27 inst, (0 mov, 0 mvi, 2 tex, 9 ipa, 1 complex, 15 math)
#    18 64-bit, 9 32-bit, 0 32-bit-const
