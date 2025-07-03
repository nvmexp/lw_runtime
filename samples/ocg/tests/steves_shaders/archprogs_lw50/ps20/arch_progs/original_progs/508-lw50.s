!!SPA1.0
.CONST_MODE  PAGE
.THREAD_TYPE PIXEL
.MAX_REG     15
.MAX_ATTR    11
# parseasm build date Feb  3 2004 15:17:32
# -profile fp50 -po tbat3 -po lat3 -if ps2x -i progs/508-lw40.s -o progs/508-lw50.s
#vendor LWPU
#version parseasm.0.0
#profile lwinst
#program fp30entry
#semantic Coit1m35rlpml7.Coit1m35rlpml7
#semantic <null atom>
#semantic <null atom>
#semantic <null atom>
#semantic <null atom>
#semantic <null atom>
#var float4 o[COLH0] : $vout.O : O[0] : -1 : 0
#var float4 Coit1m35rlpml7 :  : c[1] : -1 : 0
#var float4 f[COL0] : $vin.F : F[0] : -1 : 0
#var float4 f[TEX4] : $vin.F : F[0] : -1 : 0
#var sampler2D  : texunit 0 : -1 : 0
#var float4 f[TEX3] : $vin.F : F[0] : -1 : 0
#var sampler2D  : texunit 0 : -1 : 0
#var float4 f[TEX2] : $vin.F : F[0] : -1 : 0
#var sampler2D  : texunit 0 : -1 : 0
#var float4 f[TEX1] : $vin.F : F[0] : -1 : 0
#var sampler2D  : texunit 0 : -1 : 0
#var float4 f[TEX0] : $vin.F : F[0] : -1 : 0
#var sampler2D  : texunit 0 : -1 : 0
#tram 0 = f[WPOS].w
#tram 1 = f[COL0].w
#tram 2 = f[TEX0].x
#tram 3 = f[TEX0].y
#tram 4 = f[TEX1].x
#tram 5 = f[TEX1].y
#tram 6 = f[TEX2].x
#tram 7 = f[TEX2].y
#tram 8 = f[TEX3].x
#tram 9 = f[TEX3].y
#tram 10 = f[TEX4].x
#tram 11 = f[TEX4].y
BB0:
IPA      R0, 0;
RCP      R14, R0;
IPA      R8, 2, R14;
IPA      R9, 3, R14;
IPA      R4, 4, R14;
IPA      R5, 5, R14;
IPA      R0, 6, R14;
IPA      R1, 7, R14;
TEX      R8, 0, 0, 2D;
IPA      R12, 8, R14;
IPA      R13, 9, R14;
TEX      R4, 0, 0, 2D;
TEX      R0, 0, 0, 2D;
IPA      R8, 10, R14;
IPA      R9, 11, R14;
FADD     R1, R11, R7;
IPA      R0, 1, R14;
TEX      R12, 0, 0, 2D;
FADD     R1, R3, R1;
TEX      R8, 0, 0, 2D;
FADD     R1, R15, R1;
FADD     R1, R11, R1;
FMAD.SAT R3, R1, c[0], -R0;
FMAD     R0, R3, c[4], -R3;
FMAD     R1, R3, c[5], -R3;
FMAD     R2, R3, c[6], -R3;
FADD     R0, R0, 1.0;
FADD     R1, R1, 1.0;
FADD     R2, R2, 1.0;
FMAD     R3, R3, c[7], -R3;
FADD     R3, R3, 1.0;
END
# 31 instructions, 16 R-regs, 12 interpolants
# 31 inst, (0 mov, 0 mvi, 5 tex, 12 ipa, 1 complex, 13 math)
#    23 64-bit, 4 32-bit, 4 32-bit-const
