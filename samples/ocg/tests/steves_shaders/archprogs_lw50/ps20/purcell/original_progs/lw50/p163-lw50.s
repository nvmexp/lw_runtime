!!SPA1.0
.CONST_MODE  PAGE
.THREAD_TYPE PIXEL
.MAX_REG     15
.MAX_ATTR    12
# parseasm build date Mar 10 2004 15:40:49
# -profile fp50 -po tbat3 -po lat3 -if ps2x -i allprogs-new32//p163-lw40.s -o allprogs-new32//p163-lw50.s
#vendor LWPU
#version parseasm.0.0
#profile fp50
#program fp30entry
#semantic <null atom>
#semantic <null atom>
#semantic <null atom>
#semantic <null atom>
#var float4 o[COLH0] : $vout.O : O[0] : -1 : 0
#var float4 f[COL0] : $vin.F : F[0] : -1 : 0
#var float4 f[TEX3] : $vin.F : F[0] : -1 : 0
#var sampler2D  : texunit 3 : -1 : 0
#var float4 f[TEX2] : $vin.F : F[0] : -1 : 0
#var sampler2D  : texunit 2 : -1 : 0
#var float4 f[TEX1] : $vin.F : F[0] : -1 : 0
#var sampler2D  : texunit 1 : -1 : 0
#var float4 f[TEX0] : $vin.F : F[0] : -1 : 0
#var sampler2D  : texunit 0 : -1 : 0
#tram 0 = f[WPOS].w
#tram 1 = f[COL0].x
#tram 2 = f[COL0].y
#tram 3 = f[COL0].z
#tram 4 = f[COL0].w
#tram 5 = f[TEX0].x
#tram 6 = f[TEX0].y
#tram 7 = f[TEX1].x
#tram 8 = f[TEX1].y
#tram 9 = f[TEX2].x
#tram 10 = f[TEX2].y
#tram 11 = f[TEX3].x
#tram 12 = f[TEX3].y
BB0:
IPA      R0, 0;
RCP      R8, R0;
IPA      R0, 5, R8;
IPA      R1, 6, R8;
IPA      R4, 11, R8;
IPA      R5, 12, R8;
IPA      R12, 1, R8;
IPA      R9, 2, R8;
IPA      R10, 3, R8;
IPA      R11, 4, R8;
TEX      R0, 0, 0, 2D;
TEX      R4, 3, 3, 2D;
FADD32   R4, R12, R4;
FADD32   R5, R9, R5;
FADD32   R6, R10, R6;
FADD32   R7, R11, R7;
FMUL32   R9, R1, R5;
FMUL32   R10, R2, R6;
FMUL32   R11, R3, R7;
FMUL32   R12, R0, R4;
IPA      R0, 7, R8;
IPA      R1, 8, R8;
IPA      R4, 9, R8;
IPA      R5, 10, R8;
TEX      R0, 1, 1, 2D;
TEX      R4, 2, 2, 2D;
FMAD     R0, R0, R4, R12;
FMAD     R1, R1, R5, R9;
FMAD     R2, R2, R6, R10;
FMAD     R3, R3, R7, R11;
MOV32.SAT R0, R0;
MOV32.SAT R1, R1;
MOV32.SAT R2, R2;
F2F.SAT  R0, R0;
F2F.SAT  R1, R1;
F2F.SAT  R2, R2;
MOV32.SAT R3, R3;
F2F.SAT  R3, R3;
END
# 38 instructions, 16 R-regs, 13 interpolants
# 38 inst, (4 mov, 0 mvi, 4 tex, 13 ipa, 1 complex, 16 math)
#    26 64-bit, 12 32-bit, 0 32-bit-const
