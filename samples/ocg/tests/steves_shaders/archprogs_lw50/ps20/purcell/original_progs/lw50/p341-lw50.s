!!SPA1.0
.CONST_MODE  PAGE
.THREAD_TYPE PIXEL
.MAX_REG     15
.MAX_ATTR    9
# parseasm build date Mar 10 2004 15:40:49
# -profile fp50 -po tbat3 -po lat3 -if ps2x -i allprogs-new32//p341-lw40.s -o allprogs-new32//p341-lw50.s
#vendor LWPU
#version parseasm.0.0
#profile fp50
#program fp30entry
#semantic <null atom>
#semantic <null atom>
#semantic <null atom>
#var float4 o[COLH0] : $vout.O : O[0] : -1 : 0
#var float4 f[COL0] : $vin.F : F[0] : -1 : 0
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
#tram 4 = f[TEX0].x
#tram 5 = f[TEX0].y
#tram 6 = f[TEX1].x
#tram 7 = f[TEX1].y
#tram 8 = f[TEX2].x
#tram 9 = f[TEX2].y
BB0:
IPA      R0, 0;
RCP      R12, R0;
IPA      R8, 6, R12;
IPA      R9, 7, R12;
IPA      R4, 8, R12;
IPA      R5, 9, R12;
IPA      R0, 4, R12;
IPA      R1, 5, R12;
IPA      R14, 3, R12;
TEX      R8, 1, 1, 2D;
TEX      R4, 2, 2, 2D;
TEX      R0, 0, 0, 2D;
IPA      R13, 2, R12;
FMAD     R2, -R11, R2, R2;
FMAD     R1, -R11, R1, R1;
FMAD     R0, -R11, R0, R0;
FMAD     R2, R11, R10, R2;
FMAD     R1, R11, R9, R1;
FMAD     R0, R11, R8, R0;
FMUL32   R2, R14, R2;
FMUL32   R1, R13, R1;
F2F.M2   R2, R2;
F2F.M2   R1, R1;
IPA      R7, 1, R12;
FADD32   R1, R1, R5;
FMUL32   R0, R7, R0;
FADD32   R2, R2, R6;
MOV32.SAT R1, R1;
F2F.M2   R0, R0;
MOV32.SAT R2, R2;
F2F.SAT  R1, R1;
FADD32   R0, R0, R4;
F2F.SAT  R2, R2;
MOV32.SAT R3, R3;
MOV32.SAT R0, R0;
F2F.SAT  R3, R3;
F2F.SAT  R0, R0;
END
# 37 instructions, 16 R-regs, 10 interpolants
# 37 inst, (4 mov, 0 mvi, 3 tex, 10 ipa, 1 complex, 19 math)
#    27 64-bit, 10 32-bit, 0 32-bit-const
