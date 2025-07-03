!!SPA1.0
.CONST_MODE  PAGE
.THREAD_TYPE PIXEL
.MAX_REG     15
.MAX_ATTR    6
# parseasm build date Mar 10 2004 15:40:49
# -profile fp50 -po tbat3 -po lat3 -if ps2x -i allprogs-new32//p336-lw40.s -o allprogs-new32//p336-lw50.s
#vendor LWPU
#version parseasm.0.0
#profile fp50
#program fp30entry
#semantic C76pv1sbdfq7lf.C76pv1sbdfq7lf
#semantic <null atom>
#semantic <null atom>
#semantic <null atom>
#var float4 o[COLH0] : $vout.O : O[0] : -1 : 0
#var float4 C76pv1sbdfq7lf :  : c[320] : -1 : 0
#var float4 f[TEX2] : $vin.F : F[0] : -1 : 0
#var sampler2D  : texunit 2 : -1 : 0
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
BB0:
IPA      R0, 0;
RCP      R2, R0;
IPA      R0, 5, R2;
IPA      R1, 6, R2;
IPA      R8, 1, R2;
IPA      R9, 2, R2;
IPA      R4, 3, R2;
IPA      R5, 4, R2;
TEX      R0, 2, 2, 2D;
TEX      R8, 0, 0, 2D;
TEX      R4, 1, 1, 2D;
FADD32I  R12, -R3, 1.0;
MOV32.SAT R12, R12;
F2F.SAT  R12, R12;
FMAD     R11, -R12, R11, R11;
FMAD     R10, -R12, R10, R10;
FMAD     R9, -R12, R9, R9;
FMAD     R8, -R12, R8, R8;
FMAD     R5, R12, R5, R9;
FMAD     R4, R12, R4, R8;
FMUL32   R1, R5, R1;
FMUL32   R0, R4, R0;
FMUL32   R1, R1, c[1281];
FMAD     R4, R12, R6, R10;
FMUL32   R0, R0, c[1280];
F2F.M2   R1, R1;
FMUL32   R2, R4, R2;
F2F.M2   R0, R0;
MOV32.SAT R1, R1;
FMAD     R4, R12, R7, R11;
MOV32.SAT R0, R0;
F2F.SAT  R1, R1;
FMUL32   R3, R4, R3;
F2F.SAT  R0, R0;
FMUL32   R2, R2, c[1282];
MOV32.SAT R3, R3;
F2F.M2   R2, R2;
F2F.SAT  R3, R3;
MOV32.SAT R2, R2;
F2F.SAT  R2, R2;
END
# 40 instructions, 16 R-regs, 7 interpolants
# 40 inst, (5 mov, 0 mvi, 3 tex, 7 ipa, 1 complex, 24 math)
#    27 64-bit, 12 32-bit, 1 32-bit-const
