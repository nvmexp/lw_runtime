!!SPA1.0
.CONST_MODE  PAGE
.THREAD_TYPE PIXEL
.MAX_REG     7
.MAX_ATTR    4
# parseasm build date Mar 10 2004 15:40:49
# -profile fp50 -po tbat3 -po lat3 -if ps2x -i allprogs-new32//p354-lw40.s -o allprogs-new32//p354-lw50.s
#vendor LWPU
#version parseasm.0.0
#profile fp50
#program fp30entry
#semantic C76pv1sbdfq7lf.C76pv1sbdfq7lf
#semantic <null atom>
#semantic <null atom>
#var float4 o[COLH0] : $vout.O : O[0] : -1 : 0
#var float4 C76pv1sbdfq7lf :  : c[320] : -1 : 0
#var float4 f[TEX1] : $vin.F : F[0] : -1 : 0
#var sampler2D  : texunit 1 : -1 : 0
#var float4 f[TEX0] : $vin.F : F[0] : -1 : 0
#var sampler2D  : texunit 0 : -1 : 0
#tram 0 = f[WPOS].w
#tram 1 = f[TEX0].x
#tram 2 = f[TEX0].y
#tram 3 = f[TEX1].x
#tram 4 = f[TEX1].y
BB0:
IPA      R0, 0;
RCP      R1, R0;
IPA      R4, 1, R1;
IPA      R5, 2, R1;
IPA      R0, 3, R1;
IPA      R1, 4, R1;
TEX      R4, 0, 0, 2D;
TEX      R0, 1, 1, 2D;
FMUL32   R0, R0, R4;
FMUL32   R1, R1, R5;
FMUL32   R0, R0, c[1280];
FMUL32   R2, R2, R6;
FMUL32   R1, R1, c[1281];
F2F.M2   R0, R0;
FMUL32   R2, R2, c[1282];
F2F.M2   R1, R1;
MOV32.SAT R0, R0;
F2F.M2   R2, R2;
MOV32.SAT R1, R1;
F2F.SAT  R0, R0;
MOV32.SAT R2, R2;
F2F.SAT  R1, R1;
FADD32I  R3, -R3, 1.0;
F2F.SAT  R2, R2;
MOV32.SAT R3, R3;
F2F.SAT  R3, R3;
END
# 26 instructions, 8 R-regs, 5 interpolants
# 26 inst, (4 mov, 0 mvi, 2 tex, 5 ipa, 1 complex, 14 math)
#    15 64-bit, 10 32-bit, 1 32-bit-const
