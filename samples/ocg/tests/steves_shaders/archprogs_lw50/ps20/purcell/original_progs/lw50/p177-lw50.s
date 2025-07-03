!!SPA1.0
.CONST_MODE  PAGE
.THREAD_TYPE PIXEL
.MAX_REG     15
.MAX_ATTR    7
# parseasm build date Mar 10 2004 15:40:49
# -profile fp50 -po tbat3 -po lat3 -if ps2x -i allprogs-new32//p177-lw40.s -o allprogs-new32//p177-lw50.s
#vendor LWPU
#version parseasm.0.0
#profile fp50
#program fp30entry
#semantic C76pv1sbdfq7lf.C76pv1sbdfq7lf
#semantic Cpl4am2f1m5mib.Cpl4am2f1m5mib
#semantic <null atom>
#semantic <null atom>
#var float4 o[COLH0] : $vout.O : O[0] : -1 : 0
#var float4 f[COL0] : $vin.F : F[0] : -1 : 0
#var float4 C76pv1sbdfq7lf :  : c[320] : -1 : 0
#var float4 Cpl4am2f1m5mib :  : c[321] : -1 : 0
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
BB0:
IPA      R0, 0;
RCP      R8, R0;
MVI      R9, -1.0;
IPA      R0, 4, R8;
IPA      R1, 5, R8;
IPA      R4, 6, R8;
IPA      R5, 7, R8;
MVI      R10, -1.0;
MVI      R12, -1.0;
MVI      R13, -1.0;
TEX      R0, 0, 0, 2D;
TEX      R4, 1, 1, 2D;
MVI      R11, -1.0;
MVI      R3, -1.0;
FMAD     R2, R2, c[0], R9;
FMAD     R0, R0, c[0], R10;
FMAD     R1, R1, c[0], R12;
FMAD     R5, R5, c[0], R13;
FMAD     R4, R4, c[0], R11;
FMAD     R3, R6, c[0], R3;
FMUL32   R1, R1, R5;
MOV32    R5, c[1284];
IPA      R6, 1, R8;
FMAD     R0, R0, R4, R1;
MOV32    R1, R5;
MVI      R4, -1.0;
FMAD     R0, R2, R3, R0;
MOV32    R2, c[1287];
FMAD     R3, R7, c[0], R4;
FMAD.SAT R1, R0, c[1280], R1;
MOV32    R4, c[1285];
FMUL32   R1, R1, R6;
FMAD.SAT R2, R3, c[1283], R2;
IPA      R3, 2, R8;
FADD32.SAT R2, R2, R2;
FMAD.SAT R4, R0, c[1281], R4;
F2F.SAT.M4 R2, R2;
FMUL32   R3, R4, R3;
MOV32    R4, c[1286];
FMUL32   R1, R1, R2;
FMUL32   R3, R3, R2;
MOV32.SAT R1, R1;
MOV32.SAT R3, R3;
FMAD.SAT R4, R0, c[1282], R4;
F2F.SAT  R0, R1;
F2F.SAT  R1, R3;
IPA      R3, 3, R8;
MOV32.SAT R5, R2;
FMUL32   R4, R4, R3;
F2F.SAT  R3, R5;
FMUL32   R2, R4, R2;
MOV32.SAT R2, R2;
F2F.SAT  R2, R2;
END
# 53 instructions, 16 R-regs, 8 interpolants
# 53 inst, (9 mov, 7 mvi, 2 tex, 8 ipa, 1 complex, 26 math)
#    36 64-bit, 17 32-bit, 0 32-bit-const
