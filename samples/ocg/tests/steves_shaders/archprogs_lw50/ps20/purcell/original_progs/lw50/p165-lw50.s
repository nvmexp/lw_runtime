!!SPA1.0
.CONST_MODE  PAGE
.THREAD_TYPE PIXEL
.MAX_REG     15
.MAX_ATTR    4
# parseasm build date Mar 10 2004 15:40:49
# -profile fp50 -po tbat3 -po lat3 -if ps2x -i allprogs-new32//p165-lw40.s -o allprogs-new32//p165-lw50.s
#vendor LWPU
#version parseasm.0.0
#profile fp50
#program fp30entry
#semantic C76pv1sbdfq7lf.C76pv1sbdfq7lf
#semantic Cpl4am2f1m5mib.Cpl4am2f1m5mib
#semantic <null atom>
#semantic <null atom>
#var float4 o[COLH0] : $vout.O : O[0] : -1 : 0
#var float4 C76pv1sbdfq7lf :  : c[320] : -1 : 0
#var float4 Cpl4am2f1m5mib :  : c[321] : -1 : 0
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
RCP      R2, R0;
MVI      R8, -1.0;
IPA      R0, 1, R2;
IPA      R1, 2, R2;
IPA      R4, 3, R2;
IPA      R5, 4, R2;
MVI      R10, -1.0;
MVI      R12, -1.0;
MVI      R9, -1.0;
TEX      R0, 0, 0, 2D;
MVI      R11, -1.0;
TEX      R4, 1, 1, 2D;
MVI      R3, -1.0;
FMAD     R2, R2, c[0], R8;
FMAD     R0, R0, c[0], R10;
FMAD     R1, R1, c[0], R12;
FMAD     R6, R6, c[0], R9;
FMAD     R4, R4, c[0], R11;
FMAD     R3, R5, c[0], R3;
MOV32    R5, c[1284];
MOV32    R7, c[1285];
FMUL32   R1, R1, R3;
MOV32    R3, R5;
MOV32    R5, R7;
FMAD     R1, R0, R4, R1;
MOV32    R0, c[1286];
FMAD.SAT R4, R2, R6, R1;
FMAD     R1, R4, c[1280], R3;
FMAD     R2, R4, c[1281], R5;
FMAD     R0, R4, c[1282], R0;
MOV32.SAT R1, R1;
MOV32.SAT R2, R2;
MOV32.SAT R3, R0;
F2F.SAT  R0, R1;
F2F.SAT  R1, R2;
F2F.SAT  R2, R3;
MOV32    R3, c[1287];
FMAD     R3, R4, c[1283], R3;
MOV32.SAT R3, R3;
F2F.SAT  R3, R3;
END
# 41 instructions, 16 R-regs, 5 interpolants
# 41 inst, (10 mov, 6 mvi, 2 tex, 5 ipa, 1 complex, 17 math)
#    30 64-bit, 11 32-bit, 0 32-bit-const
