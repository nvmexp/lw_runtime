!!SPA1.0
.CONST_MODE  PAGE
.THREAD_TYPE PIXEL
.MAX_REG     15
.MAX_ATTR    8
# parseasm build date Mar 10 2004 15:40:49
# -profile fp50 -po tbat3 -po lat3 -if ps2x -i allprogs-new32//p338-lw40.s -o allprogs-new32//p338-lw50.s
#vendor LWPU
#version parseasm.0.0
#profile fp50
#program fp30entry
#semantic Cjd5bsdrkffpub.Cjd5bsdrkffpub
#semantic Cvn4u1146lkq4e.Cvn4u1146lkq4e
#semantic Cpl4am2f1m5mib.Cpl4am2f1m5mib
#semantic C76pv1sbdfq7lf.C76pv1sbdfq7lf
#semantic <null atom>
#semantic <null atom>
#semantic <null atom>
#semantic <null atom>
#var float4 o[COLH0] : $vout.O : O[0] : -1 : 0
#var float4 Cjd5bsdrkffpub :  : c[323] : -1 : 0
#var float4 Cvn4u1146lkq4e :  : c[322] : -1 : 0
#var float4 Cpl4am2f1m5mib :  : c[321] : -1 : 0
#var float4 C76pv1sbdfq7lf :  : c[320] : -1 : 0
#var float4 f[TEX3] : $vin.F : F[0] : -1 : 0
#var sampler2D  : texunit 3 : -1 : 0
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
#tram 7 = f[TEX3].x
#tram 8 = f[TEX3].y
BB0:
IPA      R0, 0;
RCP      R12, R0;
IPA      R4, 3, R12;
IPA      R5, 4, R12;
IPA      R0, 5, R12;
IPA      R1, 6, R12;
IPA      R8, 1, R12;
IPA      R9, 2, R12;
MVI      R14, -1.0;
MVI      R15, -1.0;
TEX      R4, 1, 1, 2D;
TEX      R0, 2, 2, 2D;
MVI      R13, -1.0;
TEX      R8, 0, 0, 2D;
FMAD     R8, R8, c[0], R14;
FMAD     R9, R9, c[0], R15;
FMAD     R10, R10, c[0], R13;
FMUL32   R13, R9, c[1281];
FMUL32   R11, R9, c[1285];
FMAD     R13, R8, c[1280], R13;
FMAD     R11, R8, c[1284], R11;
FMAD.SAT R13, R10, c[1282], R13;
FMAD.SAT R11, R10, c[1286], R11;
FMUL32   R4, R4, R13;
FMUL32   R5, R5, R13;
FMUL32   R6, R6, R13;
FMUL32   R7, R7, R13;
FMAD     R5, R11, R1, R5;
FMAD     R6, R11, R2, R6;
FMAD     R7, R11, R3, R7;
FMAD     R4, R11, R0, R4;
FMUL32   R2, R9, c[1289];
IPA      R0, 7, R12;
IPA      R1, 8, R12;
FMAD     R8, R8, c[1288], R2;
TEX      R0, 3, 3, 2D;
FMAD.SAT R8, R10, c[1290], R8;
FMAD     R0, R8, R0, R4;
FMAD     R1, R8, R1, R5;
FMAD     R2, R8, R2, R6;
FMUL32   R0, R0, c[1292];
FMUL32   R1, R1, c[1293];
FMAD     R3, R8, R3, R7;
MOV32.SAT R0, R0;
MOV32.SAT R1, R1;
FMUL32   R2, R2, c[1294];
F2F.SAT  R0, R0;
F2F.SAT  R1, R1;
MOV32.SAT R2, R2;
FMUL32   R3, R3, c[1295];
F2F.SAT  R2, R2;
MOV32.SAT R3, R3;
F2F.SAT  R3, R3;
END
# 53 instructions, 16 R-regs, 9 interpolants
# 53 inst, (4 mov, 3 mvi, 4 tex, 9 ipa, 1 complex, 32 math)
#    38 64-bit, 15 32-bit, 0 32-bit-const
