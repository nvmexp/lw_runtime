!!SPA1.0
.CONST_MODE  PAGE
.THREAD_TYPE PIXEL
.MAX_REG     15
.MAX_ATTR    16
# parseasm build date Mar 10 2004 15:40:49
# -profile fp50 -po tbat3 -po lat3 -if ps2x -i allprogs-new32//p345-lw40.s -o allprogs-new32//p345-lw50.s
#vendor LWPU
#version parseasm.0.0
#profile fp50
#program fp30entry
#semantic Cou6opqh6euqv9.Cou6opqh6euqv9
#semantic Cg7vl2cgg0jjn7.Cg7vl2cgg0jjn7
#semantic Cvn4u1146lkq4e.Cvn4u1146lkq4e
#semantic Cjd5bsdrkffpub.Cjd5bsdrkffpub
#semantic Cpl4am2f1m5mib.Cpl4am2f1m5mib
#semantic C76pv1sbdfq7lf.C76pv1sbdfq7lf
#semantic <null atom>
#semantic <null atom>
#semantic <null atom>
#var float4 o[COLH0] : $vout.O : O[0] : -1 : 0
#var float4 Cou6opqh6euqv9 :  : c[326] : -1 : 0
#var float4 Cg7vl2cgg0jjn7 :  : c[324] : -1 : 0
#var float4 Cvn4u1146lkq4e :  : c[322] : -1 : 0
#var float4 Cjd5bsdrkffpub :  : c[323] : -1 : 0
#var float4 Cpl4am2f1m5mib :  : c[321] : -1 : 0
#var float4 C76pv1sbdfq7lf :  : c[320] : -1 : 0
#var float4 f[COL0] : $vin.F : F[0] : -1 : 0
#var sampler2D  : texunit 3 : -1 : 0
#var float4 f[TEX4] : $vin.F : F[0] : -1 : 0
#var sampler2D  : texunit 4 : -1 : 0
#var float4 f[TEX3] : $vin.F : F[0] : -1 : 0
#var float4 f[TEX2] : $vin.F : F[0] : -1 : 0
#var float4 f[TEX1] : $vin.F : F[0] : -1 : 0
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
#tram 8 = f[TEX1].z
#tram 9 = f[TEX2].x
#tram 10 = f[TEX2].y
#tram 11 = f[TEX2].z
#tram 12 = f[TEX3].x
#tram 13 = f[TEX3].y
#tram 14 = f[TEX3].z
#tram 15 = f[TEX4].x
#tram 16 = f[TEX4].y
BB0:
IPA      R0, 0;
RCP      R8, R0;
MVI      R12, -1.0;
IPA      R4, 15, R8;
IPA      R5, 16, R8;
IPA      R0, 4, R8;
IPA      R1, 5, R8;
MVI      R9, -1.0;
MVI      R13, -1.0;
MVI      R10, -1.0;
TEX      R4, 4, 4, 2D;
MVI      R11, -1.0;
TEX      R0, 0, 0, 2D;
IPA      R7, 10, R8;
FMAD     R6, R6, c[0], R12;
FMAD     R4, R4, c[0], R9;
FMAD     R5, R5, c[0], R13;
FMAD     R0, R0, c[0], R10;
FMAD     R1, R1, c[0], R11;
IPA      R10, 9, R8;
MVI      R11, -1.0;
FMUL32   R7, R7, R1;
IPA      R9, 11, R8;
FMAD     R2, R2, c[0], R11;
FMAD     R11, R10, R0, R7;
IPA      R10, 7, R8;
IPA      R7, 6, R8;
FMAD     R12, R9, R2, R11;
FMUL32   R9, R10, R1;
IPA      R10, 8, R8;
FMUL32   R13, R12, R5;
FMAD     R11, R7, R0, R9;
IPA      R7, 13, R8;
IPA      R9, 12, R8;
FMAD     R11, R10, R2, R11;
FMUL32   R10, R7, R1;
IPA      R7, 14, R8;
FMAD     R13, R11, R4, R13;
FMAD     R9, R9, R0, R10;
FMUL32   R10, R12, R12;
FMAD     R7, R7, R2, R9;
FMAD     R9, R11, R11, R10;
FMAD     R6, R7, R6, R13;
FMAD     R7, R7, R7, R9;
FMUL32I  R6, R6, 2.0;
FMUL32   R9, R11, R6;
FMUL32   R6, R12, R6;
FMAD     R4, -R4, R7, R9;
FMAD     R5, -R5, R7, R6;
TEX      R4, 3, 3, 2D;
FMUL32   R4, R4, c[1280];
FMUL32   R9, R5, c[1281];
FMUL32   R5, R4, R4;
FMUL32   R7, R6, c[1282];
FMAD     R4, R4, -c[1284], R4;
FMUL32   R6, R9, R9;
FMAD     R9, R9, -c[1285], R9;
FMAD     R4, R5, c[1284], R4;
FMUL32   R5, R7, R7;
FMAD     R6, R6, c[1285], R9;
FMAD     R9, R7, -c[1286], R7;
IPA      R10, 2, R8;
FMUL32   R7, R6, c[1293];
FMAD     R5, R5, c[1286], R9;
MVI      R11, -1.0;
FMAD     R7, R4, c[1292], R7;
IPA      R9, 1, R8;
FMAD     R10, R10, c[0], R11;
FMAD     R7, R5, c[1294], R7;
IPA      R8, 3, R8;
FMUL32   R12, R10, R1;
FMAD     R1, R7, -c[1288], R7;
MVI      R11, -1.0;
MVI      R10, -1.0;
FMAD     R1, R4, c[1288], R1;
FMAD     R4, R9, c[0], R11;
FMAD     R8, R8, c[0], R10;
MOV32    R9, c[1299];
FMAD     R0, R4, R0, R12;
MOV32    R4, R9;
FMAD.SAT R0, R8, R2, R0;
FADD32I  R0, -R0, 1.0;
FMAD     R8, R7, -c[1289], R7;
FMUL32   R2, R0, R0;
FMAD     R6, R6, c[1289], R8;
FMUL32   R2, R2, R2;
FMUL32   R0, R2, R0;
FMAD     R2, R7, -c[1290], R7;
FMAD     R0, R0, c[1307], R4;
FMAD     R2, R5, c[1290], R2;
FMUL32   R1, R1, R0;
FMUL32   R4, R6, R0;
FMUL32   R0, R2, R0;
MOV32.SAT R1, R1;
MOV32.SAT R2, R4;
MOV32.SAT R4, R0;
F2F.SAT  R0, R1;
F2F.SAT  R1, R2;
F2F.SAT  R2, R4;
MOV32.SAT R3, R3;
F2F.SAT  R3, R3;
END
# 101 instructions, 16 R-regs, 17 interpolants
# 101 inst, (6 mov, 9 mvi, 3 tex, 17 ipa, 1 complex, 65 math)
#    72 64-bit, 27 32-bit, 2 32-bit-const
