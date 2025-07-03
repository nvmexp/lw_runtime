!!SPA1.0
.CONST_MODE  PAGE
.THREAD_TYPE PIXEL
.MAX_REG     51
.MAX_ATTR    2
# parseasm build date Mar 10 2004 15:40:49
# -profile fp50 -po tbat3 -po lat3 -if ps2x -i allprogs-new32//p359-lw40.s -o allprogs-new32//p359-lw50.s
#vendor LWPU
#version parseasm.0.0
#profile fp50
#program fp30entry
#semantic <null atom>
#semantic <null atom>
#semantic <null atom>
#semantic <null atom>
#semantic <null atom>
#semantic Cee994ndhnk1rc.Cee994ndhnk1rc
#semantic Cmur1hd5g8ngo3.Cmur1hd5g8ngo3
#semantic <null atom>
#semantic <null atom>
#semantic <null atom>
#semantic <null atom>
#semantic <null atom>
#semantic <null atom>
#semantic <null atom>
#semantic Clofsu13ov1o1e.Clofsu13ov1o1e
#semantic C6r0tbn6evb4u3.C6r0tbn6evb4u3
#semantic Coit1m35rlpml7.Coit1m35rlpml7
#semantic C144q3o24uukb6.C144q3o24uukb6
#semantic Ct2sb36fhpb7ea.Ct2sb36fhpb7ea
#semantic Cg86p5d28dfggc.Cg86p5d28dfggc
#semantic C85f7kkc64o5fe.C85f7kkc64o5fe
#semantic Cg1k4oc9vs2b8e.Cg1k4oc9vs2b8e
#semantic <null atom>
#var float4 o[COLH0] : $vout.O : O[0] : -1 : 0
#var sampler2D  : texunit 1 : -1 : 0
#var sampler2D  : texunit 1 : -1 : 0
#var sampler2D  : texunit 1 : -1 : 0
#var sampler2D  : texunit 1 : -1 : 0
#var sampler2D  : texunit 1 : -1 : 0
#var float4 Cee994ndhnk1rc :  : c[5] : -1 : 0
#var float4 Cmur1hd5g8ngo3 :  : c[4] : -1 : 0
#var sampler2D  : texunit 1 : -1 : 0
#var sampler2D  : texunit 1 : -1 : 0
#var sampler2D  : texunit 1 : -1 : 0
#var sampler2D  : texunit 1 : -1 : 0
#var sampler2D  : texunit 1 : -1 : 0
#var sampler2D  : texunit 1 : -1 : 0
#var sampler2D  : texunit 1 : -1 : 0
#var float4 Clofsu13ov1o1e :  : c[3] : -1 : 0
#var float4 C6r0tbn6evb4u3 :  : c[2] : -1 : 0
#var float4 Coit1m35rlpml7 :  : c[1] : -1 : 0
#var float4 C144q3o24uukb6 :  : c[0] : -1 : 0
#var float4 Ct2sb36fhpb7ea :  : c[8] : -1 : 0
#var float4 Cg86p5d28dfggc :  : c[7] : -1 : 0
#var float4 C85f7kkc64o5fe :  : c[6] : -1 : 0
#var float4 Cg1k4oc9vs2b8e :  : c[9] : -1 : 0
#var float4 f[TEX0] : $vin.F : F[0] : -1 : 0
#var sampler2D  : texunit 0 : -1 : 0
#tram 0 = f[WPOS].w
#tram 1 = f[TEX0].x
#tram 2 = f[TEX0].y
BB0:
IPA      R0, 0;
RCP      R0, R0;
IPA      R4, 1, R0;
IPA      R5, 2, R0;
TEX      R4, 0, 0, 2D;
FMUL32   R1, R5, c[37];
FMUL32   R0, R5, c[25];
FMAD     R1, R4, c[36], R1;
FMAD     R2, R4, c[24], R0;
FMUL32   R0, R5, c[29];
FMAD     R1, R6, c[38], R1;
FMAD     R2, R6, c[26], R2;
FMAD     R0, R4, c[28], R0;
FADD32   R1, R1, c[39];
FADD32   R3, R2, c[27];
FMAD     R0, R6, c[30], R0;
RCP      R48, R1;
FMUL32   R1, R5, c[33];
FADD32   R2, R0, c[31];
FMAD     R0, R48, R3, c[20];
FMAD     R4, R4, c[32], R1;
FMAD     R1, R48, R2, c[21];
FMAD     R44, R48, R3, c[12];
FMAD     R4, R6, c[34], R4;
FMAD     R45, R48, R2, c[13];
FMAD     R28, R48, R3, c[4];
FADD32   R49, R4, c[35];
FMAD     R29, R48, R2, c[5];
FMAD     R16, R48, R3, c[12];
FMAD     R17, R48, R2, c[13];
FMAD     R20, R48, R3, c[8];
FMAD     R21, R48, R2, c[9];
FMAD     R12, R48, R3, c[8];
FMAD     R13, R48, R2, c[9];
FMAD     R8, R48, R3, c[4];
FMAD     R9, R48, R2, c[5];
FMAD     R40, R48, R3, c[0];
FMAD     R41, R48, R2, c[1];
FMAD     R36, R48, R3, c[0];
FMAD     R37, R48, R2, c[1];
FMAD     R32, R48, R3, c[20];
FMAD     R33, R48, R2, c[21];
FMAD     R24, R48, R3, c[16];
FMAD     R4, R48, R3, c[16];
FMUL32I.SAT R3, R6, 0.066667;
FMAD     R25, R48, R2, c[17];
FMAD     R5, R48, R2, c[17];
F2F.SAT  R10, R3;
TEX      R0, 1, 1, 2D;
TEX      R4, 1, 1, 2D;
FMAD     R50, R10, -R10, c[0];
TEX      R24, 1, 1, 2D;
FMAD     R0, R48, R49, -R0;
FMAD     R1, R48, R49, -R4;
FSET     R0, -R0, c[0], GE;
FSET     R1, -R1, c[0], GE;
FMAD     R2, R48, R49, -R24;
TEX      R32, 1, 1, 2D;
FSET     R2, -R2, c[0], GE;
TEX      R36, 1, 1, 2D;
FMUL32I  R2, R2, 0.25;
FMAD     R3, R48, R49, -R32;
FMAD     R1, R1, c[0], R2;
FSET     R2, -R3, c[0], GE;
FMAD     R3, R48, R49, -R36;
TEX      R40, 1, 1, 2D;
FMAD     R2, R2, c[0], R1;
FSET     R1, -R3, c[0], GE;
FMAD     R0, R0, c[0], R2;
TEX      R8, 1, 1, 2D;
FMAD     R2, R48, R49, -R40;
TEX      R12, 1, 1, 2D;
FSET     R2, -R2, c[0], GE;
FMAD     R3, R48, R49, -R8;
FMUL32I  R2, R2, 0.25;
FMAD     R4, R48, R49, -R12;
FSET     R3, -R3, c[0], GE;
FMAD     R1, R1, c[0], R2;
FSET     R2, -R4, c[0], GE;
TEX      R20, 1, 1, 2D;
FMAD     R1, R3, c[0], R1;
TEX      R16, 1, 1, 2D;
FMAD     R3, R48, R49, -R20;
TEX      R28, 1, 1, 2D;
FMAD     R4, R48, R49, -R16;
FSET     R3, -R3, c[0], GE;
FSET     R4, -R4, c[0], GE;
FMUL32I  R3, R3, 0.25;
F2F      R5, -R28;
TEX      R44, 1, 1, 2D;
FMAD     R2, R2, c[0], R3;
FMAD     R3, R48, R49, R5;
FMAD     R2, R4, c[0], R2;
FSET     R3, -R3, c[0], GE;
FMAD     R4, R48, R49, -R44;
FMAD     R1, R3, c[0], R1;
FSET     R3, -R4, c[0], GE;
FMAD     R2, R3, c[0], R2;
FMUL32I  R2, R2, 0.333333;
FMAD     R1, R1, c[0], R2;
FMAD     R0, R0, c[0], R1;
FMUL32   R3, R0, R50;
MOV32    R0, R3;
MOV32    R1, R3;
MOV32    R2, R3;
END
# 105 instructions, 52 R-regs, 3 interpolants
# 105 inst, (3 mov, 0 mvi, 13 tex, 3 ipa, 2 complex, 84 math)
#    88 64-bit, 12 32-bit, 5 32-bit-const
