!!SPA1.0
.CONST_MODE  PAGE
.THREAD_TYPE PIXEL
.MAX_REG     51
.MAX_ATTR    2
# parseasm build date Mar 10 2004 15:40:49
# -profile fp50 -po tbat3 -po lat3 -if ps2x -i allprogs-new32//p360-lw40.s -o allprogs-new32//p360-lw50.s
#vendor LWPU
#version parseasm.0.0
#profile fp50
#program fp30entry
#semantic <null atom>
#semantic <null atom>
#semantic <null atom>
#semantic <null atom>
#semantic <null atom>
#semantic <null atom>
#semantic <null atom>
#semantic Cee994ndhnk1rc.Cee994ndhnk1rc
#semantic Cmur1hd5g8ngo3.Cmur1hd5g8ngo3
#semantic Clofsu13ov1o1e.Clofsu13ov1o1e
#semantic <null atom>
#semantic <null atom>
#semantic <null atom>
#semantic <null atom>
#semantic <null atom>
#semantic <null atom>
#semantic C6r0tbn6evb4u3.C6r0tbn6evb4u3
#semantic Coit1m35rlpml7.Coit1m35rlpml7
#semantic C144q3o24uukb6.C144q3o24uukb6
#semantic Ct2sb36fhpb7ea.Ct2sb36fhpb7ea
#semantic Cg86p5d28dfggc.Cg86p5d28dfggc
#semantic C85f7kkc64o5fe.C85f7kkc64o5fe
#semantic Cg1k4oc9vs2b8e.Cg1k4oc9vs2b8e
#semantic <null atom>
#var float4 o[COLH0] : $vout.O : O[0] : -1 : 0
#var sampler2D  : texunit 2 : -1 : 0
#var sampler2D  : texunit 1 : -1 : 0
#var sampler2D  : texunit 1 : -1 : 0
#var sampler2D  : texunit 1 : -1 : 0
#var sampler2D  : texunit 1 : -1 : 0
#var sampler2D  : texunit 1 : -1 : 0
#var sampler2D  : texunit 1 : -1 : 0
#var float4 Cee994ndhnk1rc :  : c[5] : -1 : 0
#var float4 Cmur1hd5g8ngo3 :  : c[4] : -1 : 0
#var float4 Clofsu13ov1o1e :  : c[3] : -1 : 0
#var sampler2D  : texunit 1 : -1 : 0
#var sampler2D  : texunit 1 : -1 : 0
#var sampler2D  : texunit 1 : -1 : 0
#var sampler2D  : texunit 1 : -1 : 0
#var sampler2D  : texunit 1 : -1 : 0
#var sampler2D  : texunit 1 : -1 : 0
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
RCP      R1, R0;
IPA      R0, 1, R1;
IPA      R1, 2, R1;
TEX      R0, 0, 0, 2D;
FMUL32I  R4, R2, 0.01;
FMUL32   R3, R1, c[37];
MOV32    R40, R4;
MOV32    R41, R4;
FMAD     R3, R0, c[36], R3;
FMUL32   R4, R1, c[25];
TEX      R40, 2, 2, 2D;
FMAD     R3, R2, c[38], R3;
FMAD     R4, R0, c[24], R4;
FADD32   R5, R3, c[39];
FMAD     R4, R2, c[26], R4;
FMUL32   R3, R1, c[29];
RCP      R41, R5;
FADD32   R5, R4, c[27];
FMAD     R3, R0, c[28], R3;
FMAD     R44, R41, R5, c[20];
FMAD     R3, R2, c[30], R3;
FMAD     R48, R41, R5, c[12];
FMAD     R24, R41, R5, c[4];
FADD32   R3, R3, c[31];
FMAD     R28, R41, R5, c[12];
FMAD     R36, R41, R5, c[8];
FMAD     R45, R41, R3, c[21];
FMAD     R49, R41, R3, c[13];
FMAD     R25, R41, R3, c[5];
FMAD     R29, R41, R3, c[13];
FMAD     R37, R41, R3, c[9];
FMAD     R32, R41, R5, c[8];
FMAD     R33, R41, R3, c[9];
FMAD     R12, R41, R5, c[4];
FMAD     R13, R41, R3, c[5];
FMAD     R16, R41, R5, c[0];
FMAD     R17, R41, R3, c[1];
FMAD     R20, R41, R5, c[0];
FMUL32   R1, R1, c[33];
FMAD     R21, R41, R3, c[1];
FMAD     R8, R41, R5, c[20];
FMAD     R0, R0, c[32], R1;
FMAD     R9, R41, R3, c[21];
FMAD     R4, R41, R5, c[16];
FMAD     R2, R2, c[34], R0;
FMAD     R0, R41, R5, c[16];
FMAD     R5, R41, R3, c[17];
FMAD     R1, R41, R3, c[17];
FADD32   R42, R2, c[35];
TEX      R4, 1, 1, 2D;
TEX      R0, 1, 1, 2D;
TEX      R8, 1, 1, 2D;
FMAD     R1, R41, R42, -R4;
FMAD     R0, R41, R42, -R0;
FSET     R2, -R1, c[0], GE;
FSET     R1, -R0, c[0], GE;
FMAD     R0, R41, R42, -R8;
FMUL32I  R2, R2, 0.25;
TEX      R20, 1, 1, 2D;
FSET     R0, -R0, c[0], GE;
FMAD     R1, R1, c[0], R2;
TEX      R16, 1, 1, 2D;
FMAD     R0, R0, c[0], R1;
FMAD     R1, R41, R42, -R20;
TEX      R12, 1, 1, 2D;
FSET     R1, -R1, c[0], GE;
FMAD     R2, R41, R42, -R16;
TEX      R32, 1, 1, 2D;
FSET     R2, -R2, c[0], GE;
FMAD     R3, R41, R42, -R12;
FMUL32I  R2, R2, 0.25;
FSET     R3, -R3, c[0], GE;
FMAD     R4, R41, R42, -R32;
FMAD     R1, R1, c[0], R2;
TEX      R36, 1, 1, 2D;
FSET     R2, -R4, c[0], GE;
FMAD     R1, R3, c[0], R1;
TEX      R28, 1, 1, 2D;
F2F      R3, -R36;
TEX      R24, 1, 1, 2D;
FMAD     R3, R41, R42, R3;
FMAD     R4, R41, R42, -R28;
FSET     R5, -R3, c[0], GE;
FSET     R3, -R4, c[0], GE;
FMAD     R4, R41, R42, -R24;
FMUL32I  R5, R5, 0.25;
TEX      R48, 1, 1, 2D;
FSET     R4, -R4, c[0], GE;
FMAD     R2, R2, c[0], R5;
FMAD     R1, R4, c[0], R1;
FMAD     R2, R3, c[0], R2;
FMAD     R3, R41, R42, -R48;
TEX      R44, 1, 1, 2D;
FSET     R3, -R3, c[0], GE;
FMAD     R2, R3, c[0], R2;
FMAD     R3, R41, R42, -R44;
FMUL32I  R2, R2, 0.333333;
FSET     R3, -R3, c[0], GE;
FMAD     R1, R1, c[0], R2;
FMAD     R0, R3, c[0], R0;
FMAD     R0, R0, c[0], R1;
FMAD     R3, R40, R0, R43;
MOV32    R0, R3;
MOV32    R1, R3;
MOV32    R2, R3;
END
# 106 instructions, 52 R-regs, 3 interpolants
# 106 inst, (5 mov, 0 mvi, 14 tex, 3 ipa, 2 complex, 82 math)
#    88 64-bit, 13 32-bit, 5 32-bit-const
