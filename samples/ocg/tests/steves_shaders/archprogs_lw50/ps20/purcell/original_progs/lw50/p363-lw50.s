!!SPA1.0
.CONST_MODE  PAGE
.THREAD_TYPE PIXEL
.MAX_REG     39
.MAX_ATTR    2
# parseasm build date Mar 10 2004 15:40:49
# -profile fp50 -po tbat3 -po lat3 -if ps2x -i allprogs-new32//p363-lw40.s -o allprogs-new32//p363-lw50.s
#vendor LWPU
#version parseasm.0.0
#profile fp50
#program fp30entry
#semantic Ct2sb36fhpb7ea.Ct2sb36fhpb7ea
#semantic <null atom>
#semantic <null atom>
#semantic <null atom>
#semantic <null atom>
#semantic <null atom>
#semantic <null atom>
#semantic <null atom>
#semantic <null atom>
#semantic <null atom>
#semantic <null atom>
#semantic Cg86p5d28dfggc.Cg86p5d28dfggc
#semantic C85f7kkc64o5fe.C85f7kkc64o5fe
#semantic Cee994ndhnk1rc.Cee994ndhnk1rc
#semantic Cmur1hd5g8ngo3.Cmur1hd5g8ngo3
#semantic Clofsu13ov1o1e.Clofsu13ov1o1e
#semantic C6r0tbn6evb4u3.C6r0tbn6evb4u3
#semantic Coit1m35rlpml7.Coit1m35rlpml7
#semantic C144q3o24uukb6.C144q3o24uukb6
#semantic <null atom>
#semantic Cg1k4oc9vs2b8e.Cg1k4oc9vs2b8e
#semantic <null atom>
#var float4 o[COLH0] : $vout.O : O[0] : -1 : 0
#var float4 Ct2sb36fhpb7ea :  : c[8] : -1 : 0
#var sampler2D  : texunit 3 : -1 : 0
#var sampler2D  : texunit 1 : -1 : 0
#var samplerLWBE  : texunit 2 : -1 : 0
#var samplerLWBE  : texunit 2 : -1 : 0
#var samplerLWBE  : texunit 2 : -1 : 0
#var samplerLWBE  : texunit 2 : -1 : 0
#var samplerLWBE  : texunit 2 : -1 : 0
#var samplerLWBE  : texunit 2 : -1 : 0
#var samplerLWBE  : texunit 2 : -1 : 0
#var samplerLWBE  : texunit 2 : -1 : 0
#var float4 Cg86p5d28dfggc :  : c[7] : -1 : 0
#var float4 C85f7kkc64o5fe :  : c[6] : -1 : 0
#var float4 Cee994ndhnk1rc :  : c[5] : -1 : 0
#var float4 Cmur1hd5g8ngo3 :  : c[4] : -1 : 0
#var float4 Clofsu13ov1o1e :  : c[3] : -1 : 0
#var float4 C6r0tbn6evb4u3 :  : c[2] : -1 : 0
#var float4 Coit1m35rlpml7 :  : c[1] : -1 : 0
#var float4 C144q3o24uukb6 :  : c[0] : -1 : 0
#var samplerLWBE  : texunit 2 : -1 : 0
#var float4 Cg1k4oc9vs2b8e :  : c[9] : -1 : 0
#var float4 f[TEX0] : $vin.F : F[0] : -1 : 0
#var sampler2D  : texunit 0 : -1 : 0
#tram 0 = f[TEX0].x
#tram 1 = f[TEX0].y
#tram 2 = f[TEX0].w
BB0:
IPA      R0, 2;
RCP      R0, R0;
IPA      R36, 0, R0;
IPA      R37, 1, R0;
MOV32    R32, R36;
MOV32    R33, R37;
TEX      R32, 0, 0, 2D;
FADD32   R23, R32, -c[36];
FADD32   R19, R33, -c[37];
FADD32   R11, R34, -c[38];
FMAX     R0, |R23|, |R19|;
FMUL32   R1, R19, R19;
FMAX     R0, |R11|, R0;
FMAD     R1, R23, R23, R1;
RCP      R0, R0;
FMAD     R3, R11, R11, R1;
FMUL32   R12, R23, R0;
FMUL32   R13, R19, R0;
FMUL32   R14, R11, R0;
RSQ      R7, R3;
TEX      R12, 2, 2, LWBE;
FMAD     R2, R7, R3, -R12;
FMAD     R12, R2, c[28], R23;
FMAD     R9, R2, c[29], R19;
FMAD     R6, R2, c[30], R11;
FMAD     R5, R2, c[12], R23;
FMAX     R1, |R12|, |R9|;
FMAD     R4, R2, c[13], R19;
FMAD     R0, R2, c[14], R11;
FMAX     R8, |R6|, R1;
FMAX     R1, |R5|, |R4|;
RCP      R8, R8;
FMAX     R1, |R0|, R1;
FMAD     R10, R2, c[24], R23;
FMUL32   R28, R12, R8;
FMUL32   R29, R9, R8;
FMUL32   R30, R6, R8;
RCP      R1, R1;
FMAD     R9, R2, c[25], R19;
FMAD     R6, R2, c[26], R11;
FMUL32   R24, R5, R1;
FMUL32   R25, R4, R1;
FMUL32   R26, R0, R1;
FMAX     R0, |R10|, |R9|;
FMAD     R5, R2, c[20], R23;
FMAD     R4, R2, c[21], R19;
FMAX     R8, |R6|, R0;
FMAD     R0, R2, c[22], R11;
FMAX     R1, |R5|, |R4|;
RCP      R8, R8;
FMAX     R1, |R0|, R1;
FMUL32   R16, R10, R8;
FMUL32   R17, R9, R8;
FMUL32   R18, R6, R8;
RCP      R1, R1;
FMAD     R10, R2, c[16], R23;
FMAD     R9, R2, c[17], R19;
FMUL32   R20, R5, R1;
FMUL32   R21, R4, R1;
FMUL32   R22, R0, R1;
FMAX     R0, |R10|, |R9|;
FMAD     R4, R2, c[18], R11;
FMAD     R8, R2, c[8], R23;
FMAD     R6, R2, c[9], R19;
FMAX     R5, |R4|, R0;
FMAD     R0, R2, c[10], R11;
FMAX     R1, |R8|, |R6|;
RCP      R5, R5;
FMAX     R1, |R0|, R1;
FMUL32   R12, R10, R5;
FMUL32   R13, R9, R5;
FMUL32   R14, R4, R5;
RCP      R1, R1;
FMAD     R4, R2, c[4], R23;
FMAD     R5, R2, c[5], R19;
FMUL32   R8, R8, R1;
FMUL32   R9, R6, R1;
FMUL32   R10, R0, R1;
FMAX     R15, |R4|, |R5|;
FMAD     R6, R2, c[6], R11;
FMAD     R0, R2, c[0], R23;
FMAD     R1, R2, c[1], R19;
FMAX     R27, |R6|, R15;
FMAD     R2, R2, c[2], R11;
FMAX     R15, |R0|, |R1|;
RCP      R27, R27;
FMAX     R15, |R2|, R15;
FMUL32   R4, R4, R27;
FMUL32   R5, R5, R27;
FMUL32   R6, R6, R27;
RCP      R31, R15;
FMUL32   R15, R7, R3;
FMUL32   R27, R33, R33;
FMUL32   R0, R0, R31;
FMUL32   R1, R1, R31;
FMUL32   R2, R2, R31;
FMAD     R27, R32, R32, R27;
RCP      R31, R15;
FMAD     R27, R34, R34, R27;
FMUL32   R38, R23, R31;
FMUL32   R39, R19, R31;
FMUL32   R35, R11, R31;
RSQ      R23, R27;
FMAD     R11, R32, -R23, -R38;
FMAD     R19, R33, -R23, -R39;
FMAD     R23, R34, -R23, -R35;
MVI      R31, 1.0;
FMUL32   R27, R19, R19;
FMAD     R33, R7, R3, c[0];
FMAD     R7, R11, R11, R27;
FMAD     R32, -R15, c[39], R31;
TEX      R0, 2, 2, LWBE;
FMAD     R15, R23, R23, R7;
TEX      R4, 2, 2, LWBE;
RSQ      R1, R15;
FADD32   R3, -R0, R33;
FMUL32   R0, R23, R1;
FMUL32   R2, R11, R1;
FMUL32   R1, R19, R1;
FSET     R3, -R3, c[0], GE;
FADD32   R4, R33, -R4;
TEX      R8, 2, 2, LWBE;
TEX      R12, 2, 2, LWBE;
FSET     R4, -R4, c[0], GE;
FMUL32I  R4, R4, 0.125;
FADD32   R5, R33, -R8;
FADD32   R6, R33, -R12;
FMAD     R3, R3, c[0], R4;
FSET     R5, -R5, c[0], GE;
FSET     R4, -R6, c[0], GE;
TEX      R20, 2, 2, LWBE;
FMAD     R3, R5, c[0], R3;
TEX      R16, 2, 2, LWBE;
FADD32   R5, R33, -R20;
TEX      R24, 2, 2, LWBE;
FADD32   R6, R33, -R16;
FSET     R5, -R5, c[0], GE;
FSET     R6, -R6, c[0], GE;
FMUL32I  R5, R5, 0.125;
FADD32   R7, R33, -R24;
TEX      R28, 2, 2, LWBE;
FMAD     R4, R4, c[0], R5;
FSET     R5, -R7, c[0], GE;
FMAD     R8, R6, c[0], R4;
FMAD     R3, R5, c[0], R3;
FADD32   R6, R33, -R28;
MOV32    R4, R36;
MOV32    R5, R37;
FSET     R9, -R6, c[0], GE;
TEX      R4, 1, 1, 2D;
FMAD     R8, R9, c[0], R8;
FADD32   R7, R3, R8;
FMUL32   R1, R1, R5;
FMUL32   R3, -R39, R5;
FMAD     R1, R2, R4, R1;
FMAD     R2, -R38, R4, R3;
FMAD     R1, R0, R6, R1;
FMAD     R0, -R35, R6, R2;
TEX      R0, 3, 3, 2D;
FMUL32   R3, R3, c[35];
FMUL32   R2, R2, c[34];
FMUL32   R1, R1, c[33];
FMUL32   R0, R0, c[32];
FMUL32   R2, R7, R2;
FMUL32   R1, R7, R1;
FMUL32   R0, R7, R0;
FMUL32   R3, R7, R3;
FMUL32   R1, R1, R32;
FMUL32   R0, R0, R32;
FMUL32   R2, R2, R32;
FMUL32   R3, R3, R32;
END
# 171 instructions, 40 R-regs, 3 interpolants
# 171 inst, (4 mov, 1 mvi, 12 tex, 3 ipa, 14 complex, 137 math)
#    102 64-bit, 67 32-bit, 2 32-bit-const
