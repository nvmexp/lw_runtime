!!SPA1.0
.CONST_MODE  PAGE
.THREAD_TYPE VERTEX
.MAX_REG     19
.MAX_IBUF    7
.MAX_OBUF    19
# parseasm build date Mar 10 2004 15:40:49
# -profile vp50 -po tbat3 -po lat3 -if vs2x -i allprogs-new32//v937-lw40.s -o allprogs-new32//v937-lw50.s
#vendor LWPU
#version parseasm.0.0
#profile vp50
#program fp30entry
#semantic C[95].C[95]
#semantic C[94].C[94]
#semantic C[93].C[93]
#semantic C[92].C[92]
#semantic C[16].C[16]
#semantic C[1].C[1]
#semantic C[3].C[3]
#semantic c.c
#semantic C[91].C[91]
#semantic C[90].C[90]
#semantic C[11].C[11]
#semantic C[10].C[10]
#semantic C[9].C[9]
#semantic C[8].C[8]
#semantic C[0].C[0]
#semantic C[44].C[44]
#semantic C[43].C[43]
#semantic C[42].C[42]
#var float4 o[TEX2] : $vout.O : O[0] : -1 : 0
#var float4 o[TEX1] : $vout.O : O[0] : -1 : 0
#var float4 o[TEX0] : $vout.O : O[0] : -1 : 0
#var float4 o[FOGC] : $vout.O : O[0] : -1 : 0
#var float4 o[BCOL0] : $vout.O : O[0] : -1 : 0
#var float4 o[HPOS] : $vout.O : O[0] : -1 : 0
#var float4 C[95] :  : c[95] : -1 : 0
#var float4 C[94] :  : c[94] : -1 : 0
#var float4 C[93] :  : c[93] : -1 : 0
#var float4 C[92] :  : c[92] : -1 : 0
#var float4 v[TEX9] : $vin.F : F[0] : -1 : 0
#var float4 C[16] :  : c[16] : -1 : 0
#var float4 C[1] :  : c[1] : -1 : 0
#var float4 C[3] :  : c[3] : -1 : 0
#var float4 c :  : c[0] : -1 : 0
#var float4 C[91] :  : c[91] : -1 : 0
#var float4 C[90] :  : c[90] : -1 : 0
#var float4 C[11] :  : c[11] : -1 : 0
#var float4 C[10] :  : c[10] : -1 : 0
#var float4 C[9] :  : c[9] : -1 : 0
#var float4 C[8] :  : c[8] : -1 : 0
#var float4 C[0] :  : c[0] : -1 : 0
#var float4 C[44] :  : c[44] : -1 : 0
#var float4 C[43] :  : c[43] : -1 : 0
#var float4 C[42] :  : c[42] : -1 : 0
#var float4 v[OPOS] : $vin.F : F[0] : -1 : 0
#ibuf 0 = v[OPOS].x
#ibuf 1 = v[OPOS].y
#ibuf 2 = v[OPOS].z
#ibuf 3 = v[OPOS].w
#ibuf 4 = v[UNUSED1].x
#ibuf 5 = v[UNUSED1].y
#ibuf 6 = v[UNUSED1].z
#ibuf 7 = v[UNUSED1].w
#obuf 0 = o[HPOS].x
#obuf 1 = o[HPOS].y
#obuf 2 = o[HPOS].z
#obuf 3 = o[HPOS].w
#obuf 4 = o[BCOL0].x
#obuf 5 = o[BCOL0].y
#obuf 6 = o[BCOL0].z
#obuf 7 = o[BCOL0].w
#obuf 8 = o[TEX0].x
#obuf 9 = o[TEX0].y
#obuf 10 = o[TEX0].z
#obuf 11 = o[TEX0].w
#obuf 12 = o[TEX1].x
#obuf 13 = o[TEX1].y
#obuf 14 = o[TEX2].x
#obuf 15 = o[TEX2].y
#obuf 16 = o[FOGC].x
#obuf 17 = o[FOGC].y
#obuf 18 = o[FOGC].z
#obuf 19 = o[FOGC].w
BB0:
FMUL     R1, v[1], c[177];
FMUL     R0, v[1], c[169];
FMAD     R4, v[0], c[176], R1;
FMAD     R0, v[0], c[168], R0;
FMUL     R2, v[1], c[173];
FMAD     R4, v[2], c[178], R4;
FMAD     R0, v[2], c[170], R0;
FMAD     R2, v[0], c[172], R2;
FMAD     R12, v[3], c[179], R4;
FMAD     R14, v[3], c[171], R0;
FMAD     R2, v[2], c[174], R2;
FADD32   R4, R12, -c[362];
FADD32   R0, R14, -c[360];
FMAD     R7, v[3], c[175], R2;
FADD32   R2, R7, -c[361];
FMUL32   R3, R2, c[365];
FMAD     R3, R0, c[364], R3;
FMAD     R3, R4, c[366], R3;
FMUL32   R3, R3, c[3];
FMAD     R4, -R3, c[366], R4;
FMAD     R8, -R3, c[364], R0;
FMAD     R17, -R3, c[365], R2;
MOV32    R0, c[12];
FMUL32   R1, R17, R17;
F2I.FLOOR R0, R0;
FMAD     R1, R8, R8, R1;
I2I.M4   R0, R0;
FMAD     R1, R4, R4, R1;
R2A      A0, R0;
RSQ      R16, |R1|;
FADD32   R2, c[A0 + 8], -R14;
FADD32   R0, c[A0 + 10], -R12;
FMUL32   R10, R8, R16;
FMUL32   R13, R17, R16;
FADD32   R8, c[A0 + 9], -R7;
FMUL32   R15, R8, R8;
FMUL32   R4, R4, R16;
FMAD     R15, R2, R2, R15;
FMAD     R15, R0, R0, R15;
RSQ      R18, |R15|;
FMUL32   R8, R8, R18;
FMUL32   R2, R2, R18;
FMUL32   R0, R0, R18;
FMUL32   R16, R13, R8;
FMUL32   R19, c[A0 + 5], -R8;
FMAD     R8, R10, R2, R16;
FMAD     R11, c[A0 + 4], -R2, R19;
FMAD     R2, R4, R0, R8;
FMAD     R8, c[A0 + 6], -R0, R11;
MVI      R5, -127.996;
FMAX     R0, R2, c[0];
FADD32   R2, R8, -c[A0 + 14];
FSET     R0, R0, c[0], GT;
FMUL32   R3, R2, c[A0 + 15];
FMAX     R2, c[A0 + 12], R5;
FMAX     R9, R3, c[0];
FMIN     R8, R2, c[0];
FMUL32   R2, R13, R13;
LG2      R9, R9;
FSET     R11, R13, c[0], LT;
FMUL32   R8, R8, R9;
F2I.FLOOR R9, R11;
FMUL32   R11, R10, R10;
RRO      R8, R8, 1;
I2I.M4   R9, R9;
FSET     R10, R10, c[0], LT;
EX2      R8, R8;
R2A      A2, R9;
F2I.FLOOR R9, R10;
FCMP     R0, -R0, R8, c[0];
FSET     R8, R4, c[0], LT;
I2I.M4   R9, R9;
FMIN     R0, R0, c[1];
F2I.FLOOR R8, R8;
R2A      A3, R9;
FMUL32   R10, R1, R0;
I2I.M4   R0, R8;
FMUL32   R8, R15, R18;
FMUL32   R1, R11, c[A3 + 84];
R2A      A1, R0;
MOV32    R9, c[A0 + 16];
FMAD     R13, R2, c[A2 + 92], R1;
FMUL32   R0, R4, R4;
FMAD     R6, R8, c[A0 + 17], R9;
FMAD     R4, R0, c[A1 + 100], R13;
FMAD     R6, R15, c[A0 + 18], R6;
FMAX     R5, R5, c[4];
RCP      R6, R6;
FMIN     R15, R5, c[0];
FMUL32   R5, c[A0], R6;
FMAD     R4, R5, R10, R4;
FMAX     R9, R4, c[0];
FMAX     R8, R4, c[0];
FMUL32   R4, R11, c[A3 + 85];
FSET     R13, R9, c[0], GT;
LG2      R16, R8;
FMAD     R8, R2, c[A2 + 93], R4;
FMUL32   R4, c[A0 + 1], R6;
FMUL32   R2, R15, R16;
FMAD     R8, R0, c[A1 + 101], R8;
RRO      R0, R2, 1;
FMAD     R4, R4, R10, R8;
EX2      R0, R0;
FMAX     R2, R5, c[0];
FMAX     R4, R4, c[0];
FCMP     R0, -R13, R0, c[0];
FSET     R2, R2, c[0], GT;
LG2      R4, R4;
FMUL32   R5, R11, c[A0 + 86];
FMUL32   R0, R0, c[7];
FMUL32   R4, R15, R4;
FMAD     R3, R3, c[A0 + 94], R5;
FMUL32   R5, c[A2 + 2], R6;
RRO      R4, R4, 1;
FMAD     R1, R1, c[A3 + 102], R3;
EX2      R3, R4;
FMAD     R1, R5, R10, R1;
FCMP     R3, -R2, R3, c[0];
FMAX     R2, R1, c[0];
FMAX     R4, R1, c[0];
FMUL32   R1, R3, c[7];
FSET     R2, R2, c[0], GT;
LG2      R4, R4;
FMAX     R3, R0, R1;
FMUL32   R4, R15, R4;
RRO      R4, R4, 1;
EX2      R4, R4;
FCMP     R4, -R2, R4, c[0];
FMUL32   R2, R7, c[377];
FMUL32   R5, R4, c[7];
FMAD     R4, R14, c[376], R2;
MOV32    R2, c[1];
FMAX     R3, R3, R5;
FMAD     R4, R12, c[378], R4;
FMAX     R6, R3, c[1];
FMUL32   R3, R7, c[381];
FMAD     o[14], R2, c[379], R4;
RCP      R6, R6;
FMAD     R3, R14, c[380], R3;
FMUL32   R4, R7, c[369];
FMUL32   o[4], R6, R0;
FMUL32   o[5], R6, R1;
FMUL32   o[6], R6, R5;
FMAD     R0, R12, c[382], R3;
FMAD     R1, R14, c[368], R4;
FMUL32   R3, R7, c[373];
FMAD     o[15], R2, c[383], R0;
FMAD     R0, R12, c[370], R1;
FMAD     R1, R14, c[372], R3;
FMUL32   R3, R7, c[41];
FMAD     o[12], R2, c[371], R0;
FMAD     R0, R12, c[374], R1;
FMAD     R3, R14, c[40], R3;
MOV32    R1, c[43];
FMAD     o[13], R2, c[375], R0;
FMAD     R0, R12, c[42], R3;
MOV32    R2, c[67];
FMUL32   R3, R7, c[33];
FMAD     R0, R1, c[1], R0;
MOV32    R1, c[35];
FMAD     R3, R14, c[32], R3;
FMAD     R2, -R0, R2, c[64];
FMAD     R3, R12, c[34], R3;
MOV32    o[16], R2;
MOV32    o[17], R2;
FMAD     o[0], R1, c[1], R3;
MOV32    o[18], R2;
MOV32    o[19], R2;
MOV32    o[2], R0;
FMUL32   R0, R7, c[37];
FMUL32   R2, R7, c[45];
MOV32    R1, c[39];
FMAD     R0, R14, c[36], R0;
FMAD     R2, R14, c[44], R2;
FMAD     R0, R12, c[38], R0;
FMAD     R3, R12, c[46], R2;
MOV32    R2, c[47];
FMAD     o[1], R1, c[1], R0;
MOV      o[8], v[4];
MOV32    R0, R2;
MOV      o[9], v[5];
MOV      o[10], v[6];
FMAD     o[3], R0, c[1], R3;
MOV      o[11], v[7];
MOV32    R0, c[1];
MOV32    o[7], R0;
END
# 186 instructions, 20 R-regs
# 186 inst, (20 mov, 1 mvi, 0 tex, 12 complex, 153 math)
#    121 64-bit, 65 32-bit, 0 32-bit-const
