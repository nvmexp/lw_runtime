!!SPA1.0
.CONST_MODE  PAGE
.THREAD_TYPE VERTEX
.MAX_REG     23
.MAX_IBUF    7
.MAX_OBUF    19
# parseasm build date Mar 10 2004 15:40:49
# -profile vp50 -po tbat3 -po lat3 -if vs2x -i allprogs-new32//v902-lw40.s -o allprogs-new32//v902-lw50.s
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
FMUL     R0, v[1], c[177];
FMUL     R2, v[1], c[169];
FMAD     R0, v[0], c[176], R0;
FMAD     R2, v[0], c[168], R2;
FMUL     R4, v[1], c[173];
FMAD     R0, v[2], c[178], R0;
FMAD     R2, v[2], c[170], R2;
FMAD     R4, v[0], c[172], R4;
FMAD     R0, v[3], c[179], R0;
FMAD     R2, v[3], c[171], R2;
FMAD     R7, v[2], c[174], R4;
FADD32   R4, R0, -c[362];
FADD32   R8, R2, -c[360];
FMAD     R7, v[3], c[175], R7;
FADD32   R10, R7, -c[361];
FMUL32   R11, R10, c[365];
FMAD     R11, R8, c[364], R11;
FMAD     R11, R4, c[366], R11;
FMUL32   R11, R11, c[3];
FMAD     R4, -R11, c[366], R4;
FMAD     R8, -R11, c[364], R8;
FMAD     R16, -R11, c[365], R10;
MOV32    R10, c[13];
FMUL32   R11, R16, R16;
F2I.FLOOR R10, R10;
FMAD     R11, R8, R8, R11;
I2I.M4   R10, R10;
FMAD     R11, R4, R4, R11;
R2A      A0, R10;
RSQ      R17, |R11|;
FADD32   R10, c[A0 + 8], -R2;
FADD32   R12, c[A0 + 9], -R7;
FMUL32   R14, R8, R17;
FMUL32   R18, R16, R17;
FMUL32   R16, R12, R12;
FADD32   R8, c[A0 + 10], -R0;
FMUL32   R4, R4, R17;
FMAD     R16, R10, R10, R16;
FMAD     R19, R8, R8, R16;
RSQ      R20, |R19|;
FMUL32   R12, R12, R20;
FMUL32   R10, R10, R20;
FMUL32   R8, R8, R20;
FMUL32   R16, R18, R12;
FMUL32   R21, c[A0 + 5], -R12;
FMAD     R12, R14, R10, R16;
FMAD     R16, c[A0 + 4], -R10, R21;
FMAD     R10, R4, R8, R12;
FMAD     R13, c[A0 + 6], -R8, R16;
MVI      R12, -127.996;
FMAX     R8, R10, c[0];
FADD32   R10, R13, -c[A0 + 14];
MOV32    R21, R12;
FSET     R8, R8, c[0], GT;
FMUL32   R11, R10, c[A0 + 15];
FMAX     R10, c[A0 + 12], R21;
FMAX     R11, R11, c[0];
FMIN     R10, R10, c[0];
MOV32    R12, c[12];
LG2      R11, R11;
F2I.FLOOR R12, R12;
FMUL32   R10, R10, R11;
I2I.M4   R11, R12;
RRO      R10, R10, 1;
R2A      A1, R11;
EX2      R12, R10;
FADD32   R10, c[A1 + 8], -R2;
FCMP     R3, -R8, R12, c[0];
FADD32   R12, c[A1 + 10], -R0;
FADD32   R16, c[A1 + 9], -R7;
FMIN     R1, R3, c[1];
FMUL32   R3, R16, R16;
FMUL32   R1, R9, R1;
FMAD     R3, R10, R10, R3;
FMAD     R3, R12, R12, R3;
RSQ      R22, |R3|;
FMUL32   R8, R16, R22;
FMUL32   R10, R10, R22;
FMUL32   R12, R12, R22;
FMUL32   R16, R18, R8;
FMUL32   R23, c[A1 + 5], -R8;
FMAD     R8, R14, R10, R16;
FMAD     R10, c[A1 + 4], -R10, R23;
FMAD     R8, R4, R12, R8;
FMAD     R10, c[A1 + 6], -R12, R10;
FMAX     R5, c[A1 + 12], R21;
FMAX     R8, R8, c[0];
FADD32   R10, R10, -c[A1 + 14];
FMIN     R5, R5, c[0];
FSET     R8, R8, c[0], GT;
FMUL32   R10, R10, c[A1 + 15];
FMAX     R12, R10, c[0];
FMUL32   R10, R18, R18;
FSET     R13, R18, c[0], LT;
LG2      R12, R12;
F2I.FLOOR R13, R13;
FMUL32   R12, R5, R12;
FMUL32   R5, R14, R14;
I2I.M4   R13, R13;
RRO      R12, R12, 1;
FSET     R15, R14, c[0], LT;
R2A      A3, R13;
EX2      R14, R12;
F2I.FLOOR R15, R15;
FMUL32   R12, R4, R4;
FCMP     R6, -R8, R14, c[0];
I2I.M4   R8, R15;
FSET     R4, R4, c[0], LT;
FMIN     R6, R6, c[1];
R2A      A4, R8;
F2I.FLOOR R8, R4;
FMUL32   R4, R9, R6;
FMUL32   R6, R3, R22;
I2I.M4   R9, R8;
FMUL32   R14, R5, c[A4 + 84];
MOV32    R8, c[A1 + 16];
R2A      A2, R9;
FMAD     R9, R10, c[A3 + 92], R14;
FMAD     R8, R6, c[A1 + 17], R8;
FMUL32   R6, R19, R20;
FMAD     R9, R12, c[A2 + 100], R9;
FMAD     R3, R3, c[A1 + 18], R8;
MOV32    R8, c[A0 + 16];
RCP      R3, R3;
FMAD     R6, R6, c[A0 + 17], R8;
FMUL32   R8, c[A1], R3;
FMAD     R6, R19, c[A0 + 18], R6;
FMAD     R8, R8, R4, R9;
RCP      R6, R6;
FMAX     R14, R21, c[4];
FMUL32   R9, c[A0], R6;
FMIN     R19, R14, c[0];
FMAD     R8, R9, R1, R8;
FMAX     R15, R8, c[0];
FMAX     R14, R8, c[0];
FMUL32   R8, R5, c[A4 + 85];
FSET     R18, R15, c[0], GT;
LG2      R16, R14;
FMAD     R14, R10, c[A3 + 93], R8;
FMUL32   R8, c[A1 + 1], R3;
FMUL32   R10, R19, R16;
FMAD     R16, R12, c[A2 + 101], R14;
FMUL32   R14, c[A0 + 1], R6;
RRO      R10, R10, 1;
FMAD     R8, R8, R4, R16;
EX2      R10, R10;
FMAD     R8, R14, R1, R8;
FCMP     R10, -R18, R10, c[0];
FMAX     R9, R9, c[0];
FMAX     R8, R8, c[0];
FMUL32   R10, R10, c[7];
FSET     R9, R9, c[0], GT;
FMUL32   R5, R5, c[A6 + 86];
LG2      R8, R8;
FMAD     R5, R11, c[A0 + 94], R5;
FMUL32   R3, c[A3 + 2], R3;
FMUL32   R8, R19, R8;
FMAD     R5, R13, c[A0 + 102], R5;
FMUL32   R6, c[A2 + 2], R6;
RRO      R8, R8, 1;
FMAD     R3, R3, R4, R5;
EX2      R4, R8;
FMAD     R1, R6, R1, R3;
FCMP     R4, -R9, R4, c[0];
FMAX     R3, R1, c[0];
FMAX     R5, R1, c[0];
FMUL32   R1, R4, c[7];
FSET     R3, R3, c[0], GT;
LG2      R5, R5;
FMAX     R4, R10, R1;
FMUL32   R5, R19, R5;
RRO      R5, R5, 1;
EX2      R5, R5;
FCMP     R5, -R3, R5, c[0];
FMUL32   R3, R7, c[377];
FMUL32   R6, R5, c[7];
FMAD     R5, R2, c[376], R3;
MOV32    R3, c[1];
FMAX     R4, R4, R6;
FMAD     R5, R0, c[378], R5;
FMAX     R8, R4, c[1];
FMUL32   R4, R7, c[381];
FMAD     o[14], R3, c[379], R5;
RCP      R8, R8;
FMAD     R4, R2, c[380], R4;
FMUL32   R5, R7, c[369];
FMUL32   o[4], R8, R10;
FMUL32   o[5], R8, R1;
FMUL32   o[6], R8, R6;
FMAD     R1, R0, c[382], R4;
FMAD     R4, R2, c[368], R5;
FMUL32   R5, R7, c[373];
FMAD     o[15], R3, c[383], R1;
FMAD     R1, R0, c[370], R4;
FMAD     R4, R2, c[372], R5;
FMUL32   R5, R7, c[41];
FMAD     o[12], R3, c[371], R1;
FMAD     R1, R0, c[374], R4;
FMAD     R5, R2, c[40], R5;
MOV32    R4, c[43];
FMAD     o[13], R3, c[375], R1;
FMAD     R1, R0, c[42], R5;
MOV32    R3, R4;
MOV32    R4, c[67];
FMUL32   R5, R7, c[33];
FMAD     R1, R3, c[1], R1;
MOV32    R3, c[35];
FMAD     R5, R2, c[32], R5;
FMAD     R4, -R1, R4, c[64];
FMAD     R5, R0, c[34], R5;
MOV32    o[16], R4;
MOV32    o[17], R4;
FMAD     o[0], R3, c[1], R5;
MOV32    o[18], R4;
MOV32    o[19], R4;
MOV32    o[2], R1;
FMUL32   R1, R7, c[37];
FMUL32   R4, R7, c[45];
MOV32    R3, c[39];
FMAD     R1, R2, c[36], R1;
FMAD     R2, R2, c[44], R4;
FMAD     R1, R0, c[38], R1;
FMAD     R2, R0, c[46], R2;
MOV32    R0, c[47];
FMAD     o[1], R3, c[1], R1;
MOV      o[8], v[4];
MOV      o[9], v[5];
MOV      o[10], v[6];
FMAD     o[3], R0, c[1], R2;
MOV      o[11], v[7];
MOV32    R0, c[1];
MOV32    o[7], R0;
END
# 232 instructions, 24 R-regs
# 232 inst, (23 mov, 1 mvi, 0 tex, 16 complex, 192 math)
#    147 64-bit, 85 32-bit, 0 32-bit-const
