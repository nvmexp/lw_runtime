!!SPA1.0
.CONST_MODE  PAGE
.THREAD_TYPE VERTEX
.MAX_REG     19
.MAX_IBUF    7
.MAX_OBUF    19
# parseasm build date Mar 10 2004 15:40:49
# -profile vp50 -po tbat3 -po lat3 -if vs2x -i allprogs-new32//v1073-lw40.s -o allprogs-new32//v1073-lw50.s
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
FMUL     R1, v[1], c[169];
FMUL     R0, v[1], c[173];
FMAD     R1, v[0], c[168], R1;
FMAD     R0, v[0], c[172], R0;
FMAD     R2, v[2], c[170], R1;
FMAD     R1, v[2], c[174], R0;
FMUL     R0, v[1], c[177];
FMAD     R7, v[3], c[171], R2;
FMAD     R8, v[3], c[175], R1;
FMAD     R2, v[0], c[176], R0;
FADD32   R0, R7, -c[360];
FADD32   R1, R8, -c[361];
FMAD     R3, v[2], c[178], R2;
FMUL32   R2, R1, c[365];
FMAD     R6, v[3], c[179], R3;
FMAD     R3, R0, c[364], R2;
FADD32   R2, R6, -c[362];
FMAD     R3, R2, c[366], R3;
FMUL32   R3, R3, c[3];
FMAD     R0, -R3, c[364], R0;
FMAD     R1, -R3, c[365], R1;
FMAD     R4, -R3, c[366], R2;
FMUL32   R2, R1, R1;
FMAD     R2, R0, R0, R2;
FMAD     R2, R4, R4, R2;
RSQ      R5, |R2|;
FMUL32   R11, R1, R5;
FMUL32   R10, R0, R5;
FSET     R0, R11, c[0], LT;
FMUL32   R12, R10, R10;
FSET     R1, R10, c[0], LT;
F2I.FLOOR R0, R0;
FMUL32   R2, R11, R11;
FMUL32   R9, R4, R5;
I2I.M4   R4, R0;
F2I.FLOOR R1, R1;
FSET     R0, R9, c[0], LT;
R2A      A2, R4;
I2I.M4   R4, R1;
F2I.FLOOR R1, R0;
MOV32    R0, c[12];
R2A      A3, R4;
I2I.M4   R5, R1;
F2I.FLOOR R4, R0;
FMUL32   R0, R12, c[A3 + 85];
R2A      A1, R5;
I2I.M4   R13, R4;
FMAD     R4, R2, c[A2 + 93], R0;
FMUL32   R0, R9, R9;
R2A      A0, R13;
FMAD     R4, R0, c[A1 + 101], R4;
FADD32   R14, c[A0 + 8], -R7;
FADD32   R15, c[A0 + 9], -R8;
FADD32   R13, c[A0 + 10], -R6;
FMUL32   R16, R15, R15;
FMAD     R16, R14, R14, R16;
FMAD     R16, R13, R13, R16;
RSQ      R17, |R16|;
FMUL32   R15, R15, R17;
FMUL32   R14, R14, R17;
FMUL32   R13, R13, R17;
FMUL32   R11, R15, R11;
FMUL32   R15, R16, R17;
MOV32    R17, c[A0 + 16];
FMAD     R10, R14, R10, R11;
FMAD     R11, R15, c[A0 + 17], R17;
FMAD     R9, R13, R9, R10;
FMAD     R10, R16, c[A0 + 18], R11;
FMAX     R9, R9, c[0];
MVI      R11, -127.996;
RCP      R10, R10;
FMUL32   R9, R9, R10;
FMAX     R10, R11, c[4];
FMAD     R4, c[A0 + 1], R9, R4;
FMIN     R10, R10, c[0];
FMAX     R5, R5, c[0];
FMAX     R11, R4, c[0];
FMUL32   R4, R12, c[A3 + 84];
FSET     R5, R5, c[0], GT;
LG2      R11, R11;
FMAD     R2, R2, c[A2 + 92], R4;
FMUL32   R4, R10, R11;
FMAD     R0, R0, c[A1 + 100], R2;
RRO      R2, R4, 1;
FMAD     R0, c[A0], R9, R0;
EX2      R4, R2;
FMAX     R2, R0, c[0];
FMAX     R0, R0, c[0];
FCMP     R4, -R5, R4, c[0];
FSET     R2, R2, c[0], GT;
LG2      R0, R0;
FMUL32   R5, R12, c[A0 + 86];
FMUL32   R4, R4, c[7];
FMUL32   R0, R10, R0;
FMAD     R3, R3, c[A0 + 94], R5;
RRO      R0, R0, 1;
FMAD     R1, R1, c[A3 + 102], R3;
EX2      R0, R0;
FMAD     R1, c[A2 + 2], R9, R1;
FCMP     R0, -R2, R0, c[0];
FMAX     R2, R1, c[0];
FMAX     R1, R1, c[0];
FMUL32   R0, R0, c[7];
FSET     R3, R2, c[0], GT;
LG2      R1, R1;
FMAX     R2, R0, R4;
FMUL32   R1, R10, R1;
RRO      R1, R1, 1;
EX2      R1, R1;
FCMP     R3, -R3, R1, c[0];
FMUL32   R1, R8, c[377];
FMUL32   R5, R3, c[7];
FMAD     R3, R7, c[376], R1;
MOV32    R1, c[1];
FMAX     R2, R2, R5;
FMAD     R3, R6, c[378], R3;
FMAX     R9, R2, c[1];
FMUL32   R2, R8, c[381];
FMAD     o[14], R1, c[379], R3;
RCP      R9, R9;
FMAD     R2, R7, c[380], R2;
FMUL32   R3, R8, c[369];
FMUL32   o[5], R9, R4;
FMUL32   o[4], R9, R0;
FMUL32   o[6], R9, R5;
FMAD     R0, R6, c[382], R2;
FMAD     R2, R7, c[368], R3;
FMUL32   R3, R8, c[373];
FMAD     o[15], R1, c[383], R0;
FMAD     R0, R6, c[370], R2;
FMAD     R2, R7, c[372], R3;
FMUL32   R3, R8, c[41];
FMAD     o[12], R1, c[371], R0;
FMAD     R0, R6, c[374], R2;
FMAD     R3, R7, c[40], R3;
MOV32    R2, c[43];
FMAD     o[13], R1, c[375], R0;
FMAD     R0, R6, c[42], R3;
MOV32    R1, R2;
MOV32    R2, c[67];
FMUL32   R3, R8, c[33];
FMAD     R0, R1, c[1], R0;
MOV32    R1, c[35];
FMAD     R3, R7, c[32], R3;
FMAD     R2, -R0, R2, c[64];
FMAD     R3, R6, c[34], R3;
MOV32    o[16], R2;
MOV32    o[17], R2;
FMAD     o[0], R1, c[1], R3;
MOV32    o[18], R2;
MOV32    o[19], R2;
MOV32    o[2], R0;
FMUL32   R0, R8, c[37];
FMUL32   R2, R8, c[45];
MOV32    R1, c[39];
FMAD     R0, R7, c[36], R0;
FMAD     R2, R7, c[44], R2;
FMAD     R0, R6, c[38], R0;
FMAD     R3, R6, c[46], R2;
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
# 169 instructions, 20 R-regs
# 169 inst, (21 mov, 1 mvi, 0 tex, 10 complex, 137 math)
#    110 64-bit, 59 32-bit, 0 32-bit-const
