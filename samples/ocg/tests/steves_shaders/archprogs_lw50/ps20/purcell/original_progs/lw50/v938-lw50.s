!!SPA1.0
.CONST_MODE  PAGE
.THREAD_TYPE VERTEX
.MAX_REG     19
.MAX_IBUF    11
.MAX_OBUF    15
# parseasm build date Mar 10 2004 15:40:49
# -profile vp50 -po tbat3 -po lat3 -if vs2x -i allprogs-new32//v938-lw40.s -o allprogs-new32//v938-lw50.s
#vendor LWPU
#version parseasm.0.0
#profile vp50
#program fp30entry
#semantic C[1].C[1]
#semantic C[90].C[90]
#semantic C[16].C[16]
#semantic C[11].C[11]
#semantic C[10].C[10]
#semantic C[9].C[9]
#semantic C[8].C[8]
#semantic C[0].C[0]
#semantic c.c
#semantic C[3].C[3]
#var float4 o[TEX0] : $vout.O : O[0] : -1 : 0
#var float4 o[FOGC] : $vout.O : O[0] : -1 : 0
#var float4 o[BCOL0] : $vout.O : O[0] : -1 : 0
#var float4 o[HPOS] : $vout.O : O[0] : -1 : 0
#var float4 v[TEX9] : $vin.F : F[0] : -1 : 0
#var float4 C[1] :  : c[1] : -1 : 0
#var float4 C[90] :  : c[90] : -1 : 0
#var float4 C[16] :  : c[16] : -1 : 0
#var float4 C[11] :  : c[11] : -1 : 0
#var float4 C[10] :  : c[10] : -1 : 0
#var float4 C[9] :  : c[9] : -1 : 0
#var float4 C[8] :  : c[8] : -1 : 0
#var float4 v[COL0] : $vin.F : F[0] : -1 : 0
#var float4 C[0] :  : c[0] : -1 : 0
#var float4 c :  : c[0] : -1 : 0
#var float4 v[OPOS] : $vin.F : F[0] : -1 : 0
#var float4 C[3] :  : c[3] : -1 : 0
#var float4 v[NRML] : $vin.F : F[0] : -1 : 0
#ibuf 0 = v[OPOS].x
#ibuf 1 = v[OPOS].y
#ibuf 2 = v[OPOS].z
#ibuf 3 = v[OPOS].w
#ibuf 4 = v[NOR].z
#ibuf 5 = v[COL0].x
#ibuf 6 = v[COL0].y
#ibuf 7 = v[COL0].z
#ibuf 8 = v[UNUSED1].x
#ibuf 9 = v[UNUSED1].y
#ibuf 10 = v[UNUSED1].z
#ibuf 11 = v[UNUSED1].w
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
#obuf 12 = o[FOGC].x
#obuf 13 = o[FOGC].y
#obuf 14 = o[FOGC].z
#obuf 15 = o[FOGC].w
BB0:
MOV32    R0, c[14];
FMAD     R0, v[4], R0, c[15];
MOV32    R1, c[12];
F2I.FLOOR R0, R0;
F2I.FLOOR R1, R1;
I2I.M4   R0, R0;
I2I.M4   R1, R1;
R2A      A1, R0;
R2A      A0, R1;
FMUL     R2, v[1], c[A1 + 5];
FMUL     R0, v[1], c[A1 + 1];
FMAD     R2, v[0], c[A1 + 4], R2;
FMAD     R0, v[0], c[A1], R0;
FMAD     R4, v[2], c[A1 + 6], R2;
FMAD     R2, v[2], c[A1 + 2], R0;
FMUL     R0, v[1], c[A1 + 9];
FMAD     R14, v[3], c[A1 + 7], R4;
FMAD     R12, v[3], c[A1 + 3], R2;
FMAD     R0, v[0], c[A1 + 8], R0;
FADD32   R8, c[A0 + 9], -R14;
FADD32   R6, c[A0 + 8], -R12;
FMAD     R0, v[2], c[A1 + 10], R0;
FMUL32   R2, R8, R8;
FMUL     R3, v[6], c[A1 + 5];
FMAD     R10, v[3], c[A1 + 11], R0;
FMAD     R2, R6, R6, R2;
FMAD     R3, v[5], c[A1 + 4], R3;
FADD32   R4, c[A0 + 10], -R10;
FMUL     R0, v[6], c[A1 + 1];
FMAD     R11, v[7], c[A1 + 6], R3;
FMAD     R13, R4, R4, R2;
FMAD     R2, v[5], c[A1], R0;
FMUL     R0, v[6], c[A1 + 9];
RSQ      R15, |R13|;
FMAD     R2, v[7], c[A1 + 2], R2;
FMAD     R0, v[5], c[A1 + 8], R0;
FMUL32   R6, R6, R15;
FMUL32   R8, R8, R15;
FMAD     R0, v[7], c[A1 + 10], R0;
FMUL32   R4, R4, R15;
FMUL32   R16, R11, R8;
FMUL32   R18, c[A0 + 5], -R8;
FMAD     R8, R2, R6, R16;
FMAD     R3, c[A0 + 4], -R6, R18;
FMAD     R6, R0, R4, R8;
FMAD     R3, c[A0 + 6], -R4, R3;
MVI      R1, -127.996;
FMAX     R4, R6, c[0];
FADD32   R6, R3, -c[A0 + 14];
FSET     R3, R4, c[0], GT;
FMUL32   R6, R6, c[A0 + 15];
FMAX     R4, c[A0 + 12], R1;
FMAX     R6, R6, c[0];
FMIN     R4, R4, c[0];
FMUL32   R7, R13, R15;
LG2      R8, R6;
MOV32    R6, c[A0 + 16];
FMUL32   R4, R4, R8;
FMAD     R6, R7, c[A0 + 17], R6;
RRO      R4, R4, 1;
FMAD     R7, R13, c[A0 + 18], R6;
FMUL32   R13, R2, R2;
EX2      R6, R4;
RCP      R8, R7;
FSET     R4, R2, c[0], LT;
FCMP     R3, -R3, R6, c[0];
FMUL32   R6, c[A0], R8;
F2I.FLOOR R4, R4;
FMIN     R7, R3, c[1];
FSET     R3, R11, c[0], LT;
I2I.M4   R4, R4;
FMUL32   R9, R5, R7;
F2I.FLOOR R5, R3;
R2A      A3, R4;
FSET     R3, R0, c[0], LT;
I2I.M4   R4, R5;
FMUL32   R7, R13, c[A3 + 84];
F2I.FLOOR R3, R3;
R2A      A2, R4;
FMUL32   R4, R11, R11;
I2I.M4   R3, R3;
FMUL32   R11, R11, c[361];
FMAD     R7, R4, c[A2 + 92], R7;
R2A      A1, R3;
FMAD     R11, R2, c[360], R11;
FMUL32   R2, R0, R0;
FMAD     R0, R0, c[362], R11;
FMAD     R7, R2, c[A1 + 100], R7;
FMAX     R11, R0, c[0];
FMAD     R0, R6, R9, R7;
FMAX     R1, R1, c[4];
FMUL32   R0, R0, c[363];
FMIN     R16, R1, c[0];
FMUL32   R0, R11, R0;
FMAX     R7, R0, c[0];
FMAX     R6, R0, c[0];
FMUL32   R0, c[A0 + 1], R8;
FSET     R15, R7, c[0], GT;
LG2      R17, R6;
FMUL32   R6, R13, c[A3 + 85];
FMUL32   R17, R16, R17;
FMAD     R6, R4, c[A2 + 93], R6;
RRO      R4, R17, 1;
FMAD     R6, R2, c[A1 + 101], R6;
EX2      R2, R4;
FMAD     R0, R0, R9, R6;
FCMP     R2, -R15, R2, c[0];
FMUL32   R0, R0, c[363];
FMUL32   o[4], R2, c[7];
FMUL32   R0, R11, R0;
FMAX     R1, R1, c[0];
FMAX     R0, R0, c[0];
FMUL32   R2, c[A2 + 2], R8;
FMUL32   R4, R13, c[A0 + 86];
FSET     R1, R1, c[0], GT;
LG2      R0, R0;
FMAD     R4, R5, c[A0 + 94], R4;
FMUL32   R0, R16, R0;
FMAD     R3, R3, c[A3 + 102], R4;
RRO      R0, R0, 1;
FMAD     R2, R2, R9, R3;
EX2      R0, R0;
FMUL32   R2, R2, c[363];
FCMP     R0, -R1, R0, c[0];
FMUL32   R1, R11, R2;
FMUL32   o[5], R0, c[7];
FMAX     R2, R1, c[0];
FMAX     R1, R1, c[0];
FMUL32   R0, R14, c[41];
FSET     R2, R2, c[0], GT;
LG2      R3, R1;
FMAD     R1, R12, c[40], R0;
MOV32    R0, c[43];
FMUL32   R3, R16, R3;
FMAD     R1, R10, c[42], R1;
RRO      R4, R3, 1;
MOV32    R3, c[67];
FMAD     R0, R0, c[1], R1;
EX2      R1, R4;
FMUL32   R4, R14, c[33];
FMAD     R3, -R0, R3, c[64];
FCMP     R1, -R2, R1, c[0];
FMAD     R2, R12, c[32], R4;
MOV32    o[12], R3;
FMUL32   o[6], R1, c[7];
FMAD     R1, R10, c[34], R2;
MOV32    o[13], R3;
MOV32    o[14], R3;
MOV32    o[15], R3;
MOV32    R2, c[35];
FMUL32   R3, R14, c[37];
MOV32    o[2], R0;
MOV32    R0, R2;
FMAD     R2, R12, c[36], R3;
FMUL32   R3, R14, c[45];
FMAD     o[0], R0, c[1], R1;
FMAD     R0, R10, c[38], R2;
FMAD     R3, R12, c[44], R3;
MOV32    R1, c[39];
MOV32    R2, c[47];
FMAD     R3, R10, c[46], R3;
MOV      o[8], v[8];
FMAD     o[1], R1, c[1], R0;
FMAD     o[3], R2, c[1], R3;
MOV      o[9], v[9];
MOV      o[10], v[10];
MOV      o[11], v[11];
MOV32    R0, c[1];
MOV32    o[7], R0;
END
# 169 instructions, 20 R-regs
# 169 inst, (20 mov, 1 mvi, 0 tex, 10 complex, 138 math)
#    113 64-bit, 56 32-bit, 0 32-bit-const
