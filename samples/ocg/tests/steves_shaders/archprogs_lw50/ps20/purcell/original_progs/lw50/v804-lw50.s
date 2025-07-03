!!SPA1.0
.CONST_MODE  PAGE
.THREAD_TYPE VERTEX
.MAX_REG     15
.MAX_IBUF    12
.MAX_OBUF    15
# parseasm build date Mar 10 2004 15:40:49
# -profile vp50 -po tbat3 -po lat3 -if vs2x -i allprogs-new32//v804-lw40.s -o allprogs-new32//v804-lw50.s
#vendor LWPU
#version parseasm.0.0
#profile vp50
#program fp30entry
#semantic C[8].C[8]
#semantic C[10].C[10]
#semantic C[9].C[9]
#semantic C[14].C[14]
#semantic C[7].C[7]
#semantic C[6].C[6]
#semantic C[5].C[5]
#semantic C[4].C[4]
#semantic C[13].C[13]
#semantic C[3].C[3]
#semantic C[2].C[2]
#semantic C[1].C[1]
#semantic C[0].C[0]
#semantic c.c
#var float4 o[TEX3] : $vout.O : O[0] : -1 : 0
#var float4 o[TEX2] : $vout.O : O[0] : -1 : 0
#var float4 o[TEX1] : $vout.O : O[0] : -1 : 0
#var float4 o[TEX0] : $vout.O : O[0] : -1 : 0
#var float4 o[BCOL0] : $vout.O : O[0] : -1 : 0
#var float4 o[HPOS] : $vout.O : O[0] : -1 : 0
#var float4 C[8] :  : c[8] : -1 : 0
#var float4 C[10] :  : c[10] : -1 : 0
#var float4 C[9] :  : c[9] : -1 : 0
#var float4 C[14] :  : c[14] : -1 : 0
#var float4 C[7] :  : c[7] : -1 : 0
#var float4 C[6] :  : c[6] : -1 : 0
#var float4 C[5] :  : c[5] : -1 : 0
#var float4 C[4] :  : c[4] : -1 : 0
#var float4 C[13] :  : c[13] : -1 : 0
#var float4 v[COL1] : $vin.F : F[0] : -1 : 0
#var float4 C[3] :  : c[3] : -1 : 0
#var float4 C[2] :  : c[2] : -1 : 0
#var float4 C[1] :  : c[1] : -1 : 0
#var float4 C[0] :  : c[0] : -1 : 0
#var float4 v[COL0] : $vin.F : F[0] : -1 : 0
#var float4 v[OPOS] : $vin.F : F[0] : -1 : 0
#var float4 c :  : c[0] : -1 : 0
#var float4 v[FOGC] : $vin.F : F[0] : -1 : 0
#ibuf 0 = v[OPOS].x
#ibuf 1 = v[OPOS].y
#ibuf 2 = v[OPOS].z
#ibuf 3 = v[OPOS].w
#ibuf 4 = v[COL0].x
#ibuf 5 = v[COL0].y
#ibuf 6 = v[COL0].z
#ibuf 7 = v[COL1].x
#ibuf 8 = v[COL1].y
#ibuf 9 = v[FOG].x
#ibuf 10 = v[FOG].y
#ibuf 11 = v[FOG].z
#ibuf 12 = v[FOG].w
#obuf 0 = o[HPOS].x
#obuf 1 = o[HPOS].y
#obuf 2 = o[HPOS].z
#obuf 3 = o[HPOS].w
#obuf 4 = o[BCOL0].x
#obuf 5 = o[BCOL0].y
#obuf 6 = o[BCOL0].z
#obuf 7 = o[TEX0].x
#obuf 8 = o[TEX0].y
#obuf 9 = o[TEX1].x
#obuf 10 = o[TEX1].y
#obuf 11 = o[TEX1].z
#obuf 12 = o[TEX2].x
#obuf 13 = o[TEX3].x
#obuf 14 = o[TEX3].y
#obuf 15 = o[TEX3].z
BB0:
F2I.FLOOR R0, v[12];
F2I.FLOOR R1, v[10];
I2I.M4   R0, R0;
I2I.M4   R1, R1;
R2A      A0, R0;
R2A      A1, R1;
FMUL     R1, v[1], c[A0 + 1];
FMUL     R0, v[1], c[A1 + 1];
FMAD     R1, v[0], c[A0], R1;
FMAD     R2, v[0], c[A1], R0;
FMUL     R0, v[1], c[A0 + 5];
FMAD     R1, v[2], c[A0 + 2], R1;
FMAD     R2, v[2], c[A1 + 2], R2;
FMAD     R0, v[0], c[A0 + 4], R0;
FMAD     R1, v[3], c[A0 + 3], R1;
FMAD     R3, v[3], c[A1 + 3], R2;
FMAD     R2, v[2], c[A0 + 6], R0;
FMUL     R0, v[1], c[A1 + 5];
FMUL     R3, v[9], R3;
FMAD     R2, v[3], c[A0 + 7], R2;
FMAD     R0, v[0], c[A1 + 4], R0;
FMAD     R4, v[11], R1, R3;
FMAD     R1, v[2], c[A1 + 6], R0;
FADD32   R6, -R4, c[52];
FMUL     R0, v[1], c[A0 + 9];
FMAD     R1, v[3], c[A1 + 7], R1;
FMAD     R0, v[0], c[A0 + 8], R0;
FMUL     R3, v[9], R1;
FMUL     R1, v[1], c[A1 + 9];
FMAD     R0, v[2], c[A0 + 10], R0;
FMAD     R5, v[11], R2, R3;
FMAD     R1, v[0], c[A1 + 8], R1;
FMAD     R3, v[3], c[A0 + 11], R0;
FADD32   R0, -R5, c[53];
FMAD     R1, v[2], c[A1 + 10], R1;
FMUL32   R7, R0, R0;
FMAD     R2, v[3], c[A1 + 11], R1;
FMUL     R1, v[5], c[A0 + 1];
FMAD     R9, R6, R6, R7;
FMUL     R7, v[9], R2;
FMAD     R1, v[4], c[A0], R1;
FMUL     R2, v[5], c[A1 + 1];
FMAD     R3, v[11], R3, R7;
FMAD     R1, v[6], c[A0 + 2], R1;
FMAD     R8, v[4], c[A1], R2;
FADD32   R7, -R3, c[54];
FMUL     R2, v[5], c[A0 + 5];
FMAD     R8, v[6], c[A1 + 2], R8;
FMAD     R9, R7, R7, R9;
FMAD     R2, v[4], c[A0 + 4], R2;
FMUL     R8, v[9], R8;
RSQ      R10, |R9|;
FMAD     R2, v[6], c[A0 + 6], R2;
FMAD     R1, v[11], R1, R8;
FMUL32   R9, R0, R10;
FMUL     R8, v[5], c[A1 + 5];
FMUL     R0, v[5], c[A0 + 9];
FMAD     R11, v[4], c[A1 + 4], R8;
FMAD     R0, v[4], c[A0 + 8], R0;
FMUL     R8, v[5], c[A1 + 9];
FMAD     R11, v[6], c[A1 + 6], R11;
FMAD     R0, v[6], c[A0 + 10], R0;
FMAD     R8, v[4], c[A1 + 8], R8;
FMUL     R11, v[9], R11;
FMAD     R8, v[6], c[A1 + 10], R8;
FMAD     R2, v[11], R2, R11;
FMUL     R11, v[9], R8;
FMUL32   R8, R6, R10;
FMUL32   R6, R2, R2;
FMAD     R0, v[11], R0, R11;
FMUL32   R7, R7, R10;
FMAD     R6, R1, R1, R6;
FMAD     R6, R0, R0, R6;
RSQ      R6, |R6|;
FMUL32   R2, R2, R6;
FMUL32   R1, R1, R6;
FMUL32   R0, R0, R6;
FMUL32   R6, R9, R2;
FMAD     R6, R8, R1, R6;
FMAD     R6, R7, R0, R6;
FADD32   R10, R6, R6;
FMAD     R8, R10, R1, -R8;
FMAD     R9, R10, R2, -R9;
FMAD     R7, R10, R0, -R7;
FMUL32   R10, R9, c[17];
FMUL32   R11, R9, c[21];
FMUL32   R9, R9, c[25];
FMAD     R10, R8, c[16], R10;
FMAD     R11, R8, c[20], R11;
FMAD     R8, R8, c[24], R9;
FMAD     o[9], R7, c[18], R10;
FMAD     o[10], R7, c[22], R11;
FMAD     o[11], R7, c[26], R8;
FADD32   R8, -R4, c[28];
FADD32   R9, -R5, c[29];
FADD32   R7, -R3, c[30];
FMUL32   R10, R9, R9;
FMAD     R10, R8, R8, R10;
FMAD     R10, R7, R7, R10;
RSQ      R10, |R10|;
FMUL32   R9, R9, R10;
FMUL32   R8, R8, R10;
FMUL32   R7, R7, R10;
FMUL32   R9, R9, R2;
MVI      R10, -127.996;
FMAD     R8, R8, R1, R9;
MOV32    R9, R10;
FMAD     R10, R7, R0, R8;
FMAX     R8, R9, c[36];
FMAX     R7, R10, c[0];
FMIN     R8, R8, c[0];
FMAX     R11, R10, c[0];
FSET     R7, R7, c[0], GT;
MOV32    R9, c[36];
LG2      R11, R11;
FADD32   R10, R10, c[56];
FMUL32   R8, R8, R11;
FMUL32   R10, R10, c[58];
MOV32    R11, c[37];
RRO      R8, R8, 1;
FMAX     R10, R10, c[31];
EX2      R8, R8;
MOV32    R12, c[38];
FCMP     R7, -R7, R8, c[0];
MOV32    R8, R12;
MOV32    o[12], R6;
FMAD     R6, R7, c[40], R9;
FMAD     R9, R7, c[41], R11;
FMAD     R7, R7, c[42], R8;
FMAD     o[4], R6, R10, c[32];
FMAD     o[5], R9, R10, c[33];
FMAD     o[6], R7, R10, c[34];
FMUL32   R6, R2, c[17];
FMUL32   R7, R2, c[21];
FMUL32   R2, R2, c[25];
FMAD     R6, R1, c[16], R6;
FMAD     R7, R1, c[20], R7;
FMAD     R1, R1, c[24], R2;
FMAD     o[13], R0, c[18], R6;
FMAD     o[14], R0, c[22], R7;
FMAD     o[15], R0, c[26], R1;
FMUL32   R0, R5, c[1];
FMUL32   R1, R5, c[5];
FMUL32   R2, R5, c[9];
FMAD     R0, R4, c[0], R0;
FMAD     R1, R4, c[4], R1;
FMAD     R2, R4, c[8], R2;
FMAD     R0, R3, c[2], R0;
FMAD     R1, R3, c[6], R1;
FMAD     R2, R3, c[10], R2;
FMAD     o[0], v[3], c[3], R0;
FMAD     o[1], v[3], c[7], R1;
FMAD     o[2], v[3], c[11], R2;
FMUL32   R0, R5, c[13];
MOV      o[7], v[7];
MOV      o[8], v[8];
FMAD     R0, R4, c[12], R0;
FMAD     R0, R3, c[14], R0;
FMAD     o[3], v[3], c[15], R0;
END
# 159 instructions, 16 R-regs
# 159 inst, (8 mov, 1 mvi, 0 tex, 5 complex, 145 math)
#    119 64-bit, 40 32-bit, 0 32-bit-const
