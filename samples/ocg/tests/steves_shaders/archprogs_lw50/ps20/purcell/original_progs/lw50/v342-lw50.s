!!SPA1.0
.CONST_MODE  PAGE
.THREAD_TYPE VERTEX
.MAX_REG     15
.MAX_IBUF    18
.MAX_OBUF    16
# parseasm build date Mar 10 2004 15:40:49
# -profile vp50 -po tbat3 -po lat3 -if vs2x -i allprogs-new32//v342-lw40.s -o allprogs-new32//v342-lw50.s
#vendor LWPU
#version parseasm.0.0
#profile vp50
#program fp30entry
#semantic C[14].C[14]
#semantic C[13].C[13]
#semantic C[12].C[12]
#semantic C[11].C[11]
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
#var float4 C[14] :  : c[14] : -1 : 0
#var float4 C[13] :  : c[13] : -1 : 0
#var float4 C[12] :  : c[12] : -1 : 0
#var float4 C[11] :  : c[11] : -1 : 0
#var float4 v[COL1] : $vin.F : F[0] : -1 : 0
#var float4 C[3] :  : c[3] : -1 : 0
#var float4 C[2] :  : c[2] : -1 : 0
#var float4 C[1] :  : c[1] : -1 : 0
#var float4 C[0] :  : c[0] : -1 : 0
#var float4 v[COL0] : $vin.F : F[0] : -1 : 0
#var float4 v[NRML] : $vin.F : F[0] : -1 : 0
#var float4 v[WGHT] : $vin.F : F[0] : -1 : 0
#var float4 v[OPOS] : $vin.F : F[0] : -1 : 0
#var float4 c :  : c[0] : -1 : 0
#var float4 v[FOGC] : $vin.F : F[0] : -1 : 0
#ibuf 0 = v[OPOS].x
#ibuf 1 = v[OPOS].y
#ibuf 2 = v[OPOS].z
#ibuf 3 = v[OPOS].w
#ibuf 4 = v[WGT].x
#ibuf 5 = v[WGT].y
#ibuf 6 = v[WGT].z
#ibuf 7 = v[NOR].x
#ibuf 8 = v[NOR].y
#ibuf 9 = v[NOR].z
#ibuf 10 = v[COL0].x
#ibuf 11 = v[COL0].y
#ibuf 12 = v[COL0].z
#ibuf 13 = v[COL1].x
#ibuf 14 = v[COL1].y
#ibuf 15 = v[FOG].x
#ibuf 16 = v[FOG].y
#ibuf 17 = v[FOG].z
#ibuf 18 = v[FOG].w
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
#obuf 11 = o[TEX2].x
#obuf 12 = o[TEX2].y
#obuf 13 = o[TEX2].z
#obuf 14 = o[TEX3].x
#obuf 15 = o[TEX3].y
#obuf 16 = o[TEX3].z
BB0:
F2I.FLOOR R0, v[18];
F2I.FLOOR R1, v[16];
I2I.M4   R0, R0;
I2I.M4   R1, R1;
R2A      A0, R0;
R2A      A1, R1;
FMUL     R2, v[5], c[A0 + 1];
FMUL     R0, v[5], c[A1 + 1];
FMUL     R1, v[5], c[A0 + 5];
FMAD     R2, v[4], c[A0], R2;
FMAD     R0, v[4], c[A1], R0;
FMAD     R1, v[4], c[A0 + 4], R1;
FMAD     R3, v[6], c[A0 + 2], R2;
FMAD     R0, v[6], c[A1 + 2], R0;
FMAD     R2, v[6], c[A0 + 6], R1;
FMUL     R1, v[5], c[A1 + 5];
FMUL     R4, v[15], R0;
FMUL     R0, v[5], c[A0 + 9];
FMAD     R1, v[4], c[A1 + 4], R1;
FMAD     R7, v[17], R3, R4;
FMAD     R0, v[4], c[A0 + 8], R0;
FMAD     R3, v[6], c[A1 + 6], R1;
FMUL     R1, v[5], c[A1 + 9];
FMAD     R0, v[6], c[A0 + 10], R0;
FMUL     R3, v[15], R3;
FMAD     R1, v[4], c[A1 + 8], R1;
FMAD     R6, v[17], R2, R3;
FMAD     R1, v[6], c[A1 + 10], R1;
FMUL32   R3, R6, R6;
FMUL     R2, v[15], R1;
FADD     R1, -v[0], c[48];
FMAD     R3, R7, R7, R3;
FMAD     R4, v[17], R0, R2;
FADD     R2, -v[1], c[49];
FADD     R0, -v[2], c[50];
FMAD     R5, R4, R4, R3;
FMUL32   R3, R2, R2;
RSQ      R5, |R5|;
FMAD     R3, R1, R1, R3;
FMUL32   R8, R7, R5;
FMUL32   R10, R6, R5;
FMUL32   R6, R4, R5;
FMAD     R3, R0, R0, R3;
FADD     R4, -v[0], c[44];
FADD     R5, -v[1], c[45];
RSQ      R7, |R3|;
FMUL32   R3, R5, R5;
FMUL32   R2, R2, R7;
FMUL32   R1, R1, R7;
FMUL32   R0, R0, R7;
FMAD     R9, R4, R4, R3;
FMUL32   R7, R2, R10;
FADD     R3, -v[2], c[46];
FMAD     R7, R1, R8, R7;
FMAD     R9, R3, R3, R9;
FMAD     R7, R0, R6, R7;
RSQ      R11, |R9|;
FMUL     R9, v[8], c[A0 + 1];
FMUL32   R5, R5, R11;
FMUL32   R4, R4, R11;
FMUL32   R3, R3, R11;
FMUL32   R11, R5, R10;
FMAD     R10, v[7], c[A0], R9;
FMUL     R9, v[8], c[A1 + 1];
FMAD     R8, R4, R8, R11;
FMAD     R10, v[9], c[A0 + 2], R10;
FMAD     R9, v[7], c[A1], R9;
FMAD     R6, R3, R6, R8;
FMUL     R8, v[8], c[A0 + 5];
FMAD     R9, v[9], c[A1 + 2], R9;
FADD32   R7, R7, R6;
FMAD     R8, v[7], c[A0 + 4], R8;
FMUL     R11, v[15], R9;
FMUL     R9, v[8], c[A1 + 5];
FMAD     R8, v[9], c[A0 + 6], R8;
FMAD     R11, v[17], R10, R11;
FMAD     R12, v[7], c[A1 + 4], R9;
FMUL     R9, v[8], c[A0 + 9];
FMUL     R10, v[8], c[A1 + 9];
FMAD     R12, v[9], c[A1 + 6], R12;
FMAD     R9, v[7], c[A0 + 8], R9;
FMAD     R10, v[7], c[A1 + 8], R10;
FMUL     R12, v[15], R12;
FMAD     R9, v[9], c[A0 + 10], R9;
FMAD     R10, v[9], c[A1 + 10], R10;
FMAD     R8, v[17], R8, R12;
FMUL     R12, v[15], R10;
FMUL32   R10, R8, R8;
FMAD     R9, v[17], R9, R12;
FMAD     R10, R11, R11, R10;
FMAD     R10, R9, R9, R10;
RSQ      R10, |R10|;
FMUL32   R11, R11, R10;
FMUL32   R8, R8, R10;
FMUL32   R10, R9, R10;
FMUL32   R9, R2, R8;
FMUL32   R12, R5, R8;
FMUL     R8, v[11], c[A0 + 1];
FMAD     R9, R1, R11, R9;
FMAD     R11, R4, R11, R12;
FMAD     R8, v[10], c[A0], R8;
FMAD     R9, R0, R10, R9;
FMAD     R13, R3, R10, R11;
FMAD     R10, v[12], c[A0 + 2], R8;
FMUL     R8, v[11], c[A1 + 1];
FADD32   R14, R9, R13;
FMUL     R9, v[11], c[A0 + 5];
FMAD     R8, v[10], c[A1], R8;
FMUL32   R11, R14, R14;
FMAD     R9, v[10], c[A0 + 4], R9;
FMAD     R8, v[12], c[A1 + 2], R8;
FMAD     R15, R7, R7, R11;
FMAD     R11, v[12], c[A0 + 6], R9;
FMUL     R12, v[15], R8;
FMUL     R9, v[11], c[A1 + 5];
FMUL     R8, v[11], c[A0 + 9];
FMAD     R10, v[17], R10, R12;
FMAD     R12, v[10], c[A1 + 4], R9;
FMAD     R8, v[10], c[A0 + 8], R8;
FMUL     R9, v[11], c[A1 + 9];
FMAD     R12, v[12], c[A1 + 6], R12;
FMAD     R8, v[12], c[A0 + 10], R8;
FMAD     R9, v[10], c[A1 + 8], R9;
FMUL     R12, v[15], R12;
FMAD     R9, v[12], c[A1 + 10], R9;
FMAD     R11, v[17], R11, R12;
FMUL     R9, v[15], R9;
FMUL32   R12, R11, R11;
FMAD     R8, v[17], R8, R9;
FMAD     R9, R10, R10, R12;
FMAD     R9, R8, R8, R9;
RSQ      R9, |R9|;
FMUL32   R10, R10, R9;
FMUL32   R11, R11, R9;
FMUL32   R8, R8, R9;
FMUL32   R2, R2, R11;
FMUL32   R5, R5, R11;
FMAD     R1, R1, R10, R2;
FMAD     R2, R4, R10, R5;
FMAD     R0, R0, R8, R1;
FMAD     R1, R3, R8, R2;
FADD32   R0, R0, R1;
FMAX     R2, R1, c[0];
FMAX     R3, R1, c[0];
FMAD     R4, R0, R0, R15;
FSET     R2, R2, c[0], GT;
LG2      R3, R3;
RSQ      R4, |R4|;
MVI      R5, -127.996;
FMUL32   o[14], R7, R4;
FMUL32   o[15], R14, R4;
FMUL32   o[16], R0, R4;
MOV32    R0, R5;
MOV32    R4, c[52];
FMAX     R0, R0, c[44];
MOV32    R5, c[53];
FMIN     R0, R0, c[0];
MOV32    R7, c[54];
FMUL32   R0, R0, R3;
MOV32    R3, R7;
MOV32    o[11], R6;
RRO      R0, R0, 1;
MOV32    o[12], R13;
MOV32    o[13], R1;
EX2      R0, R0;
FMUL     R1, v[1], c[A0 + 1];
FMUL     R6, v[1], c[A1 + 1];
FCMP     R0, -R2, R0, c[0];
FMAD     R1, v[0], c[A0], R1;
FMAD     R2, v[0], c[A1], R6;
FMAD     o[4], R0, c[56], R4;
FMAD     o[5], R0, c[57], R5;
FMAD     o[6], R0, c[58], R3;
FMAD     R1, v[2], c[A0 + 2], R1;
FMAD     R3, v[2], c[A1 + 2], R2;
FMUL     R0, v[1], c[A0 + 5];
FMAD     R2, v[3], c[A0 + 3], R1;
FMAD     R3, v[3], c[A1 + 3], R3;
FMAD     R1, v[0], c[A0 + 4], R0;
FMUL     R0, v[1], c[A1 + 5];
FMUL     R3, v[15], R3;
FMAD     R1, v[2], c[A0 + 6], R1;
FMAD     R0, v[0], c[A1 + 4], R0;
FMAD     R2, v[17], R2, R3;
FMAD     R3, v[3], c[A0 + 7], R1;
FMAD     R1, v[2], c[A1 + 6], R0;
FMUL     R0, v[1], c[A0 + 9];
FMAD     R4, v[3], c[A1 + 7], R1;
FMAD     R0, v[0], c[A0 + 8], R0;
FMUL     R1, v[1], c[A1 + 9];
FMUL     R4, v[15], R4;
FMAD     R0, v[2], c[A0 + 10], R0;
FMAD     R1, v[0], c[A1 + 8], R1;
FMAD     R3, v[17], R3, R4;
FMAD     R0, v[3], c[A0 + 11], R0;
FMAD     R1, v[2], c[A1 + 10], R1;
FMUL32   R4, R3, c[1];
FMUL32   R5, R3, c[5];
FMAD     R1, v[3], c[A1 + 11], R1;
FMAD     R4, R2, c[0], R4;
FMAD     R5, R2, c[4], R5;
FMUL     R1, v[15], R1;
FMUL32   R6, R3, c[9];
FMAD     R0, v[17], R0, R1;
FMAD     R1, R2, c[8], R6;
FMUL32   R3, R3, c[13];
FMAD     R4, R0, c[2], R4;
FMAD     R5, R0, c[6], R5;
FMAD     R2, R2, c[12], R3;
FMAD     o[0], v[3], c[3], R4;
FMAD     o[1], v[3], c[7], R5;
FMAD     R1, R0, c[10], R1;
FMAD     R0, R0, c[14], R2;
MOV      o[9], v[13];
FMAD     o[2], v[3], c[11], R1;
FMAD     o[3], v[3], c[15], R0;
MOV      o[10], v[14];
MOV      o[7], v[13];
MOV      o[8], v[14];
END
# 219 instructions, 16 R-regs
# 219 inst, (12 mov, 1 mvi, 0 tex, 8 complex, 198 math)
#    173 64-bit, 46 32-bit, 0 32-bit-const
