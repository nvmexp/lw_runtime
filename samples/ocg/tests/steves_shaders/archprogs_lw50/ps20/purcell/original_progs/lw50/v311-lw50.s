!!SPA1.0
.CONST_MODE  PAGE
.THREAD_TYPE VERTEX
.MAX_REG     19
.MAX_IBUF    18
.MAX_OBUF    16
# parseasm build date Mar 10 2004 15:40:49
# -profile vp50 -po tbat3 -po lat3 -if vs2x -i allprogs-new32//v311-lw40.s -o allprogs-new32//v311-lw50.s
#vendor LWPU
#version parseasm.0.0
#profile vp50
#program fp30entry
#semantic C[12].C[12]
#semantic C[9].C[9]
#semantic C[11].C[11]
#semantic C[13].C[13]
#semantic C[7].C[7]
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
#var float4 C[12] :  : c[12] : -1 : 0
#var float4 C[9] :  : c[9] : -1 : 0
#var float4 C[11] :  : c[11] : -1 : 0
#var float4 C[13] :  : c[13] : -1 : 0
#var float4 C[7] :  : c[7] : -1 : 0
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
FMUL     R3, v[15], R3;
FMAD     R2, v[3], c[A0 + 7], R2;
FMAD     R0, v[0], c[A1 + 4], R0;
FMAD     R1, v[17], R1, R3;
FMAD     R3, v[2], c[A1 + 6], R0;
FADD32   R4, -R1, c[28];
FMUL     R0, v[1], c[A0 + 9];
FMAD     R3, v[3], c[A1 + 7], R3;
FMAD     R0, v[0], c[A0 + 8], R0;
FMUL     R5, v[15], R3;
FMUL     R3, v[1], c[A1 + 9];
FMAD     R0, v[2], c[A0 + 10], R0;
FMAD     R2, v[17], R2, R5;
FMAD     R3, v[0], c[A1 + 8], R3;
FMAD     R0, v[3], c[A0 + 11], R0;
FADD32   R5, -R2, c[29];
FMAD     R3, v[2], c[A1 + 10], R3;
FMUL32   R7, R5, R5;
FMAD     R6, v[3], c[A1 + 11], R3;
FMUL     R3, v[5], c[A0 + 1];
FMAD     R8, R4, R4, R7;
FMUL     R7, v[15], R6;
FMAD     R6, v[4], c[A0], R3;
FMUL     R3, v[5], c[A1 + 1];
FMAD     R0, v[17], R0, R7;
FMAD     R6, v[6], c[A0 + 2], R6;
FMAD     R7, v[4], c[A1], R3;
FADD32   R3, -R0, c[30];
FMUL     R9, v[5], c[A0 + 5];
FMAD     R7, v[6], c[A1 + 2], R7;
FMAD     R8, R3, R3, R8;
FMAD     R9, v[4], c[A0 + 4], R9;
FMUL     R7, v[15], R7;
RSQ      R8, |R8|;
FMAD     R9, v[6], c[A0 + 6], R9;
FMAD     R11, v[17], R6, R7;
FMUL32   R5, R5, R8;
FMUL     R7, v[5], c[A1 + 5];
FMUL     R6, v[5], c[A0 + 9];
FMAD     R10, v[4], c[A1 + 4], R7;
FMAD     R7, v[4], c[A0 + 8], R6;
FMUL     R6, v[5], c[A1 + 9];
FMAD     R10, v[6], c[A1 + 6], R10;
FMAD     R7, v[6], c[A0 + 10], R7;
FMAD     R6, v[4], c[A1 + 8], R6;
FMUL     R10, v[15], R10;
FMAD     R6, v[6], c[A1 + 10], R6;
FMAD     R12, v[17], R9, R10;
FMUL     R9, v[15], R6;
FMUL32   R4, R4, R8;
FMUL32   R6, R12, R12;
FMAD     R7, v[17], R7, R9;
FMUL32   R3, R3, R8;
FMAD     R6, R11, R11, R6;
FMAD     R6, R7, R7, R6;
FADD32   R8, -R1, c[52];
FADD32   R9, -R2, c[53];
RSQ      R10, |R6|;
FMUL32   R6, R9, R9;
FMUL32   R12, R12, R10;
FMUL32   R11, R11, R10;
FMUL32   R10, R7, R10;
FMAD     R13, R8, R8, R6;
FMUL32   R6, R5, R12;
FADD32   R7, -R0, c[54];
FMAD     R6, R4, R11, R6;
FMAD     R13, R7, R7, R13;
FMAD     R6, R3, R10, R6;
RSQ      R14, |R13|;
FMUL     R13, v[8], c[A0 + 1];
FMUL32   R9, R9, R14;
FMUL32   R8, R8, R14;
FMUL32   R7, R7, R14;
FMUL32   R12, R9, R12;
FMAD     R13, v[7], c[A0], R13;
FMUL     R14, v[8], c[A1 + 1];
FMAD     R11, R8, R11, R12;
FMAD     R12, v[9], c[A0 + 2], R13;
FMAD     R13, v[7], c[A1], R14;
FMAD     R10, R7, R10, R11;
FMUL     R11, v[8], c[A0 + 5];
FMAD     R13, v[9], c[A1 + 2], R13;
FADD32   R10, R6, R10;
FMAD     R11, v[7], c[A0 + 4], R11;
FMUL     R13, v[15], R13;
FMUL     R14, v[8], c[A1 + 5];
FMAD     R11, v[9], c[A0 + 6], R11;
FMAD     R13, v[17], R12, R13;
FMAD     R15, v[7], c[A1 + 4], R14;
FMUL     R12, v[8], c[A0 + 9];
FMUL     R14, v[8], c[A1 + 9];
FMAD     R15, v[9], c[A1 + 6], R15;
FMAD     R12, v[7], c[A0 + 8], R12;
FMAD     R14, v[7], c[A1 + 8], R14;
FMUL     R15, v[15], R15;
FMAD     R12, v[9], c[A0 + 10], R12;
FMAD     R14, v[9], c[A1 + 10], R14;
FMAD     R11, v[17], R11, R15;
FMUL     R15, v[15], R14;
FMUL32   R14, R11, R11;
FMAD     R12, v[17], R12, R15;
FMAD     R14, R13, R13, R14;
FMAD     R14, R12, R12, R14;
RSQ      R14, |R14|;
FMUL32   R11, R11, R14;
FMUL32   R13, R13, R14;
FMUL32   R12, R12, R14;
FMUL32   R14, R5, R11;
FMUL32   R15, R9, R11;
FMUL     R11, v[11], c[A0 + 1];
FMAD     R14, R4, R13, R14;
FMAD     R13, R8, R13, R15;
FMAD     R11, v[10], c[A0], R11;
FMAD     R16, R3, R12, R14;
FMAD     R12, R7, R12, R13;
FMAD     R13, v[12], c[A0 + 2], R11;
FMUL     R11, v[11], c[A1 + 1];
FADD32   R17, R16, R12;
FMUL     R12, v[11], c[A0 + 5];
FMAD     R11, v[10], c[A1], R11;
FMUL32   R14, R17, R17;
FMAD     R12, v[10], c[A0 + 4], R12;
FMAD     R11, v[12], c[A1 + 2], R11;
FMAD     R18, R10, R10, R14;
FMAD     R14, v[12], c[A0 + 6], R12;
FMUL     R15, v[15], R11;
FMUL     R12, v[11], c[A1 + 5];
FMUL     R11, v[11], c[A0 + 9];
FMAD     R13, v[17], R13, R15;
FMAD     R15, v[10], c[A1 + 4], R12;
FMAD     R11, v[10], c[A0 + 8], R11;
FMUL     R12, v[11], c[A1 + 9];
FMAD     R15, v[12], c[A1 + 6], R15;
FMAD     R11, v[12], c[A0 + 10], R11;
FMAD     R12, v[10], c[A1 + 8], R12;
FMUL     R15, v[15], R15;
FMAD     R12, v[12], c[A1 + 10], R12;
FMAD     R14, v[17], R14, R15;
FMUL     R12, v[15], R12;
FMUL32   R15, R14, R14;
FMAD     R11, v[17], R11, R12;
FMAD     R12, R13, R13, R15;
FMAD     R12, R11, R11, R12;
RSQ      R12, |R12|;
FMUL32   R14, R14, R12;
FMUL32   R13, R13, R12;
FMUL32   R11, R11, R12;
FMUL32   R12, R9, R14;
FMUL32   R9, R5, R14;
FMAD     R8, R8, R13, R12;
FMAD     R9, R4, R13, R9;
FMAD     R7, R7, R11, R8;
FMAD     R8, R3, R11, R9;
FADD32   R7, R8, R7;
FMAD     R9, R7, R7, R18;
MVI      R12, -127.996;
RSQ      R9, |R9|;
FMUL32   o[14], R10, R9;
FMUL32   o[15], R17, R9;
FMUL32   R7, R7, R9;
FMAX     R10, R12, c[44];
FMAX     R9, R8, c[0];
MOV32    o[16], R7;
FMAX     R7, R7, c[0];
FMUL32   R5, R14, R5;
FMIN     R10, R10, c[0];
LG2      R7, R7;
FMAD     R4, R13, R4, R5;
FSET     R5, R9, c[0], GT;
FMUL32   R7, R10, R7;
FMAD     R3, R11, R3, R4;
FMAX     R4, R12, c[36];
RRO      R9, R7, 1;
FMAX     R7, R3, c[0];
FMAX     R3, R3, c[0];
EX2      R9, R9;
FSET     R7, R7, c[0], GT;
LG2      R3, R3;
FCMP     R5, -R5, R9, c[0];
FMIN     R4, R4, c[0];
MOV32    R9, c[44];
MOV32    R10, c[45];
FMUL32   R3, R4, R3;
MOV32    R4, R9;
MOV32    R9, R10;
RRO      R3, R3, 1;
MOV32    R10, c[46];
EX2      R3, R3;
MOV32    o[11], R6;
FCMP     R3, -R7, R3, c[0];
MOV32    o[12], R16;
MOV32    o[13], R8;
FMAD     R4, R3, c[48], R4;
FMAD     R6, R3, c[49], R9;
FMAD     R3, R3, c[50], R10;
FMUL32   o[4], R4, R5;
FMUL32   o[5], R6, R5;
FMUL32   o[6], R3, R5;
FMUL32   R3, R2, c[1];
FMUL32   R4, R2, c[5];
FMUL32   R5, R2, c[9];
FMAD     R3, R1, c[0], R3;
FMAD     R4, R1, c[4], R4;
FMAD     R5, R1, c[8], R5;
FMAD     R3, R0, c[2], R3;
FMAD     R4, R0, c[6], R4;
FMAD     R5, R0, c[10], R5;
FMAD     o[0], v[3], c[3], R3;
FMAD     o[1], v[3], c[7], R4;
FMAD     o[2], v[3], c[11], R5;
FMUL32   R2, R2, c[13];
MOV      o[9], v[13];
MOV      o[10], v[14];
FMAD     R1, R1, c[12], R2;
MOV      o[7], v[13];
MOV      o[8], v[14];
FMAD     R0, R0, c[14], R1;
FMAD     o[3], v[3], c[15], R0;
END
# 236 instructions, 20 R-regs
# 236 inst, (13 mov, 1 mvi, 0 tex, 10 complex, 212 math)
#    178 64-bit, 58 32-bit, 0 32-bit-const
