!!SPA1.0
.CONST_MODE  PAGE
.THREAD_TYPE VERTEX
.MAX_REG     11
.MAX_IBUF    14
.MAX_OBUF    22
# parseasm build date Mar 10 2004 15:40:49
# -profile vp50 -po tbat3 -po lat3 -if vs2x -i allprogs-new32//v955-lw40.s -o allprogs-new32//v955-lw50.s
#vendor LWPU
#version parseasm.0.0
#profile vp50
#program fp30entry
#semantic C[10].C[10]
#semantic C[9].C[9]
#semantic C[8].C[8]
#semantic C[11].C[11]
#semantic C[17].C[17]
#semantic C[16].C[16]
#semantic C[14].C[14]
#semantic C[13].C[13]
#semantic C[12].C[12]
#semantic C[15].C[15]
#semantic C[3].C[3]
#semantic C[2].C[2]
#semantic C[1].C[1]
#semantic C[0].C[0]
#semantic C[7].C[7]
#semantic C[6].C[6]
#semantic C[5].C[5]
#semantic C[4].C[4]
#var float4 o[TEX5] : $vout.O : O[0] : -1 : 0
#var float4 o[TEX4] : $vout.O : O[0] : -1 : 0
#var float4 o[TEX3] : $vout.O : O[0] : -1 : 0
#var float4 o[TEX2] : $vout.O : O[0] : -1 : 0
#var float4 o[TEX1] : $vout.O : O[0] : -1 : 0
#var float4 o[TEX0] : $vout.O : O[0] : -1 : 0
#var float4 o[HPOS] : $vout.O : O[0] : -1 : 0
#var float4 v[WGHT] : $vin.F : F[0] : -1 : 0
#var float4 C[10] :  : c[10] : -1 : 0
#var float4 C[9] :  : c[9] : -1 : 0
#var float4 C[8] :  : c[8] : -1 : 0
#var float4 C[11] :  : c[11] : -1 : 0
#var float4 C[17] :  : c[17] : -1 : 0
#var float4 C[16] :  : c[16] : -1 : 0
#var float4 v[COL1] : $vin.F : F[0] : -1 : 0
#var float4 v[COL0] : $vin.F : F[0] : -1 : 0
#var float4 v[NRML] : $vin.F : F[0] : -1 : 0
#var float4 C[14] :  : c[14] : -1 : 0
#var float4 C[13] :  : c[13] : -1 : 0
#var float4 C[12] :  : c[12] : -1 : 0
#var float4 C[15] :  : c[15] : -1 : 0
#var float4 C[3] :  : c[3] : -1 : 0
#var float4 C[2] :  : c[2] : -1 : 0
#var float4 C[1] :  : c[1] : -1 : 0
#var float4 C[0] :  : c[0] : -1 : 0
#var float4 C[7] :  : c[7] : -1 : 0
#var float4 C[6] :  : c[6] : -1 : 0
#var float4 C[5] :  : c[5] : -1 : 0
#var float4 C[4] :  : c[4] : -1 : 0
#var float4 v[OPOS] : $vin.F : F[0] : -1 : 0
#ibuf 0 = v[OPOS].x
#ibuf 1 = v[OPOS].y
#ibuf 2 = v[OPOS].z
#ibuf 3 = v[OPOS].w
#ibuf 4 = v[WGT].x
#ibuf 5 = v[WGT].y
#ibuf 6 = v[NOR].x
#ibuf 7 = v[NOR].y
#ibuf 8 = v[NOR].z
#ibuf 9 = v[COL0].x
#ibuf 10 = v[COL0].y
#ibuf 11 = v[COL0].z
#ibuf 12 = v[COL1].x
#ibuf 13 = v[COL1].y
#ibuf 14 = v[COL1].z
#obuf 0 = o[HPOS].x
#obuf 1 = o[HPOS].y
#obuf 2 = o[HPOS].z
#obuf 3 = o[HPOS].w
#obuf 4 = o[TEX0].x
#obuf 5 = o[TEX0].y
#obuf 6 = o[TEX1].x
#obuf 7 = o[TEX1].y
#obuf 8 = o[TEX1].z
#obuf 9 = o[TEX1].w
#obuf 10 = o[TEX2].x
#obuf 11 = o[TEX2].y
#obuf 12 = o[TEX2].z
#obuf 13 = o[TEX2].w
#obuf 14 = o[TEX3].x
#obuf 15 = o[TEX3].y
#obuf 16 = o[TEX3].z
#obuf 17 = o[TEX4].x
#obuf 18 = o[TEX4].y
#obuf 19 = o[TEX4].z
#obuf 20 = o[TEX5].x
#obuf 21 = o[TEX5].y
#obuf 22 = o[TEX5].z
BB0:
FMUL     R0, v[0], c[1];
MOV32    R1, -c[63];
FMUL     R3, v[0], c[0];
FMAD     R2, v[1], c[5], R0;
MOV32    R0, -c[63];
FMAD     R3, v[1], c[4], R3;
FMAD     R4, v[2], c[9], R2;
FMUL     R2, v[0], c[2];
FMAD     R3, v[2], c[8], R3;
FMAD     R7, v[3], c[13], R4;
FMAD     R2, v[1], c[6], R2;
FMAD     R8, v[3], c[12], R3;
FMAD     R1, R7, R1, c[61];
FMAD     R3, v[2], c[10], R2;
FMAD     R2, R8, R0, c[60];
MOV32    R0, -c[63];
FMAD     R6, v[3], c[14], R3;
FMUL32   R3, R2, c[48];
FMUL32   R4, R2, c[49];
FMAD     R0, R6, R0, c[62];
FMAD     R3, R1, c[52], R3;
FMAD     R4, R1, c[53], R4;
FMUL32   R2, R2, c[50];
FMAD     R3, R0, c[56], R3;
FMAD     R4, R0, c[57], R4;
FMAD     R1, R1, c[54], R2;
FMUL     R2, v[7], R4;
FMAD     R0, R0, c[58], R1;
FMUL     R1, v[10], R4;
FMAD     R2, v[6], R3, R2;
FMUL     R4, v[13], R4;
FMAD     R1, v[9], R3, R1;
FMAD     o[14], v[8], R0, R2;
FMAD     R3, v[12], R3, R4;
FMAD     o[15], v[11], R0, R1;
MOV32    R2, c[52];
FMAD     o[16], v[14], R0, R3;
MOV32    R1, c[48];
MOV32    R0, R2;
MOV32    R2, c[56];
MOV32    R3, R1;
MOV32    R1, c[53];
FMUL32   R3, R3, c[68];
MOV32    R4, R1;
MOV32    R1, c[49];
FMAD     R3, R0, c[69], R3;
MOV32    R0, c[57];
FMAD     R3, R2, c[70], R3;
FMUL32   R5, R1, c[68];
MOV32    R1, c[54];
MOV32    R2, c[50];
FMAD     R4, R4, c[69], R5;
FMAD     R4, R0, c[70], R4;
MOV32    R0, c[58];
FMUL32   R2, R2, c[68];
FMUL     R5, v[7], R4;
FMAD     R1, R1, c[69], R2;
FMAD     R2, v[6], R3, R5;
FMUL     R5, v[10], R4;
FMAD     R0, R0, c[70], R1;
FMUL     R1, v[13], R4;
FMAD     R4, v[9], R3, R5;
FMAD     o[20], v[8], R0, R2;
FMAD     R2, v[12], R3, R1;
FMAD     o[21], v[11], R0, R4;
FADD32   R1, -R7, c[65];
FMAD     o[22], v[14], R0, R2;
FADD32   R2, -R8, c[64];
FADD32   R0, -R6, c[66];
FMUL32   R3, R2, c[48];
FMUL32   R4, R2, c[49];
FMUL32   R2, R2, c[50];
FMAD     R3, R1, c[52], R3;
FMAD     R4, R1, c[53], R4;
FMAD     R1, R1, c[54], R2;
FMAD     R2, R0, c[56], R3;
FMAD     R3, R0, c[57], R4;
FMAD     R0, R0, c[58], R1;
FMUL     R1, v[7], R3;
FMUL     R4, v[10], R3;
FMUL     R3, v[13], R3;
FMAD     R1, v[6], R2, R1;
FMAD     R4, v[9], R2, R4;
FMAD     R2, v[12], R2, R3;
FMAD     o[17], v[8], R0, R1;
FMAD     o[18], v[11], R0, R4;
FMAD     o[19], v[14], R0, R2;
FMUL32   R1, R7, c[36];
FMUL     R0, v[0], c[3];
FMUL32   R2, R7, c[37];
FMAD     R1, R8, c[32], R1;
FMAD     R0, v[1], c[7], R0;
FMAD     R2, R8, c[33], R2;
FMAD     R1, R6, c[40], R1;
FMAD     R0, v[2], c[11], R0;
FMAD     R2, R6, c[41], R2;
FMUL32   R3, R7, c[38];
FMAD     R0, v[3], c[15], R0;
FMAD     R3, R8, c[34], R3;
FMAD     R1, R0, c[44], R1;
FMAD     R2, R0, c[45], R2;
FMAD     R3, R6, c[42], R3;
FMUL32   R4, R7, c[39];
MOV32    o[6], R1;
FMAD     R3, R0, c[46], R3;
FMAD     R4, R8, c[35], R4;
MOV32    o[0], R1;
MOV32    o[7], R2;
FMAD     R1, R6, c[43], R4;
MOV32    o[1], R2;
MOV32    o[8], R3;
FMAD     R0, R0, c[47], R1;
MOV32    o[2], R3;
FMUL     R1, v[0], c[16];
MOV32    o[9], R0;
MOV32    o[3], R0;
FMAD     R0, v[1], c[20], R1;
FMUL     R1, v[0], c[17];
FMUL     R2, v[0], c[18];
FMAD     R0, v[2], c[24], R0;
FMAD     R1, v[1], c[21], R1;
FMAD     R2, v[1], c[22], R2;
FMAD     o[10], v[3], c[28], R0;
FMAD     R0, v[2], c[25], R1;
FMAD     R1, v[2], c[26], R2;
FMUL     R2, v[0], c[19];
FMAD     o[11], v[3], c[29], R0;
FMAD     o[12], v[3], c[30], R1;
FMAD     R0, v[1], c[23], R2;
MOV      o[4], v[4];
MOV      o[5], v[5];
FMAD     R0, v[2], c[27], R0;
FMAD     o[13], v[3], c[31], R0;
END
# 133 instructions, 12 R-regs
# 133 inst, (25 mov, 0 mvi, 0 tex, 0 complex, 108 math)
#    94 64-bit, 39 32-bit, 0 32-bit-const
