!!SPA1.0
.CONST_MODE  PAGE
.THREAD_TYPE VERTEX
.MAX_REG     11
.MAX_IBUF    7
.MAX_OBUF    19
# parseasm build date Mar 10 2004 15:40:49
# -profile vp50 -po tbat3 -po lat3 -if vs2x -i allprogs-new32//v321-lw40.s -o allprogs-new32//v321-lw50.s
#vendor LWPU
#version parseasm.0.0
#profile vp50
#program fp30entry
#semantic c.c
#semantic C[10].C[10]
#semantic C[8].C[8]
#semantic C[6].C[6]
#semantic C[4].C[4]
#semantic C[3].C[3]
#semantic C[2].C[2]
#semantic C[1].C[1]
#semantic C[0].C[0]
#semantic C[7].C[7]
#semantic C[17].C[17]
#semantic C[16].C[16]
#semantic C[15].C[15]
#semantic C[14].C[14]
#var float4 o[TEX3] : $vout.O : O[0] : -1 : 0
#var float4 o[TEX2] : $vout.O : O[0] : -1 : 0
#var float4 o[TEX1] : $vout.O : O[0] : -1 : 0
#var float4 o[TEX0] : $vout.O : O[0] : -1 : 0
#var float4 o[BCOL0] : $vout.O : O[0] : -1 : 0
#var float4 o[HPOS] : $vout.O : O[0] : -1 : 0
#var float4 c :  : c[0] : -1 : 0
#var float4 C[10] :  : c[10] : -1 : 0
#var float4 C[8] :  : c[8] : -1 : 0
#var float4 C[6] :  : c[6] : -1 : 0
#var float4 C[4] :  : c[4] : -1 : 0
#var float4 C[3] :  : c[3] : -1 : 0
#var float4 C[2] :  : c[2] : -1 : 0
#var float4 C[1] :  : c[1] : -1 : 0
#var float4 C[0] :  : c[0] : -1 : 0
#var float4 C[7] :  : c[7] : -1 : 0
#var float4 v[WGHT] : $vin.F : F[0] : -1 : 0
#var float4 C[17] :  : c[17] : -1 : 0
#var float4 C[16] :  : c[16] : -1 : 0
#var float4 C[15] :  : c[15] : -1 : 0
#var float4 C[14] :  : c[14] : -1 : 0
#var float4 v[OPOS] : $vin.F : F[0] : -1 : 0
#ibuf 0 = v[OPOS].x
#ibuf 1 = v[OPOS].y
#ibuf 2 = v[OPOS].z
#ibuf 3 = v[OPOS].w
#ibuf 4 = v[WGT].x
#ibuf 5 = v[WGT].y
#ibuf 6 = v[WGT].z
#ibuf 7 = v[WGT].w
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
#obuf 16 = o[TEX2].z
#obuf 17 = o[TEX2].w
#obuf 18 = o[TEX3].x
#obuf 19 = o[TEX3].y
BB0:
MOV32    R0, c[60];
MOV32    R1, c[61];
MOV32    R2, c[62];
FMAD     R0, v[0], c[56], R0;
FMAD     R1, v[0], c[57], R1;
MOV32    R4, R2;
MOV32    R3, c[63];
FMUL32   R2, R1, c[64];
FMAD     R4, v[2], c[58], R4;
FMAD     R5, R0, c[68], R2;
FMUL32   R2, R1, c[66];
FMAD     R3, v[2], c[59], R3;
MOV      R7, v[0];
FMAD     R6, R0, c[70], R2;
MOV      R8, v[1];
FMUL32   R2, R1, c[65];
FMUL32   R6, R6, R4;
FMUL32   R1, R1, c[67];
FMAD     R2, R0, c[69], R2;
FMAD     R5, R5, R3, R6;
FMAD     R1, R0, c[71], R1;
FMUL     R0, v[7], R5;
FMUL32   R5, R1, R4;
MOV      R1, v[2];
FMAD     R4, R0, c[30], R7;
FMAD     R2, R2, R3, R5;
FMAD     R5, R0, c[29], R8;
MOV      R3, v[3];
FMUL     R2, v[7], R2;
FMUL32   R6, R5, c[33];
FMUL32   R7, R5, c[41];
FMAD     R1, R2, c[30], R1;
FMAD     R6, R4, c[32], R6;
FMAD     R3, R2, c[29], R3;
FMAD     R7, R4, c[40], R7;
FMAD     R6, R1, c[34], R6;
FMUL32   R8, R5, c[17];
FMAD     R7, R1, c[42], R7;
FMAD     o[18], R3, c[35], R6;
FMAD     R6, R4, c[16], R8;
FMAD     o[19], R3, c[43], R7;
FMUL32   R7, R5, c[25];
FMAD     R6, R1, c[18], R6;
FMUL32   R8, R5, c[1];
FMAD     R7, R4, c[24], R7;
FMAD     o[12], R3, c[19], R6;
FMAD     R6, R4, c[0], R8;
FMAD     R7, R1, c[26], R7;
FMUL32   R8, R5, c[5];
FMAD     R6, R1, c[2], R6;
FMAD     o[13], R3, c[27], R7;
FMAD     R7, R4, c[4], R8;
FMAD     o[0], R3, c[3], R6;
FMUL32   R6, R5, c[9];
FMAD     R7, R1, c[6], R7;
FMUL32   R5, R5, c[13];
FMAD     R6, R4, c[8], R6;
FMAD     o[1], R3, c[7], R7;
FMAD     R4, R4, c[12], R5;
FMAD     R5, R1, c[10], R6;
MOV32    o[14], R0;
FMAD     R0, R1, c[14], R4;
FMAD     o[2], R3, c[11], R5;
MOV32    o[15], R2;
FMAD     o[3], R3, c[15], R0;
MOV32    o[16], R2;
MOV32    o[17], R2;
MOV      o[8], v[4];
MOV      o[9], v[5];
MOV      o[10], v[5];
MOV      o[11], v[5];
MOV32    R0, c[28];
FMAD     R0, v[6], R0, c[31];
F2I.FLOOR R0, R0;
I2I.M4   R0, R0;
R2A      A0, R0;
MOV32    o[4], c[A0];
MOV32    o[5], c[A0 + 1];
MOV32    o[6], c[A0 + 2];
MOV32    o[7], c[A0 + 3];
END
# 80 instructions, 12 R-regs
# 80 inst, (22 mov, 0 mvi, 0 tex, 0 complex, 58 math)
#    52 64-bit, 28 32-bit, 0 32-bit-const
