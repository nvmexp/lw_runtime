!!SPA1.0
.CONST_MODE  PAGE
.THREAD_TYPE VERTEX
.MAX_REG     7
.MAX_IBUF    21
.MAX_OBUF    29
# parseasm build date Mar 10 2004 15:40:49
# -profile vp50 -po tbat3 -po lat3 -if vs2x -i allprogs-new32//v319-lw40.s -o allprogs-new32//v319-lw50.s
#vendor LWPU
#version parseasm.0.0
#profile vp50
#program fp30entry
#semantic C[8].C[8]
#semantic C[6].C[6]
#semantic C[5].C[5]
#semantic C[4].C[4]
#semantic C[3].C[3]
#semantic C[2].C[2]
#semantic C[1].C[1]
#semantic C[0].C[0]
#var float4 o[TEX5] : $vout.O : O[0] : -1 : 0
#var float4 o[TEX4] : $vout.O : O[0] : -1 : 0
#var float4 o[TEX3] : $vout.O : O[0] : -1 : 0
#var float4 o[TEX2] : $vout.O : O[0] : -1 : 0
#var float4 o[TEX1] : $vout.O : O[0] : -1 : 0
#var float4 o[TEX0] : $vout.O : O[0] : -1 : 0
#var float4 o[BCOL0] : $vout.O : O[0] : -1 : 0
#var float4 o[HPOS] : $vout.O : O[0] : -1 : 0
#var float4 C[8] :  : c[8] : -1 : 0
#var float4 C[6] :  : c[6] : -1 : 0
#var float4 C[5] :  : c[5] : -1 : 0
#var float4 C[4] :  : c[4] : -1 : 0
#var float4 v[WGHT] : $vin.F : F[0] : -1 : 0
#var float4 v[FOGC] : $vin.F : F[0] : -1 : 0
#var float4 v[COL1] : $vin.F : F[0] : -1 : 0
#var float4 v[COL0] : $vin.F : F[0] : -1 : 0
#var float4 v[NRML] : $vin.F : F[0] : -1 : 0
#var float4 C[3] :  : c[3] : -1 : 0
#var float4 C[2] :  : c[2] : -1 : 0
#var float4 C[1] :  : c[1] : -1 : 0
#var float4 C[0] :  : c[0] : -1 : 0
#var float4 v[OPOS] : $vin.F : F[0] : -1 : 0
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
#ibuf 10 = v[NOR].w
#ibuf 11 = v[COL0].x
#ibuf 12 = v[COL0].y
#ibuf 13 = v[COL0].z
#ibuf 14 = v[COL0].w
#ibuf 15 = v[COL1].x
#ibuf 16 = v[COL1].y
#ibuf 17 = v[COL1].z
#ibuf 18 = v[COL1].w
#ibuf 19 = v[FOG].x
#ibuf 20 = v[FOG].y
#ibuf 21 = v[FOG].z
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
#obuf 14 = o[TEX1].z
#obuf 15 = o[TEX1].w
#obuf 16 = o[TEX2].x
#obuf 17 = o[TEX2].y
#obuf 18 = o[TEX2].z
#obuf 19 = o[TEX2].w
#obuf 20 = o[TEX3].x
#obuf 21 = o[TEX3].y
#obuf 22 = o[TEX3].z
#obuf 23 = o[TEX3].w
#obuf 24 = o[TEX4].x
#obuf 25 = o[TEX4].y
#obuf 26 = o[TEX4].z
#obuf 27 = o[TEX4].w
#obuf 28 = o[TEX5].x
#obuf 29 = o[TEX5].y
BB0:
MOV      R2, v[6];
MOV      R1, v[21];
MOV      R0, v[4];
MOV      R4, v[19];
FMUL     R3, v[6], R1;
MOV      R1, v[5];
FMUL     R4, v[4], R4;
FMAD     R3, -v[20], R2, R3;
MOV      R2, v[20];
FMAD     R4, -v[21], R0, R4;
FMUL     R0, v[1], c[25];
FMUL     R2, v[4], R2;
FMUL32   R4, R4, c[17];
FMAD     R0, v[0], c[24], R0;
FMAD     R1, -v[19], R1, R2;
FMAD     R2, R3, c[16], R4;
FMAD     R0, v[2], c[26], R0;
FMUL     R3, v[1], c[33];
FMAD     o[25], R1, c[18], R2;
FMAD     o[28], v[3], c[27], R0;
FMAD     R0, v[0], c[32], R3;
FMUL     R1, v[20], c[17];
FMUL     R2, v[5], c[17];
FMAD     R0, v[2], c[34], R0;
FMAD     R1, v[19], c[16], R1;
FMAD     R2, v[4], c[16], R2;
FMAD     o[29], v[3], c[35], R0;
FMAD     o[24], v[21], c[18], R1;
FMAD     o[26], v[6], c[18], R2;
MOV      o[27], v[3];
MOV      o[20], v[11];
MOV      o[21], v[12];
MOV      o[22], v[13];
MOV      o[23], v[14];
MOV      o[16], v[15];
MOV      o[17], v[16];
MOV      o[18], v[17];
MOV      o[19], v[18];
MOV      o[12], v[11];
MOV      o[13], v[12];
MOV      o[14], v[13];
MOV      o[15], v[14];
MOV      o[8], v[7];
MOV      o[9], v[8];
MOV      o[10], v[9];
MOV      o[11], v[10];
MOV32    o[4], c[20];
MOV32    o[5], c[21];
MOV32    o[6], c[22];
MOV32    o[7], c[23];
FMUL     R0, v[1], c[1];
FMUL     R1, v[1], c[5];
FMUL     R2, v[1], c[9];
FMAD     R0, v[0], c[0], R0;
FMAD     R1, v[0], c[4], R1;
FMAD     R2, v[0], c[8], R2;
FMAD     R0, v[2], c[2], R0;
FMAD     R1, v[2], c[6], R1;
FMAD     R2, v[2], c[10], R2;
FMAD     o[0], v[3], c[3], R0;
FMAD     o[1], v[3], c[7], R1;
FMAD     o[2], v[3], c[11], R2;
FMUL     R0, v[1], c[13];
FMAD     R0, v[0], c[12], R0;
FMAD     R0, v[2], c[14], R0;
FMAD     o[3], v[3], c[15], R0;
END
# 66 instructions, 8 R-regs
# 66 inst, (27 mov, 0 mvi, 0 tex, 0 complex, 39 math)
#    61 64-bit, 5 32-bit, 0 32-bit-const
