!!SPA1.0
.CONST_MODE  PAGE
.THREAD_TYPE VERTEX
.MAX_REG     7
.MAX_IBUF    11
.MAX_OBUF    26
# parseasm build date Mar 10 2004 15:40:49
# -profile vp50 -po tbat3 -po lat3 -if vs2x -i allprogs-new32//v297-lw40.s -o allprogs-new32//v297-lw50.s
#vendor LWPU
#version parseasm.0.0
#profile vp50
#program fp30entry
#semantic C[12].C[12]
#semantic C[11].C[11]
#semantic C[10].C[10]
#semantic C[9].C[9]
#semantic C[15].C[15]
#semantic C[14].C[14]
#semantic C[13].C[13]
#semantic C[8].C[8]
#semantic C[7].C[7]
#semantic C[6].C[6]
#semantic C[4].C[4]
#semantic C[3].C[3]
#semantic C[2].C[2]
#semantic C[1].C[1]
#semantic C[0].C[0]
#var float4 o[TEX4] : $vout.O : O[0] : -1 : 0
#var float4 o[TEX3] : $vout.O : O[0] : -1 : 0
#var float4 o[TEX2] : $vout.O : O[0] : -1 : 0
#var float4 o[TEX1] : $vout.O : O[0] : -1 : 0
#var float4 o[TEX0] : $vout.O : O[0] : -1 : 0
#var float4 o[BCOL0] : $vout.O : O[0] : -1 : 0
#var float4 o[HPOS] : $vout.O : O[0] : -1 : 0
#var float4 C[12] :  : c[12] : -1 : 0
#var float4 C[11] :  : c[11] : -1 : 0
#var float4 C[10] :  : c[10] : -1 : 0
#var float4 C[9] :  : c[9] : -1 : 0
#var float4 v[NRML] : $vin.F : F[0] : -1 : 0
#var float4 C[15] :  : c[15] : -1 : 0
#var float4 C[14] :  : c[14] : -1 : 0
#var float4 C[13] :  : c[13] : -1 : 0
#var float4 C[8] :  : c[8] : -1 : 0
#var float4 C[7] :  : c[7] : -1 : 0
#var float4 C[6] :  : c[6] : -1 : 0
#var float4 C[4] :  : c[4] : -1 : 0
#var float4 v[WGHT] : $vin.F : F[0] : -1 : 0
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
#ibuf 7 = v[WGT].w
#ibuf 8 = v[NOR].x
#ibuf 9 = v[NOR].y
#ibuf 10 = v[NOR].z
#ibuf 11 = v[NOR].w
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
BB0:
FADD     R0, -v[0], c[16];
FADD     R1, -v[1], c[17];
FADD     R2, -v[2], c[18];
FMUL32   R3, R1, R1;
FMAD     R3, R0, R0, R3;
FMAD     R3, R2, R2, R3;
RSQ      R4, |R3|;
MOV32    R3, c[32];
FMUL32   R0, R0, R4;
FMUL32   R1, R1, R4;
FMUL32   R2, R2, R4;
MOV32    R4, R3;
FMUL32   R3, R1, -c[25];
MOV32    R5, c[33];
MOV32    R6, c[34];
FMAD     R3, R0, -c[24], R3;
FMAD     R3, R2, -c[26], R3;
MOV32    o[24], R0;
MOV32    o[25], R1;
FADD     R0, v[3], -R3;
MOV32    o[26], R2;
FMUL     R1, v[1], c[37];
FMAD     o[4], R0, c[28], R4;
FMAD     o[5], R0, c[29], R5;
FMAD     R0, R0, c[30], R6;
FMAD     R1, v[0], c[36], R1;
FMUL     R2, v[1], c[41];
MOV32    o[6], R0;
MOV32    o[7], R0;
FMAD     R0, v[2], c[38], R1;
FMAD     R1, v[0], c[40], R2;
FMUL     R2, v[1], c[45];
FMAD     o[20], v[3], c[39], R0;
FMAD     R0, v[2], c[42], R1;
FMAD     R1, v[0], c[44], R2;
FMUL     R2, v[1], c[49];
FMAD     o[21], v[3], c[43], R0;
FMAD     R0, v[2], c[46], R1;
FMAD     R1, v[0], c[48], R2;
MOV      o[16], v[8];
FMAD     o[22], v[3], c[47], R0;
FMAD     R0, v[2], c[50], R1;
MOV      o[17], v[9];
MOV      o[18], v[10];
FMAD     o[23], v[3], c[51], R0;
MOV      o[19], v[11];
FMUL     R0, v[1], c[53];
FMUL     R1, v[1], c[57];
FMUL     R2, v[1], c[61];
FMAD     R0, v[0], c[52], R0;
FMAD     R1, v[0], c[56], R1;
FMAD     R2, v[0], c[60], R2;
FMAD     R0, v[2], c[54], R0;
FMAD     R1, v[2], c[58], R1;
FMAD     R2, v[2], c[62], R2;
FMAD     o[12], v[3], c[55], R0;
FMAD     o[13], v[3], c[59], R1;
FMAD     o[14], v[3], c[63], R2;
MOV      o[15], v[3];
MOV      o[8], v[4];
MOV      o[9], v[5];
MOV      o[10], v[6];
MOV      o[11], v[7];
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
# 79 instructions, 8 R-regs
# 79 inst, (18 mov, 0 mvi, 0 tex, 1 complex, 60 math)
#    65 64-bit, 14 32-bit, 0 32-bit-const
