!!SPA1.0
.CONST_MODE  PAGE
.THREAD_TYPE VERTEX
.MAX_REG     7
.MAX_IBUF    12
.MAX_OBUF    15
# parseasm build date Mar 10 2004 15:40:49
# -profile vp50 -po tbat3 -po lat3 -if vs2x -i allprogs-new32//v940-lw40.s -o allprogs-new32//v940-lw50.s
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
#var float4 o[TEX2] : $vout.O : O[0] : -1 : 0
#var float4 o[TEX1] : $vout.O : O[0] : -1 : 0
#var float4 o[TEX0] : $vout.O : O[0] : -1 : 0
#var float4 o[BCOL0] : $vout.O : O[0] : -1 : 0
#var float4 o[HPOS] : $vout.O : O[0] : -1 : 0
#var float4 v[NRML] : $vin.F : F[0] : -1 : 0
#var float4 v[COL0] : $vin.F : F[0] : -1 : 0
#var float4 v[WGHT] : $vin.F : F[0] : -1 : 0
#var float4 C[8] :  : c[8] : -1 : 0
#var float4 C[6] :  : c[6] : -1 : 0
#var float4 C[5] :  : c[5] : -1 : 0
#var float4 C[4] :  : c[4] : -1 : 0
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
#ibuf 9 = v[COL0].x
#ibuf 10 = v[COL0].y
#ibuf 11 = v[COL0].z
#ibuf 12 = v[COL0].w
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
#obuf 10 = o[TEX1].x
#obuf 11 = o[TEX1].y
#obuf 12 = o[TEX1].z
#obuf 13 = o[TEX2].x
#obuf 14 = o[TEX2].y
#obuf 15 = o[TEX2].z
BB0:
MOV32    R1, c[32];
MOV32    R2, c[32];
MOV32    R0, c[32];
FMAD     R1, v[4], R1, c[33];
FMAD     R2, v[5], R2, c[33];
FMAD     R0, v[6], R0, c[33];
FMUL32   R3, R2, c[17];
FMUL32   R4, R2, c[21];
FMUL32   R2, R2, c[25];
FMAD     R3, R1, c[16], R3;
FMAD     R4, R1, c[20], R4;
FMAD     R1, R1, c[24], R2;
FMAD     o[13], R0, c[18], R3;
FMAD     o[14], R0, c[22], R4;
FMAD     o[15], R0, c[26], R1;
FMUL     R0, v[1], c[17];
FMUL     R1, v[1], c[21];
FMUL     R2, v[1], c[25];
FMAD     R0, v[0], c[16], R0;
FMAD     R1, v[0], c[20], R1;
FMAD     R2, v[0], c[24], R2;
FMAD     R0, v[2], c[18], R0;
FMAD     R1, v[2], c[22], R1;
FMAD     R2, v[2], c[26], R2;
FMAD     o[10], v[3], c[19], R0;
FMAD     o[11], v[3], c[23], R1;
FMAD     o[12], v[3], c[27], R2;
MOV      o[8], v[7];
MOV      o[9], v[8];
MOV      o[4], v[9];
MOV      o[5], v[10];
MOV      o[6], v[11];
MOV      o[7], v[12];
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
# 49 instructions, 8 R-regs
# 49 inst, (9 mov, 0 mvi, 0 tex, 0 complex, 40 math)
#    43 64-bit, 6 32-bit, 0 32-bit-const
