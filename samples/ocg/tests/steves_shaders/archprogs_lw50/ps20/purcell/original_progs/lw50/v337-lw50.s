!!SPA1.0
.CONST_MODE  PAGE
.THREAD_TYPE VERTEX
.MAX_REG     3
.MAX_IBUF    11
.MAX_OBUF    27
# parseasm build date Mar 10 2004 15:40:49
# -profile vp50 -po tbat3 -po lat3 -if vs2x -i allprogs-new32//v337-lw40.s -o allprogs-new32//v337-lw50.s
#vendor LWPU
#version parseasm.0.0
#profile vp50
#program fp30entry
#semantic C[9].C[9]
#semantic C[5].C[5]
#semantic C[4].C[4]
#semantic C[8].C[8]
#semantic C[6].C[6]
#semantic C[12].C[12]
#semantic C[10].C[10]
#semantic C[3].C[3]
#semantic C[2].C[2]
#semantic C[1].C[1]
#semantic C[0].C[0]
#var float4 o[TEX4] : $vout.O : O[0] : -1 : 0
#var float4 o[TEX3] : $vout.O : O[0] : -1 : 0
#var float4 o[TEX2] : $vout.O : O[0] : -1 : 0
#var float4 o[TEX1] : $vout.O : O[0] : -1 : 0
#var float4 o[TEX0] : $vout.O : O[0] : -1 : 0
#var float4 o[BCOL1] : $vout.O : O[0] : -1 : 0
#var float4 o[BCOL0] : $vout.O : O[0] : -1 : 0
#var float4 o[HPOS] : $vout.O : O[0] : -1 : 0
#var float4 C[9] :  : c[9] : -1 : 0
#var float4 C[5] :  : c[5] : -1 : 0
#var float4 C[4] :  : c[4] : -1 : 0
#var float4 v[WGHT] : $vin.F : F[0] : -1 : 0
#var float4 C[8] :  : c[8] : -1 : 0
#var float4 C[6] :  : c[6] : -1 : 0
#var float4 C[12] :  : c[12] : -1 : 0
#var float4 C[10] :  : c[10] : -1 : 0
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
#obuf 8 = o[BCOL1].x
#obuf 9 = o[BCOL1].y
#obuf 10 = o[BCOL1].z
#obuf 11 = o[BCOL1].w
#obuf 12 = o[TEX0].x
#obuf 13 = o[TEX0].y
#obuf 14 = o[TEX0].z
#obuf 15 = o[TEX0].w
#obuf 16 = o[TEX1].x
#obuf 17 = o[TEX1].y
#obuf 18 = o[TEX2].x
#obuf 19 = o[TEX2].y
#obuf 20 = o[TEX3].x
#obuf 21 = o[TEX3].y
#obuf 22 = o[TEX3].z
#obuf 23 = o[TEX3].w
#obuf 24 = o[TEX4].x
#obuf 25 = o[TEX4].y
#obuf 26 = o[TEX4].z
#obuf 27 = o[TEX4].w
BB0:
MOV32    o[24], c[16];
MOV32    o[25], c[17];
MOV32    o[26], c[18];
MOV32    o[27], c[19];
MOV      o[20], v[4];
MOV      o[21], v[5];
MOV      o[22], v[6];
MOV      o[23], v[7];
FMUL     R0, v[1], c[25];
FMUL     R1, v[1], c[33];
FMUL     R2, v[1], c[41];
FMAD     R0, v[0], c[24], R0;
FMAD     R1, v[0], c[32], R1;
FMAD     R2, v[0], c[40], R2;
FMAD     R0, v[2], c[26], R0;
FMAD     R1, v[2], c[34], R1;
FMAD     R2, v[2], c[42], R2;
FMAD     o[18], v[3], c[27], R0;
FMAD     o[19], v[3], c[35], R1;
FMAD     o[16], v[3], c[43], R2;
FMUL     R0, v[1], c[49];
MOV      o[12], v[8];
MOV      o[13], v[9];
FMAD     R0, v[0], c[48], R0;
MOV      o[14], v[10];
MOV      o[15], v[11];
FMAD     R0, v[2], c[50], R0;
MOV32    o[8], c[36];
MOV32    o[9], c[37];
FMAD     o[17], v[3], c[51], R0;
MOV32    o[10], c[38];
MOV32    o[11], c[39];
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
# 52 instructions, 4 R-regs
# 52 inst, (20 mov, 0 mvi, 0 tex, 0 complex, 32 math)
#    40 64-bit, 12 32-bit, 0 32-bit-const
