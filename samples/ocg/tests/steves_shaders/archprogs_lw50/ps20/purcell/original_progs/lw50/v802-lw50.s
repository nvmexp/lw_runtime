!!SPA1.0
.CONST_MODE  PAGE
.THREAD_TYPE VERTEX
.MAX_REG     7
.MAX_IBUF    9
.MAX_OBUF    23
# parseasm build date Mar 10 2004 15:40:49
# -profile vp50 -po tbat3 -po lat3 -if vs2x -i allprogs-new32//v802-lw40.s -o allprogs-new32//v802-lw50.s
#vendor LWPU
#version parseasm.0.0
#profile vp50
#program fp30entry
#semantic C[9].C[9]
#semantic C[7].C[7]
#semantic C[8].C[8]
#semantic C[6].C[6]
#semantic C[5].C[5]
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
#var float4 o[HPOS] : $vout.O : O[0] : -1 : 0
#var float4 v[WGHT] : $vin.F : F[0] : -1 : 0
#var float4 v[NRML] : $vin.F : F[0] : -1 : 0
#var float4 C[9] :  : c[9] : -1 : 0
#var float4 C[7] :  : c[7] : -1 : 0
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
#ibuf 7 = v[WGT].w
#ibuf 8 = v[NOR].x
#ibuf 9 = v[NOR].y
#obuf 0 = o[HPOS].x
#obuf 1 = o[HPOS].y
#obuf 2 = o[HPOS].z
#obuf 3 = o[HPOS].w
#obuf 4 = o[TEX0].x
#obuf 5 = o[TEX0].y
#obuf 6 = o[TEX0].z
#obuf 7 = o[TEX0].w
#obuf 8 = o[TEX1].x
#obuf 9 = o[TEX1].y
#obuf 10 = o[TEX1].z
#obuf 11 = o[TEX1].w
#obuf 12 = o[TEX2].x
#obuf 13 = o[TEX2].y
#obuf 14 = o[TEX2].z
#obuf 15 = o[TEX2].w
#obuf 16 = o[TEX3].x
#obuf 17 = o[TEX3].y
#obuf 18 = o[TEX3].z
#obuf 19 = o[TEX3].w
#obuf 20 = o[TEX4].x
#obuf 21 = o[TEX4].y
#obuf 22 = o[TEX4].z
#obuf 23 = o[TEX4].w
BB0:
FADD     R0, -v[0], c[32];
FADD     R2, -v[1], c[33];
FADD     R1, -v[2], c[34];
FADD     R4, -v[0], c[28];
FMUL32   R3, R2, R2;
FADD     R6, -v[1], c[29];
FADD     R5, -v[2], c[30];
FMAD     R3, R0, R0, R3;
FMUL32   R7, R6, R6;
FMAD     R3, R1, R1, R3;
FMAD     R7, R4, R4, R7;
RSQ      R3, R3;
FMAD     R7, R5, R5, R7;
RSQ      R7, R7;
FMUL32   R4, R4, R7;
FMUL32   R6, R6, R7;
FMUL32   R5, R5, R7;
FMAD     o[16], R0, R3, R4;
FMAD     o[17], R2, R3, R6;
FMAD     o[18], R1, R3, R5;
MOV32    o[12], R4;
MOV32    o[13], R6;
MOV32    o[14], R5;
MOV      o[20], v[4];
MOV      o[21], v[5];
MOV      o[22], v[6];
MOV      o[23], v[7];
MOV32    R0, c[36];
MOV32    R1, c[36];
MOV      o[8], v[8];
MOV32    o[19], R0;
MOV32    o[15], R1;
MOV      o[9], v[9];
MOV32    R0, c[36];
MOV32    R1, c[36];
FMUL     R2, v[1], c[17];
MOV32    o[10], R0;
MOV32    o[11], R1;
FMAD     R0, v[0], c[16], R2;
FMUL     R1, v[1], c[21];
FMUL     R2, v[1], c[25];
FMAD     R0, v[2], c[18], R0;
FMAD     R1, v[0], c[20], R1;
FMAD     R2, v[0], c[24], R2;
FMAD     R0, v[3], c[19], R0;
FMAD     R1, v[2], c[22], R1;
FMAD     R2, v[2], c[26], R2;
FADD32   o[4], R0, R0;
FMAD     R0, v[3], c[23], R1;
FMAD     R1, v[3], c[27], R2;
MOV32    R2, c[36];
FADD32   o[5], R0, R0;
FADD32   o[6], R1, R1;
MOV32    o[7], R2;
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
# 70 instructions, 8 R-regs
# 70 inst, (19 mov, 0 mvi, 0 tex, 2 complex, 49 math)
#    49 64-bit, 21 32-bit, 0 32-bit-const
