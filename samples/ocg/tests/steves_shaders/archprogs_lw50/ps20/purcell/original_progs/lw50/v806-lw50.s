!!SPA1.0
.CONST_MODE  PAGE
.THREAD_TYPE VERTEX
.MAX_REG     3
.MAX_IBUF    11
.MAX_OBUF    27
# parseasm build date Mar 10 2004 15:40:49
# -profile vp50 -po tbat3 -po lat3 -if vs2x -i allprogs-new32//v806-lw40.s -o allprogs-new32//v806-lw50.s
#vendor LWPU
#version parseasm.0.0
#profile vp50
#program fp30entry
#semantic C[4].C[4]
#semantic C[8].C[8]
#semantic C[7].C[7]
#semantic C[6].C[6]
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
#var float4 o[HPOS] : $vout.O : O[0] : -1 : 0
#var float4 C[4] :  : c[4] : -1 : 0
#var float4 C[8] :  : c[8] : -1 : 0
#var float4 C[7] :  : c[7] : -1 : 0
#var float4 C[6] :  : c[6] : -1 : 0
#var float4 v[NRML] : $vin.F : F[0] : -1 : 0
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
#obuf 24 = o[TEX5].x
#obuf 25 = o[TEX5].y
#obuf 26 = o[TEX5].z
#obuf 27 = o[TEX5].w
BB0:
FMUL     R0, v[1], c[13];
FMAD     R0, v[0], c[12], R0;
FMAD     R0, v[2], c[14], R0;
MOV32    o[20], c[32];
MOV32    o[21], c[33];
FMAD     R0, v[3], c[15], R0;
MOV32    o[22], c[34];
MOV32    o[23], c[35];
FADD32   R1, R0, -c[16];
MOV32    o[16], c[28];
MOV32    o[17], c[29];
FMUL32   R1, R1, c[17];
MOV32    o[18], c[30];
MOV32    o[19], c[31];
MOV32    o[24], R1;
MOV32    o[25], R1;
MOV32    o[26], R1;
MOV32    o[27], R1;
MOV32    o[12], c[24];
MOV32    o[13], c[25];
MOV32    o[14], c[26];
MOV32    o[15], c[27];
MOV      o[8], v[8];
MOV      o[9], v[9];
MOV      o[10], v[10];
MOV      o[11], v[11];
MOV      o[4], v[4];
MOV      o[5], v[5];
MOV      o[6], v[6];
MOV      o[7], v[7];
FMUL     R1, v[1], c[1];
FMUL     R2, v[1], c[5];
FMUL     R3, v[1], c[9];
FMAD     R1, v[0], c[0], R1;
FMAD     R2, v[0], c[4], R2;
FMAD     R3, v[0], c[8], R3;
FMAD     R1, v[2], c[2], R1;
FMAD     R2, v[2], c[6], R2;
FMAD     R3, v[2], c[10], R3;
FMAD     o[0], v[3], c[3], R1;
FMAD     o[1], v[3], c[7], R2;
FMAD     o[2], v[3], c[11], R3;
MOV32    o[3], R0;
END
# 43 instructions, 4 R-regs
# 43 inst, (25 mov, 0 mvi, 0 tex, 0 complex, 18 math)
#    24 64-bit, 19 32-bit, 0 32-bit-const
