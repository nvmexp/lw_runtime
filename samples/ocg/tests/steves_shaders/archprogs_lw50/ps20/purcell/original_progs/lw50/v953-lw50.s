!!SPA1.0
.CONST_MODE  PAGE
.THREAD_TYPE VERTEX
.MAX_REG     7
.MAX_IBUF    8
.MAX_OBUF    11
# parseasm build date Mar 10 2004 15:40:49
# -profile vp50 -po tbat3 -po lat3 -if vs2x -i allprogs-new32//v953-lw40.s -o allprogs-new32//v953-lw50.s
#vendor LWPU
#version parseasm.0.0
#profile vp50
#program fp30entry
#semantic C[8].C[8]
#semantic C[7].C[7]
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
#var float4 o[HPOS] : $vout.O : O[0] : -1 : 0
#var float4 v[NRML] : $vin.F : F[0] : -1 : 0
#var float4 v[WGHT] : $vin.F : F[0] : -1 : 0
#var float4 C[8] :  : c[8] : -1 : 0
#var float4 C[7] :  : c[7] : -1 : 0
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
#obuf 0 = o[HPOS].x
#obuf 1 = o[HPOS].y
#obuf 2 = o[HPOS].z
#obuf 3 = o[HPOS].w
#obuf 4 = o[TEX0].x
#obuf 5 = o[TEX0].y
#obuf 6 = o[TEX0].z
#obuf 7 = o[TEX1].x
#obuf 8 = o[TEX1].y
#obuf 9 = o[TEX1].z
#obuf 10 = o[TEX2].x
#obuf 11 = o[TEX2].y
BB0:
FMUL     R1, v[0], c[1];
FMUL     R0, v[0], c[0];
FMAD     R2, v[1], c[5], R1;
FMAD     R1, v[1], c[4], R0;
FMUL     R0, v[0], c[2];
FMAD     R2, v[2], c[9], R2;
FMAD     R1, v[2], c[8], R1;
FMAD     R0, v[1], c[6], R0;
FMAD     R2, v[3], c[13], R2;
FMAD     R3, v[3], c[12], R1;
FMAD     R1, v[2], c[10], R0;
FMUL     R0, v[0], c[3];
FMUL32   R4, R3, c[16];
FMAD     R1, v[3], c[14], R1;
FMAD     R0, v[1], c[7], R0;
FMAD     R4, R2, c[20], R4;
FMUL32   R5, R3, c[17];
FMAD     R0, v[2], c[11], R0;
FMAD     R4, R1, c[24], R4;
FMAD     R5, R2, c[21], R5;
FMAD     R0, v[3], c[15], R0;
FMUL32   R6, R3, c[18];
FMAD     R5, R1, c[25], R5;
FMAD     o[0], R0, c[28], R4;
FMAD     R4, R2, c[22], R6;
FMAD     o[1], R0, c[29], R5;
FMUL32   R3, R3, c[19];
FMAD     R4, R1, c[26], R4;
MOV      o[10], v[7];
FMAD     R2, R2, c[23], R3;
FMAD     o[2], R0, c[30], R4;
MOV      o[11], v[8];
FMAD     R1, R1, c[27], R2;
FADD     o[7], -v[0], c[32];
FADD     o[8], -v[1], c[33];
FMAD     o[3], R0, c[31], R1;
FADD     o[9], -v[2], c[34];
MOV      o[4], v[4];
MOV      o[5], v[5];
MOV      o[6], v[6];
END
# 40 instructions, 8 R-regs
# 40 inst, (5 mov, 0 mvi, 0 tex, 0 complex, 35 math)
#    36 64-bit, 4 32-bit, 0 32-bit-const
