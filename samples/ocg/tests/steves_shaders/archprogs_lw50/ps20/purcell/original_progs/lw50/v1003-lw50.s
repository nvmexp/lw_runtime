!!SPA1.0
.CONST_MODE  PAGE
.THREAD_TYPE VERTEX
.MAX_REG     7
.MAX_IBUF    10
.MAX_OBUF    17
# parseasm build date Mar 10 2004 15:40:49
# -profile vp50 -po tbat3 -po lat3 -if vs2x -i allprogs-new32//v1003-lw40.s -o allprogs-new32//v1003-lw50.s
#vendor LWPU
#version parseasm.0.0
#profile vp50
#program fp30entry
#semantic C[7].C[7]
#semantic C[6].C[6]
#semantic C[5].C[5]
#semantic C[4].C[4]
#semantic C[3].C[3]
#semantic C[2].C[2]
#semantic C[1].C[1]
#semantic C[0].C[0]
#var float4 o[TEX3] : $vout.O : O[0] : -1 : 0
#var float4 o[TEX2] : $vout.O : O[0] : -1 : 0
#var float4 o[TEX1] : $vout.O : O[0] : -1 : 0
#var float4 o[TEX0] : $vout.O : O[0] : -1 : 0
#var float4 o[HPOS] : $vout.O : O[0] : -1 : 0
#var float4 C[7] :  : c[7] : -1 : 0
#var float4 C[6] :  : c[6] : -1 : 0
#var float4 C[5] :  : c[5] : -1 : 0
#var float4 C[4] :  : c[4] : -1 : 0
#var float4 v[WGHT] : $vin.F : F[0] : -1 : 0
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
#obuf 11 = o[TEX2].x
#obuf 12 = o[TEX2].y
#obuf 13 = o[TEX2].z
#obuf 14 = o[TEX3].x
#obuf 15 = o[TEX3].y
#obuf 16 = o[TEX3].z
#obuf 17 = o[TEX3].w
BB0:
FADD     R1, -v[0], c[28];
FADD     R2, -v[1], c[29];
FADD     R0, -v[2], c[30];
FMUL32   R3, R2, R2;
FMAD     R3, R1, R1, R3;
FMAD     R3, R0, R0, R3;
RSQ      R3, |R3|;
FMUL32   R1, R1, R3;
FMUL32   R2, R2, R3;
FMUL32   R0, R0, R3;
FMUL     R3, v[5], R2;
FMAD     R3, v[4], R1, R3;
FMAD     R3, v[6], R0, R3;
FADD32   R3, R3, R3;
FMAD     R1, v[4], R3, -R1;
FMAD     R2, v[5], R3, -R2;
FMAD     R0, v[6], R3, -R0;
FMUL32   R3, R2, c[17];
FMUL32   R4, R2, c[21];
FMUL32   R2, R2, c[25];
FMAD     R3, R1, c[16], R3;
FMAD     R4, R1, c[20], R4;
FMAD     R1, R1, c[24], R2;
FMAD     o[11], R0, c[18], R3;
FMAD     o[12], R0, c[22], R4;
FMAD     o[13], R0, c[26], R1;
MOV      o[14], v[7];
MOV      o[15], v[8];
MOV      o[16], v[9];
MOV      o[17], v[10];
FMUL     R0, v[5], c[17];
FMUL     R1, v[5], c[21];
FMUL     R2, v[5], c[25];
FMAD     R0, v[4], c[16], R0;
FMAD     R1, v[4], c[20], R1;
FMAD     R2, v[4], c[24], R2;
FMAD     o[8], v[6], c[18], R0;
FMAD     o[9], v[6], c[22], R1;
FMAD     o[10], v[6], c[26], R2;
MOV      o[4], v[7];
MOV      o[5], v[8];
MOV      o[6], v[9];
MOV      o[7], v[10];
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
# 59 instructions, 8 R-regs
# 59 inst, (8 mov, 0 mvi, 0 tex, 1 complex, 50 math)
#    51 64-bit, 8 32-bit, 0 32-bit-const
