!!SPA1.0
.CONST_MODE  PAGE
.THREAD_TYPE VERTEX
.MAX_REG     7
.MAX_IBUF    22
.MAX_OBUF    23
# parseasm build date Mar 10 2004 15:40:49
# -profile vp50 -po tbat3 -po lat3 -if vs2x -i allprogs-new32//v824-lw40.s -o allprogs-new32//v824-lw50.s
#vendor LWPU
#version parseasm.0.0
#profile vp50
#program fp30entry
#semantic C[0].C[0]
#semantic C[90].C[90]
#semantic C[16].C[16]
#semantic C[7].C[7]
#semantic C[6].C[6]
#semantic C[5].C[5]
#semantic C[4].C[4]
#var float4 o[TEX1] : $vout.O : O[0] : -1 : 0
#var float4 o[TEX0] : $vout.O : O[0] : -1 : 0
#var float4 o[FOGC] : $vout.O : O[0] : -1 : 0
#var float4 o[BCOL1] : $vout.O : O[0] : -1 : 0
#var float4 o[BCOL0] : $vout.O : O[0] : -1 : 0
#var float4 o[HPOS] : $vout.O : O[0] : -1 : 0
#var float4 v[FOGC] : $vin.F : F[0] : -1 : 0
#var float4 v[TEX0] : $vin.F : F[0] : -1 : 0
#var float4 v[TEX9] : $vin.F : F[0] : -1 : 0
#var float4 C[0] :  : c[0] : -1 : 0
#var float4 C[90] :  : c[90] : -1 : 0
#var float4 v[TEX4] : $vin.F : F[0] : -1 : 0
#var float4 v[TEX3] : $vin.F : F[0] : -1 : 0
#var float4 C[16] :  : c[16] : -1 : 0
#var float4 C[7] :  : c[7] : -1 : 0
#var float4 C[6] :  : c[6] : -1 : 0
#var float4 C[5] :  : c[5] : -1 : 0
#var float4 C[4] :  : c[4] : -1 : 0
#var float4 v[OPOS] : $vin.F : F[0] : -1 : 0
#ibuf 0 = v[OPOS].x
#ibuf 1 = v[OPOS].y
#ibuf 2 = v[OPOS].z
#ibuf 3 = v[OPOS].w
#ibuf 4 = v[FOG].x
#ibuf 5 = v[FOG].y
#ibuf 6 = v[FOG].z
#ibuf 7 = v[FOG].w
#ibuf 8 = v[UNUSED1].x
#ibuf 9 = v[UNUSED1].y
#ibuf 10 = v[UNUSED1].z
#ibuf 11 = v[UNUSED1].w
#ibuf 12 = v[TEX0].x
#ibuf 13 = v[TEX0].y
#ibuf 14 = v[TEX0].z
#ibuf 15 = v[TEX0].w
#ibuf 16 = v[TEX3].x
#ibuf 17 = v[TEX3].y
#ibuf 18 = v[TEX3].z
#ibuf 19 = v[TEX3].w
#ibuf 20 = v[TEX4].x
#ibuf 21 = v[TEX4].y
#ibuf 22 = v[TEX4].z
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
#obuf 18 = o[TEX1].z
#obuf 19 = o[TEX1].w
#obuf 20 = o[FOGC].x
#obuf 21 = o[FOGC].y
#obuf 22 = o[FOGC].z
#obuf 23 = o[FOGC].w
BB0:
MOV      R3, v[21];
MOV      R0, v[22];
MOV      R1, v[20];
FMUL     R4, v[16], R3;
FMUL     R5, v[18], R0;
FMUL     R2, v[16], R1;
FMAD     R1, -v[16], R1, R4;
FMAD     R4, -v[17], R3, R5;
FMAD     R0, -v[16], R0, R2;
FMUL32   R2, R0, R0;
FMAD     R2, R4, R4, R2;
FMAD     R2, R1, R1, R2;
RSQ      R2, |R2|;
FMUL32   R3, R1, R2;
FMUL32   R0, R0, R2;
FMUL32   R1, R4, R2;
FMUL     R5, v[16], R3;
FMUL     R4, v[18], R0;
FMUL     R2, v[17], R1;
FMAD     R5, v[18], -R1, R5;
FMAD     R4, v[17], -R3, R4;
FMAD     R2, v[16], -R0, R2;
FMUL32   R5, R5, c[361];
FMUL32   R6, R0, c[361];
MOV      R0, v[16];
FMAD     R4, R4, c[360], R5;
FMAD     R5, R1, c[360], R6;
MOV      R1, v[17];
FMAD     R2, R2, c[362], R4;
FMAD     R3, R3, c[362], R5;
FMUL     R1, v[17], R1;
FADD32   o[9], R2, c[3];
FADD32   o[10], R3, c[3];
FMAD     R1, R0, v[16], R1;
MOV      R0, v[18];
FMAD     R0, R0, v[18], R1;
RSQ      R2, |R0|;
FMUL     R0, v[1], c[25];
FMUL     R3, v[16], R2;
FMAD     R0, v[0], c[24], R0;
FMUL     R1, v[17], R2;
FMUL     R2, v[18], R2;
FMAD     R0, v[2], c[26], R0;
FMUL32   R4, R1, c[361];
MOV32    R1, c[67];
FMAD     R0, v[3], c[27], R0;
FMAD     R3, R3, c[360], R4;
MOV      o[16], v[12];
FMAD     R1, -R0, R1, c[64];
FMAD     R2, R2, c[362], R3;
MOV      o[17], v[13];
MOV32    o[20], R1;
FADD32   o[8], R2, c[3];
MOV32    o[21], R1;
MOV32    o[22], R1;
MOV32    o[23], R1;
MOV      o[18], v[14];
MOV      o[19], v[15];
MOV      o[12], v[8];
MOV      o[13], v[9];
MOV      o[14], v[10];
MOV      o[15], v[11];
FADD     o[11], v[19], c[3];
MOV      o[4], v[4];
MOV      o[5], v[5];
MOV      o[6], v[6];
MOV      o[7], v[7];
FMUL     R1, v[1], c[17];
MOV32    o[2], R0;
FMUL     R0, v[1], c[21];
FMAD     R1, v[0], c[16], R1;
FMUL     R2, v[1], c[29];
FMAD     R0, v[0], c[20], R0;
FMAD     R1, v[2], c[18], R1;
FMAD     R2, v[0], c[28], R2;
FMAD     R0, v[2], c[22], R0;
FMAD     o[0], v[3], c[19], R1;
FMAD     R1, v[2], c[30], R2;
FMAD     o[1], v[3], c[23], R0;
FMAD     o[3], v[3], c[31], R1;
END
# 80 instructions, 8 R-regs
# 80 inst, (24 mov, 0 mvi, 0 tex, 2 complex, 54 math)
#    64 64-bit, 16 32-bit, 0 32-bit-const
