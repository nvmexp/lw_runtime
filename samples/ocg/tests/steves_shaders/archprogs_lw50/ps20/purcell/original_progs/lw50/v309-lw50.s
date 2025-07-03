!!SPA1.0
.CONST_MODE  PAGE
.THREAD_TYPE VERTEX
.MAX_REG     11
.MAX_IBUF    8
.MAX_OBUF    7
# parseasm build date Mar 10 2004 15:40:49
# -profile vp50 -po tbat3 -po lat3 -if vs2x -i allprogs-new32//v309-lw40.s -o allprogs-new32//v309-lw50.s
#vendor LWPU
#version parseasm.0.0
#profile vp50
#program fp30entry
#semantic C[4].C[4]
#semantic C[3].C[3]
#semantic C[2].C[2]
#semantic C[1].C[1]
#semantic C[0].C[0]
#var float4 o[TEX0] : $vout.O : O[0] : -1 : 0
#var float4 o[HPOS] : $vout.O : O[0] : -1 : 0
#var float4 v[NRML] : $vin.F : F[0] : -1 : 0
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
#ibuf 7 = v[NOR].x
#ibuf 8 = v[NOR].y
#obuf 0 = o[HPOS].x
#obuf 1 = o[HPOS].y
#obuf 2 = o[HPOS].z
#obuf 3 = o[HPOS].w
#obuf 4 = o[TEX0].x
#obuf 5 = o[TEX0].y
#obuf 6 = o[TEX0].z
#obuf 7 = o[TEX0].w
BB0:
MOV      R0, v[0];
FADD     R5, -v[1], c[17];
FADD     R3, -v[0], c[16];
FADD     R0, v[4], -R0;
MOV      R2, v[1];
MOV      R1, v[2];
FMUL32   R4, R0, R5;
FADD     R6, v[5], -R2;
FADD     R2, v[6], -R1;
FADD     R1, -v[2], c[18];
FMAD     R8, -R3, R6, R4;
FMUL32   R4, R2, R3;
FMUL32   R7, R6, R1;
FMUL32   R10, R8, R0;
FMAD     R4, -R1, R0, R4;
FMAD     R7, -R5, R2, R7;
FMUL32   R9, R4, R2;
FMAD     R2, -R2, R7, R10;
FMUL32   R7, R7, R6;
FMAD     R6, -R6, R8, R9;
FMUL32   R8, R2, R2;
FMAD     R0, -R0, R4, R7;
FMUL32   R4, R5, R5;
FMAD     R7, R6, R6, R8;
FMAD     R4, R3, R3, R4;
FMAD     R7, R0, R0, R7;
FMAD     R4, R1, R1, R4;
RSQ      R7, |R7|;
RSQ      R4, |R4|;
FMUL32   R6, R6, R7;
FMUL32   R2, R2, R7;
FMUL32   R0, R0, R7;
FMUL32   R5, R5, R4;
FMUL32   R3, R3, R4;
FMUL32   R1, R1, R4;
FMUL32   R2, R2, R5;
FADD     R4, -v[3], c[19];
MOV      o[6], v[3];
FMAD     R2, R6, R3, R2;
FMUL     o[5], v[8], R4;
MOV      o[7], v[3];
FMAD     R0, R0, R1, R2;
FMUL     R1, v[1], c[1];
FMUL     R2, v[1], c[5];
FMUL     o[4], v[7], R0;
FMAD     R0, v[0], c[0], R1;
FMAD     R1, v[0], c[4], R2;
FMUL     R2, v[1], c[9];
FMAD     R0, v[2], c[2], R0;
FMAD     R1, v[2], c[6], R1;
FMAD     R2, v[0], c[8], R2;
FMAD     o[0], v[3], c[3], R0;
FMAD     o[1], v[3], c[7], R1;
FMAD     R0, v[2], c[10], R2;
FMUL     R1, v[1], c[13];
FMAD     o[2], v[3], c[11], R0;
FMAD     R0, v[0], c[12], R1;
FMAD     R0, v[2], c[14], R0;
FMAD     o[3], v[3], c[15], R0;
END
# 59 instructions, 12 R-regs
# 59 inst, (5 mov, 0 mvi, 0 tex, 2 complex, 52 math)
#    44 64-bit, 15 32-bit, 0 32-bit-const
