!!SPA1.0
.CONST_MODE  PAGE
.THREAD_TYPE VERTEX
.MAX_REG     3
.MAX_IBUF    3
.MAX_OBUF    7
# parseasm build date Mar 10 2004 15:40:49
# -profile vp50 -po tbat3 -po lat3 -if vs2x -i allprogs-new32//v947-lw40.s -o allprogs-new32//v947-lw50.s
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
#var float4 o[TEX0] : $vout.O : O[0] : -1 : 0
#var float4 o[HPOS] : $vout.O : O[0] : -1 : 0
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
#obuf 0 = o[HPOS].x
#obuf 1 = o[HPOS].y
#obuf 2 = o[HPOS].z
#obuf 3 = o[HPOS].w
#obuf 4 = o[TEX0].x
#obuf 5 = o[TEX0].y
#obuf 6 = o[TEX0].z
#obuf 7 = o[TEX0].w
BB0:
FMUL     R0, v[1], c[17];
FMUL     R1, v[1], c[21];
FMUL     R2, v[1], c[25];
FMAD     R0, v[0], c[16], R0;
FMAD     R1, v[0], c[20], R1;
FMAD     R2, v[0], c[24], R2;
FMAD     R0, v[2], c[18], R0;
FMAD     R1, v[2], c[22], R1;
FMAD     R2, v[2], c[26], R2;
FMAD     o[4], v[3], c[19], R0;
FMAD     o[5], v[3], c[23], R1;
FMAD     o[6], v[3], c[27], R2;
FMUL     R0, v[1], c[29];
FMUL     R1, v[1], c[1];
FMUL     R2, v[1], c[5];
FMAD     R0, v[0], c[28], R0;
FMAD     R1, v[0], c[0], R1;
FMAD     R2, v[0], c[4], R2;
FMAD     R0, v[2], c[30], R0;
FMAD     R1, v[2], c[2], R1;
FMAD     R2, v[2], c[6], R2;
FMAD     o[7], v[3], c[31], R0;
FMAD     o[0], v[3], c[3], R1;
FMAD     o[1], v[3], c[7], R2;
FMUL     R0, v[1], c[9];
FMUL     R1, v[1], c[13];
FMAD     R0, v[0], c[8], R0;
FMAD     R1, v[0], c[12], R1;
FMAD     R0, v[2], c[10], R0;
FMAD     R1, v[2], c[14], R1;
FMAD     o[2], v[3], c[11], R0;
FMAD     o[3], v[3], c[15], R1;
END
# 32 instructions, 4 R-regs
# 32 inst, (0 mov, 0 mvi, 0 tex, 0 complex, 32 math)
#    32 64-bit, 0 32-bit, 0 32-bit-const
