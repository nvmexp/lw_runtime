!!SPA1.0
.CONST_MODE  PAGE
.THREAD_TYPE VERTEX
.MAX_REG     3
.MAX_IBUF    7
.MAX_OBUF    5
# parseasm build date Mar 10 2004 15:40:49
# -profile vp50 -po tbat3 -po lat3 -if vs2x -i allprogs-new32//v1010-lw40.s -o allprogs-new32//v1010-lw50.s
#vendor LWPU
#version parseasm.0.0
#profile vp50
#program fp30entry
#semantic C[91].C[91]
#semantic C[90].C[90]
#semantic C[7].C[7]
#semantic C[6].C[6]
#semantic C[5].C[5]
#semantic C[4].C[4]
#var float4 o[TEX0] : $vout.O : O[0] : -1 : 0
#var float4 o[HPOS] : $vout.O : O[0] : -1 : 0
#var float4 C[91] :  : c[91] : -1 : 0
#var float4 C[90] :  : c[90] : -1 : 0
#var float4 v[WGHT] : $vin.F : F[0] : -1 : 0
#var float4 C[7] :  : c[7] : -1 : 0
#var float4 C[6] :  : c[6] : -1 : 0
#var float4 C[5] :  : c[5] : -1 : 0
#var float4 C[4] :  : c[4] : -1 : 0
#var float4 v[OPOS] : $vin.F : F[0] : -1 : 0
#ibuf 0 = v[OPOS].x
#ibuf 1 = v[OPOS].y
#ibuf 2 = v[OPOS].z
#ibuf 3 = v[OPOS].w
#ibuf 4 = v[WGT].x
#ibuf 5 = v[WGT].y
#ibuf 6 = v[WGT].z
#ibuf 7 = v[WGT].w
#obuf 0 = o[HPOS].x
#obuf 1 = o[HPOS].y
#obuf 2 = o[HPOS].z
#obuf 3 = o[HPOS].w
#obuf 4 = o[TEX0].x
#obuf 5 = o[TEX0].y
BB0:
FMUL     R0, v[5], c[361];
FMUL     R1, v[5], c[365];
FMUL     R2, v[1], c[17];
FMAD     R0, v[4], c[360], R0;
FMAD     R1, v[4], c[364], R1;
FMAD     R2, v[0], c[16], R2;
FMAD     R0, v[6], c[362], R0;
FMAD     R1, v[6], c[366], R1;
FMAD     R2, v[2], c[18], R2;
FMAD     o[4], v[7], c[363], R0;
FMAD     o[5], v[7], c[367], R1;
FMAD     o[0], v[3], c[19], R2;
FMUL     R0, v[1], c[21];
FMUL     R1, v[1], c[25];
FMUL     R2, v[1], c[29];
FMAD     R0, v[0], c[20], R0;
FMAD     R1, v[0], c[24], R1;
FMAD     R2, v[0], c[28], R2;
FMAD     R0, v[2], c[22], R0;
FMAD     R1, v[2], c[26], R1;
FMAD     R2, v[2], c[30], R2;
FMAD     o[1], v[3], c[23], R0;
FMAD     o[2], v[3], c[27], R1;
FMAD     o[3], v[3], c[31], R2;
END
# 24 instructions, 4 R-regs
# 24 inst, (0 mov, 0 mvi, 0 tex, 0 complex, 24 math)
#    24 64-bit, 0 32-bit, 0 32-bit-const
