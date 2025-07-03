!!SPA1.0
.CONST_MODE  PAGE
.THREAD_TYPE VERTEX
.MAX_REG     3
.MAX_IBUF    8
.MAX_OBUF    11
# parseasm build date Mar 10 2004 15:40:49
# -profile vp50 -po tbat3 -po lat3 -if vs2x -i allprogs-new32//v1004-lw40.s -o allprogs-new32//v1004-lw50.s
#vendor LWPU
#version parseasm.0.0
#profile vp50
#program fp30entry
#semantic C[5].C[5]
#semantic C[4].C[4]
#semantic C[3].C[3]
#semantic C[2].C[2]
#semantic C[1].C[1]
#semantic C[0].C[0]
#var float4 o[PSIZE] : $vout.O : O[0] : -1 : 0
#var float4 o[BCOL0] : $vout.O : O[0] : -1 : 0
#var float4 o[HPOS] : $vout.O : O[0] : -1 : 0
#var float4 v[NRML] : $vin.F : F[0] : -1 : 0
#var float4 C[5] :  : c[5] : -1 : 0
#var float4 v[WGHT] : $vin.F : F[0] : -1 : 0
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
#ibuf 5 = v[NOR].x
#ibuf 6 = v[NOR].y
#ibuf 7 = v[NOR].z
#ibuf 8 = v[NOR].w
#obuf 0 = o[HPOS].x
#obuf 1 = o[HPOS].y
#obuf 2 = o[HPOS].z
#obuf 3 = o[HPOS].w
#obuf 4 = o[BCOL0].x
#obuf 5 = o[BCOL0].y
#obuf 6 = o[BCOL0].z
#obuf 7 = o[BCOL0].w
#obuf 8 = o[PSIZ].x
#obuf 9 = o[PSIZ].y
#obuf 10 = o[PSIZ].z
#obuf 11 = o[PSIZ].w
BB0:
FMUL     R0, v[1], c[17];
FMAD     R0, v[0], c[16], R0;
FMAD     R0, v[2], c[18], R0;
FMAD     R0, v[3], c[19], R0;
RCP      R0, R0;
FMUL     R0, v[4], R0;
MOV      o[4], v[5];
MOV      o[5], v[6];
FMUL32   R0, R0, c[22];
MOV      o[6], v[7];
MOV      o[7], v[8];
FMAX     R0, R0, c[20];
FMUL     R1, v[1], c[1];
FMIN     R0, R0, c[21];
FMAD     R1, v[0], c[0], R1;
FMUL     R2, v[1], c[5];
MOV32    o[8], R0;
FMAD     R1, v[2], c[2], R1;
FMAD     R2, v[0], c[4], R2;
MOV32    o[9], R0;
FMAD     o[0], v[3], c[3], R1;
FMAD     R1, v[2], c[6], R2;
MOV32    o[10], R0;
MOV32    o[11], R0;
FMAD     o[1], v[3], c[7], R1;
FMUL     R0, v[1], c[9];
FMUL     R1, v[1], c[13];
FMAD     R0, v[0], c[8], R0;
FMAD     R1, v[0], c[12], R1;
FMAD     R0, v[2], c[10], R0;
FMAD     R1, v[2], c[14], R1;
FMAD     o[2], v[3], c[11], R0;
FMAD     o[3], v[3], c[15], R1;
END
# 33 instructions, 4 R-regs
# 33 inst, (8 mov, 0 mvi, 0 tex, 1 complex, 24 math)
#    28 64-bit, 5 32-bit, 0 32-bit-const
