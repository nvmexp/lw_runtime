!!SPA1.0
.CONST_MODE  PAGE
.THREAD_TYPE VERTEX
.MAX_REG     3
.MAX_IBUF    11
.MAX_OBUF    19
# parseasm build date Mar 10 2004 15:40:49
# -profile vp50 -po tbat3 -po lat3 -if vs2x -i allprogs-new32//v900-lw40.s -o allprogs-new32//v900-lw50.s
#vendor LWPU
#version parseasm.0.0
#profile vp50
#program fp30entry
#semantic C[37].C[37]
#semantic C[16].C[16]
#semantic C[44].C[44]
#semantic C[7].C[7]
#semantic C[6].C[6]
#semantic C[5].C[5]
#semantic C[4].C[4]
#var float4 o[TEX2] : $vout.O : O[0] : -1 : 0
#var float4 o[TEX1] : $vout.O : O[0] : -1 : 0
#var float4 o[TEX0] : $vout.O : O[0] : -1 : 0
#var float4 o[FOGC] : $vout.O : O[0] : -1 : 0
#var float4 o[HPOS] : $vout.O : O[0] : -1 : 0
#var float4 v[TEX0] : $vin.F : F[0] : -1 : 0
#var float4 v[TEX9] : $vin.F : F[0] : -1 : 0
#var float4 C[37] :  : c[37] : -1 : 0
#var float4 C[16] :  : c[16] : -1 : 0
#var float4 C[44] :  : c[44] : -1 : 0
#var float4 C[7] :  : c[7] : -1 : 0
#var float4 C[6] :  : c[6] : -1 : 0
#var float4 C[5] :  : c[5] : -1 : 0
#var float4 C[4] :  : c[4] : -1 : 0
#var float4 v[OPOS] : $vin.F : F[0] : -1 : 0
#ibuf 0 = v[OPOS].x
#ibuf 1 = v[OPOS].y
#ibuf 2 = v[OPOS].z
#ibuf 3 = v[OPOS].w
#ibuf 4 = v[UNUSED1].x
#ibuf 5 = v[UNUSED1].y
#ibuf 6 = v[UNUSED1].z
#ibuf 7 = v[UNUSED1].w
#ibuf 8 = v[TEX0].x
#ibuf 9 = v[TEX0].y
#ibuf 10 = v[TEX0].z
#ibuf 11 = v[TEX0].w
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
#obuf 16 = o[FOGC].x
#obuf 17 = o[FOGC].y
#obuf 18 = o[FOGC].z
#obuf 19 = o[FOGC].w
BB0:
MOV32    R0, -c[148];
FMUL     R1, v[1], c[177];
FMAD     R1, v[0], c[176], R1;
FMUL     R2, v[1], c[25];
FMAD     R1, v[2], c[178], R1;
FMAD     R2, v[0], c[24], R2;
FMAD     R1, v[3], c[179], R1;
FMAD     R2, v[2], c[26], R2;
MOV32    R3, c[67];
FMAD     R0, R0, R1, c[149];
FMAD     R1, v[3], c[27], R2;
MOV32    o[12], R0;
FMAD     R2, -R1, R3, c[64];
MOV32    o[13], R0;
MOV32    o[14], R0;
MOV32    o[15], R0;
MOV32    o[16], R2;
MOV32    o[17], R2;
MOV32    o[18], R2;
MOV32    o[19], R2;
MOV      o[8], v[8];
MOV      o[9], v[9];
MOV      o[10], v[10];
MOV      o[11], v[11];
MOV      o[4], v[4];
MOV      o[5], v[5];
MOV      o[6], v[6];
MOV      o[7], v[7];
FMUL     R0, v[1], c[17];
MOV32    o[2], R1;
FMUL     R1, v[1], c[21];
FMAD     R0, v[0], c[16], R0;
FMUL     R2, v[1], c[29];
FMAD     R1, v[0], c[20], R1;
FMAD     R0, v[2], c[18], R0;
FMAD     R2, v[0], c[28], R2;
FMAD     R1, v[2], c[22], R1;
FMAD     o[0], v[3], c[19], R0;
FMAD     R0, v[2], c[30], R2;
FMAD     o[1], v[3], c[23], R1;
FMAD     o[3], v[3], c[31], R0;
END
# 41 instructions, 4 R-regs
# 41 inst, (19 mov, 0 mvi, 0 tex, 0 complex, 22 math)
#    30 64-bit, 11 32-bit, 0 32-bit-const
