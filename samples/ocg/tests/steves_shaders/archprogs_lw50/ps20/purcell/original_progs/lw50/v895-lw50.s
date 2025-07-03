!!SPA1.0
.CONST_MODE  PAGE
.THREAD_TYPE VERTEX
.MAX_REG     3
.MAX_IBUF    14
.MAX_OBUF    21
# parseasm build date Mar 10 2004 15:40:49
# -profile vp50 -po tbat3 -po lat3 -if vs2x -i allprogs-new32//v895-lw40.s -o allprogs-new32//v895-lw50.s
#vendor LWPU
#version parseasm.0.0
#profile vp50
#program fp30entry
#semantic C[16].C[16]
#semantic C[6].C[6]
#semantic C[2].C[2]
#semantic C[44].C[44]
#semantic C[43].C[43]
#semantic C[42].C[42]
#semantic C[7].C[7]
#semantic C[5].C[5]
#semantic C[4].C[4]
#var float4 o[TEX4] : $vout.O : O[0] : -1 : 0
#var float4 o[TEX3] : $vout.O : O[0] : -1 : 0
#var float4 o[TEX2] : $vout.O : O[0] : -1 : 0
#var float4 o[TEX1] : $vout.O : O[0] : -1 : 0
#var float4 o[TEX0] : $vout.O : O[0] : -1 : 0
#var float4 o[FOGC] : $vout.O : O[0] : -1 : 0
#var float4 o[HPOS] : $vout.O : O[0] : -1 : 0
#var float4 v[NRML] : $vin.F : F[0] : -1 : 0
#var float4 C[16] :  : c[16] : -1 : 0
#var float4 C[6] :  : c[6] : -1 : 0
#var float4 C[2] :  : c[2] : -1 : 0
#var float4 v[WGHT] : $vin.F : F[0] : -1 : 0
#var float4 v[COL1] : $vin.F : F[0] : -1 : 0
#var float4 C[44] :  : c[44] : -1 : 0
#var float4 C[43] :  : c[43] : -1 : 0
#var float4 C[42] :  : c[42] : -1 : 0
#var float4 v[COL0] : $vin.F : F[0] : -1 : 0
#var float4 C[7] :  : c[7] : -1 : 0
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
#ibuf 7 = v[NOR].x
#ibuf 8 = v[NOR].y
#ibuf 9 = v[COL0].x
#ibuf 10 = v[COL0].y
#ibuf 11 = v[COL0].z
#ibuf 12 = v[COL1].x
#ibuf 13 = v[COL1].y
#ibuf 14 = v[COL1].z
#obuf 0 = o[HPOS].x
#obuf 1 = o[HPOS].y
#obuf 2 = o[HPOS].z
#obuf 3 = o[HPOS].w
#obuf 4 = o[TEX0].x
#obuf 5 = o[TEX0].y
#obuf 6 = o[TEX1].x
#obuf 7 = o[TEX1].y
#obuf 8 = o[TEX1].z
#obuf 9 = o[TEX2].x
#obuf 10 = o[TEX2].y
#obuf 11 = o[TEX2].z
#obuf 12 = o[TEX3].x
#obuf 13 = o[TEX3].y
#obuf 14 = o[TEX3].z
#obuf 15 = o[TEX4].x
#obuf 16 = o[TEX4].y
#obuf 17 = o[TEX4].z
#obuf 18 = o[FOGC].x
#obuf 19 = o[FOGC].y
#obuf 20 = o[FOGC].z
#obuf 21 = o[FOGC].w
BB0:
FMUL     R0, v[1], c[25];
MOV32    R2, c[67];
FMAD     R1, v[0], c[24], R0;
FMUL     R0, v[5], c[169];
FMAD     R3, v[2], c[26], R1;
FMAD     R0, v[4], c[168], R0;
FMUL     R1, v[5], c[173];
FMAD     R3, v[3], c[27], R3;
FMAD     o[15], v[6], c[170], R0;
FMAD     R0, v[4], c[172], R1;
FMAD     R2, -R3, R2, c[64];
FMUL     R1, v[5], c[177];
FMAD     o[16], v[6], c[174], R0;
MOV32    o[18], R2;
FMAD     R0, v[4], c[176], R1;
MOV32    o[19], R2;
MOV32    o[20], R2;
MOV32    o[21], R2;
FMAD     o[17], v[6], c[178], R0;
FMUL     R0, v[13], c[169];
FMUL     R1, v[13], c[173];
FMUL     R2, v[13], c[177];
FMAD     R0, v[12], c[168], R0;
FMAD     R1, v[12], c[172], R1;
FMAD     R2, v[12], c[176], R2;
FMAD     o[12], v[14], c[170], R0;
FMAD     o[13], v[14], c[174], R1;
FMAD     o[14], v[14], c[178], R2;
FMUL     R0, v[10], c[169];
FMUL     R1, v[10], c[173];
FMUL     R2, v[10], c[177];
FMAD     R0, v[9], c[168], R0;
FMAD     R1, v[9], c[172], R1;
FMAD     R2, v[9], c[176], R2;
FMAD     o[9], v[11], c[170], R0;
FMAD     o[10], v[11], c[174], R1;
FMAD     o[11], v[11], c[178], R2;
FMUL     R0, v[1], c[169];
FMUL     R1, v[1], c[173];
FMAD     R0, v[0], c[168], R0;
FMAD     R1, v[0], c[172], R1;
FMUL     R2, v[1], c[177];
FMAD     R0, v[2], c[170], R0;
FMAD     R1, v[2], c[174], R1;
FMAD     R2, v[0], c[176], R2;
FMAD     R0, v[3], c[171], R0;
FMAD     R1, v[3], c[175], R1;
FMAD     R2, v[2], c[178], R2;
FADD32   o[6], -R0, c[8];
FADD32   o[7], -R1, c[9];
FMAD     R0, v[3], c[179], R2;
MOV      o[4], v[7];
MOV      o[5], v[8];
FADD32   o[8], -R0, c[10];
FMUL     R0, v[1], c[17];
MOV32    o[2], R3;
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
# 67 instructions, 4 R-regs
# 67 inst, (8 mov, 0 mvi, 0 tex, 0 complex, 59 math)
#    58 64-bit, 9 32-bit, 0 32-bit-const
