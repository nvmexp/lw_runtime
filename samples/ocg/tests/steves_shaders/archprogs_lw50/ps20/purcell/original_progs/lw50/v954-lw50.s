!!SPA1.0
.CONST_MODE  PAGE
.THREAD_TYPE VERTEX
.MAX_REG     11
.MAX_IBUF    16
.MAX_OBUF    11
# parseasm build date Mar 10 2004 15:40:49
# -profile vp50 -po tbat3 -po lat3 -if vs2x -i allprogs-new32//v954-lw40.s -o allprogs-new32//v954-lw50.s
#vendor LWPU
#version parseasm.0.0
#profile vp50
#program fp30entry
#semantic C[232].C[232]
#semantic C[231].C[231]
#semantic C[230].C[230]
#semantic C[229].C[229]
#semantic C[228].C[228]
#semantic C[227].C[227]
#semantic C[233].C[233]
#semantic C[226].C[226]
#semantic C[225].C[225]
#semantic c.c
#semantic C[234].C[234]
#var float4 o[TEX2] : $vout.O : O[0] : -1 : 0
#var float4 o[TEX1] : $vout.O : O[0] : -1 : 0
#var float4 o[TEX0] : $vout.O : O[0] : -1 : 0
#var float4 o[HPOS] : $vout.O : O[0] : -1 : 0
#var float4 v[COL1] : $vin.F : F[0] : -1 : 0
#var float4 C[232] :  : c[232] : -1 : 0
#var float4 C[231] :  : c[231] : -1 : 0
#var float4 C[230] :  : c[230] : -1 : 0
#var float4 C[229] :  : c[229] : -1 : 0
#var float4 C[228] :  : c[228] : -1 : 0
#var float4 C[227] :  : c[227] : -1 : 0
#var float4 C[233] :  : c[233] : -1 : 0
#var float4 C[226] :  : c[226] : -1 : 0
#var float4 C[225] :  : c[225] : -1 : 0
#var float4 v[OPOS] : $vin.F : F[0] : -1 : 0
#var float4 v[COL0] : $vin.F : F[0] : -1 : 0
#var float4 c :  : c[0] : -1 : 0
#var float4 v[WGHT] : $vin.F : F[0] : -1 : 0
#var float4 C[234] :  : c[234] : -1 : 0
#var float4 v[NRML] : $vin.F : F[0] : -1 : 0
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
#ibuf 11 = v[COL0].x
#ibuf 12 = v[COL0].y
#ibuf 13 = v[COL0].z
#ibuf 14 = v[COL0].w
#ibuf 15 = v[COL1].x
#ibuf 16 = v[COL1].y
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
FMUL     R0, v[8], c[936];
FMUL     R1, v[7], c[936];
R2A      A2, R0;
R2A      A3, R1;
FMUL     R1, v[1], c[A2 + 5];
FMUL     R0, v[1], c[A3 + 5];
FMAD     R2, v[0], c[A2 + 4], R1;
FMAD     R0, v[0], c[A3 + 4], R0;
FMUL     R1, v[9], c[936];
FMAD     R2, v[2], c[A2 + 6], R2;
FMAD     R0, v[2], c[A3 + 6], R0;
R2A      A1, R1;
FMAD     R2, v[3], c[A2 + 7], R2;
FMAD     R3, v[3], c[A3 + 7], R0;
FMUL     R0, v[1], c[A1 + 5];
FMUL     R1, v[10], c[936];
FMUL     R3, v[11], R3;
FMAD     R0, v[0], c[A1 + 4], R0;
R2A      A0, R1;
FMAD     R3, v[12], R2, R3;
FMAD     R2, v[2], c[A1 + 6], R0;
FMUL     R1, v[1], c[A0 + 5];
FMUL     R0, v[1], c[A2 + 1];
FMAD     R2, v[3], c[A1 + 7], R2;
FMAD     R1, v[0], c[A0 + 4], R1;
FMAD     R0, v[0], c[A2], R0;
FMAD     R2, v[13], R2, R3;
FMAD     R1, v[2], c[A0 + 6], R1;
FMAD     R3, v[2], c[A2 + 2], R0;
FMUL     R0, v[1], c[A3 + 1];
FMAD     R1, v[3], c[A0 + 7], R1;
FMAD     R3, v[3], c[A2 + 3], R3;
FMAD     R0, v[0], c[A3], R0;
FMAD     R1, v[14], R1, R2;
FMUL     R2, v[1], c[A1 + 1];
FMAD     R4, v[2], c[A3 + 2], R0;
FMUL     R0, v[1], c[A0 + 1];
FMAD     R2, v[0], c[A1], R2;
FMAD     R4, v[3], c[A3 + 3], R4;
FMAD     R0, v[0], c[A0], R0;
FMAD     R2, v[2], c[A1 + 2], R2;
FMUL     R4, v[11], R4;
FMAD     R0, v[2], c[A0 + 2], R0;
FMAD     R2, v[3], c[A1 + 3], R2;
FMAD     R4, v[12], R3, R4;
FMAD     R0, v[3], c[A0 + 3], R0;
FMUL     R3, v[1], c[A2 + 9];
FMAD     R4, v[13], R2, R4;
FMUL     R2, v[1], c[A3 + 9];
FMAD     R3, v[0], c[A2 + 8], R3;
FMAD     R0, v[14], R0, R4;
FMAD     R2, v[0], c[A3 + 8], R2;
FMAD     R4, v[2], c[A2 + 10], R3;
FMUL32   R3, R0, c[901];
FMAD     R2, v[2], c[A3 + 10], R2;
FMAD     R4, v[3], c[A2 + 11], R4;
FMAD     R5, R1, c[905], R3;
FMAD     R6, v[3], c[A3 + 11], R2;
FMUL     R3, v[1], c[A1 + 9];
FMUL     R2, v[1], c[A0 + 9];
FMUL     R6, v[11], R6;
FMAD     R3, v[0], c[A1 + 8], R3;
FMAD     R2, v[0], c[A0 + 8], R2;
FMAD     R6, v[12], R4, R6;
FMAD     R3, v[2], c[A1 + 10], R3;
FMAD     R2, v[2], c[A0 + 10], R2;
FMUL32   R4, R0, c[900];
FMAD     R3, v[3], c[A1 + 11], R3;
FMAD     R2, v[3], c[A0 + 11], R2;
FMAD     R4, R1, c[904], R4;
FMAD     R6, v[13], R3, R6;
FMUL32   R3, R0, c[902];
FMAD     R2, v[14], R2, R6;
FMAD     R3, R1, c[906], R3;
FMAD     R5, R2, c[909], R5;
FMAD     R4, R2, c[908], R4;
FMAD     R3, R2, c[910], R3;
FADD32   R5, R5, c[913];
FADD32   R6, R4, c[912];
FADD32   R4, R3, c[914];
FMUL32   R3, R0, c[903];
FMUL32   R7, R6, c[916];
FMUL32   R8, R6, c[917];
FMAD     R3, R1, c[907], R3;
FMAD     R7, R5, c[920], R7;
FMAD     R8, R5, c[921], R8;
FMAD     R3, R2, c[911], R3;
FMAD     R7, R4, c[924], R7;
FMAD     R8, R4, c[925], R8;
FADD32   R3, R3, c[915];
FMUL32   R9, R6, c[918];
FMUL32   R6, R6, c[919];
FMAD     o[0], R3, c[928], R7;
FMAD     o[1], R3, c[929], R8;
FMAD     R7, R5, c[922], R9;
FMAD     R6, R5, c[923], R6;
MOV32    R5, -c[935];
FMAD     R7, R4, c[926], R7;
FMAD     R4, R4, c[927], R6;
FMAD     o[7], R0, R5, c[932];
FMAD     o[2], R3, c[930], R7;
FMAD     o[3], R3, c[931], R4;
MOV32    R0, -c[935];
MOV32    R3, -c[935];
FMUL     R4, v[5], c[A2 + 1];
FMAD     o[8], R1, R0, c[933];
FMAD     o[9], R2, R3, c[934];
FMAD     R2, v[4], c[A2], R4;
FMUL     R1, v[5], c[A3 + 1];
FMUL     R0, v[5], c[A1 + 1];
FMAD     R2, v[6], c[A2 + 2], R2;
FMAD     R3, v[4], c[A3], R1;
FMAD     R1, v[4], c[A1], R0;
FMUL     R0, v[5], c[A0 + 1];
FMAD     R3, v[6], c[A3 + 2], R3;
FMAD     R1, v[6], c[A1 + 2], R1;
FMAD     R0, v[4], c[A0], R0;
FMUL     R3, v[11], R3;
FMUL     R4, v[5], c[A2 + 5];
FMAD     R0, v[6], c[A0 + 2], R0;
FMAD     R2, v[12], R2, R3;
FMAD     R3, v[4], c[A2 + 4], R4;
FMUL     R4, v[5], c[A3 + 5];
FMAD     R1, v[13], R1, R2;
FMAD     R2, v[6], c[A2 + 6], R3;
FMAD     R3, v[4], c[A3 + 4], R4;
FMAD     o[4], v[14], R0, R1;
FMUL     R0, v[5], c[A1 + 5];
FMAD     R3, v[6], c[A3 + 6], R3;
FMUL     R1, v[5], c[A0 + 5];
FMAD     R0, v[4], c[A1 + 4], R0;
FMUL     R3, v[11], R3;
FMAD     R1, v[4], c[A0 + 4], R1;
FMAD     R0, v[6], c[A1 + 6], R0;
FMAD     R2, v[12], R2, R3;
FMAD     R1, v[6], c[A0 + 6], R1;
FMUL     R3, v[5], c[A2 + 9];
FMAD     R0, v[13], R0, R2;
FMUL     R2, v[5], c[A3 + 9];
FMAD     R3, v[4], c[A2 + 8], R3;
FMAD     o[5], v[14], R1, R0;
FMAD     R0, v[4], c[A3 + 8], R2;
FMAD     R2, v[6], c[A2 + 10], R3;
FMUL     R1, v[5], c[A1 + 9];
FMAD     R3, v[6], c[A3 + 10], R0;
FMUL     R0, v[5], c[A0 + 9];
FMAD     R1, v[4], c[A1 + 8], R1;
FMUL     R3, v[11], R3;
FMAD     R0, v[4], c[A0 + 8], R0;
FMAD     R1, v[6], c[A1 + 10], R1;
FMAD     R2, v[12], R2, R3;
FMAD     R0, v[6], c[A0 + 10], R0;
MOV      o[10], v[15];
FMAD     R1, v[13], R1, R2;
MOV      o[11], v[16];
FMAD     o[6], v[14], R0, R1;
END
# 156 instructions, 12 R-regs
# 156 inst, (9 mov, 0 mvi, 0 tex, 0 complex, 147 math)
#    141 64-bit, 15 32-bit, 0 32-bit-const
