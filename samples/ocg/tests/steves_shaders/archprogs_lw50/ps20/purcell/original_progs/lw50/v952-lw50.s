!!SPA1.0
.CONST_MODE  PAGE
.THREAD_TYPE VERTEX
.MAX_REG     7
.MAX_IBUF    11
.MAX_OBUF    6
# parseasm build date Mar 10 2004 15:40:49
# -profile vp50 -po tbat3 -po lat3 -if vs2x -i allprogs-new32//v952-lw40.s -o allprogs-new32//v952-lw50.s
#vendor LWPU
#version parseasm.0.0
#profile vp50
#program fp30entry
#semantic C[228].C[228]
#semantic C[227].C[227]
#semantic C[226].C[226]
#semantic C[225].C[225]
#semantic c.c
#semantic C[229].C[229]
#var float4 o[TEX0] : $vout.O : O[0] : -1 : 0
#var float4 o[HPOS] : $vout.O : O[0] : -1 : 0
#var float4 C[228] :  : c[228] : -1 : 0
#var float4 C[227] :  : c[227] : -1 : 0
#var float4 C[226] :  : c[226] : -1 : 0
#var float4 C[225] :  : c[225] : -1 : 0
#var float4 v[NRML] : $vin.F : F[0] : -1 : 0
#var float4 c :  : c[0] : -1 : 0
#var float4 v[OPOS] : $vin.F : F[0] : -1 : 0
#var float4 C[229] :  : c[229] : -1 : 0
#var float4 v[WGHT] : $vin.F : F[0] : -1 : 0
#ibuf 0 = v[OPOS].x
#ibuf 1 = v[OPOS].y
#ibuf 2 = v[OPOS].z
#ibuf 3 = v[OPOS].w
#ibuf 4 = v[WGT].x
#ibuf 5 = v[WGT].y
#ibuf 6 = v[WGT].z
#ibuf 7 = v[WGT].w
#ibuf 8 = v[NOR].x
#ibuf 9 = v[NOR].y
#ibuf 10 = v[NOR].z
#ibuf 11 = v[NOR].w
#obuf 0 = o[HPOS].x
#obuf 1 = o[HPOS].y
#obuf 2 = o[HPOS].z
#obuf 3 = o[HPOS].w
#obuf 4 = o[TEX0].x
#obuf 5 = o[TEX0].y
#obuf 6 = o[TEX0].z
BB0:
FMUL     R0, v[5], c[916];
FMUL     R1, v[4], c[916];
R2A      A2, R0;
R2A      A3, R1;
FMUL     R1, v[1], c[A2 + 5];
FMUL     R0, v[1], c[A3 + 5];
FMAD     R2, v[0], c[A2 + 4], R1;
FMAD     R0, v[0], c[A3 + 4], R0;
FMUL     R1, v[6], c[916];
FMAD     R2, v[2], c[A2 + 6], R2;
FMAD     R0, v[2], c[A3 + 6], R0;
R2A      A1, R1;
FMAD     R2, v[3], c[A2 + 7], R2;
FMAD     R3, v[3], c[A3 + 7], R0;
FMUL     R0, v[1], c[A1 + 5];
FMUL     R1, v[7], c[916];
FMUL     R3, v[8], R3;
FMAD     R0, v[0], c[A1 + 4], R0;
R2A      A0, R1;
FMAD     R3, v[9], R2, R3;
FMAD     R2, v[2], c[A1 + 6], R0;
FMUL     R0, v[1], c[A0 + 5];
FMUL     R1, v[1], c[A2 + 1];
FMAD     R2, v[3], c[A1 + 7], R2;
FMAD     R0, v[0], c[A0 + 4], R0;
FMAD     R1, v[0], c[A2], R1;
FMAD     R3, v[10], R2, R3;
FMAD     R0, v[2], c[A0 + 6], R0;
FMAD     R2, v[2], c[A2 + 2], R1;
FMUL     R1, v[1], c[A3 + 1];
FMAD     R0, v[3], c[A0 + 7], R0;
FMAD     R2, v[3], c[A2 + 3], R2;
FMAD     R1, v[0], c[A3], R1;
FMAD     R4, v[11], R0, R3;
FMUL     R0, v[1], c[A1 + 1];
FMAD     R3, v[2], c[A3 + 2], R1;
FMUL     R1, v[1], c[A0 + 1];
FMAD     R0, v[0], c[A1], R0;
FMAD     R3, v[3], c[A3 + 3], R3;
FMAD     R1, v[0], c[A0], R1;
FMAD     R0, v[2], c[A1 + 2], R0;
FMUL     R3, v[8], R3;
FMAD     R1, v[2], c[A0 + 2], R1;
FMAD     R0, v[3], c[A1 + 3], R0;
FMAD     R3, v[9], R2, R3;
FMAD     R2, v[3], c[A0 + 3], R1;
FMUL     R1, v[1], c[A2 + 9];
FMAD     R3, v[10], R0, R3;
FMUL     R0, v[1], c[A3 + 9];
FMAD     R1, v[0], c[A2 + 8], R1;
FMAD     R5, v[11], R2, R3;
FMAD     R0, v[0], c[A3 + 8], R0;
FMAD     R2, v[2], c[A2 + 10], R1;
FMUL32   R1, R5, c[900];
FMAD     R0, v[2], c[A3 + 10], R0;
FMAD     R2, v[3], c[A2 + 11], R2;
FMAD     R6, R4, c[904], R1;
FMAD     R3, v[3], c[A3 + 11], R0;
FMUL     R1, v[1], c[A1 + 9];
FMUL     R0, v[1], c[A0 + 9];
FMUL     R3, v[8], R3;
FMAD     R1, v[0], c[A1 + 8], R1;
FMAD     R0, v[0], c[A0 + 8], R0;
FMAD     R2, v[9], R2, R3;
FMAD     R1, v[2], c[A1 + 10], R1;
FMAD     R0, v[2], c[A0 + 10], R0;
FMUL32   R3, R5, c[901];
FMAD     R1, v[3], c[A1 + 11], R1;
FMAD     R0, v[3], c[A0 + 11], R0;
FMAD     R3, R4, c[905], R3;
FMAD     R1, v[10], R1, R2;
FMUL32   R2, R5, c[903];
FMAD     R0, v[11], R0, R1;
FMAD     R1, R4, c[907], R2;
FMAD     R2, R0, c[908], R6;
FMAD     R3, R0, c[909], R3;
FMAD     R1, R0, c[911], R1;
FADD32   R2, R2, c[912];
FADD32   R3, R3, c[913];
FADD32   R1, R1, c[915];
FMUL32   R5, R5, c[902];
MOV32    o[4], R2;
MOV32    o[0], R2;
FMAD     R2, R4, c[906], R5;
MOV32    o[5], R3;
MOV32    o[1], R3;
FMAD     R0, R0, c[910], R2;
MOV32    o[6], R1;
MOV32    o[3], R1;
FADD32   o[2], R0, c[914];
END
# 90 instructions, 8 R-regs
# 90 inst, (10 mov, 0 mvi, 0 tex, 0 complex, 80 math)
#    76 64-bit, 14 32-bit, 0 32-bit-const
