!!SPA1.0
.CONST_MODE  PAGE
.THREAD_TYPE VERTEX
.MAX_REG     7
.MAX_IBUF    7
.MAX_OBUF    13
# parseasm build date Mar 10 2004 15:40:49
# -profile vp50 -po tbat3 -po lat3 -if vs2x -i allprogs-new32//v846-lw40.s -o allprogs-new32//v846-lw50.s
#vendor LWPU
#version parseasm.0.0
#profile vp50
#program fp30entry
#semantic C[38].C[38]
#semantic C[91].C[91]
#semantic C[90].C[90]
#semantic C[16].C[16]
#semantic C[11].C[11]
#semantic C[10].C[10]
#semantic C[9].C[9]
#semantic C[8].C[8]
#semantic C[0].C[0]
#semantic C[44].C[44]
#semantic C[43].C[43]
#semantic C[42].C[42]
#var float4 o[TEX0] : $vout.O : O[0] : -1 : 0
#var float4 o[FOGC] : $vout.O : O[0] : -1 : 0
#var float4 o[BCOL0] : $vout.O : O[0] : -1 : 0
#var float4 o[HPOS] : $vout.O : O[0] : -1 : 0
#var float4 C[38] :  : c[38] : -1 : 0
#var float4 C[91] :  : c[91] : -1 : 0
#var float4 C[90] :  : c[90] : -1 : 0
#var float4 v[TEX9] : $vin.F : F[0] : -1 : 0
#var float4 C[16] :  : c[16] : -1 : 0
#var float4 C[11] :  : c[11] : -1 : 0
#var float4 C[10] :  : c[10] : -1 : 0
#var float4 C[9] :  : c[9] : -1 : 0
#var float4 C[8] :  : c[8] : -1 : 0
#var float4 C[0] :  : c[0] : -1 : 0
#var float4 C[44] :  : c[44] : -1 : 0
#var float4 C[43] :  : c[43] : -1 : 0
#var float4 C[42] :  : c[42] : -1 : 0
#var float4 v[OPOS] : $vin.F : F[0] : -1 : 0
#ibuf 0 = v[OPOS].x
#ibuf 1 = v[OPOS].y
#ibuf 2 = v[OPOS].z
#ibuf 3 = v[OPOS].w
#ibuf 4 = v[UNUSED1].x
#ibuf 5 = v[UNUSED1].y
#ibuf 6 = v[UNUSED1].z
#ibuf 7 = v[UNUSED1].w
#obuf 0 = o[HPOS].x
#obuf 1 = o[HPOS].y
#obuf 2 = o[HPOS].z
#obuf 3 = o[HPOS].w
#obuf 4 = o[BCOL0].x
#obuf 5 = o[BCOL0].y
#obuf 6 = o[BCOL0].z
#obuf 7 = o[BCOL0].w
#obuf 8 = o[TEX0].x
#obuf 9 = o[TEX0].y
#obuf 10 = o[FOGC].x
#obuf 11 = o[FOGC].y
#obuf 12 = o[FOGC].z
#obuf 13 = o[FOGC].w
BB0:
FMUL     R1, v[1], c[169];
FMUL     R0, v[1], c[173];
FMAD     R1, v[0], c[168], R1;
FMAD     R2, v[0], c[172], R0;
FMUL     R0, v[1], c[177];
FMAD     R1, v[2], c[170], R1;
FMAD     R2, v[2], c[174], R2;
FMAD     R0, v[0], c[176], R0;
FMAD     R1, v[3], c[171], R1;
FMAD     R2, v[3], c[175], R2;
FMAD     R0, v[2], c[178], R0;
MOV32    R3, c[43];
FMUL32   R4, R2, c[41];
FMAD     R0, v[3], c[179], R0;
FMAD     R5, R1, c[40], R4;
MOV32    R4, c[67];
FMUL32   R6, R2, c[33];
FMAD     R5, R0, c[42], R5;
FMAD     R6, R1, c[32], R6;
FMAD     R3, R3, c[1], R5;
MOV32    R5, c[35];
FMAD     R6, R0, c[34], R6;
FMAD     R4, -R3, R4, c[64];
FMUL32   R7, R2, c[37];
MOV32    o[10], R4;
FMAD     o[0], R5, c[1], R6;
FMAD     R5, R1, c[36], R7;
MOV32    o[2], R3;
FMUL32   R2, R2, c[45];
FMAD     R3, R0, c[38], R5;
MOV32    o[11], R4;
FMAD     R1, R1, c[44], R2;
MOV32    o[12], R4;
MOV32    o[13], R4;
FMAD     R4, R0, c[46], R1;
MOV32    R1, c[39];
MOV32    R2, c[47];
FMUL     R0, v[5], c[361];
FMAD     R0, v[4], c[360], R0;
FMAD     o[1], R1, c[1], R3;
FMAD     o[3], R2, c[1], R4;
FMAD     R0, v[6], c[362], R0;
FMUL     R1, v[5], c[365];
MOV32    o[4], c[152];
FMAD     o[8], v[7], c[363], R0;
FMAD     R0, v[4], c[364], R1;
MOV32    o[5], c[153];
MOV32    o[6], c[154];
FMAD     R0, v[6], c[366], R0;
MOV32    o[7], c[155];
FMAD     o[9], v[7], c[367], R0;
END
# 51 instructions, 8 R-regs
# 51 inst, (14 mov, 0 mvi, 0 tex, 0 complex, 37 math)
#    33 64-bit, 18 32-bit, 0 32-bit-const
