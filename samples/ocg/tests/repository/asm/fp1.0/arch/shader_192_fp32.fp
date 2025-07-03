!!FP1.0
DECLARE C0={0.1, 0.2, 0.3, 0.4};
DECLARE C1={4, 3, 2, 1};
DECLARE C2={0.500000, 0.500000, 0.250000, 0.000000};
DECLARE C3={6.283190, -0.000000, 0.000025, -3.141590};
DECLARE C4={-0.001389, -0.500000, 1.000000, 0.041667};
DECLARE C5={0.000000, 1.200000, 0.400000, 20.000000};
DECLARE C6={2.000000, -1.000000, 0.000000, 0.000000};
MADR R7.xyz, f[TEX0], C2.x, C2.x;
TEX R2, R7, TEX6, 3D;
TEX R9, f[TEX1], TEX1, 2D;
MADR R9.w, C6.x, R2.x, C6.y;
MADR R9.w, f[TEX0].x, C0.w, R9.w;
MADR R9.w, R9.w, C2.y, C2.z;
FRCR R9.w, R9.w;
MADR R9.w, R9.w, C3.x, C3.w;
MULR R9.w, R9.w, R9.w;
MADR R6.w, R9.w, C3.y, C3.z;
MADR R8.w, R9.w, R6.w, C4.x;
MADR R10.w, R9.w, R8.w, C4.w;
MADR R0.w, R9.w, R10.w, C4.y;
MADR R9.w, R9.w, R0.w, C4.z;
LG2R R9.w, |R9.w|;
MULR R9.w, R9.w, C2.x;
EX2R R9.w, R9.w;
MOVR R2.xyz, -C1;
ADDR R4.xyz, R2, C0;
MADR R6.xyz, R4, R9.w, C1;
MULR R1.xyz, R9, R6;
MULR R8.xyz, R1, C5.y;
DP4R R8.w, f[TEX3], f[TEX3];
RSQR R8.w, R8.w;
MULR R3.xyz, R8.w, f[TEX3];
DP3R R8.w, f[TEX4], R3;
MOVRC RC, R8.w;
MOVR R8.w(GE), R8.w;
MOVR R8.w(LT), C5.x;
LG2R R8.w, R8.w;
MULR R8.w, R8.w, C5.w;
EX2R R8.w, R8.w;
MULR R8.w, R8.w, C5.z;
ADDR R8.w, R8.w, C4.z;
MULR R0.xyz, R8, R8.w;
MOVR R0.w, C4.z;
MOVR o[COLR], R0; 
END

# Passes = 29 

# Registers = 11 

# Textures = 4 
