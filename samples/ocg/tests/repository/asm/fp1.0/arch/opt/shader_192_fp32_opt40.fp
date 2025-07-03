!!FP2.0 
DECLARE C0={0.1, 0.2, 0.3, 0.4};
DECLARE C1={4, 3, 2, 1};
DECLARE C2={0.500000, 0.500000, 0.250000, 0.000000};
DECLARE C3={6.283190, -0.000000, 0.000025, -3.141590};
DECLARE C4={-0.001389, -0.500000, 1.000000, 0.041667};
DECLARE C5={0.000000, 1.200000, 0.400000, 20.000000};
DECLARE C6={2.000000, -1.000000, 0.000000, 0.000000};
MADR R0.xyz, f[TEX0], C2.x, C2.x;
TEX R1, R0, TEX6, 3D;
TEX R0, f[TEX1], TEX1, 2D;
MADR R0.w, C6.x, R1.x, C6.y;
MADR R0.w, f[TEX0].x, C0.w, R0.w;
MADR R0.w, R0.w, C2.y, C2.z;
FRCR R0.w, R0.w;
MADR R0.w, R0.w, C3.x, C3.w;
MULR R0.w, R0.w, R0.w;
MADR R1.w, R0.w, C3.y, C3.z;
MADR R1.w, R0.w, R1.w, C4.x;
MADR R1.w, R0.w, R1.w, C4.w;
MADR R1.w, R0.w, R1.w, C4.y;
MADR R0.w, R0.w, R1.w, C4.z;
LG2R R0.w, |R0.w|;
MULR R0.w, R0.w, C2.x;
EX2R R0.w, R0.w;
MOVR R1.xyz, -C1;
ADDR R1.xyz, R1, C0;
MADR R1.xyz, R1, R0.w, C1;
MULR R1.xyz, R0, R1;
MULR R0.xyz, R1, C5.y;
DP4R R0.w, f[TEX3], f[TEX3];
LG2R_d2 R0.w, R0.w;
EX2R R0.w, -R0.w;
MULR R1.xyz, R0.w, f[TEX3];
DP3R R0.w, f[TEX4], R1;
MOVR RC, R0.w;
MOVR R0.w(LT), C5.x;
LG2R R0.w, R0.w;
MULR R0.w, R0.w, C5.w;
EX2R R0.w, R0.w;
MULR R0.w, R0.w, C5.z;
ADDR R0.w, R0.w, C4.z;
MADR R0.xyz, R0, R0.w, {0, 0, 0, 0}.x;
MOVR R0.w, C4.z;
END

# Passes = 27 

# Registers = 2 

# Textures = 4 
