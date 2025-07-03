!!FP2.0
TEX R2, f[TEX0], TEX3, 2D;
MOVR R2.w, {1.000000, 0.000000, 0.000000, 0.000000}.x;
DIVR R0.xyz, f[TEX1], f[TEX1].w;
MULR R3.xyz, R2.y, f[TEX6];
TEX R1, f[TEX0], TEX0, 2D;
DP3R R0.x, R0, R1;
MADR R3.xyz, f[TEX5], R2.x, R3;
MADR R2.xyz, f[TEX7], R2.z, R3;
DP3R R0.y, R2, f[TEX4];
TEX R4, f[TEX0], TEX5, 2D;
DP3R R2.z, R2, R2;
RCPR R1.w, R2.z;
MULR R1.w, R1, R0.y;
ADDR R1.w, R1, R1;
MADR R0.yz, R1.w, R2.xxyy, -f[TEX4].xxyy;
RCPR R1.w, f[TEX2].w;
TEX R2.xyz, R0.yzyy, TEX4, 2D;
MULR R0.yzw, R0.w, R2.xxyz;
MULR R2.xyz, R0.yzwy, {1.000000, 0.8900000, 0.7600000, 0.550000};
MADR R3.xyz, R2, R2, -R2;
MULR R0.yzw, R1.w, f[TEX2].wxyz;
DP3R R0.y, R0.yzwy, R1;
TEX R0, R0, TEX2, 2D;
MADR R1.xyz, {0.900000, 0.450000, 0.560000, 0.880000}, R3, R2;
MULR R0.xyz, R0, {0.5640000, 0.780000, 0.640000, 0.77000};
MULR R2.xyz, R4, R0;
DP3R R0.x, R1, {0.333333, 0.000000, 0.000000, 0.000000}.x;
MADR R0.xyz, {0.8800000, 0, 0, 0}, -R0.x, R0.x;
MADR R0.xyz, {0.8800000, 0, 0, 0}, R1, R0;
MADR H0.xyz, {0.910000, 0.000000, 0.000000, 0.000000}.x, R2, R0;
MOVR H0.w, R2.w;
END

# Passes = -1 

# Registers = -1 

# Textures = -1 
