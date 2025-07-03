!!FP2.0 
DECLARE C0={0.1, 0.2, 0.3, 0.4};
TEX R0, f[TEX0], TEX0, 2D;
DP3R R1.x, f[TEX1], R0;
DP3R R1.y, f[TEX2], R0;
DP3R R1.z, f[TEX3], R0;
MOVR R2.xyz, f[15];
DP3R_m2 R0.x, R1, R2;
MOVR R0.w, f[TEX0].w;
DP3R R2.w, R2, R2;
DIVR R0.x, R0.x, R2.w;
MADR R1.xyz, R0.x, R2, -R1;
TEX R1.xyz, R1, TEX6, 3D;
MULR R0.xyz, R1, C0;
END

# Passes = 8 

# Registers = 3 

# Textures = 4 
