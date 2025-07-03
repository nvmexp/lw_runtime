!!FP2.0 
TEX R0, f[TEX0], TEX0, 2D;
DP3R R1.x, f[TEX1], R0;
DP3R R1.y, f[TEX2], R0;
DP3R R1.z, f[TEX3], R0;
MOVR R0.xyz, f[15];
DP3R R1.w, R0, R0;
DP3R R0.w, R0, R1;
DIVR_m2 R0.w, R0.w, R1.w;
MADR R1.xyz, R0.w, R0, -R1;
TEX R0, R1, TEX6, 3D;
MOVR R0.w, f[COL0].w;
END

# Passes = 9 

# Registers = 2 

# Textures = 4 
