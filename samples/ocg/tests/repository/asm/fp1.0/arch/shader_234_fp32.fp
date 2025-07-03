!!FP1.0
TEX R0, f[TEX0], TEX0, 2D;
DP3R R1.x, f[TEX1], R0;
DP3R R1.y, f[TEX2], R0;
DP3R R1.z, f[TEX3], R0;
RFLR R1, f[TEX4], R1;
TEX R0, R1, TEX6, 3D;
MOVR R0.w, f[COL0].w;
MOVR o[COLR], R0; 
END

# Passes = 10 

# Registers = 2 

# Textures = 4 
