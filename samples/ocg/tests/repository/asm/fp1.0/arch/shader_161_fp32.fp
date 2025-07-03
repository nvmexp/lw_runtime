!!FP1.0 
TEX R0, f[TEX0], TEX0, 2D;
TEX R1, f[TEX1], TEX1, 2D;
DP3R R2.x, R1, f[TEX2];
DP3R R2.y, R1, f[TEX3];
DP3R R2.z, R1, f[TEX4];
TEX R2, R2, TEX2, 2D;
MULR R0.xyz, R0, R2;
MOVR R0.w, f[TEX4].z;
MOVR o[COLR], R0; 
END

# Passes = 7 

# Registers = 3 

# Textures = 5 
