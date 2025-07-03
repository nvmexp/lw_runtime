!!FP2.0 
TEX R0, f[TEX0], TEX0, 2D;
DP3R R1.x, f[TEX1], R0;
DP3R R1.y, f[TEX2], R0;
DP3R R1.z, f[TEX3], R0;
TEX R1, R1, TEX6, 3D;
MULR R0, R1, f[COL0];
END

# Passes = 6 

# Registers = 2 

# Textures = 4 
