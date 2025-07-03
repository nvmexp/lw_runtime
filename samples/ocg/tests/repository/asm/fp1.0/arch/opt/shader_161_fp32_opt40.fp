!!FP2.0 
TEX R1, f[TEX1], TEX1, 2D;
DP3R R0.x, R1, f[TEX2];
DP3R R0.y, R1, f[TEX3];
MOVR R0.w, f[TEX4].z;
DP3R R0.z, R1, f[TEX4];
TEX R1.xyz, R0, TEX2, 2D;
TEX R0.xyz, f[TEX0], TEX0, 2D;
MULR R0.xyz, R0, R1;
END

# Passes = 6 

# Registers = 2 

# Textures = 5 
