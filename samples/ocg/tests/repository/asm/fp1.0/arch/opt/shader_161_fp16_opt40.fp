!!FP2.0 
TEX H1, f[TEX1], TEX1, 2D;
DP3H H0.x, H1, f[TEX2];
DP3H H0.y, H1, f[TEX3];
DP3H H0.z, H1, f[TEX4];
MOVH H0.w, f[TEX4].z;
TEX H1.xyz, H0, TEX2, 2D;
TEX H0.xyz, f[TEX0], TEX0, 2D;
MULH H0.xyz, H0, H1;
END

# Passes = 6 

# Registers = 1 

# Textures = 5 
