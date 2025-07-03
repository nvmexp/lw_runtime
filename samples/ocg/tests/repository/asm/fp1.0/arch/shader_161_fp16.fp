!!FP1.0 
TEX H0, f[TEX0], TEX0, 2D;
TEX H1, f[TEX1], TEX1, 2D;
DP3H H2.x, H1, f[TEX2];
DP3H H2.y, H1, f[TEX3];
DP3H H2.z, H1, f[TEX4];
TEX H2, H2, TEX2, 2D;
MULH H0.xyz, H0, H2;
MOVH H0.w, f[TEX4].z;
MOVH o[COLH], H0; 
END

# Passes = 7 

# Registers = 2 

# Textures = 5 
