!!FP1.0
TEX H0, f[TEX0], TEX0, 2D;
DP3H H1.x, f[TEX1], H0;
DP3H H1.y, f[TEX2], H0;
DP3H H1.z, f[TEX3], H0;
RFLH H1, f[TEX4], H1;
TEX H0, H1, TEX6, 3D;
MOVH H0.w, f[COL0].w;
MOVH o[COLH], H0; 
END

# Passes = 10 

# Registers = 1 

# Textures = 4 
