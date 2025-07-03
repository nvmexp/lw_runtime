!!FP2.0
TEX H0, f[TEX0], TEX0, 2D;
DP3H H1.x, f[TEX1], H0;
DP3H H1.y, f[TEX2], H0;
DP3H H1.z, f[TEX3], H0;
MOVH H0.xyz, f[15];
DP3H H1.w, H0, H0;
DP3H H0.w, H0, H1;
DIVH_m2 H0.w, H0.w, H1.w;
MADH H1.xyz, H0.w, H0, -H1;
TEX H0, H1, TEX6, 3D;
MOVH H0.w, f[COL0].w;
END

# Passes = 9 

# Registers = 1 

# Textures = 4 
