!!FP2.0
DECLARE C0={0.1, 0.2, 0.3, 0.4};
TEX H0, f[TEX0], TEX0, 2D;
DP3H H1.x, f[TEX1], H0;
DP3H H1.y, f[TEX2], H0;
DP3H H1.z, f[TEX3], H0;
MOVH H2.xyz, f[15];
DP3H_m2 H0.x, H1, H2;
MOVH H0.w, f[TEX0].w;
DP3H H2.w, H2, H2;
DIVH H0.x, H0.x, H2.w;
MADH H1.xyz, H0.x, H2, -H1;
TEX H1.xyz, H1, TEX6, 3D;
MULH H0.xyz, H1, C0;
END

# Passes = 8 

# Registers = 2 

# Textures = 4 
