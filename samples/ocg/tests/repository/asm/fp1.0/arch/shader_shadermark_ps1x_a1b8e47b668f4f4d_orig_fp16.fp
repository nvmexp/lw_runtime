!!FP1.0
MOVH H0.xyz, f[TEX0];
MOVH H3.xyz, f[TEX1];
MOVH H2.xyz, f[TEX3];
TEX H1, f[TEX4], TEX1, 2D;
MADH H1.xyz, H1, {2, -1, 0, 0}.x, {2, -1, 0, 0}.y;
DP3H H0.x, H0, H1;
MADH H4.xyz, H1, {2, -1, 0, 0}.x, {2, -1, 0, 0}.y;
MOVH H1.xyz, f[TEX1];
DP3H H0.y, H3, H4;
MADH H3.xyz, H4, {2, -1, 0, 0}.x, {2, -1, 0, 0}.y;
DP3H H0.z, H2, H3;
DP3H H0.x, H0, H1;
MULH H0.x, H0, H0;
TEX H0, H0.x, TEX5, 2D;
MOVH H0.w, H0.y;
MULH H0.xyz, {0.1, 0.5, 0.3, 1.0}, H0;
MOVH o[COLH], H0; 
END

# Passes = -1 

# Registers = -1 

# Textures = -1 
