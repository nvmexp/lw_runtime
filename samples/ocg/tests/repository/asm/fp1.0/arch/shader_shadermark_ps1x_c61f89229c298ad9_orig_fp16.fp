!!FP1.0
MOVH H0.xyz, f[TEX1];
MOVH H2.xyz, f[TEX3];
TEX H1, f[TEX0], TEX0, 2D;
MADH H3.xyz, H1, {2, -1, 0, 0}.x, {2, -1, 0, 0}.y;
MOVH H1.xyz, f[TEX2];
DP3H H0.x, H0, H3;
MADH H3.xyz, H3, {2, -1, 0, 0}.x, {2, -1, 0, 0}.y;
DP3H H0.y, H1, H3;
TEX H1, f[TEX4], TEX1, 2D;
MADH H3.xyz, H3, {2, -1, 0, 0}.x, {2, -1, 0, 0}.y;
DP3H H0.z, H2, H3;
DP3H H0.w, H0, H0;
MADH H2.xyz, H1, {2, -1, 0, 0}.x, {2, -1, 0, 0}.y;
DP3H H0.z, H0, H2;
MULH H0.z, H0, {2, 2, 2, 2}.x;
TEX H2, H0, TEX4, 2D;
MULH H2.xyz, H2, {2, 2, 2, 2}.x;
MADH H1.zw, -H1.xxxy, {2, -1, 0, 0}.x, {2, -1, 0, 0}.y;
MULH H1.xy, H0, H0.z;
MADH H0.xy, -H1.zwzz, H0.w, H1;
MOVH H0.zw, f[TEX5].xxxy;
TEX H1, f[TEX0], TEX5, 2D;
ADDH H0.xy, H0, H0.zwzz;
TEX H0, H0, TEX2, 2D;
MULH H0.xyz, H1, H0;
TEX H1, f[TEX0], TEX3, 2D;
MADH H0.xyz, H1, H2, H0;
MOVH H1.xyz, H0;
MOVH H0, H1;
MOVH o[COLH], H0; 
END

# Passes = -1 

# Registers = -1 

# Textures = -1 
