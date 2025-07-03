!!FP1.0
RCPH H1.w, f[TEX1].w;
MOVH H3.w, {1.000000, 0.000000, 0.000000, 0.000000}.x;
TEX H2, f[TEX0], TEX3, 2D;
MULH H0.xyz, H1.w, f[TEX1];
MULH H3.xyz, H2.y, f[TEX6];
TEX H1, f[TEX0], TEX0, 2D;
DP3H H0.x, H0, H1;
TEX H4, f[TEX0], TEX5, 2D;
MADH H3.xyz, f[TEX5], H2.x, H3;
MADH H2.xyz, f[TEX7], H2.z, H3;
DP3H H0.y, H2, f[TEX4];
DP3H H2.z, H2, H2;
RCPH H1.w, H2.z;
MULH H1.w, H1, H0.y;
ADDH H1.w, H1, H1;
MADH H0.yz, H1.w, H2.xxyy, -f[TEX4].xxyy;
RCPH H1.w, f[TEX2].w;
TEX H2, H0.yzyy, TEX4, 2D;
MULH H0.yzw, H0.w, H2.xxyz;
MULH H2.xyz, H0.yzwy, {1.000000, 0.8900000, 0.7600000, 0.550000};
MADH H3.xyz, H2, H2, -H2;
MULH H0.yzw, H1.w, f[TEX2].wxyz;
DP3H H0.y, H0.yzwy, H1;
TEX H0, H0, TEX2, 2D;
MADH H1.xyz, {0.900000, 0.450000, 0.560000, 0.880000}, H3, H2;
MULH H0.xyz, H0, {0.5640000, 0.780000, 0.640000, 0.77000};
MULH H2.xyz, H4, H0;
DP3H H0.x, H1, {0.333333, 0.000000, 0.000000, 0.000000}.x;
MADH H0.xyz, {0.8800000, 0, 0, 0}, -H0.x, H0.x;
MADH H0.xyz, {0.8800000, 0, 0, 0}, H1, H0;
MADH H0.xyz, {0.910000, 0.000000, 0.000000, 0.000000}.x, H2, H0;
MOVH H0.w, H3.w;
MOVH o[COLH], H0; 
END

# Passes = -1 

# Registers = -1 

# Textures = -1 
