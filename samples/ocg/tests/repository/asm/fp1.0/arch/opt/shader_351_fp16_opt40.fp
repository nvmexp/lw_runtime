!!FP2.0
MOVH H1.xyz, f[TEX1];
MOVH H0.w, f[TEX1];
RCPH H0.w, H0.w;
MULH H1.xyz, H0.w, H1;
TEX H0, f[TEX0], TEX0, 2D;
DP3H H2.x, H1, H0;
DIVH H1.xyz, f[TEX2], f[TEX2].w;
DP3H H2.y, H1, H0;
TEX H1, f[TEX3], TEX3, 2D;
MADH H1.xyz, {1.987654, 0, 0, 0}.x, H1, {1.987654, 0, 0, 0}.y;
TEX H0, H2, TEX2, 2D;
DP3H H0.w, f[TEX4], f[TEX4];
MOVH H2.yzw, f[TEX4].wxyz;
LG2H H0.w, |H0.w|;
MULH H2.x, H0.w, {1.304100, 0, 0, 0}.x;
EX2H H0.w, H2.x;
MULH H2.xyz, H0.w, H2.yzwy;
DP3H H1.x, H2, H1;
SGEH H0.w, H1.x, {0, 0, 0, 0}.x;
ADDH H1.z, H1.x, -{0.561372, 0, 0, 0}.x;
MULH H0.w, H0, H1.z;
ADDH H0.w, {0.368316, 0, 0, 0}.x, H0;
ADDH H1.z, -H0.w, {0.741652, 0, 0, 0}.x;
MULH H0.w, H1.z, H1.z;
MULH H0.w, H0, H0;
MULH H0.w, H1.z, H0;
MULH H0.xyz, H0, H0.w;
MOVH H0.w, H1;
END

# Passes = 17 

# Registers = 2 

# Textures = 5 
