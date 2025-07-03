!!FP1.0
DECLARE CCONST0={0.34, 0.45, 0.55, 0.66};
DECLARE CCONST01={0.45, 0.66, 0.55, 0.77};
DECLARE CCONST02={0.32, 0.33, 0.33, 0.55};
DECLARE CCONST03={0.66, 0.55, 0.65, 0.65};
DECLARE CCONST04={0.76, 0.56, 0.57, 0.34};
DECLARE CCONST06={0.34, 0.45, 0.56, 0.67};
TEX H0, f[TEX0], TEX0, 2D;
MADH H2.xyz, f[COL0], {2, -1, 0, 0}.x, {2, -1, 0, 0}.y;
MADH H3.xyz, H0, {2, -1, 0, 0}.x, {2, -1, 0, 0}.y;
DP3H_SAT H2.w, H2, H3;
DP3H H2.x, f[TEX1], H3;
DP3H H2.y, f[TEX2], H3;
MOVH H0.x, H0.w;
MOVH H0.w, H0.x;
MOVH H0.xyz, f[TEX3];
DP3H H2.z, H0, H3;
DP3H H0.x, H2, H2;
TEX H1, f[TEX4], TEX4, 2D;
MADH H3, H1, {2, -1, 0, 0}.x, {2, -1, 0, 0}.y;
DP3H H0.y, H2, H3;
MULH H1.xyz, H0.y, {2, 2, 2, 2};
MULH H1.xyz, H2, H1;
MADH H0.xyz, -H3, H0.x, H1;
TEX H1, H0, TEX3, 2D;
MULH H0.xyz, H1, CCONST0;
MULH H2.xyz, H0, H0;
ADDH H3, H2, -H0;
MADH H0.xyz, CCONST01, H3, H0;
DP3H H2.x, H0, CCONST03;
ADDH H1.xyz, H0, -H2.x;
MADH H1.xyz, CCONST02, H1, H2.x;
ADDH H3.w, {1, 1, 1, 1}, -H2;
MULH H1.w, H3, H1;
MULH H1.w, H1, H1;
MULH H1.w, H1, H3.w;
MOVH H0.x, CCONST04.w;
MADH H1.w, H1, CCONST06, H0.x;
MULH H0.xyz, H1, H1.w;
MOVH H0.w, H0.w;
MOVH o[COLH], H0; 
END

# Passes = 22 

# Registers = 2 

# Textures = 5 
