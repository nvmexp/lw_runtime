!!FP1.0
TEX H1, f[TEX0], TEX1, 2D;
ADDH H1.xyz, H1, {1.987654, 1.788889, 1.610000, 1.449000}.x;
DP3H H0.x, f[TEX2], H1;
DP3H H0.z, f[TEX4], H1;
DP3H H0.y, f[TEX3], H1;
DP3H H0.w, H0, H0;
LG2H H0.w, |H0.w|;
MULH H0.w, H0, {0.5, 0, 0, 0}.x; 
EX2H H0.w, -H0.w;
MULH H4.xyz, H0, H0.w;
MOVH H0.w, {0.855620, 0.770058, 0.693052, 0.623747}.y;
MULH H0.xyz, H4, {0.561372, 0.505235, 0.454712, 0.409240}.x;
ADDH H0.xyz, f[TEX1], H0;
MOVH H4.w, f[COL0].x;
TEX H6.xyz, f[TEX0], TEX0, 2D;
MOVH H6.w, H1.w;
MOVH o[COLH], H0; 
END

# Passes = 14 

# Registers = 4 

# Textures = 5 
