!!FP2.0
TEX H6, f[TEX0], TEX1, 2D;
ADDH H6.xyz, H6, {1.987654, 1.788889, 1.610000, 1.449000}.x;
DP3H H0.x, f[TEX2], H6;
DP3H H0.z, f[TEX4], H6;
DP3H H0.y, f[TEX3], H6;
DP3H H0.w, H0, H0;
MOVH H4.w, f[COL0].x;
LG2H_d2 H0.w, |H0.w|;
TEX H6.xyz, f[TEX0], TEX0, 2D;
EX2H H0.w, -H0.w;
MULH H4.xyz, H0, H0.w;
MOVH H0.w, {0.855620, 0.770058, 0.693052, 0.623747}.y;
MULH H0.xyz, H4, {0.561372, 0.505235, 0.454712, 0.409240}.x;
MOVH H1.xy, f[TEX1];
MOVH H1.w, f[TEX1].xywz;
ADDH H0.xyz, H1.xywz, H0;
END

# Passes = 12 

# Registers = 4 

# Textures = 5 
