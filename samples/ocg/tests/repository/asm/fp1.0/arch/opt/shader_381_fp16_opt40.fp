!!FP2.0
TEX H0, f[TEX1], TEX3, 2D;
DP3H_SAT H1.w, H0, {1.304100, 1.173690, 1.056321, 0.950689};
DP3H_SAT H2.x, H0, {0.855620, 0.770058, 0.693052, 0.623747};
DP3H_SAT H0.x, H0, {0.561372, 0.505235, 0.454712, 0.409240};
MULH H1.xyz, H0.x, f[COL1];
TEX H0, f[TEX0], TEX0, 2D;
MADH H1.xyz, H2.x, f[COL0], H1;
MADH H1.xyz, H1.w, f[TEX7], H1;
MULH H1.xyz, H1, {0.368316, 0, 0, 0}.x;
MULH H0.w, H0, {0.541652, 0, 0, 0}.x;
MULH H0.xyz, H0, H1;
MADH_m2 H0.xyz, H0, {0.758548, 0, 0, 0}.x, {0.758548, 0, 0, 0}.w;
MOVH H0.w, H0;
END

# Passes = 8 

# Registers = 2 

# Textures = 3 
