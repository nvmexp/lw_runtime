!!FP2.0
TEX H0, f[TEX0], TEX0, 2D;
MULH H1.xyz, f[COL0], {1.987654, 1.788889, 1.610000, 1.449000}.x;
MULH H1.xyz, H0, H1;
MULH H1.xyz, H1, {0.855620, 0.770058, 0.693052, 0.623747}.x;
MOVH H1.w, {1.987654, 1.788889, 1.610000, 1.449000}.w;
ADDH H1.xyz, H1, H1;
MADH H0.xyz, {0.561372, 0.505235, 0.454712, 0.409240}.x, H0, -H1;
MADH H0.xyz, H0.w, H0, H1;
MOVH H0.w, H1;
END

# Passes = 5 

# Registers = 2 

# Textures = 1 
