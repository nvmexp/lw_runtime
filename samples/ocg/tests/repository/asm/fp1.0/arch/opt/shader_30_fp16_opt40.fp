!!FP2.0 
DECLARE C0={0.1, 0.2, 0.3, 0.4};
DECLARE C2={0.5, 0.6, 0.7, 0.8};
TEX H1, f[TEX0], TEX0, 2D;
MULH H0, H1, f[COL0];
TEX H1, f[TEX3], TEX3, 2D;
ADDH H1.w, {1, 1, 1, 1}, -H1;
TEX H1.xyz, f[TEX2], TEX2, 2D;
MULH H1, H1, H1.w;
TEX H2, f[TEX1], TEX1, 2D;
MADH H0.xyz, H1, C2, H0;
MULH H0.xyz, H2, H0;
MULH_m2 H0.xyz, C0, H0;
END

# Passes = 6 

# Registers = 2 

# Textures = 4 
