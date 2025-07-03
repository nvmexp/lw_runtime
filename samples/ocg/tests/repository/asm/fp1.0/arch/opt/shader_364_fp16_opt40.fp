!!FP2.0
TEX H0, f[TEX0], TEX2, 2D;
TEX H1, f[TEX1], TEX2, 2D;
ADDH H0, H0, H1;
TEX H1, f[TEX2], TEX2, 2D;
ADDH H1, H1, H0;
TEX H0, f[TEX3], TEX2, 2D;
ADDH H0, H0, H1;
TEX H2, f[TEX0], TEX0, 2D;
MULH H1, H0, {0.98, 0, 0, 0}.x;
TEX H0, f[TEX1], TEX0, 2D;
ADDH H0.w, H2, H0;
TEX H2, f[TEX2], TEX0, 2D;
ADDH H2.w, H2, H0;
TEX H0, f[TEX3], TEX0, 2D;
ADDH H0.w, H0, H2;
TEX H2, f[TEX0], TEX1, 2D;
MADH H1, H0.w, {0.3, 0, 0, 0}.x, H1;
TEX H0, f[TEX1], TEX1, 2D;
ADDH H0, H2, H0;
TEX H2, f[TEX2], TEX1, 2D;
ADDH H2, H2, H0;
TEX H0, f[TEX3], TEX1, 2D;
ADDH H0, H0, H2;
MULH H0, H0, H1;
ADDH H0, H0.w, H0;
ADDH H0, H0, {-9.36, 0, 0, 0}.x;
END

# Passes = 14 

# Registers = 2 

# Textures = 4 
