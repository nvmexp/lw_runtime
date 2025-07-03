!!FP2.0 
TEX H0, f[TEX0], TEX0, 2D;
TEX H1, f[TEX1], TEX1, 2D;
ADDH H2, H1, -H0;
TEX H1, H1.wxxx, TEX2, 2D;
MADH H0, H1, H2, H0;
END

# Passes = 3 

# Registers = 2 

# Textures = 2 
