!!FP1.0 
TEX H0, f[TEX0], TEX0, 2D;
TEX H1, f[TEX1], TEX1, 2D;
TEX H2, H1.wxxx, TEX2, 2D;
MADH H3, H2, -H0, H0;
MADH H0, H2, H1, H3;
MOVH o[COLH], H0; 
END

# Passes = 4 

# Registers = 2 

# Textures = 2 
