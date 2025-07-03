!!FP1.0
TEX H0, f[TEX0], TEX0, 2D;
MULH H0.w, H0, {1.987654, 1.788889, 1.610000, 1.449000}.x;
MULH H0, H0, f[COL0];
MULH H0.xyz, H0, {1.304100, 1.173690, 1.056321, 0.950689};
MOVH o[COLH], H0; 
END

# Passes = 4 

# Registers = 2 

# Textures = 1 
