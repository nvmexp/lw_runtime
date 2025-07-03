!!FP1.0 
# Pixelshader 066
# TSS count 2
TXP H0, f[TEX0], TEX0, 2D; # eliminated a MOV
TXP H0.w, f[TEX1], TEX1, 2D; # eliminated a MOV
MOVH o[COLH], H0; 
END

# Passes = 2 

# Registers = 1 

# Textures = 2 
