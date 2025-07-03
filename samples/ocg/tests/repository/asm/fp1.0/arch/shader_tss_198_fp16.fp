!!FP1.0 
# Pixelshader 198
# TSS count 2
TXP H1, f[TEX0], TEX0, LWBE;
MOVH H0.xyz, H1;
MOVH H0.w, f[COL0];
TXP H0.w, f[TEX1], TEX1, 2D; # eliminated a MOV
MOVH o[COLH], H0; 
END

# Passes = 3 

# Registers = 1 

# Textures = 2 
