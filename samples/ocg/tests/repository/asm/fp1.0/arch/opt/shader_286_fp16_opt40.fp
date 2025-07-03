!!FP1.0 
DEFINE C0={0.7, 0.7, 0.7, 0.7};
TEX H0, f[TEX0], TEX0, 2D;
MULH H0.w, f[COL0].w, H0.w;
MOVH H0.xyz, C0;
END

# Passes = 1 

# Registers = 1 

# Textures = 1 
