!!FP1.0 
DEFINE C0={0.7, 0.7, 0.7, 0.7};
TEX R0, f[TEX0], TEX0, 2D;
MULR R0.w, f[COL0].w, R0.w;
MOVR R0.xyz, C0;
END

# Passes = 1 

# Registers = 1 

# Textures = 1 
