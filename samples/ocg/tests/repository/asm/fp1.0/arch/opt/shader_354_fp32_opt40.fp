!!FP2.0
DECLARE C0={1, 2, 3, 0};
TEX R0, f[TEX0], TEX0, 2D;
MADR_m2 R0.xyz, C0, R0, C0.w;
TEX R1, f[TEX1], TEX1, 2D;
MULR R0.xyz, R1, R0;
ADDR R0.w, {1, 1, 1, 1}, -R1;
END

# Passes = 2 

# Registers = 2 

# Textures = 2 
