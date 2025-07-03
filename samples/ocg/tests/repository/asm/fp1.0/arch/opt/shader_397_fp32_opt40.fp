!!FP2.0
TEX R0, f[TEX0], TEX0, 2D;
MULR R1.xyz, f[COL0], {1.987654, 1.788889, 1.610000, 1.449000}.x;
MULR R1.xyz, R0, R1;
MULR R1.xyz, R1, {0.855620, 0.770058, 0.693052, 0.623747}.x;
MOVR R1.w, {1.987654, 1.788889, 1.610000, 1.449000}.w;
ADDR R1.xyz, R1, R1;
MADR R0.xyz, {0.561372, 0.505235, 0.454712, 0.409240}.x, R0, -R1;
MADR H0.xyz, R0.w, R0, R1;
MOVR H0.w, R1;
END

# Passes = 5 

# Registers = 2 

# Textures = 1 
