!!FP2.0
TEX R0, f[TEX0], TEX2, 2D;
TEX R1, f[TEX1], TEX2, 2D;
ADDR R1.w, R0, R1;
TEX R0, f[TEX2], TEX2, 2D;
ADDR R1.w, R0, R1;
TEX R0, f[TEX3], TEX2, 2D;
ADDR R1.w, R0, R1;
TEX R0, f[TEX4], TEX2, 2D;
ADDR R0.w, R0, R1;
MADH_SAT R0.w, R0, {1.487654, 0, 0, 0}.x, -f[COL0].w;
ADDR R0.w, -R0, {1.304100, 0, 0, 0}.x;
ADDR R0.z, -R0.w, {0.855620, 0, 0, 0}.x;
MADR H0, {0.561372, 0, 0, 0}, R0.z, R0.w;
END

# Passes = 9 

# Registers = 2 

# Textures = 5 
