!!FP2.0 
DECLARE C0={0.1, 0.2, 0.3, 0.4};
DECLARE C1={4, 3, 2, 1};
DECLARE C2={0.5, 0.6, 0.7, 0.8};
ADDR R1, C0, -f[TEX2];
DP3R R1.w, R1, R1;
ADDR R1.w, R1.w, C2.y;
DIVR R1, C1, R1.w;
TEX R0, f[TEX0], TEX0, 2D;
ADDR R0, R0, R1;
TEX R1, f[TEX1], TEX1, 2D;
MADR R0, R1.w, -R0, R0;
MADR R0, R1.w, R1, R0;
END

# Passes = 7 

# Registers = 2 

# Textures = 3 
