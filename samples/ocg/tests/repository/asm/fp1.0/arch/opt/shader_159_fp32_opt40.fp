!!FP2.0 
DECLARE C0={10, 20, 30, 40};
TEX R2, f[TEX2], TEX2, 2D;
MOVR RC, R2.z;
TEX R0, f[TEX0], TEX0, 2D;
DP3R_SAT R3.x, R2, f[TEX4];
TEX R1, f[TEX3], TEX4, 2D;
MULR R3, R3.x, R0;
DP3R_SAT R0, f[TEX1], f[TEX1];
MOVR R2(GE), C0.x;
ADDR R0.x, {1, 1, 1, 1}, -R0;
MULR R0, R1, R0.x;
MULR R0, R0, R3;
MOVR R2(LT), C0.y;
MULR R0, R0, R2;
MULR_m2 R0, R0, f[COL0];
END

# Passes = 9 

# Registers = 4 

# Textures = 5 
