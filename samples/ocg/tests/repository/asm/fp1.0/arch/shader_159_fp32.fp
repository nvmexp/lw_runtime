!!FP1.0 
DECLARE C0={10, 20, 30, 40};
TEX R2, f[TEX2], TEX2, 2D;
DP3R_SAT R3.x, R2, f[TEX4];
TEX R0, f[TEX0], TEX0, 2D;
TEX R4, f[TEX3], TEX4, 2D;
MULR R3, R3.x, R0;
DP3R_SAT R0, f[TEX1], f[TEX1];
ADDR R0.x, {1, 1, 1, 1}, -R0;
MULR R0, R4, R0.x;
MULR R0, R0, R3;
MOVRC RC, R2.z;
MOVR R1(GE), C0.x;
MOVR R1(LT), C0.y;
MULR R0, R0, R1;
MULR R0, R0, f[COL0];
MULR R0, R0, {2, 0, 0, 0}.x; 
MOVR o[COLR], R0; 
END

# Passes = 10 

# Registers = 5 

# Textures = 5 
