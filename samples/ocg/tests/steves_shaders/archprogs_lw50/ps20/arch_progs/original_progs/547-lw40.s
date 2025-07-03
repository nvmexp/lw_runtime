;; Id: 547   pixel count: 12134714 lw40 ppc: 0.888888888889
ps_2_0
def c1, 0.816497, 0.000000, 0.577350, 0.062500
def c2, -0.408248, 0.707107, 0.577350, 0.000000
def c3, -0.408248, -0.707107, 0.577350, 0.000000
def c5, 2.000000, -1.000000, 16.000000, 1.000000
def c6, 0.300000, 0.590000, 0.110000, 0.000000
dcl t0.rg
dcl t1.rg
dcl_pp t2
dcl_pp t3
dcl t4.rgb
dcl t5.rgb
dcl t6.rgb
dcl t7.rgb
dcl v0.rgb
dcl_2d s0
dcl_2d s1
dcl_lwbe s2
dcl_2d s4
texld r4, t1, s4
nrm r1.rgb, t4
mad r6.rgb, c5.r, r4, c5.g
dp3 r0.r, r6, t5
dp3 r0.g, r6, t6
dp3 r0.b, r6, t7
dp3 r2.r, r0, r0
dp3 r3.r, r0, r1
mul r1.rgb, r1, r2.r
add r0.a, r3.r, r3.r
mad r0.rgb, r0.a, r0, -r1
mov r2.r, t2.b
mov r2.g, t2.a
add r6.a, -r3.r, c5.a
mov r1.r, t3.b
mov r1.g, t3.a
texld r5, r0, s2
texld r3, r2, s1
texld r2, t2, s1
texld r1, r1, s1
texld r0, t0, s0
mul r0.a, r6.a, r6.a
mul r0.a, r0.a, r0.a
mul r0.a, r6.a, r0.a
mad r0.a, r0.a, c4.r, c4.g
mul r5, r5, r0.a
mul r4.rgb, r5.a, r5
mul r4.rgb, r4.a, r4
mul r4.rgb, r4, c0
mul r4.rgb, r4, c5.b
mul r3.rgb, r3.a, r3
mul r3.rgb, r3, c5.b
dp3_sat r5.r, r6, c2
mul r3.rgb, r3, r5.r
mul r2.rgb, r2.a, r2
mul r2.rgb, r2, c5.b
dp3_sat r7.r, r6, c1
dp3_sat r5.r, r6, c3
mad r2.rgb, r7.r, r2, r3
mul r1.rgb, r1.a, r1
mul r1.rgb, r1, c5.b
mad r1.rgb, r5.r, r1, r2
mul r0.rgb, r0, v0
mad r0.rgb, r1, r0, r4
dp3 r1.r, r0, c6
mul r0.a, r1.r, c1.a
mov oC0, r0
