//;; Id: 575   pixel count: 1336726 lw40 ppc: 1.06666666667
ps_2_0
def c2, 2.000000, -1.000000, 0.000000, 1.000000
dcl t0.rg
dcl t1.rgb
dcl t2.rgb
dcl t3.rgb
dcl t4.rgb
dcl_lwbe s0
dcl_2d s1
texld r0, t0, s1
dp3 r3.r, t1, t1
rsq r0.a, r3.r
mul r1.rgb, r0.a, t1
mad r2.rgb, c2.r, r0, c2.g
dp3 r0.r, r2, t2
dp3 r0.g, r2, t3
dp3 r0.b, r2, t4
dp3 r2.r, r0, r0
rsq r4.a, r3.r
mul r1.a, r4.a, r3.r
dp3 r3.r, r0, r1
mul r1.rgb, r1, r2.r
add r0.a, r3.r, r3.r
max r2.a, r3.r, c2.b
mad r0.rgb, r0.a, r0, -r1
texld r0, r0, s0
add r0.a, -r2.a, c2.a
mul r2.a, r0.a, r0.a
mul r2.a, r2.a, r2.a
mul r0.a, r0.a, r2.a
mad r0.rgb, r0, r0.a, c0
mad r0.a, r1.a, c1.b, -c1.a
mov oC0, r0
