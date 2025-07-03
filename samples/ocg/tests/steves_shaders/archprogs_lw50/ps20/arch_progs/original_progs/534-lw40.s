;; Id: 534   pixel count: 907812 lw40 ppc: 0.8
ps_2_0
def c4, 2.000000, -1.000000, -0.000977, 0.001953
def c6, 0.001953, -0.000977, 0.222222, 0.444444
def c7, -0.000977, 0.001953, 0.111111, 0.333333
dcl t0.rg
dcl t1.rgb
dcl t2.rgb
dcl t3.rgb
dcl t4.rgb
dcl t5.rgb
dcl_2d s2
dcl_2d s3
dcl_lwbe s4
dcl_2d s5
texld r0, t0, s3
mad r1.rgb, c4.r, r0, c4.g
mul r0.rgb, r1.g, t3
mad r0.rgb, t2, r1.r, r0
mad r0.rgb, t4, r1.b, r0
dp3 r3.r, r0, t1
dp3 r2.r, r0, r0
add r1.a, r3.r, r3.r
mul r2.rgb, r2.r, t1
mad r0.rgb, r1.a, r0, -r2
rcp r1.a, t5.b
mul r2.rg, r1.a, t5
mul r1.a, r0.a, c5.r
mad r1.rg, r1, r1.a, r2
add r4.rg, r1, c6
add r3.rg, r1, c4.b
add r2.rg, r1, c7
add r1.rg, r1, c4.a
texld r6, r0, s4
texld r5, t0, s5
texld r4, r4, s2
texld r3, r3, s2
texld r2, r2, s2
texld r1, r1, s2
mul r0.rgb, r0.a, r6
mul r6.rgb, r0, c0
mad r0.rgb, r6, r6, -r6
mad r0.rgb, c2, r0, r6
dp3 r7.r, r0, c7.a
lrp r6.rgb, c3, r0, r7.r
add r0.rgb, c1, c1
mul r5.rgb, r5, r0
mul r0.rgb, r4, c6.b
mad r0.rgb, r3, c6.a, r0
mad r0.rgb, r2, c6.b, r0
mad r0.rgb, r1, c7.b, r0
mad r0.rgb, r0, r5, r6
mov oC0, r0
