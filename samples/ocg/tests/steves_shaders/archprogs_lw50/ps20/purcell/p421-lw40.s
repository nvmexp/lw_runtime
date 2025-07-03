ps_2_0
dcl t0
dcl t1
dcl t2
dcl t3
dcl t4
dcl_2d s0
dcl_2d s1
dcl_2d s2
dcl_2d s3
dcl_lwbe s4
dcl_2d s5
texld r0, t0, s3
texld r1, t3, s3
texld r2, t0, s0
mad r2, r2, c1.r, c1.g
rcp r3.a, t1.a
mul r3, r3.a, t1
mad r3, r2, c0, r3
mad r0, r0, c1.r, c1.g
mad r1, r1, c1.r, c1.g
add r0, r0, r1
mul r0, r0, c2.r
dp3 r1.r, r0, t2
add r2.r, r1.r, r1.r
mad r0, r2.r, r0, -t2
add r4.r, c1.a, -r1
pow r4.r, r4.r, c1.b
texld r0, r0, s4
texld r1, r3, s1
texld r2, r3, s2
texld r3, t4, s5
mul r4.r, r4.r, r3.a
add r1, r0, r1
lrp r0, r4.r, r1, r2
mul r0, r0, r3
mov oC0, r0
