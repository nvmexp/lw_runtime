ps_2_0
dcl_2d s0
dcl_2d s1
dcl t0
dcl t1
dcl t2
texld r0, t0, s0
texld r1, t1, s1
add r2, c0, -t2
dp3 r2.a, r2, r2
add r2.a, r2.a, c2.g
rcp r2.a, r2.a
mul r2, c1, r2.a
add r0, r0, r2
lrp r2, r1.a, r1, r0
mov oC0, r2
