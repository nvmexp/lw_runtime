ps_2_0
def c0, 0.000000, 0.000000, 0.000000, 1.000000
def c1, 0.000000, 0.000000, 0.500000, 0.454545
def c2, 0.058594, 0.000000, 0.000000, 0.000000
def c3, 2.000000, -1.000000, 0.000000, 0.000000
dcl t2.rg
dcl_2d s0
dcl_2d s1
texld r0, t2, s0
texld r7, t2, s1
cmp r0.rgb, r0, r0, c1.r
mov r2.rgb, r0.a
log r5.r, r0.r
log r5.g, r0.g
log r5.b, r0.b
mov r0.rgb, r5
mul r9.rgb, r0, c1.a
exp r4.r, r9.r
exp r4.g, r9.g
exp r4.b, r9.b
mov r4.a, c0.a
mov oC1, r4
mad r6.rgb, c3.r, r7, c3.g
dp3 r6.a, r6, r6
rsq r6.a, r6.a
mul r1.rgb, r6, r6.a
mad r3.rgb, r1, c1.b, c1.b
mov r3.a, c0.a
mov oC3, r3
mov r10, c0
mov oC0, r10
mov r2.a, c2.r
mov oC2, r2
