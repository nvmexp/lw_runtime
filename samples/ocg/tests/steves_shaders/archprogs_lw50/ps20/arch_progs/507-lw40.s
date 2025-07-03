//;; Id: 507   pixel count: 5327244 lw40 ppc: 4.0
ps_2_0
def c0, 16.000000, 0.062500, 0.000000, 0.000000
def c1, 1.000000, 0.000000, 0.000000, 0.000000
def c2, 0.300000, 0.590000, 0.110000, 0.000000
dcl t0.rg
dcl_pp t3.rg
dcl v0
dcl_2d s0
dcl_2d s1
texld r1, t3, s1
texld r0, t0, s0
mul r1.rgb, r1.a, r1
mul r1.rgb, r1, c0.r
mul r0.rgb, r0, v0
mul r0.rgb, r1, r0
mul r0.a, r0.a, v0.a
dp3 r0.r, r0, c2
mul r0.a, r0.a, r0.r
mov r1.gb, c1.brga
mul r0.a, r0.a, c0.g
mov r0.rg, r1.gbra
mov r0.b, r1.g
mov oC0, r0
