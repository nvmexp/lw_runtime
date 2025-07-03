ps_3_0
def c5, 2.00000000, -1.00000000, 16.00000000, 0.00000000 ; 0x40000000 0xbf800000 0x41800000 0x80000000
dcl_texcoord0 v0.rg
dcl_texcoord1 v1.rg
dcl_texcoord2_pp v2.rgb
dcl_texcoord3_pp v3.rgb
dcl_texcoord4_pp v4.rgb
dcl_texcoord5_pp v5.rgb
dcl_texcoord6 v6.rgb
dcl_texcoord7 v7
dcl_2d s0
dcl_2d s1
dcl_2d s2
dcl_2d s3
dcl_2d s4
dcl_lwbe s5
dcl_2d s6
texldp_pp r0, v7, s0
mov r1.xyz, c0
mul r1.xyz, r1, c1.x
mul_pp r3.xyz, r0.x, r1
mov r0.zw, c5.w
texld_pp r1, v0.xyx, s2
dp3_pp r0.y, v5, v5
rsq_pp r0.y, r0.y
nrm_pp r4.xyz, v6
mad_pp r2.xyz, v5, r0.y, r4
mul_pp r0.y, r1.w, c4.x
nrm_pp r5.xyz, r2
texld r2, v0.xyx, s3
mad_pp r2.xyz, c5.x, r2.wyzw, c5.y
mul_pp r1.xyz, r1, c3
dp3_pp r0.x, r2, r5
texldl_pp r0, r0, s6
mul_pp r0.xyz, r3, r0.x
dp3_pp r0.w, r2, r4
mul_pp r5.xyz, r1, r0
mul_sat_pp r3.w, r0.w, c5.z
mov_sat_pp r2.w, r0.w
dp3_pp r0.x, r2, v2
dp3_pp r0.y, r2, v3
dp3_pp r0.z, r2, v4
texld_pp r1, r0, s5
texld_pp r0, v1.xyx, s4
mul_pp r4.xyz, r3, r2.w
texld_pp r2, v0.xyx, s1
mul_pp r3.xyz, r2, c2
mul_pp r4.xyz, r4, r3
mul_pp r2.xyz, r1, r0
mul_pp r4.xyz, r0, r4
mul_pp r1.xyz, r5, r3.w
mad r2.xyz, r2, r3, r4
mad oC0.xyz, r1, r0, r2
mov oC0.w, c5.w
