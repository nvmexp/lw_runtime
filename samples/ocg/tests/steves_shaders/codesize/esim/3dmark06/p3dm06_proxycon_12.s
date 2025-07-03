ps_3_0

def c5, 0.00000000, 1.00000000, 2.00000000, -1.00000000 ; 0x000000 0x3f800000 0x40000000 0xbf800000
def c6, 16.00000000, 0.00000000, 0.00000000, 0.00000000 ; 0x41800000 0x000000 0x000000 0x000000
dcl_texcoord0 v0.rg
dcl_texcoord5_pp v1.rgb
dcl_texcoord6 v2.rgb
dcl_texcoord7 v3.rgb
dcl_2d s0
dcl_2d s1
dcl_2d s2
dcl_2d s3
cmp r0.w, -v2.z, c5.x, c5.y
if_ne r0.w, -r0.w
dp3_pp r0.w, v1, v1
rsq_pp r0.z, r0.w
nrm_pp r5.xyz, v2
dp3 r0.w, v3, v3
mad_pp r0.xyz, v1, r0.z, r5
add_sat_pp r1.w, -r0.w, c5.y
nrm_pp r1.xyz, r0
texld r0, v0.xyx, s2
mad_pp r4.xyz, c5.z, r0.wyzw, c5.w
mov r0, c0
mul r0, r0, c1.x
dp3_pp r2.x, r4, r1
mul_pp r0, r1.w, r0
texld_pp r3, v0.xyx, s1
mul_pp r2.w, r3.w, c4.x
mov_pp r2.yz, c5.x
texldl_pp r1, r2.xwzy, s3
mul_pp r1, r0, r1.x
mul_pp r2.xyz, r3, c3
dp3_pp r3.w, r4, r5
mul_pp r1, r2, r1
mul_sat_pp r2.w, r3.w, c6.x
mul_pp r2, r1, r2.w
mov_sat_pp r3.w, r3.w
texld_pp r1, v0.xyx, s0
mul_pp r1, r1, c2
mul_pp r0, r0, r3.w
mad oC0, r0, r1, r2
else
mov oC0, c5.x
endif
;Auto options added
;#PASM_OPTS: -srcalpha 0 -fog 0 -signtex fff7 -texrange 7f
