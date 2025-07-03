vs_3_0
def c9, 2.00000000, -1.00000000, 0.00000000, 0.00000000
dcl_position0 v0
dcl_tangent0 v1
dcl_binormal0 v2
dcl_normal0 v3
dcl_texcoord0 v4
dcl_texcoord0 o0.xy
dcl_texcoord1 o1.xyz
dcl_texcoord6 o2.xyz
dcl_texcoord7 o3.xyz
dcl_position0 o4
dp4 o4.x, v0, c0
dp4 o4.y, v0, c1
dp4 o4.z, v0, c2
dp4 o4.w, v0, c3
add r0.xyz, c7, -v0
dp3 o3.x, r0, c4
dp3 o3.y, r0, c5
dp3 o3.z, r0, c6
mad r3.xyz, c9.xxxx, v1, c9.yyyy
dp3 o2.x, r0, r3
mad r2.xyz, c9.xxxx, v2, c9.yyyy
dp3 o2.y, r0, r2
mad r1.xyz, c9.xxxx, v3, c9.yyyy
dp3 o2.z, r0, r1
add r0.xyz, c8, -v0
dp3 o1.x, r0, r3
dp3 o1.y, r0, r2
dp3 o1.z, r0, r1
mov o0.xy, v4
