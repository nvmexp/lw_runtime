vs_3_0
def c10, 2.00000000, -1.00000000, 0.00000000, 0.00000000
dcl_position0 v0
dcl_tangent0 v1
dcl_binormal0 v2
dcl_normal0 v3
dcl_texcoord0 v4
dcl_texcoord0 o0.xy
dcl_texcoord1 o1.xyz
dcl_texcoord6 o2.xyz
dcl_texcoord7 o3
dcl_position0 o4
dp4 o4.x, v0, c4
dp4 o4.y, v0, c5
dp4 o4.z, v0, c6
dp4 o4.w, v0, c7
dp4 o3.x, v0, c0
dp4 o3.y, v0, c1
dp4 o3.z, v0, c2
dp4 o3.w, v0, c3
mad r3.xyz, c10.xxxx, v1, c10.yyyy
dp3 o2.x, c8, r3
mad r2.xyz, c10.xxxx, v2, c10.yyyy
dp3 o2.y, c8, r2
mad r1.xyz, c10.xxxx, v3, c10.yyyy
dp3 o2.z, c8, r1
add r0.xyz, c9, -v0
dp3 o1.x, r0, r3
dp3 o1.y, r0, r2
dp3 o1.z, r0, r1
mov o0.xy, v4
