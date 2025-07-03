vs_3_0
def c14, 2.00000000, -1.00000000, 0.00000000, 0.00000000
dcl_position0 v0
dcl_tangent0 v1
dcl_binormal0 v2
dcl_normal0 v3
dcl_texcoord0 v4
dcl_texcoord1 v5
dcl_texcoord0 o0.xy
dcl_texcoord1 o1.xy
dcl_texcoord2 o2.xyz
dcl_texcoord3 o3.xyz
dcl_texcoord4 o4.xyz
dcl_texcoord5 o5.xyz
dcl_texcoord6 o6.xyz
dcl_texcoord7 o7
dcl_position0 o8
dp4 o8.x, v0, c4
dp4 o8.y, v0, c5
dp4 o8.z, v0, c6
add r3.xyz, c13, -v0
mad r2.xyz, c14.xxxx, v1, c14.yyyy
dp4 o8.w, v0, c7
dp3 o2.x, r3, r2
mad r0.xyz, c14.xxxx, v2, c14.yyyy
mad r1.xyz, c14.xxxx, v3, c14.yyyy
dp3 o2.y, r3, r0
dp3 o2.z, r3, r1
dp3 o3.x, r2, c8
dp3 o4.x, r2, c9
dp3 o5.x, r2, c10
dp3 o6.x, c11, r2
dp3 o3.y, r0, c8
dp3 o4.y, r0, c9
dp3 o5.y, r0, c10
dp3 o6.y, c11, r0
dp3 o3.z, r1, c8
dp3 o4.z, r1, c9
dp3 o5.z, r1, c10
mad r0.xyz, r1, c12.xxxx, v0
dp3 o6.z, c11, r1
mov r0.w, v0.wwww
dp4 o7.x, r0, c0
dp4 o7.y, r0, c1
dp4 o7.z, r0, c2
dp4 o7.w, r0, c3
mov o0.xy, v4
mov o1.xy, v5
