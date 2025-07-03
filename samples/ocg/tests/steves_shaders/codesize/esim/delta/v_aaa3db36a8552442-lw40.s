vs_3_0
def c206, 2.00000000, -1.00000000, 3.00000000, 0.00000000
dcl_position0 v0
dcl_tangent0 v1
dcl_binormal0 v2
dcl_normal0 v3
dcl_texcoord0 v4
dcl_blendindices0 v5
dcl_blendweight0 v6
dcl_texcoord0 o0.xy
dcl_texcoord1 o1.xyz
dcl_texcoord2 o2.xyz
dcl_texcoord3 o3.xyz
dcl_texcoord4 o4.xyz
dcl_texcoord6 o5.xyz
dcl_texcoord7 o6
dcl_position0 o7
mul r0.xy, c206.zzzz, v5
mova a0.xy, r0
mul r0, c[a0.y + 0], v6.yyyy
mad r3, c[a0.x + 0], v6.xxxx, r0
mov r0.w, v0.wwww
dp4 r0.x, v0, r3
mul r2, c[a0.y + 1], v6.yyyy
mul r1, c[a0.y + 2], v6.yyyy
mad r2, c[a0.x + 1], v6.xxxx, r2
mad r1, c[a0.x + 2], v6.xxxx, r1
dp4 r0.y, v0, r2
dp4 r0.z, v0, r1
dp4 o7.x, r0, c196
dp4 o7.y, r0, c197
dp4 o7.z, r0, c198
dp4 o7.w, r0, c199
mad r4.xyz, c206.xxxx, v1, c206.yyyy
add r6.xyz, -r0, c205
dp3 r5.x, r4, r3
dp3 r5.y, r4, r2
dp3 r5.z, r4, r1
dp3 o1.x, r6, r5
mad r7.xyz, c206.xxxx, v2, c206.yyyy
dp3 r4.x, r7, r3
dp3 r4.y, r7, r2
dp3 r4.z, r7, r1
mad r7.xyz, c206.xxxx, v3, c206.yyyy
dp3 o1.y, r6, r4
dp3 r3.x, r7, r3
dp3 r3.y, r7, r2
dp3 r3.z, r7, r1
dp3 o1.z, r6, r3
dp3 o2.x, r5, c200
dp3 o3.x, r5, c201
dp3 o4.x, r5, c202
dp3 o5.x, c203, r5
dp3 o2.y, r4, c200
dp3 o3.y, r4, c201
dp3 o4.y, r4, c202
dp3 o5.y, c203, r4
dp3 o2.z, r3, c200
dp3 o3.z, r3, c201
dp3 o4.z, r3, c202
mad r0.xyz, r3, c204.xxxx, r0
mov r0.w, v0.wwww
dp3 o5.z, c203, r3
dp4 o6.x, r0, c192
dp4 o6.y, r0, c193
dp4 o6.z, r0, c194
dp4 o6.w, r0, c195
mov o0.xy, v4
