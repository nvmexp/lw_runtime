vs_3_0
def c206, 3.00000000, 1.44269502, 1.00000000, 0.05968310
def c207, -1.50000000, 0.00000000, 0.00000000, 0.00000000
dcl_position0 v0
dcl_blendindices0 v1
dcl_blendweight0 v2
dcl_texcoord2 o0.xyz
dcl_texcoord3 o1.xy
dcl_texcoord4 o2.xyz
dcl_position0 o3
mul r0.xy, c206.xxxx, v1
mova a0.xy, r0
mul r0, c[a0.y + 0], v2.yyyy
mad r0, c[a0.x + 0], v2.xxxx, r0
dp4 r0.x, v0, r0
mul r1, c[a0.y + 1], v2.yyyy
mad r2, c[a0.x + 1], v2.xxxx, r1
mul r1, c[a0.y + 2], v2.yyyy
dp4 r0.y, v0, r2
mad r1, c[a0.x + 2], v2.xxxx, r1
dp4 r0.z, v0, r1
mov r0.w, v0.wwww
dp4 r1.x, r0, c199
dp4 r1.y, r0, c200
dp4 r1.z, r0, c201
mul r1.xyz, r1, c205.xxxx
dp3 r1.w, r1, r1
dp4 o3.x, r0, c192
rsq r1.w, r1.wwww
dp4 o3.y, r0, c193
rcp r2.w, r1.wwww
dp4 o3.z, r0, c194
mul r2.xyz, r2.wwww, -c203
dp4 o3.w, r0, c195
mul r2.xyz, r2, c206.yyyy
mul r1.xyz, r1, r1.wwww
exp r2.x, r2.xxxx
exp r2.y, r2.yyyy
exp r2.z, r2.zzzz
add o0.xyz, -r2, c206.zzzz
dp3 r1.z, r1, c202
mad r1.w, r1.zzzz, r1.zzzz, c206.zzzz
mad r2.w, c204.yyyy, r1.zzzz, c204.zzzz
mul o1.x, r1.wwww, c206.wwww
pow r1.w, r2.wwww, c207.xxxx
mul o1.y, r1.wwww, c204.xxxx
dp4 o2.x, r0, c196
dp4 o2.y, r0, c197
dp4 o2.z, r0, c198
