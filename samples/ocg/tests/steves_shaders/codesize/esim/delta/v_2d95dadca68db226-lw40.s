vs_3_0
def c14, 1.44269502, 1.00000000, 0.05968310, -1.50000000
dcl_position0 v0
dcl_texcoord0 v1
dcl_texcoord0 o0.xy
dcl_texcoord2 o1.xyz
dcl_texcoord3 o2.xy
dcl_texcoord4 o3.xyz
dcl_position0 o4
dp4 r0.x, v0, c7
dp4 r0.y, v0, c8
dp4 r0.z, v0, c9
mul r1.xyz, r0, c13.xxxx
dp4 o4.x, v0, c0
dp3 r0.w, r1, r1
dp4 o4.y, v0, c1
rsq r0.w, r0.wwww
dp4 o4.z, v0, c2
rcp r0.z, r0.wwww
dp4 o4.w, v0, c3
mul r0.xyz, r0.zzzz, -c11
mul r1.xyz, r1, r0.wwww
mul r0.xyz, r0, c14.xxxx
exp r0.x, r0.xxxx
exp r0.y, r0.yyyy
exp r0.z, r0.zzzz
dp3 r0.w, r1, c10
add o1.xyz, -r0, c14.yyyy
mad r1.w, c12.yyyy, r0.wwww, c12.zzzz
mad r0.z, r0.wwww, r0.wwww, c14.yyyy
pow r0.w, r1.wwww, c14.wwww
mul o2.x, r0.zzzz, c14.zzzz
mul o2.y, r0.wwww, c12.xxxx
dp4 o3.x, v0, c4
dp4 o3.y, v0, c5
dp4 o3.z, v0, c6
mov o0.xy, v1
