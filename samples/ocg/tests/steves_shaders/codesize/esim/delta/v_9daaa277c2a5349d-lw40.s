vs_3_0
def c14, 1.44269502, 1.00000000, 0.05968310, -1.50000000
dcl_position0 v0
dcl_texcoord2 o0.xyz
dcl_texcoord3 o1.xy
dcl_texcoord4 o2.xyz
dcl_position0 o3
dp4 r0.x, v0, c7
dp4 r0.y, v0, c8
dp4 r0.z, v0, c9
mul r0.xyz, r0, c13.xxxx
dp3 r0.w, r0, r0
dp4 o3.x, v0, c0
rsq r0.w, r0.wwww
dp4 o3.y, v0, c1
rcp r1.w, r0.wwww
dp4 o3.z, v0, c2
mul r1.xyz, r1.wwww, -c11
dp4 o3.w, v0, c3
mul r1.xyz, r1, c14.xxxx
mul r0.xyz, r0, r0.wwww
exp r1.x, r1.xxxx
exp r1.y, r1.yyyy
exp r1.z, r1.zzzz
add o0.xyz, -r1, c14.yyyy
dp3 r0.z, r0, c10
mad r0.w, r0.zzzz, r0.zzzz, c14.yyyy
mad r1.w, c12.yyyy, r0.zzzz, c12.zzzz
mul o1.x, r0.wwww, c14.zzzz
pow r0.w, r1.wwww, c14.wwww
mul o1.y, r0.wwww, c12.xxxx
dp4 o2.x, v0, c4
dp4 o2.y, v0, c5
dp4 o2.z, v0, c6
