vs_2_0
dcl_position0 v0
dcl_normal0 v1
dcl_texcoord0 v2
dcl_tangent0 v3
dcl_binormal0 v4
dp4 oPos.x, v0, c[4]
dp4 oPos.y, v0, c[5]
dp4 oPos.w, v0, c[7]
dp3 oT2.x, v3, c[42]
dp3 oT3.x, v3, c[43]
dp3 oT4.x, v3, c[44]
dp3 oT2.y, v4, c[42]
dp3 oT3.y, v4, c[43]
dp3 oT4.y, v4, c[44]
dp3 oT2.z, v1, c[42]
dp3 oT3.z, v1, c[43]
dp3 oT4.z, v1, c[44]
dp4 r0.x, v0, c[42]
dp4 r0.y, v0, c[43]
dp4 r0.z, v0, c[44]
dp4 r1.y, v0, c[6]
add oT1.xyz, -r0, c[2]
mad oFog, -r1.y, c[16].w, c[16].x
mov oPos.z, r1.y
mov oT0.xy, v2
