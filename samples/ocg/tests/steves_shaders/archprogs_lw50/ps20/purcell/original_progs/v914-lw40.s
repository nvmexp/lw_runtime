vs_1_1
dcl_position0 v0
dcl_normal0 v3
dcl_texcoord0 v7
dcl_tangent0 v11
dcl_binormal0 v12
dp4 r0.x, v0, c[42]
dp4 r0.y, v0, c[43]
dp4 r0.z, v0, c[44]
dp4 r1.x, v0, c[4]
dp4 r1.y, v0, c[5]
dp4 r1.z, v0, c[6]
dp4 r1.w, v0, c[7]
mov oPos, r1
mad oFog, -r1.z, c[16].w, c[16].x
dp3 oT1.x, v11, c[42]
dp3 oT2.x, v11, c[43]
dp3 oT3.x, v11, c[44]
dp3 oT1.y, v12, c[42]
dp3 oT2.y, v12, c[43]
dp3 oT3.y, v12, c[44]
dp3 oT1.z, v3, c[42]
dp3 oT2.z, v3, c[43]
dp3 oT3.z, v3, c[44]
add r1.xyz, c[2], -r0
mov oT4.xyz, r1
dp3 r0.x, r1, v11
dp3 r0.y, r1, v12
dp3 r0.z, r1, v3
dp3 r0.w, r0, r0
rsq r0.w, r0.w
mul r0, r0, r0.w
mad oD0.xyz, r0, c[0].w, c[0].w
dp4 oT0.x, v7, c[90]
dp4 oT0.y, v7, c[91]
