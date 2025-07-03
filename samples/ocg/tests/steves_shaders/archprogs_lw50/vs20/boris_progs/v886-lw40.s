vs_1_1
dcl_position0 v0
dcl_color0 v5
dcl_texcoord0 v7
dcl_texcoord1 v8
dcl_tangent0 v11
dcl_binormal0 v12
dp4 r0.x, v0, c[4]
dp4 r0.y, v0, c[5]
dp4 r0.z, v0, c[6]
dp4 r0.w, v0, c[7]
mov oPos, r0
mad oFog, -r0.z, c[16].w, c[16].x
mov r0, v11
mov r1, v12
mul r2, v11.yzxw, r1.zxyw
mad r3, -v11.zxyw, r1.yzxw, r2
dp3 r1.w, v11, v11
rsq r1.w, r1.w
mul r1.xyz, v11, r1.w
dp3 r3.w, r3, r3
rsq r3.w, r3.w
mul r3.xyz, r3, r3.w
mul r2, r3.yzxw, r0.zxyw
mad r2, -r3.zxyw, r0.yzxw, r2
dp3 r0.x, r1, c[90]
dp3 r0.y, r2, c[90]
dp3 r0.z, r3, c[90]
add oD1, r0, c[0].w
mov oT0, v7
mov oT1, v8
mov oD0, v5
