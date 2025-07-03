vs_2_0
def c[0], 1.000000, 0.816497, -0.408248, 0.577350
def c[3], 0.707107, -0.707107, 0.000000, 0.000000
dcl_position0 v0
dcl_normal0 v1
dcl_texcoord0 v2
dcl_tangent0 v4
mov r0.w, c[0].x
dp4 r0.x, v0, c[42]
dp4 r0.y, v0, c[43]
dp4 r0.z, v0, c[44]
dp4 oPos.x, r0, c[8]
dp4 oPos.y, r0, c[9]
dp4 oPos.w, r0, c[11]
dp4 r1.y, r0, c[10]
add oT3.xyz, -r0, c[2]
mad oFog, -r1.y, c[16].w, c[16].x
mov oPos.z, r1.y
dp3 r6.y, v4, c[43]
dp3 r6.z, v4, c[44]
mul r1.xy, r6.yzzw, c[3]
add r1.x, r1.y, r1.x
dp3 r6.x, v4, c[42]
dp3 r5.x, v1, c[42]
dp3 r5.y, v1, c[43]
dp3 r5.z, v1, c[44]
mul r4.xyz, r6.yzxw, r5.zxyw
mad r4.xyz, r5.yzxw, r6.zxyw, -r4
mul r0.xyz, r4, v4.w
mul r4.xy, r0.yzzw, c[3]
add r1.y, r4.y, r4.x
mul r4.xy, r5.yzzw, c[3]
add r1.z, r4.y, r4.x
slt r4.xyz, r1, c[3].z
mul r1.xyz, r1, r1
mova a0.xyz, r4
mul r4.xyz, r1.y, c[a0.y+23]
mad r4.xyz, r1.x, c[a0.x+21], r4
mad r2.xyz, r1.z, c[a0.z+25], r4
dp3 r1.x, c[0].yzzw, r6
dp3 r1.z, c[0].yzzw, r5
dp3 r1.y, c[0].yzzw, r0
slt r4.xyz, r1, c[3].z
mul r1.xyz, r1, r1
mova a0.xyz, r4
mul r4.xyz, r1.y, c[a0.y+23]
mad r4.xyz, r1.x, c[a0.x+21], r4
mad r3.xyz, r1.z, c[a0.z+25], r4
add r1.xyz, r2, r3
dp3 r4.x, c[0].w, r6
mov oT4.xyz, r6
dp3 r4.z, c[0].w, r5
dp3 r4.y, c[0].w, r0
mov oT5.xyz, r0
slt r0.xyz, r4, c[3].z
mul r4.xyz, r4, r4
mova a0.xyz, r0
mul r0.xyz, r4.y, c[a0.y+23]
mad r0.xyz, r4.x, c[a0.x+21], r0
mad r0.xyz, r4.z, c[a0.z+25], r0
add r1.xyz, r1, r0
mul r1.xyz, r1, c[0].w
rcp r4.x, r1.x
rcp r4.y, r1.y
rcp r4.z, r1.z
slt r1.xyz, r5, c[3].z
mova a0.xyz, r1
mul r1.xyz, r5, r5
mov oT6.xyz, r5
mul r5.xyz, r1.y, c[a0.y+23]
mad r5.xyz, r1.x, c[a0.x+21], r5
mad r1.xyz, r1.z, c[a0.z+25], r5
log r1.x, r1.x
log r1.y, r1.y
log r1.z, r1.z
mul r1.xyz, r1, c[1].x
exp r1.x, r1.x
exp r1.y, r1.y
exp r1.z, r1.z
mul r1.xyz, r1, c[1].w
mul r1.xyz, r4, r1
mul oD0.xyz, r3, r1
mul oD1.xyz, r2, r1
mul oT7.xyz, r0, r1
dp4 oT0.x, v2, c[90]
dp4 oT0.y, v2, c[91]
dp4 oT1.x, v2, c[92]
dp4 oT1.y, v2, c[93]
dp4 oT2.x, v2, c[94]
dp4 oT2.y, v2, c[95]
mov oD0.w, c[3].z
