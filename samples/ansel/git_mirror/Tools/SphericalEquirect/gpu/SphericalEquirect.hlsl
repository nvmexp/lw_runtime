#define MAX_TILE_GEOMETRY_IN_SHARED_MEMORY 32
static const float cPi = 3.14159265358979323846264338327950288f;
static const float cPi2 = cPi * 0.5f;
static const float cPi4 = cPi * 0.25f;

// For alignment reasons,
// this also hides 'tileNormal * -d (plane offset)'
// in w components of OA, AB, AD vectors
struct TileGeometry
{
    float4 OA;
    float4 AB;
    float4 AD;
};

struct PitchRangeIndex { uint start, end; };

cbuffer GlobalParameters : register(b0)
{
    uint width;
    uint tileCount;
    float2 recipWidthHeight;
};

Texture2DArray<float3> tiles : register(t0);
StructuredBuffer<PitchRangeIndex> tilePitchIndex : register(t1);
StructuredBuffer<TileGeometry> tileGeometry : register(t2);
RWByteAddressBuffer result : register(u0);
uniform SamplerState tileSampler : TEXUNIT0;

groupshared TileGeometry tileGeometryShared[MAX_TILE_GEOMETRY_IN_SHARED_MEMORY];

[numthreads(256, 1, 1)]
void CSMain( uint3 tid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    const uint j = gid.x * 256 + tid.x, i = gid.y; // *16 + tid.y;
    const uint start = tilePitchIndex[i].start;
    const uint end = tilePitchIndex[i].end;
    if (tid.x <= end - start)
        tileGeometryShared[tid.x] = tileGeometry[tid.x + start];

    GroupMemoryBarrierWithGroupSync();

    const uint wr = width;
    const uint hr = wr * 0.5f;
    const float x = (2.0f * j) / (wr - 1) - 1.0f, y = (2.0f * i) / (hr - 1) - 1.0f;
    const float theta = x * cPi, phi = y * cPi2;

    const float3 R = float3(sin(theta) * cos(phi), sin(phi), cos(theta) * cos(phi));
    float3 finalColor = float3(0.0f, 0.0f, 0.0f);
    float totalWeight = 0.0f;

    for (uint k = 0; k <= end - start; ++k)
    {
        const TileGeometry tile = tileGeometryShared[k];
        const float3 tileNormal = float3(tile.OA.w, tile.AB.w, tile.AD.w);
        const float recipT = dot(tileNormal, R);

        if (recipT > 1.0f)
        {
            const float3 tileIntersection = R / recipT;
            const float3 AT = tileIntersection - tile.OA.xyz;
            const float2 Proj = float2(dot(AT, tile.AD.xyz) * recipWidthHeight.y, dot(AT, tile.AB.xyz) * recipWidthHeight.x);

            if (saturate(Proj.x) == Proj.x && saturate(Proj.y) == Proj.y)
            {
                const uint tileNo = k + start;
                const float2 normalizedProj = abs(float2(2.0f * Proj.y - 1.0f, 2.0f * Proj.x - 1.0f));
                float weight = 1.0f - max(normalizedProj.x, normalizedProj.y);
                weight *= weight * weight;

                finalColor += tiles.SampleLevel(tileSampler, float3(Proj.y, 1.0f - Proj.x, tileNo), 0) * weight;
                totalWeight += weight;
            }
        }
    }
    const float3 col = 255 * finalColor / totalWeight;
    const uint color = uint(col.z) << 16 | uint(col.y) << 8 | uint(col.x);
    const uint addr = 4 * (i * wr + j);
    result.Store(addr, color);
}