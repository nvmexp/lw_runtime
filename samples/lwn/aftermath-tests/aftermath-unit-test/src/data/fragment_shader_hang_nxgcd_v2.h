#ifdef __cplusplus
extern "C" {
#endif

static const unsigned char fragment_shader_hang_nxgcd_v2[] = {
0x02,0x00,0xcd,0xdb,0x94,0x0c,0x00,0x00,0x8d,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0xa0,0x00,0x00,0x00,0x64,0x00,0x00,0x00,
0x08,0x01,0x00,0x00,0xb8,0x02,0x00,0x00,0xc4,0x03,0x00,0x00,0x98,0x06,0x00,0x00,
0x60,0x0a,0x00,0x00,0x1c,0x00,0x00,0x00,0x80,0x0a,0x00,0x00,0x18,0x00,0x00,0x00,
0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x8c,0x00,0x00,0x00,0x10,0x00,0x00,0x00,
0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
0xbc,0x0a,0x00,0x00,0xd4,0x01,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x9c,0x0a,0x00,0x00,0x08,0x00,0x00,0x00,
0xa8,0x0a,0x00,0x00,0x10,0x00,0x00,0x00,0x88,0x00,0x00,0x00,0x00,0x01,0x00,0x00,
0x02,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x02,0xc0,0x02,0x80,0x10,0x00,0x00,0x00,
0xc5,0x0b,0x28,0x01,0x01,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x02,0x0c,0x09,0x09,
0x00,0x00,0x01,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x6d,0x01,0x00,0x00,
0x40,0x81,0x02,0x00,0x00,0x00,0x00,0x00,0x01,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
0x04,0x00,0x00,0x00,0x80,0x23,0x00,0x80,0x97,0xb1,0x00,0x00,0x01,0x20,0x00,0x00,
0x80,0x03,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
0x90,0x64,0x40,0x80,0x90,0x64,0x40,0x00,0x2e,0x00,0x00,0x00,0x00,0x08,0x00,0x00,
0x00,0x08,0x00,0x00,0x64,0x00,0x00,0x00,0x6d,0xdb,0xa2,0x00,0x2d,0x5a,0x00,0x00,
0x40,0x3b,0x00,0x00,0x09,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x45,0x03,0x00,0x00,
0x00,0x00,0x00,0x00,0x6d,0x07,0x00,0x00,0x6d,0x01,0x10,0x00,0x1b,0x06,0x1c,0x20,
0x00,0x03,0x00,0x00,0x00,0x00,0x00,0x00,0x04,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
0x01,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x46,0x01,0x08,0x00,0x00,0x00,0x00,0x00,
0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x80,0x00,0x00,0x00,0x00,
0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0xec,0x15,0xe4,0x00,
0x00,0x00,0x00,0x00,0xe4,0xa2,0xe5,0x29,0x00,0x00,0x00,0x00,0x01,0x00,0x00,0x00,
0x52,0xed,0xa7,0x0a,0x00,0x00,0x00,0x00,0x00,0x70,0xe7,0x90,0x1b,0x00,0xb4,0x28,
0x6d,0x57,0xb4,0x2d,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
0x05,0x8a,0xa1,0x00,0x05,0x8a,0xa1,0x00,0x00,0x00,0x00,0x00,0x44,0x40,0x00,0x00,
0x00,0x00,0x00,0x00,0x0a,0x0a,0xa4,0x49,0x77,0x39,0x05,0xff,0x7f,0x00,0x00,0x00,
0x00,0x00,0x00,0x00,0x42,0x40,0x00,0x00,0x0a,0x0a,0xa4,0x49,0x00,0x00,0x00,0x00,
0x77,0x39,0x05,0xff,0x00,0x00,0x00,0x00,0x1f,0x00,0x00,0x00,0x42,0x40,0x00,0x00,
0x00,0x00,0x00,0x00,0x0a,0x0a,0xa4,0x49,0x77,0x39,0x05,0xff,0x44,0x00,0x00,0x00,
0x00,0x00,0x00,0x00,0x0a,0x0a,0xa0,0x49,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x1e,0x00,0x00,0x00,0x01,0x02,0x00,0x00,
0x42,0x40,0x00,0x00,0x00,0x00,0x00,0x80,0x00,0x00,0x00,0x00,0x02,0x00,0x04,0x40,
0x42,0x40,0x00,0x00,0x00,0x00,0x00,0x00,0x0a,0x0a,0xa4,0x49,0x77,0x39,0x05,0xff,
0x00,0x04,0x00,0x00,0x00,0x00,0x00,0x00,0xc0,0xff,0xff,0xff,0x0a,0x0a,0xa4,0x49,
0x77,0x39,0x05,0xff,0x00,0x00,0x20,0x20,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x03,0x00,0x00,0x00,0x55,0x00,0x00,0x00,
0x55,0x50,0x00,0x00,0x00,0x00,0x00,0x00,0x44,0x40,0x00,0x00,0xfc,0xff,0xff,0xff,
0x0a,0x0a,0xa4,0x49,0x77,0x39,0x05,0xff,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
0x00,0x00,0x00,0x00,0x00,0x00,0x00,0xff,0x44,0x40,0x00,0x00,0xe0,0xff,0xff,0xff,
0x0a,0x0a,0xa4,0x49,0x77,0x39,0x05,0xff,0x00,0x00,0x00,0x00,0x44,0x40,0x00,0x00,
0xe0,0xff,0xff,0xff,0x77,0x39,0x05,0xff,0x44,0x40,0x00,0x00,0x0a,0x0a,0xa4,0x49,
0x77,0x39,0x05,0xff,0x00,0x00,0x00,0x80,0xfe,0x01,0x00,0x00,0x0a,0x0a,0xa4,0x49,
0x77,0x39,0x05,0xff,0x42,0x00,0x00,0x00,0x01,0x18,0x00,0x00,0x00,0x00,0x00,0x00,
0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x42,0x00,0x00,0x00,
0x00,0x00,0x00,0x00,0x0a,0x0a,0xa4,0x49,0x77,0x39,0x05,0xff,0x00,0x00,0x00,0x00,
0x49,0x92,0x24,0x09,0x00,0x00,0x03,0x01,0x00,0xbe,0x40,0x00,0x00,0x00,0x00,0x00,
0x44,0x40,0x00,0x00,0xf6,0xff,0xff,0xff,0x0a,0x0a,0xa4,0x49,0x77,0x39,0x05,0xff,
0xb8,0x02,0x00,0x00,0x45,0x03,0x00,0x00,0x6d,0x07,0x00,0x00,0x6d,0x01,0x10,0x00,
0x1b,0x06,0x1c,0x20,0x18,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x80,
0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
0x00,0x00,0x00,0x00,0x00,0x8a,0x00,0x20,0x0a,0x18,0x00,0x00,0x05,0x0a,0x00,0x00,
0x29,0x00,0x00,0x00,0x00,0x00,0x00,0x80,0x00,0x00,0x00,0x80,0x00,0x00,0x00,0x00,
0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0xff,0x00,0x00,0x00,0x22,0x60,0x00,0x3f,
0x00,0x00,0x21,0x00,0x3e,0x00,0x00,0x00,0x0a,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
0x00,0x00,0x00,0x00,0x76,0xeb,0xc5,0x00,0x00,0x00,0x00,0x00,0x06,0x00,0x00,0x00,
0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x80,0x03,0x35,0x50,0x00,0x01,0x00,0x00,
0x00,0x00,0x00,0x00,0xe3,0x0b,0x01,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0xc4,0x03,0x00,0x00,0x08,0x00,0x00,0x00,
0x01,0x00,0x00,0x00,0x20,0x00,0x00,0x00,0x19,0x92,0x01,0x03,0x09,0x80,0x04,0x00,
0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,
0x18,0x3d,0x01,0x00,0x18,0x3d,0x01,0x00,0x18,0x3d,0x01,0x00,0x18,0x3d,0x01,0x00,
0x18,0x3d,0x01,0x00,0x18,0x3d,0x01,0x00,0x18,0x3d,0x01,0x00,0x18,0x3d,0x01,0x00,
0x18,0x3d,0x01,0x00,0x18,0x3d,0x01,0x00,0x18,0x3d,0x01,0x00,0x18,0x3d,0x01,0x00,
0x18,0x3d,0x01,0x00,0x18,0x3d,0x01,0x00,0x18,0x3d,0x01,0x00,0x18,0x3d,0x01,0x00,
0x18,0x3d,0x01,0x00,0x18,0x3d,0x01,0x00,0x18,0x3d,0x01,0x00,0x18,0x3d,0x01,0x00,
0x18,0x3d,0x01,0x00,0x18,0x3d,0x01,0x00,0x18,0x3d,0x01,0x00,0x18,0x3d,0x01,0x00,
0x18,0x3d,0x01,0x00,0x28,0x3d,0x01,0x00,0x18,0x3d,0x01,0x00,0x18,0x3d,0x01,0x00,
0x18,0x3d,0x01,0x00,0x18,0x3d,0x01,0x00,0x18,0x3d,0x01,0x00,0x18,0x3d,0x01,0x00,
0x18,0x3d,0x01,0x00,0x18,0x3d,0x01,0x00,0x18,0x3d,0x01,0x00,0x18,0x3d,0x01,0x00,
0x18,0x3d,0x01,0x00,0x18,0x3d,0x01,0x00,0x18,0x3d,0x01,0x00,0x18,0x3d,0x01,0x00,
0x18,0x3d,0x01,0x00,0x18,0x3d,0x01,0x00,0x18,0x3d,0x01,0x00,0x18,0x3d,0x01,0x00,
0x18,0x3d,0x01,0x00,0x18,0x3d,0x01,0x00,0x18,0x3d,0x01,0x00,0x18,0x3d,0x01,0x00,
0x18,0x3d,0x01,0x00,0x18,0x3d,0x01,0x00,0x18,0x3d,0x01,0x00,0x18,0x3d,0x01,0x00,
0x18,0x3d,0x01,0x00,0x18,0x3d,0x01,0x00,0x18,0x3d,0x01,0x00,0x18,0x3d,0x01,0x00,
0x18,0x3d,0x01,0x00,0x18,0x3d,0x01,0x00,0x18,0x3d,0x01,0x00,0x18,0x3d,0x01,0x00,
0x18,0x3d,0x01,0x00,0x18,0x3d,0x01,0x00,0x18,0x3d,0x01,0x00,0x18,0x3d,0x01,0x00,
0x18,0x3d,0x01,0x00,0x18,0x3d,0x01,0x00,0x18,0x3d,0x01,0x00,0x18,0x3d,0x01,0x00,
0x18,0x3d,0x01,0x00,0x18,0x3d,0x01,0x00,0x18,0x3d,0x01,0x00,0x18,0x3d,0x01,0x00,
0x18,0x3d,0x01,0x00,0x18,0x3d,0x01,0x00,0x18,0x3d,0x01,0x00,0x18,0x3d,0x01,0x00,
0x18,0x3d,0x01,0x00,0x18,0x3d,0x01,0x00,0x18,0x3d,0x01,0x00,0x18,0x3d,0x01,0x00,
0x18,0x3d,0x01,0x00,0x18,0x3d,0x01,0x00,0x18,0x3d,0x01,0x00,0x18,0x3d,0x01,0x00,
0x18,0x3d,0x01,0x00,0x18,0x3d,0x01,0x00,0x18,0x3d,0x01,0x00,0x18,0x3d,0x01,0x00,
0x18,0x3d,0x01,0x00,0x18,0x3d,0x01,0x00,0x18,0x3d,0x01,0x00,0x18,0x3d,0x01,0x00,
0x18,0x3d,0x01,0x00,0x18,0x3d,0x01,0x00,0x18,0x3d,0x01,0x00,0x18,0x3d,0x01,0x00,
0x18,0x3d,0x01,0x00,0x18,0x3d,0x01,0x00,0x18,0x3d,0x01,0x00,0x18,0x3d,0x01,0x00,
0x18,0x3d,0x01,0x00,0x18,0x3d,0x01,0x00,0x18,0x3d,0x01,0x00,0x18,0x3d,0x01,0x00,
0x18,0x3d,0x01,0x00,0x18,0x3d,0x01,0x00,0x18,0x3d,0x01,0x00,0x18,0x3d,0x01,0x00,
0x18,0x3d,0x01,0x00,0x18,0x3d,0x01,0x00,0x18,0x3d,0x01,0x00,0x18,0x3d,0x01,0x00,
0x18,0x3d,0x01,0x00,0x18,0x3d,0x01,0x00,0x18,0x3d,0x01,0x00,0x18,0x3d,0x01,0x00,
0x18,0x3d,0x01,0x00,0x18,0x3d,0x01,0x00,0x18,0x3d,0x01,0x00,0x18,0x3d,0x01,0x00,
0x18,0x3d,0x01,0x00,0x18,0x3d,0x01,0x00,0x18,0x3d,0x01,0x00,0x18,0x3d,0x01,0x00,
0x18,0x3d,0x01,0x00,0x18,0x3d,0x01,0x00,0x18,0x3d,0x01,0x00,0x18,0x3d,0x01,0x00,
0x45,0x03,0x00,0x00,0x6d,0x07,0x00,0x00,0x6d,0x01,0x10,0x00,0x1b,0x06,0x1c,0x20,
0x18,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x80,0x00,0x00,0x00,0x00,
0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
0x00,0x8a,0x00,0x20,0x0a,0x18,0x00,0x00,0x05,0x0a,0x00,0x00,0x29,0x00,0x00,0x00,
0x00,0x00,0x00,0x80,0x00,0x00,0x00,0x80,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
0x00,0x00,0x00,0x00,0xff,0x00,0x00,0x00,0x22,0x60,0x00,0x3f,0x00,0x00,0x21,0x00,
0x3e,0x00,0x00,0x00,0x0a,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
0x76,0xeb,0xc5,0x00,0x00,0x00,0x00,0x00,0x06,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
0x01,0x00,0x00,0x00,0x80,0x03,0x35,0x50,0x00,0x01,0x00,0x00,0x00,0x00,0x00,0x00,
0xe3,0x0b,0x01,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
0x00,0x00,0x00,0x00,0xc4,0x03,0x00,0x00,0x08,0x00,0x00,0x00,0x01,0x00,0x00,0x00,
0x20,0x00,0x00,0x00,0x19,0x92,0x01,0x03,0x09,0x80,0x04,0x00,0x00,0x00,0x00,0x00,
0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0xff,0xff,0xff,0xff,
0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0x18,0x3d,0x01,0x00,
0x28,0x3d,0x01,0x00,0x28,0x3d,0x01,0x00,0x18,0x3d,0x01,0x00,0x18,0x3d,0x01,0x00,
0x18,0x3d,0x01,0x00,0x18,0x3d,0x01,0x00,0x18,0x3d,0x01,0x00,0x28,0x3d,0x01,0x00,
0x18,0x3d,0x01,0x00,0x18,0x3d,0x01,0x00,0x18,0x3d,0x01,0x00,0x18,0x3d,0x01,0x00,
0x18,0x3d,0x01,0x00,0x18,0x3d,0x01,0x00,0x18,0x3d,0x01,0x00,0x18,0x3d,0x01,0x00,
0x18,0x3d,0x01,0x00,0x18,0x3d,0x01,0x00,0x18,0x3d,0x01,0x00,0x18,0x3d,0x01,0x00,
0x18,0x3d,0x01,0x00,0x18,0x3d,0x01,0x00,0x18,0x3d,0x01,0x00,0x18,0x3d,0x01,0x00,
0x18,0x3d,0x01,0x00,0x18,0x3d,0x01,0x00,0x18,0x3d,0x01,0x00,0x18,0x3d,0x01,0x00,
0x18,0x3d,0x01,0x00,0x18,0x3d,0x01,0x00,0x18,0x3d,0x01,0x00,0x18,0x3d,0x01,0x00,
0x18,0x3d,0x01,0x00,0x18,0x3d,0x01,0x00,0x18,0x3d,0x01,0x00,0x18,0x3d,0x01,0x00,
0x18,0x3d,0x01,0x00,0x18,0x3d,0x01,0x00,0x18,0x3d,0x01,0x00,0x18,0x3d,0x01,0x00,
0x18,0x3d,0x01,0x00,0x18,0x3d,0x01,0x00,0x18,0x3d,0x01,0x00,0x18,0x3d,0x01,0x00,
0x18,0x3d,0x01,0x00,0x18,0x3d,0x01,0x00,0x18,0x3d,0x01,0x00,0x18,0x3d,0x01,0x00,
0x18,0x3d,0x01,0x00,0x18,0x3d,0x01,0x00,0x18,0x3d,0x01,0x00,0x18,0x3d,0x01,0x00,
0x18,0x3d,0x01,0x00,0x18,0x3d,0x01,0x00,0x18,0x3d,0x01,0x00,0x18,0x3d,0x01,0x00,
0x18,0x3d,0x01,0x00,0x18,0x3d,0x01,0x00,0x18,0x3d,0x01,0x00,0x18,0x3d,0x01,0x00,
0x18,0x3d,0x01,0x00,0x18,0x3d,0x01,0x00,0x18,0x3d,0x01,0x00,0x18,0x3d,0x01,0x00,
0x18,0x3d,0x01,0x00,0x18,0x3d,0x01,0x00,0x18,0x3d,0x01,0x00,0x18,0x3d,0x01,0x00,
0x18,0x3d,0x01,0x00,0x18,0x3d,0x01,0x00,0x18,0x3d,0x01,0x00,0x18,0x3d,0x01,0x00,
0x18,0x3d,0x01,0x00,0x18,0x3d,0x01,0x00,0x18,0x3d,0x01,0x00,0x18,0x3d,0x01,0x00,
0x18,0x3d,0x01,0x00,0x18,0x3d,0x01,0x00,0x18,0x3d,0x01,0x00,0x18,0x3d,0x01,0x00,
0x18,0x3d,0x01,0x00,0x18,0x3d,0x01,0x00,0x18,0x3d,0x01,0x00,0x18,0x3d,0x01,0x00,
0x18,0x3d,0x01,0x00,0x18,0x3d,0x01,0x00,0x18,0x3d,0x01,0x00,0x18,0x3d,0x01,0x00,
0x18,0x3d,0x01,0x00,0x18,0x3d,0x01,0x00,0x18,0x3d,0x01,0x00,0x18,0x3d,0x01,0x00,
0x18,0x3d,0x01,0x00,0x18,0x3d,0x01,0x00,0x18,0x3d,0x01,0x00,0x18,0x3d,0x01,0x00,
0x18,0x3d,0x01,0x00,0x18,0x3d,0x01,0x00,0x18,0x3d,0x01,0x00,0x18,0x3d,0x01,0x00,
0x18,0x3d,0x01,0x00,0x18,0x3d,0x01,0x00,0x18,0x3d,0x01,0x00,0x18,0x3d,0x01,0x00,
0x18,0x3d,0x01,0x00,0x18,0x3d,0x01,0x00,0x18,0x3d,0x01,0x00,0x18,0x3d,0x01,0x00,
0x18,0x3d,0x01,0x00,0x18,0x3d,0x01,0x00,0x18,0x3d,0x01,0x00,0x18,0x3d,0x01,0x00,
0x18,0x3d,0x01,0x00,0x18,0x3d,0x01,0x00,0x18,0x3d,0x01,0x00,0x18,0x3d,0x01,0x00,
0x18,0x3d,0x01,0x00,0x18,0x3d,0x01,0x00,0x18,0x3d,0x01,0x00,0x18,0x3d,0x01,0x00,
0x18,0x3d,0x01,0x00,0x18,0x3d,0x01,0x00,0x18,0x3d,0x01,0x00,0x18,0x3d,0x01,0x00,
0x18,0x3d,0x01,0x00,0x18,0x3d,0x01,0x00,0x18,0x3d,0x01,0x00,0x98,0x06,0x00,0x00,
0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x1c,0x00,0x00,0x00,
0x00,0x00,0x80,0x05,0x00,0x00,0xf8,0x0a,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x80,0x18,0x00,0x00,0x00,0x01,0x00,0x00,0x00,
0x01,0x00,0x00,0x00,0x08,0x00,0x00,0x00,0x00,0x00,0x00,0x80,0xff,0xff,0xff,0x00,
0x00,0x00,0x00,0x80,0x01,0x00,0x00,0x00,0x10,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
0x01,0x00,0x00,0x00,0xf0,0xb7,0x03,0x00,0x05,0x00,0x00,0x00,0xf0,0xb7,0x03,0x00,
0x05,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0xf0,0xb7,0x03,0x00,
0x05,0x02,0x00,0x00,0x00,0x00,0x40,0x60,0x00,0x00,0x00,0x84,0x00,0x00,0x00,0x00,
0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x18,0x00,0x00,0x00,
0x56,0x04,0x00,0x00,0x01,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x56,0x04,0x00,0x00,
0x10,0x01,0x00,0x00,0x10,0x67,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x01,0x00,0x00,0x00,0x01,0x00,0x00,0x00,
0x00,0xdd,0x6b,0x01,0x05,0x00,0x00,0x00,0x00,0xdd,0x6b,0x01,0x05,0x00,0x00,0x00,
0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0xdd,0x6b,0x01,0x05,0x02,0x00,0x00,
0x00,0x00,0x40,0x60,0x00,0x00,0x00,0x84,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x19,0x00,0x00,0x00,0x99,0xbe,0x0f,0x00,
0x01,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x99,0xbe,0x0f,0x00,0x00,0x00,0x00,0x00,
0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
0x00,0x00,0x00,0x00,0x02,0x00,0x00,0x00,0x01,0x00,0x00,0x19,0x94,0x09,0xfa,0x14,
0x05,0x00,0x00,0x00,0xb8,0x50,0xfa,0x14,0x05,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
0x00,0x00,0x00,0x00,0x00,0x16,0xfa,0x14,0x05,0x72,0x75,0x00,0xa0,0x1c,0x10,0x80,
0x00,0x00,0x55,0x05,0x2c,0x07,0x00,0x80,0x30,0x07,0x00,0x80,0x34,0x07,0x00,0x80,
0x70,0x1c,0x00,0x80,0x00,0x00,0x00,0x00,0x80,0x1c,0x00,0x80,0x00,0x00,0x00,0x00,
0x90,0x1c,0x00,0x80,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
0x00,0x00,0x00,0x00,0x1a,0x00,0x00,0x00,0x4c,0x1c,0x00,0x00,0x01,0x00,0x00,0x00,
0x00,0x00,0x00,0x00,0x31,0x1c,0x00,0x00,0xb5,0xd9,0x03,0x00,0x10,0x65,0x00,0x00,
0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
0xd4,0x01,0x00,0x00
};

#ifdef __cplusplus
}
#endif
