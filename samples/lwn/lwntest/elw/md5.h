#ifndef __MD5_H
#define __MD5_H

typedef struct _MD5Context {
    unsigned int buf[4];
    unsigned int bits[2];
    unsigned char in[64];
} MD5Context;

void MD5Init(MD5Context *context);
void MD5Update(MD5Context *context, unsigned char const *buf, unsigned len);
void MD5Final(unsigned char digest[16], MD5Context *context);
void MD5Transform(uint32_t buf[4], uint32_t const in[16]);

int md5Read(char *line, char *name, unsigned char md5s[MAX_MD5_GOLDS][16]);
void md5Write(const char *name, unsigned char *md5);
void md5Generate(const char *name, unsigned char *bufp, unsigned int w, unsigned int h, unsigned short bpp, unsigned char* md5);

#endif // __MD5_H
