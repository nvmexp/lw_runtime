#ifndef XMMA_DEVICE_FUNCTION
#define XMMA_DEVICE_FUNCTION

template<typename T, int>
struct ResultPack{
};

template<> struct ResultPack<float, 4> {
    uint32_t elem0;
    uint32_t elem1;
    uint32_t elem2;
    uint32_t elem3;
    __host__ __device__ ResultPack(){
        elem0 = elem1 = elem2 = elem3 = 0;
    }
    template<typename Frag>
    inline __device__ void setFrag(Frag &frag){
       memcpy(&(frag.reg(0)), &elem0, sizeof(uint32_t));
       memcpy(&(frag.reg(1)), &elem1, sizeof(uint32_t));
       memcpy(&(frag.reg(2)), &elem2, sizeof(uint32_t));
       memcpy(&(frag.reg(3)), &elem3, sizeof(uint32_t));

    }
    template<typename Frag>
    inline __device__ void getFrag(Frag &frag){
        memcpy(&elem0, &(frag.reg(0)), sizeof(uint32_t));     
        memcpy(&elem1, &(frag.reg(1)), sizeof(uint32_t));
        memcpy(&elem2, &(frag.reg(2)), sizeof(uint32_t));
        memcpy(&elem3, &(frag.reg(3)), sizeof(uint32_t));
    }
};
template<> struct ResultPack<float, 8> {
    uint32_t elem0;
    uint32_t elem1;
    uint32_t elem2;
    uint32_t elem3;
    uint32_t elem4;
    uint32_t elem5;
    uint32_t elem6;
    uint32_t elem7;
    __host__ __device__ ResultPack(){
        elem0 = elem1 = elem2 = elem3 = 0;
        elem4 = elem5 = elem6 = elem7 = 0;
    }

    template<typename Frag>
    inline __device__ void setFrag(Frag &frag){
        memcpy(&(frag.reg(0)), &elem0, sizeof(uint32_t));
        memcpy(&(frag.reg(1)), &elem1, sizeof(uint32_t));
        memcpy(&(frag.reg(2)), &elem2, sizeof(uint32_t));
        memcpy(&(frag.reg(3)), &elem3, sizeof(uint32_t));
        memcpy(&(frag.reg(4)), &elem4, sizeof(uint32_t));
        memcpy(&(frag.reg(5)), &elem5, sizeof(uint32_t));
        memcpy(&(frag.reg(6)), &elem6, sizeof(uint32_t));
        memcpy(&(frag.reg(7)), &elem7, sizeof(uint32_t));

    }

    template<typename Frag>
    inline __device__ void getFrag(Frag &frag){
        memcpy(&elem0, &(frag.reg(0)), sizeof(uint32_t));     
        memcpy(&elem1, &(frag.reg(1)), sizeof(uint32_t));
        memcpy(&elem2, &(frag.reg(2)), sizeof(uint32_t));
        memcpy(&elem3, &(frag.reg(3)), sizeof(uint32_t));
        memcpy(&elem4, &(frag.reg(4)), sizeof(uint32_t));     
        memcpy(&elem5, &(frag.reg(5)), sizeof(uint32_t));
        memcpy(&elem6, &(frag.reg(6)), sizeof(uint32_t));
        memcpy(&elem7, &(frag.reg(7)), sizeof(uint32_t));
    }


};

template<> struct ResultPack<float, 16> {
    uint32_t elem0;
    uint32_t elem1;
    uint32_t elem2;
    uint32_t elem3;
    uint32_t elem4;
    uint32_t elem5;
    uint32_t elem6;
    uint32_t elem7;
    uint32_t elem8;
    uint32_t elem9;
    uint32_t elem10;
    uint32_t elem11;
    uint32_t elem12;
    uint32_t elem13;
    uint32_t elem14;
    uint32_t elem15;
    __host__ __device__ ResultPack(){
        elem0 = elem1 = elem2 = elem3 = 0;
        elem4 = elem5 = elem6 = elem7 = 0;
        elem8 = elem9 = elem10 = elem11 = 0;
        elem12 = elem13 = elem14 = elem15 = 0;
    }
    template<typename Frag>
    inline __device__ void setFrag(Frag &frag){
        memcpy(&(frag.reg(0)), &elem0, sizeof(uint32_t));
        memcpy(&(frag.reg(1)), &elem1, sizeof(uint32_t));
        memcpy(&(frag.reg(2)), &elem2, sizeof(uint32_t));
        memcpy(&(frag.reg(3)), &elem3, sizeof(uint32_t));
        memcpy(&(frag.reg(4)), &elem4, sizeof(uint32_t));
        memcpy(&(frag.reg(5)), &elem5, sizeof(uint32_t));
        memcpy(&(frag.reg(6)), &elem6, sizeof(uint32_t));
        memcpy(&(frag.reg(7)), &elem7, sizeof(uint32_t));
        memcpy(&(frag.reg(8)), &elem8, sizeof(uint32_t));
        memcpy(&(frag.reg(9)), &elem9, sizeof(uint32_t));
        memcpy(&(frag.reg(10)), &elem10, sizeof(uint32_t));
        memcpy(&(frag.reg(11)), &elem11, sizeof(uint32_t));
        memcpy(&(frag.reg(12)), &elem12, sizeof(uint32_t));
        memcpy(&(frag.reg(13)), &elem13, sizeof(uint32_t));
        memcpy(&(frag.reg(14)), &elem14, sizeof(uint32_t));
        memcpy(&(frag.reg(15)), &elem15, sizeof(uint32_t));

    }

    template<typename Frag>
    inline __device__ void getFrag(Frag &frag){
        memcpy(&elem0, &(frag.reg(0)), sizeof(uint32_t));     
        memcpy(&elem1, &(frag.reg(1)), sizeof(uint32_t));
        memcpy(&elem2, &(frag.reg(2)), sizeof(uint32_t));
        memcpy(&elem3, &(frag.reg(3)), sizeof(uint32_t));
        memcpy(&elem4, &(frag.reg(4)), sizeof(uint32_t));     
        memcpy(&elem5, &(frag.reg(5)), sizeof(uint32_t));
        memcpy(&elem6, &(frag.reg(6)), sizeof(uint32_t));
        memcpy(&elem7, &(frag.reg(7)), sizeof(uint32_t));
        memcpy(&elem8, &(frag.reg(8)), sizeof(uint32_t));
        memcpy(&elem9, &(frag.reg(9)), sizeof(uint32_t));
        memcpy(&elem10, &(frag.reg(10)), sizeof(uint32_t));
        memcpy(&elem11, &(frag.reg(11)), sizeof(uint32_t));
        memcpy(&elem12, &(frag.reg(12)), sizeof(uint32_t));
        memcpy(&elem13, &(frag.reg(13)), sizeof(uint32_t));
        memcpy(&elem14, &(frag.reg(14)), sizeof(uint32_t));
        memcpy(&elem15, &(frag.reg(15)), sizeof(uint32_t));
    }

};

#ifdef CASK_SDK_CASK_PLUGIN_LINK
//FIXME: current pipeline compiler doesn't support __lw_register_params__
#if __LWDACC_VER_MAJOR__ >= 11 && __LWDACC_VER_MINOR__ >= 2 &&\
        defined(__LWDA_ARCH__) && __LWDA_ARCH__ >= 800
#define ABI_PREFIX __lw_register_params__
#else
#define ABI_PREFIX
#endif

extern "C"  __device__  ABI_PREFIX  ResultPack<float, 4> activation_4(ResultPack<float,4> in);
extern "C"  __device__  ABI_PREFIX  ResultPack<float, 8> activation_8(ResultPack<float,8> in);
extern "C"  __device__  ABI_PREFIX  ResultPack<float, 16> activation_16(ResultPack<float,16> in);
#endif

#endif
