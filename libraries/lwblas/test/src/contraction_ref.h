#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <string.h>

#include <lwda_runtime.h>
#include <lwblas_v2.h>

#include <unordered_map>
#include <unordered_set>
#include <map>
#include <iostream>
#include <string>
#include <algorithm>
#include <vector>
#include <list>
#include <thread>


#include <lwtensor.h>
#include <lwtensor/internal/lwtensor.h>
#include <lwtensor/internal/utilEx.h>
#include <lwtensor/internal/operators.h>
#include <lwtensor/internal/defines.h>

namespace ReferenceTC
{
   using namespace LWTENSOR_NAMESPACE;

   template<typename T>
   void intersect(const T &a, const T &b, 
         std::list<mode_type> &intersection)
   {
      for(auto ai : a )
         for(auto bi : b )
            if( ai == bi )
            {
               intersection.emplace_back(ai);
               break;
            }
   }
   void intersect(const mode_type *a, int na, const mode_type *b, int nb, 
         std::list<mode_type> &intersection)
   {
      for(int i=0; i < na; ++i)
         for(int j=0; j < nb; ++j)
            if( a[i] == b[j] )
            {   
               intersection.emplace_back(a[i]);
               break;
            }
   }
   lwtensorStatus_t  initStride(const TensorDescriptor *desc,
         const std::list<mode_type> &mode,
         std::unordered_map<mode_type, stride_type> &strideA)
   {
      if( desc == NULL ) return LWTENSOR_STATUS_ILWALID_VALUE;

      int i = 0;
      for(auto m : mode )
      {
         strideA[m] = desc->getStride(i);
         i++;
      }
      return LWTENSOR_STATUS_SUCCESS;
   }

   lwtensorStatus_t initExtent(const TensorDescriptor *desc, const mode_type* mode, std::unordered_map<mode_type, extent_type> &extent)
   {
      if( desc == NULL || mode == NULL ) return LWTENSOR_STATUS_SUCCESS;

      for(int i=0; i < desc->getNumModes(); ++i)
      {
         auto modeSize = desc->getExtent(i);
         if( desc->getExtent(i) == 0 || extent.find(mode[i]) != extent.end() && extent.at(mode[i]) != modeSize )
         {
            fprintf(stderr, "LWTENSOR ERROR: extent of mode %c does not match.\n",mode[i]);
            return LWTENSOR_STATUS_ILWALID_VALUE;
         }
         else
            extent[mode[i]] = modeSize;
      }
      return LWTENSOR_STATUS_SUCCESS;
   }

   template<typename T>
      extent_type getSize(const T &mode,
            const std::unordered_map<mode_type, extent_type> &extent)
      {
         extent_type total_size = 1;
         for( const auto i : mode )
            total_size *= extent.at(i);
         return total_size;
      }

   stride_type getOffset(extent_type loopCounter, const std::list<mode_type> &mode, 
         const std::unordered_map<mode_type, extent_type> &extent,
         const std::unordered_map<mode_type,stride_type> &stride)
   {
      stride_type offset = 0;
      for( auto idx : mode )
      {
         extent_type offsetDim = loopCounter % extent.at(idx);
         loopCounter /= extent.at(idx);

         offset += offsetDim * stride.at(idx);
      }
      return offset;
   }

   template<int dummy>
      lwtensorStatus_t tensorMult_ref(const std::list<mode_type> &modeM, const std::list<mode_type> &modeN,
                                      const std::list<mode_type> &modeK, const std::list<mode_type> &modeL,
                                      const std::unordered_map<mode_type, extent_type> &extent,
            const void *alpha, const void *A, const lwdaDataType_t typeA, const lwtensorOperator_t opA, const std::unordered_map<mode_type,stride_type> &strideA,
                               const void *B, const lwdaDataType_t typeB, const lwtensorOperator_t opB, const std::unordered_map<mode_type,stride_type> &strideB,
            const void *beta,  const void *C, const lwdaDataType_t typeC, const lwtensorOperator_t opC, const std::unordered_map<mode_type,stride_type> &strideC,
                                     void *D, lwdaDataType_t typeScalar, lwdaDataType_t typeCompute,
            const lwtensorOperator_t opOut, const lwtensorOperator_t opReduce, const bool isReduction)
   {
       (void) modeM, (void)modeN;
       (void) modeK, (void)modeL;
       (void) extent;
       (void) alpha, (void)A, (void)typeA, (void)opA, (void)strideA;
       (void) B, (void)typeB, (void)opB, (void)strideB;
       (void) beta, (void)C, (void)typeC, (void)opC, (void)strideC;
       (void) D, (void)typeScalar, (void)typeCompute;
       (void) opOut, (void)opReduce, (void)isReduction;
       return LWTENSOR_STATUS_NOT_SUPPORTED;
   }

   template<typename T>
   constexpr bool isReal();
   template<>
   constexpr bool isReal<half>(){ return true; }
#if LWTENSOR_LWDA_VERSION_MAJOR >= 11
   template<>
   constexpr bool isReal<BFloat16>(){ return true; }
#endif
   template<>
   constexpr bool isReal<float>(){ return true; }
   template<>
   constexpr bool isReal<double>(){ return true; }
   template<>
   constexpr bool isReal<lwComplex>() { return false; }
   template<>
   constexpr bool isReal<lwDoubleComplex>() { return false; }

   template<typename T>
   constexpr bool isComplex();
   template<>
   constexpr bool isComplex<half>(){ return false; }
#if LWTENSOR_LWDA_VERSION_MAJOR >= 11
   template<>
   constexpr bool isComplex<BFloat16>(){ return false; }
#endif
   template<>
   constexpr bool isComplex<float>(){ return false; }
   template<>
   constexpr bool isComplex<double>(){ return false; }
   template<>
   constexpr bool isComplex<lwComplex>() { return true; }
   template<>
   constexpr bool isComplex<lwDoubleComplex>() { return true; }

   template<int dummy,
            typename TypeA,
            typename TypeB,
            typename TypeC,
            typename TypeScalar,
            typename TypeCompute,
            typename... Args>
      lwtensorStatus_t tensorMult_ref(const std::list<mode_type> &modeM, const std::list<mode_type> &modeN,
                                      const std::list<mode_type> &modeK, const std::list<mode_type> &modeL,
                                      const std::unordered_map<mode_type, extent_type> &extent,
            const void *alpha_, const void *A_, const lwdaDataType_t typeA, const lwtensorOperator_t opA, const std::unordered_map<mode_type,stride_type> &strideA,
                                const void *B_, const lwdaDataType_t typeB, const lwtensorOperator_t opB, const std::unordered_map<mode_type,stride_type> &strideB,
            const void *beta_,  const void *C_, const lwdaDataType_t typeC, const lwtensorOperator_t opC, const std::unordered_map<mode_type,stride_type> &strideC,
                                      void *D_, lwdaDataType_t typeScalar, lwdaDataType_t typeCompute,
            const lwtensorOperator_t opOut, const lwtensorOperator_t opReduce, const bool isReduction)
   {
       constexpr lwtensorOperator_t OP_AB = LWTENSOR_OP_MUL;
    if ( (typeA == toLwdaDataType<TypeA>()) &&
         ((B_ == nullptr) || (typeB == toLwdaDataType<TypeB>())) &&
         (typeC == toLwdaDataType<TypeC>()) &&
         (typeScalar == toLwdaDataType<TypeScalar>()) &&
         (typeCompute == toLwdaDataType<TypeCompute>()))
    {
        const TypeA* A = (const TypeA*) A_;
        const TypeB* B = (const TypeB*) B_;
        const TypeC* C = (const TypeC*) C_;
              TypeC* D = (TypeC*) D_;
        const TypeScalar* alpha = (const TypeScalar*) alpha_;
        const TypeScalar* beta = (const TypeScalar*) beta_;
        constexpr bool isRealTimesComplex = (isReal<TypeA>() && isComplex<TypeB>()) ||
                                            (isReal<TypeB>() && isComplex<TypeA>());
      extent_type extentM = getSize(modeM, extent);
      extent_type extentN = getSize(modeN, extent);
      extent_type extentK = getSize(modeK, extent);
      extent_type extentL = getSize(modeL, extent);

      auto numThreads = std::max(1u,std::thread::hardware_conlwrrency()/2);

//      printf("l: ");
//      for(auto idx : modeL )
//         printf("%d %d %d %d %d",idx, extent.at(idx),  strideA.at(idx),strideB.at(idx), strideC.at(idx));
//      printf("\nm: ");
//      for(auto idx : modeM )
//         printf("%d %d %d %d",idx, extent.at(idx), strideA.at(idx), strideC.at(idx));
//      printf("\nn: ");
//      for(auto idx : modeN )
//         printf("%d %d %d %d",idx, extent.at(idx), strideB.at(idx), strideC.at(idx));
//      printf("\nk: ");
//      for(auto idx : modeK )
//         printf("%d %d %d %d",idx, extent.at(idx), strideA.at(idx), strideB.at(idx));
//      printf("\n");

      bool betaIsZero  = lwIsEqual( *beta,  lwGet<TypeScalar>( 0 ) );
      bool alphaIsZero  = lwIsEqual( *alpha,  lwGet<TypeScalar>( 0 ) );

      for( extent_type l = 0; l < extentL; l++ )
      { stride_type offsetCl = getOffset(l, modeL, extent, strideC);
         stride_type offsetAl = getOffset(l, modeL, extent, strideA);
         stride_type offsetBl = getOffset(l, modeL, extent, strideB);
         #pragma omp parallel for num_threads(numThreads)
         for( extent_type i = 0; i < extentM; i++ )
         {
            stride_type offsetCm = getOffset(i, modeM, extent, strideC);
            stride_type offsetAm = getOffset(i, modeM, extent, strideA);
            for( extent_type j = 0; j < extentN; j++ )
            {
               stride_type offsetCn = getOffset(j, modeN, extent, strideC);
               stride_type offsetBn = getOffset(j, modeN, extent, strideB);
               stride_type offsetC = offsetCm + offsetCn + offsetCl;

               /** Handle the special case when alpha and beta are zero. */
               {
                  /** Initialize aclwmulator to zero. */
                  auto acc = getNeutralElement<TypeCompute>(opReduce);
                  if (alphaIsZero)
                  {
                      acc = lwGet<TypeCompute>(0);
                  }
                  else
                  {
                      /** Accumulate along the k-dimension. */
                      for( extent_type k = 0; k < extentK; k++ )
                      {
                          stride_type offsetAk = getOffset(k, modeK, extent, strideA);
                          stride_type offsetBk = getOffset(k, modeK, extent, strideB);
                          stride_type offsetA  = offsetAm + offsetAk + offsetAl;
                          stride_type offsetB  = offsetBn + offsetBk + offsetBl;
                          const auto b = isReduction ? getNeutralElement<TypeCompute>(OP_AB) : lwtensorUnaryOp(lwGet<TypeCompute>(B[offsetB]),opB);
                          acc = lwtensorBinaryOp(acc,
                                  Operator<TypeCompute, TypeCompute, TypeCompute, OP_AB, isRealTimesComplex>::execute(lwtensorUnaryOp( lwGet<TypeCompute>(A[offsetA]),opA), b), opReduce);
                      }
                  }
                  /** Handle the special case when beta is zero. */
                  if (betaIsZero)
                  {
                     D[offsetC] = lwGet<TypeC>( lwtensorUnaryOp(
                             lwtensorBinaryOp(
                             Operator<TypeCompute, TypeCompute, TypeCompute, LWTENSOR_OP_MUL>::execute( lwGet<TypeCompute>(*alpha), acc ), lwGet<TypeCompute>( 0 ), LWTENSOR_OP_ADD ), opOut) );
                  }
                  else
                  {
                     D[offsetC] = lwGet<TypeC>( lwtensorUnaryOp( 
                                 lwtensorBinaryOp( 
                                 Operator<TypeCompute, TypeCompute, TypeCompute, LWTENSOR_OP_MUL>::execute( lwGet<TypeCompute>(*alpha), acc ), 
                                 Operator<TypeCompute, TypeCompute, TypeCompute, LWTENSOR_OP_MUL>::execute( lwGet<TypeCompute>(*beta), lwtensorUnaryOp(lwGet<TypeCompute>( C[ offsetC ]), opC) ), LWTENSOR_OP_ADD), opOut) );
                  }
               }
            }
         }
      }
      return LWTENSOR_STATUS_SUCCESS;
    }else{
        return tensorMult_ref<dummy, Args...>(modeM, modeN, modeK, modeL, extent,
            alpha_, A_, typeA, opA, strideA,
                    B_, typeB, opB, strideB,
            beta_,  C_, typeC, opC, strideC,
                    D_, typeScalar, typeCompute, opOut, opReduce, isReduction);
    }
   }

   lwtensorStatus_t tensorMult_ref(
    const void* alpha, const void *A, const lwtensorTensorDescriptor_t* descA_, const mode_type* modeA,
                       const void *B, const lwtensorTensorDescriptor_t* descB_, const mode_type* modeB,
    const void* beta,  const void *C, const lwtensorTensorDescriptor_t* descC_, const mode_type* modeC,
                             void *D,
    lwtensorOperator_t opOut, lwtensorOperator_t opReduce,
    lwdaDataType_t typeScalar, lwdaDataType_t typeCompute, bool isReduction, std::list<mode_type>* enforced_modeK = nullptr )
   {
      std::unordered_map<mode_type, stride_type> strideA_;
      std::unordered_map<mode_type, stride_type> strideB_;
      std::unordered_map<mode_type, stride_type> strideC_;
      std::unordered_map<mode_type, extent_type> extent;

      TensorDescriptor *descA = (TensorDescriptor*) descA_;
      TensorDescriptor *descB = (TensorDescriptor*) descB_;
      TensorDescriptor *descC = (TensorDescriptor*) descC_;

      initExtent(descA, modeA, extent);
      initExtent(descB, modeB, extent);
      initExtent(descC, modeC, extent);

      auto nmodeA = descA->getNumModes();
      auto nmodeB = descB->getNumModes();
      auto nmodeC = descC->getNumModes();

      std::list<mode_type> modeA_;
      std::list<mode_type> modeB_;
      std::list<mode_type> modeC_;
      for(int i=0; i < nmodeA; ++i) modeA_.push_back(modeA[i]);
      for(int i=0; i < nmodeB; ++i) modeB_.push_back(modeB[i]);
      for(int i=0; i < nmodeC; ++i) modeC_.push_back(modeC[i]);

      if( initStride(descA, modeA_, strideA_) != LWTENSOR_STATUS_SUCCESS)
      {
          printf("ERROR %s %d\n", __FILE__, __LINE__);
          exit(-1);
      }
      if( initStride(descB, modeB_, strideB_) != LWTENSOR_STATUS_SUCCESS)
      {
          printf("ERROR %s %d\n", __FILE__, __LINE__);
          exit(-1);
      }
      if( initStride(descC, modeC_, strideC_) != LWTENSOR_STATUS_SUCCESS)
      {
          printf("ERROR %s %d\n", __FILE__, __LINE__);
          exit(-1);
      }

      lwdaDataType_t typeA = (descA) ? descA->getDataType() : LWDA_R_32F;
      lwdaDataType_t typeB = (descB) ? descB->getDataType() : LWDA_R_32F;
      lwtensorOperator_t opA = descA->getOp();
      lwtensorOperator_t opB = descB->getOp();

      if( descA && descB && (typeA == LWDA_C_32F && typeB == LWDA_R_32F) || (typeA == LWDA_C_64F && typeB == LWDA_R_64F))
      {
          // ensure that A is always read (in the case of real times complex)
          std::swap(descA, descB);
          std::swap(opA, opB);
          std::swap(typeA, typeB);
          std::swap(modeA, modeB);
          std::swap(A, B);
          std::swap(nmodeA, nmodeB);
          std::swap(strideA_, strideB_);
      }

      constexpr int kMaxNumModes = 64;
      mode_type augmentedModeA[kMaxNumModes];
      static_assert(kMaxNumModes >= LWTENSOR_NAMESPACE::kMaxNumModes, "too many modes");
      for(int i=0; i < nmodeA; ++i){
          augmentedModeA[i] = modeA[i];
      }

      // ensure that modes that are in C but not in A are implicitly broadcasted
      if (isReduction)
      {
          for(int i=0; i < nmodeC; ++i){
            auto mode = modeC[i];
            if (strideA_.find(mode) == strideA_.end())
            {
                strideA_[mode] = 0; // add artificial mode with stride of zero
                augmentedModeA[nmodeA++] = mode;
            }
          }
      }

      std::list<mode_type> modeM; 
      std::list<mode_type> modeN; 
      std::list<mode_type> modeK; 
      std::list<mode_type> modeL; // looped/batched modes
      intersect(augmentedModeA, nmodeA, modeC, nmodeC, modeM);
      intersect(modeB, nmodeB, modeC, nmodeC, modeN);
      intersect(modeB, nmodeB, augmentedModeA, nmodeA, modeK);
      intersect(modeM, modeN, modeL);
      for(auto l : modeL)
      {
         modeM.remove(l);
         modeN.remove(l);
         modeK.remove(l);
      }

      /** Fix for empty modeK. */
      if ( modeK.empty() )
      {
         const mode_type kReservedMode = 233;
         assert( extent.find( kReservedMode ) == extent.end() );
         assert( strideB_.find( kReservedMode ) == strideB_.end() );
         assert( strideA_.find( kReservedMode ) == strideA_.end() );
         extent[ kReservedMode ] = 1;
         if ( strideB_.find( modeB_.back() ) != strideB_.end() )
            strideB_[ kReservedMode ] = strideB_[ modeB_.back() ] * extent.at( modeB_.back() );
         else
            strideB_[ kReservedMode ] = 0;
         if ( strideA_.find( modeA_.back() ) != strideA_.end() )
            strideA_[ kReservedMode ] = strideA_[ modeA_.back() ] * extent.at(modeA_.back());
         else
            strideA_[ kReservedMode ] = 0;
         modeB_.push_back( kReservedMode );
         modeA_.push_back( kReservedMode );
         modeK.push_back( kReservedMode );
         //std::cout << "missing modeK" << std::endl;
      }

      if ( enforced_modeK )
      {
         /** If the order of modeK is enforced, then override modeK. */
         if ( enforced_modeK->size() == modeK.size() ) 
         {
           int n_missing_mode = 0;
           for ( auto &m : modeK ) 
             if ( std::find( enforced_modeK->begin(), enforced_modeK->end(), m ) == enforced_modeK->end() )
             {
               //std::cout << "(the same size) missing modes:" << std::string( 1, m ) << std::endl;
               n_missing_mode ++;
             }
           if ( !n_missing_mode ) modeK = *enforced_modeK;
         }
         else
         {
            //std::cout << "missing modes\n";
            //for ( auto &m : *enforced_modeK ) std::cout << std::string( 1, m ) << ",";
            //std::cout << "\n";
            //for ( auto &m : modeK ) std::cout << std::string( 1, m ) << ",";
            //std::cout << "\n";
            auto proposed_modeK = *enforced_modeK;
            for ( auto &m : modeK ) if ( std::find( proposed_modeK.begin(), proposed_modeK.end(), m ) == proposed_modeK.end() ) proposed_modeK.push_back( m );
            if ( proposed_modeK.size() == modeK.size() ) modeK = proposed_modeK;
            //std::cout << "missing modes\n";
            //for ( auto &m : *enforced_modeK ) std::cout << std::string( 1, m ) << ",";
            //std::cout << "\n";
            //for ( auto &m : modeK ) std::cout << std::string( 1, m ) << ",";
            //std::cout << "\n";
         }
      }

      lwdaDataType_t typeC = descC->getDataType();
      lwtensorOperator_t opC = descC->getOp();

      // tensor core configs remap compute-type to aclwmulator precision
      if(typeA == LWDA_R_32F && (isReduction || typeB == LWDA_R_32F) && typeC == LWDA_R_32F)
      {
          if( typeCompute == LWDA_R_16F )
              typeCompute = LWDA_R_32F;
#if LWTENSOR_LWDA_VERSION_MAJOR >= 11
          if( typeCompute == LWDA_R_16BF )
              typeCompute = LWDA_R_32F;
          if( typeCompute == LWDA_R_TF32 )
              typeCompute = LWDA_R_32F;
#endif
      }
      if(typeA == LWDA_R_16F && (isReduction || typeB == LWDA_R_16F) && typeC == LWDA_R_16F)
      {
          typeCompute = LWDA_R_64F; // promote compute-type to float for more accurate ref
      }
#if LWTENSOR_LWDA_VERSION_MAJOR >= 11
      if(typeA == LWDA_R_16BF && (isReduction || typeB == LWDA_R_16BF) && typeC == LWDA_R_16BF)
      {
          typeCompute = LWDA_R_64F; // promote compute-type to float for more accurate ref
      }
#endif
      if(typeA == LWDA_C_32F && (isReduction || typeB == LWDA_C_32F) && typeC == LWDA_C_32F)
      {
          if( typeCompute == LWDA_C_TF32 )
              typeCompute = LWDA_C_32F;
      }

        /*TypeA,  TypeB,  TypeC,  scalar,  comp*/
      return tensorMult_ref<42,
          half,   half,   half,   half,   half,
          half,   half,   half,   float,  half,
          half,   half,   half,   float,  double,
          half,   half,   half,   float,  float,
          half,   half,   float,  float,  float,
          float,  float,  float,  float,  float,
          float,  float,  float,  float,  half,
          float,  float,  float,  double, double,
          float,  float,  double, float,  float,
          float,  float,  double, double, double,
          float,  double, float,  float,  float,
          float,  double, float,  double, double,
          float,  double, double, float,  float,
          float,  double, double, double, double,
          double, float,  float,  float,  float,
          double, float,  float,  double, double,
          double, float,  double, float,  float,
          double, float,  double, double, double,
          double, double, float,  float,  float,
          double, double, float,  double, double,
          double, double, double, float,  float,
          double, double, double, double, double,
          double, double, double, double, float,
          double, double, double, double, half,
#if LWTENSOR_LWDA_VERSION_MAJOR >= 11
          BFloat16, BFloat16, BFloat16, float, double,
          BFloat16, BFloat16, BFloat16, float, float,
          BFloat16, BFloat16, BFloat16, float, BFloat16,
#endif
          
          lwComplex,       lwComplex,       lwComplex,       lwComplex,       lwComplex,
          lwDoubleComplex, lwDoubleComplex, lwDoubleComplex, lwDoubleComplex, lwDoubleComplex,

          /* mixed-precision */
          lwDoubleComplex, lwDoubleComplex, lwDoubleComplex, lwDoubleComplex, lwComplex,
          double,          lwDoubleComplex, lwDoubleComplex, lwDoubleComplex, lwComplex,
          lwDoubleComplex, double,          lwDoubleComplex, lwDoubleComplex, lwComplex,

          /* complex-times-real */
          float,           lwComplex,       lwComplex,       lwComplex,       lwComplex,
          lwComplex,       float,           lwComplex,       lwComplex,       lwComplex,
          double,          lwDoubleComplex, lwDoubleComplex, lwDoubleComplex, lwDoubleComplex,
          lwDoubleComplex, double,          lwDoubleComplex, lwDoubleComplex, lwDoubleComplex
              > (modeM, modeN, modeK, modeL, extent,
                 alpha, A, typeA, opA, strideA_,
                        B, typeB, opB, strideB_,
                 beta,  C, typeC, opC, strideC_,
                        D, typeScalar, typeCompute, opOut, opReduce, isReduction);
   }
}
