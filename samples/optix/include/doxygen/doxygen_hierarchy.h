/************************************************************************************
/*
/*            MAIN CHAPTERS - These are the main chapters for the documentation
/*
/***********************************************************************************/

/**
  @page OptiXChapters OptiX Components
  
  An extensive description of OptiX framework components and their features can be found in
  the document \a OptiX_Programming_Guide.pdf shipped with the SDK.
  
  <B>Components API Reference</B>
  
  * OptiX - a scalable framework for building ray tracing applications.
  > See @ref OptiXApiReference for details .
  
  * OptiXpp - C++ wrapper around OptiX objects and handling functions.
  > See @ref optixpp for details .
  
  * OptiXu - simple API for performing raytracing queries using OptiX or the CPU. Also includes the rtuTraversal API
  subset for ray/triangle intersection.
  > See @ref LWDACReference and @ref rtu for details .
  
  * OptiX Prime - high performance API for intersecting a set of rays against a set of triangles.
  > See @ref PrimeReference for details .
  
  * OptiX Prime++ - C++ wrapper around OptiX Prime objects and handling functions.
  > See @ref optixprimepp for details .
  
*/

/************************************************************************************
/*
/*                   OPTIX, OPTIXPP AND OPTIXU DOXYGEN HIERARCHY
/*
/***********************************************************************************/

/** @defgroup OptiXApiReference OptiX API Reference
    @brief OptiX API functions
*/

  /** @defgroup Context Context handling functions
      @brief Functions related to an OptiX context
      @ingroup OptiXApiReference
  */

  /** @defgroup rtContextLaunch rtContextLaunch functions
      @brief Functions designed to launch OptiX ray tracing
      @ingroup Context
  */

  /** @defgroup GeometryGroup GeometryGroup handling functions
      @brief Functions related to an OptiX Geometry Group node
      @ingroup OptiXApiReference
  */

  /** @defgroup GroupNode GroupNode functions
      @brief Functions related to an OptiX Group node
      @ingroup OptiXApiReference
  */

  /** @defgroup SelectorNode SelectorNode functions
      @brief Functions related to an OptiX Selector node
      @ingroup OptiXApiReference
  */

  /** @defgroup TransformNode TransformNode functions
      @brief Functions related to an OptiX Transform node
      @ingroup OptiXApiReference
  */

  /** @defgroup AccelerationStructure Acceleration functions
      @brief Functions related to an OptiX Acceleration Structure node
      @ingroup OptiXApiReference
  */

  /** @defgroup GeometryInstance GeometryInstance functions
      @brief Functions related to an OptiX Geometry Instance node
      @ingroup OptiXApiReference
  */

  /** @defgroup Geometry Geometry functions
      @brief Functions related to an OptiX Geometry node
      @ingroup OptiXApiReference
  */

  /** @defgroup GeometryTriangles GeometryTriangles functions
      @brief Functions related to an OptiX GeometryTriangles node
      @ingroup OptiXApiReference
  */

  /** @defgroup Material Material functions
      @brief Functions related to an OptiX Material
      @ingroup OptiXApiReference
  */

  /** @defgroup Program Program functions
      @brief Functions related to an OptiX program
      @ingroup OptiXApiReference
  */

  /** @defgroup Buffer Buffer functions
      @brief Functions related to an OptiX Buffer
      @ingroup OptiXApiReference
  */

  /** @defgroup TextureSampler TextureSampler functions
      @brief Functions related to an OptiX Texture Sampler
      @ingroup OptiXApiReference
  */

  /** @defgroup Variables Variable functions
      @brief Functions related to variable handling
      @ingroup OptiXApiReference
  */
  /** @{ */

    /** @defgroup rtVariableSet Variable setters
        @brief Functions designed to modify the value of a program variable
    */

    /** @defgroup rtVariableGet Variable getters
        @brief Functions designed to modify the value of a program variable
    */
    
  /** @} */

  /** @defgroup CommandList CommandList functions
      @brief Functions related to an OptiX Command List
      @ingroup OptiXApiReference
  */

  /** @defgroup ContextFreeFunctions Context-free functions
      @brief Functions that don't pertain to an OptiX context to be called
      @ingroup OptiXApiReference
  */


  /**  @defgroup LWDACReference LWCA C Reference
       @brief OptiX Functions related to host and device code
       @ingroup OptiXApiReference
  */

    /** @defgroup LWDACDeclarations OptiX LWCA C declarations
        @brief Functions designed to declare programs and types used by OptiX device code
        @ingroup LWDACReference
    */
     
    /** @defgroup LWDACTypes OptiX basic types
        @brief Basic types used in OptiX
        @ingroup LWDACReference
    */
     
    /** @defgroup LWDACFunctions OptiX LWCA C functions
        @brief OptiX Functions designed to operate on device side. Some of them can also be included explicitly in host code if desired
        @ingroup LWDACReference
    */
    
      /** @defgroup rtTex Texture fetch functions
          @ingroup LWDACFunctions
      */
      
      /** @defgroup rtPrintf rtPrintf functions
          @ingroup LWDACFunctions
      */

  /** @defgroup optixpp OptiXpp wrapper
      @ingroup OptiXApiReference
  */

  /** @defgroup rtu rtu API
      The rtu API provides a simple interface for intersecting a set of rays
      against a set of triangles. It has been superseded by OptiX Prime.
      @ingroup OptiXApiReference
  */
  
    /** @defgroup rtuTraversal rtu Traversal API
        @ingroup rtu
    */


/************************************************************************************
/*
/*                 OPTIX PRIME AND OPTIX PRIME++ DOXYGEN HIERARCHY
/*
/***********************************************************************************/

/**
  @defgroup PrimeReference OptiX Prime API Reference
*/

  /** @defgroup Prime_Context Context
      @ingroup PrimeReference
  */

  /** @defgroup Prime_BufferDesc Buffer descriptor
      @ingroup PrimeReference
  */

  /** @defgroup Prime_Model Model
      @ingroup PrimeReference
  */

  /** @defgroup Prime_Query Query
      @ingroup PrimeReference
  */

  /** @defgroup Prime_Misc Miscellaneous functions
      @ingroup PrimeReference
  */
  
  /** @defgroup optixprimepp OptiX Prime++ wrapper
      @ingroup PrimeReference
  */

/**
  @defgroup InteropTypes OptiX Interoperability Types
  This section lists OpenGL and Direct3D texture formats that are lwrrently supported for interoperability with OptiX.
*/
