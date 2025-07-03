/*
 * Copyright (c) 2016, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 *
 */

#ifndef _LWGRAPH_H_
#define _LWGRAPH_H_

#include "stddef.h"
#include "lwda_runtime_api.h"
#include <library_types.h>

#include "stdint.h"

#ifndef LWGRAPH_API
#ifdef _WIN32
#define LWGRAPH_API __stdcall
#else
#define LWGRAPH_API
#endif
#endif

#ifdef __cplusplus
  extern "C" {
#endif

/* lwGRAPH status type returns */
typedef enum
{
    LWGRAPH_STATUS_SUCCESS            =0,
    LWGRAPH_STATUS_NOT_INITIALIZED    =1,
    LWGRAPH_STATUS_ALLOC_FAILED       =2,
    LWGRAPH_STATUS_ILWALID_VALUE      =3,
    LWGRAPH_STATUS_ARCH_MISMATCH      =4,
    LWGRAPH_STATUS_MAPPING_ERROR      =5,
    LWGRAPH_STATUS_EXELWTION_FAILED   =6,
    LWGRAPH_STATUS_INTERNAL_ERROR     =7,
    LWGRAPH_STATUS_TYPE_NOT_SUPPORTED =8,
    LWGRAPH_STATUS_NOT_COLWERGED      =9,
    LWGRAPH_STATUS_GRAPH_TYPE_NOT_SUPPORTED =10

} lwgraphStatus_t;

const char* lwgraphStatusGetString( lwgraphStatus_t status);

/* Opaque structure holding lwGRAPH library context */
struct lwgraphContext;
typedef struct lwgraphContext *lwgraphHandle_t;

/* Opaque structure holding the graph descriptor */
struct lwgraphGraphDescr;
typedef struct lwgraphGraphDescr *lwgraphGraphDescr_t;

/* Semi-ring types */
typedef enum
{
   LWGRAPH_PLUS_TIMES_SR = 0,
   LWGRAPH_MIN_PLUS_SR   = 1,
   LWGRAPH_MAX_MIN_SR    = 2,
   LWGRAPH_OR_AND_SR     = 3,
} lwgraphSemiring_t;

/* Topology types */
typedef enum
{
   LWGRAPH_CSR_32 = 0,
   LWGRAPH_CSC_32 = 1,
   LWGRAPH_COO_32 = 2,
} lwgraphTopologyType_t;

typedef enum
{
   LWGRAPH_DEFAULT                = 0,  // Default is unsorted.
   LWGRAPH_UNSORTED               = 1,  //
   LWGRAPH_SORTED_BY_SOURCE       = 2,  // CSR
   LWGRAPH_SORTED_BY_DESTINATION  = 3   // CSC
} lwgraphTag_t;

typedef enum
{
  LWGRAPH_MULTIPLY                = 0,
  LWGRAPH_SUM                     = 1,
  LWGRAPH_MIN                     = 2,
  LWGRAPH_MAX                     = 3
} lwgraphSemiringOps_t;

typedef enum
{
  LWGRAPH_MODULARITY_MAXIMIZATION  = 0, //maximize modularity with Lanczos solver  
  LWGRAPH_BALANCED_LWT_LANCZOS = 1, //minimize balanced cut with Lanczos solver  
 LWGRAPH_BALANCED_LWT_LOBPCG = 2 //minimize balanced cut with LOPCG solver  
} lwgraphSpectralClusteringType_t;

struct SpectralClusteringParameter {
       int n_clusters; //number of clusters
       int n_eig_vects; // //number of eigelwectors
       lwgraphSpectralClusteringType_t algorithm ; // algorithm to use
       float evs_tolerance; // tolerance of the eigensolver
       int evs_max_iter; // maximum number of iterations of the eigensolver
       float kmean_tolerance; // tolerance of kmeans
       int kmean_max_iter; // maximum number of iterations of kemeans 
       void * opt; // optional parameter that can be used for preconditioning in the future
};

typedef enum
{
LWGRAPH_MODULARITY,  // clustering score telling how good the clustering is compared to random assignment.
LWGRAPH_EDGE_LWT,  // total number of edges between clusters.
LWGRAPH_RATIO_LWT // sum for all clusters of the number of edges going outside of the cluster divided by the number of vertex inside the cluster
} lwgraphClusteringMetric_t;

struct lwgraphCSRTopology32I_st {
  int lwertices; // n+1
  int nedges; // nnz
  int *source_offsets; // rowPtr
  int *destination_indices; // colInd
};
typedef struct lwgraphCSRTopology32I_st *lwgraphCSRTopology32I_t;

struct lwgraphCSCTopology32I_st {
  int lwertices; // n+1
  int nedges; // nnz
  int *destination_offsets; // colPtr
  int *source_indices; // rowInd
};
typedef struct lwgraphCSCTopology32I_st *lwgraphCSCTopology32I_t;

struct lwgraphCOOTopology32I_st {
  int lwertices; // n+1
  int nedges; // nnz
  int *source_indices; // rowInd
  int *destination_indices; // colInd
  lwgraphTag_t tag;
};
typedef struct lwgraphCOOTopology32I_st *lwgraphCOOTopology32I_t;
/* Return properties values for the lwGraph library, such as library version */
lwgraphStatus_t LWGRAPH_API lwgraphGetProperty(libraryPropertyType type, int *value);

/* Open the library and create the handle */
lwgraphStatus_t LWGRAPH_API lwgraphCreate(lwgraphHandle_t *handle);

/*  Close the library and destroy the handle  */
lwgraphStatus_t LWGRAPH_API lwgraphDestroy(lwgraphHandle_t handle);

/* Create an empty graph descriptor */
lwgraphStatus_t LWGRAPH_API lwgraphCreateGraphDescr(lwgraphHandle_t handle, lwgraphGraphDescr_t *descrG);

/* Destroy a graph descriptor */
lwgraphStatus_t LWGRAPH_API lwgraphDestroyGraphDescr(lwgraphHandle_t handle, lwgraphGraphDescr_t descrG);

/* Set size, topology data in the graph descriptor  */
lwgraphStatus_t LWGRAPH_API lwgraphSetGraphStructure(lwgraphHandle_t handle, lwgraphGraphDescr_t descrG, void* topologyData, lwgraphTopologyType_t TType);

/* Query size and topology information from the graph descriptor */
lwgraphStatus_t LWGRAPH_API lwgraphGetGraphStructure (lwgraphHandle_t handle, lwgraphGraphDescr_t descrG, void* topologyData, lwgraphTopologyType_t* TType);

/* Allocate numsets vectors of size V reprensenting Vertex Data and attached them the graph.
 * settypes[i] is the type of vector #i, lwrrently all Vertex and Edge data should have the same type */
lwgraphStatus_t LWGRAPH_API lwgraphAllocateVertexData(lwgraphHandle_t handle, lwgraphGraphDescr_t descrG, size_t numsets, lwdaDataType_t  *settypes);

/* Allocate numsets vectors of size E reprensenting Edge Data and attached them the graph.
 * settypes[i] is the type of vector #i, lwrrently all Vertex and Edge data should have the same type */
lwgraphStatus_t LWGRAPH_API lwgraphAllocateEdgeData(lwgraphHandle_t handle, lwgraphGraphDescr_t descrG, size_t numsets, lwdaDataType_t *settypes);

/* Update the vertex set #setnum with the data in *vertexData, sets have 0-based index
 *  Colwersions are not sopported so lwgraphTopologyType_t should match the graph structure */
lwgraphStatus_t LWGRAPH_API lwgraphSetVertexData(lwgraphHandle_t handle, lwgraphGraphDescr_t descrG, void *vertexData, size_t setnum);

/* Copy the edge set #setnum in *edgeData, sets have 0-based index
 *  Colwersions are not sopported so lwgraphTopologyType_t should match the graph structure */
lwgraphStatus_t LWGRAPH_API lwgraphGetVertexData(lwgraphHandle_t handle, lwgraphGraphDescr_t descrG, void *vertexData, size_t setnum);

/* Colwert the edge data to another topology
 */
lwgraphStatus_t LWGRAPH_API lwgraphColwertTopology(lwgraphHandle_t handle,
                                lwgraphTopologyType_t srcTType, void *srcTopology, void *srcEdgeData, lwdaDataType_t *dataType,
                                lwgraphTopologyType_t dstTType, void *dstTopology, void *dstEdgeData);

/* Colwert graph to another structure
 */
lwgraphStatus_t LWGRAPH_API lwgraphColwertGraph(lwgraphHandle_t handle, lwgraphGraphDescr_t srcDescrG, lwgraphGraphDescr_t dstDescrG, lwgraphTopologyType_t dstTType);

/* Update the edge set #setnum with the data in *edgeData, sets have 0-based index
 */
lwgraphStatus_t LWGRAPH_API lwgraphSetEdgeData(lwgraphHandle_t handle, lwgraphGraphDescr_t descrG, void *edgeData, size_t setnum);

/* Copy the edge set #setnum in *edgeData, sets have 0-based index
 */
lwgraphStatus_t LWGRAPH_API lwgraphGetEdgeData(lwgraphHandle_t handle, lwgraphGraphDescr_t descrG, void *edgeData, size_t setnum);

/* create a new graph by extracting a subgraph given a list of vertices
 */
lwgraphStatus_t LWGRAPH_API lwgraphExtractSubgraphByVertex(lwgraphHandle_t handle, lwgraphGraphDescr_t descrG, lwgraphGraphDescr_t subdescrG, int *subvertices, size_t numvertices );
/* create a new graph by extracting a subgraph given a list of edges
 */
lwgraphStatus_t LWGRAPH_API lwgraphExtractSubgraphByEdge( lwgraphHandle_t handle, lwgraphGraphDescr_t descrG, lwgraphGraphDescr_t subdescrG, int *subedges , size_t numedges);

/* lwGRAPH Semi-ring sparse matrix vector multiplication
 */
lwgraphStatus_t LWGRAPH_API lwgraphSrSpmv(lwgraphHandle_t handle,
                                 const lwgraphGraphDescr_t descrG,
                                 const size_t weight_index,
                                 const void *alpha,
                                 const size_t x_index,
                                 const void *beta,
                                 const size_t y_index,
                                 const lwgraphSemiring_t SR);

/* Helper struct for Traversal parameters
 */
typedef struct {
	size_t pad[128];
} lwgraphTraversalParameter_t; 


/* Initializes traversal parameters with default values
 */
lwgraphStatus_t LWGRAPH_API lwgraphTraversalParameterInit(lwgraphTraversalParameter_t *param);

/* Stores/retrieves index of a vertex data where target distances will be stored 
 */ 
lwgraphStatus_t LWGRAPH_API lwgraphTraversalSetDistancesIndex(lwgraphTraversalParameter_t *param, const size_t value);

lwgraphStatus_t LWGRAPH_API lwgraphTraversalGetDistancesIndex(const lwgraphTraversalParameter_t param, size_t *value);

/* Stores/retrieves index of a vertex data where path predecessors will be stored
 */
lwgraphStatus_t LWGRAPH_API lwgraphTraversalSetPredecessorsIndex(lwgraphTraversalParameter_t *param, const size_t value);

lwgraphStatus_t LWGRAPH_API lwgraphTraversalGetPredecessorsIndex(const lwgraphTraversalParameter_t param, size_t *value);

/* Stores/retrieves index of an edge data which tells traversal algorithm whether path can go through an edge or not
 */
lwgraphStatus_t LWGRAPH_API lwgraphTraversalSetEdgeMaskIndex(lwgraphTraversalParameter_t *param, const size_t value);

lwgraphStatus_t LWGRAPH_API lwgraphTraversalGetEdgeMaskIndex(const lwgraphTraversalParameter_t param, size_t *value);

/* Stores/retrieves flag that tells an algorithm whether the graph is directed or not
 */
lwgraphStatus_t LWGRAPH_API lwgraphTraversalSetUndirectedFlag(lwgraphTraversalParameter_t *param, const size_t value);

lwgraphStatus_t LWGRAPH_API lwgraphTraversalGetUndirectedFlag(const lwgraphTraversalParameter_t param, size_t *value);

/* Stores/retrieves 'alpha' and 'beta' parameters for BFS traversal algorithm
 */
lwgraphStatus_t LWGRAPH_API lwgraphTraversalSetAlpha(lwgraphTraversalParameter_t *param, const size_t value);

lwgraphStatus_t LWGRAPH_API lwgraphTraversalGetAlpha(const lwgraphTraversalParameter_t param, size_t *value);

lwgraphStatus_t LWGRAPH_API lwgraphTraversalSetBeta(lwgraphTraversalParameter_t *param, const size_t value);

lwgraphStatus_t LWGRAPH_API lwgraphTraversalGetBeta(const lwgraphTraversalParameter_t param, size_t *value);



//Traversal available
typedef enum {
	LWGRAPH_TRAVERSAL_BFS=0
} lwgraphTraversal_t;


/* lwGRAPH Traversal API
 * Compute a traversal of the graph from a single vertex using algorithm specified by traversalT parameter
 */
lwgraphStatus_t LWGRAPH_API lwgraphTraversal(lwgraphHandle_t handle,
                               const lwgraphGraphDescr_t descrG,
                               const lwgraphTraversal_t traversalT,
			       const int *source_vert,
			       const lwgraphTraversalParameter_t params);

/* lwGRAPH Single Source Shortest Path (SSSP)
 * Callwlate the shortest path distance from a single vertex in the graph to all other vertices.
 */
lwgraphStatus_t LWGRAPH_API lwgraphSssp(lwgraphHandle_t handle,
                               const lwgraphGraphDescr_t descrG,
                               const size_t weight_index,
                               const int *source_vert,
                               const size_t sssp_index);

/* lwGRAPH WidestPath
 * Find widest path potential from source_index to every other vertices.
 */
lwgraphStatus_t LWGRAPH_API lwgraphWidestPath(lwgraphHandle_t handle,
                                  const lwgraphGraphDescr_t descrG,
                                  const size_t weight_index,
                                  const int *source_vert,
                                  const size_t widest_path_index);

/* lwGRAPH PageRank
 * Find PageRank for each vertex of a graph with a given transition probabilities, a bookmark vector of dangling vertices, and the damping factor.
 */
lwgraphStatus_t LWGRAPH_API lwgraphPagerank(lwgraphHandle_t handle,
                                   const lwgraphGraphDescr_t descrG,
                                   const size_t weight_index,
                                   const void *alpha,
                                   const size_t bookmark_index,
                                   const int has_guess,
                                   const size_t pagerank_index,
                                   const float tolerance,
                                   const int max_iter );

/* lwGRAPH contraction
 * given array of agregates contract graph with 
 * given (Combine, Reduce) operators for Vertex Set
 * and Edge Set;
 */ 
lwgraphStatus_t LWGRAPH_API lwgraphContractGraph(lwgraphHandle_t handle, 
                                                 lwgraphGraphDescr_t descrG, 
                                                 lwgraphGraphDescr_t contrdescrG, 
                                                 int *aggregates, 
                                                 size_t numaggregates,
                                                 lwgraphSemiringOps_t VertexCombineOp,
                                                 lwgraphSemiringOps_t VertexReduceOp,
                                                 lwgraphSemiringOps_t EdgeCombineOp,
                                                 lwgraphSemiringOps_t EdgeReduceOp,
                                                 int flag );

/* lwGRAPH spectral clustering
 * given a graph and solver parameters of struct SpectralClusteringParameter, 
 * assign vertices to groups such as 
 * intra-group connections are strong and/or inter-groups connections are weak 
 * using spectral technique.
 */ 
lwgraphStatus_t LWGRAPH_API lwgraphSpectralClustering(lwgraphHandle_t handle, 
                                   const lwgraphGraphDescr_t graph_descr,
                                   const size_t weight_index, 
                                   const struct SpectralClusteringParameter *params,
                                   int* clustering, 
                                   void* eig_vals,  
                                   void* eig_vects); 

/* lwGRAPH analyze clustering
 * Given a graph, a clustering, and a metric
 * compute the score that measures the clustering quality according to the metric.
 */ 
lwgraphStatus_t LWGRAPH_API lwgraphAnalyzeClustering( lwgraphHandle_t handle, 
                                  const lwgraphGraphDescr_t graph_descr, 
                                  const size_t weight_index,
                                  const int n_clusters, 
                                  const int* clustering,
                                  lwgraphClusteringMetric_t metric, 
                                  float * score); 

/* lwGRAPH Triangles counting
 * count number of triangles (cycles of size 3) formed by graph edges
 */
lwgraphStatus_t LWGRAPH_API lwgraphTriangleCount(lwgraphHandle_t handle, 
                                   const lwgraphGraphDescr_t graph_descr, 
                                   uint64_t* result); 

#if defined(__cplusplus)
} /* extern "C" */
#endif

#endif /* _LWGRAPH_H_ */

