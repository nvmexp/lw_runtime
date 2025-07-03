local SpatialBatchNormalization, parent =
   torch.class('lwdnn.SpatialBatchNormalization', 'lwdnn.BatchNormalization')

SpatialBatchNormalization.mode = 'LWDNN_BATCHNORM_SPATIAL'
SpatialBatchNormalization.nDim = 4
SpatialBatchNormalization.__version = 2
