local VolumetricBatchNormalization =
   torch.class('lwdnn.VolumetricBatchNormalization', 'lwdnn.BatchNormalization')

VolumetricBatchNormalization.mode = 'LWDNN_BATCHNORM_SPATIAL'
VolumetricBatchNormalization.nDim = 5
VolumetricBatchNormalization.__version = 2
