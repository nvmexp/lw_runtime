-- modules that can be colwerted to nn seamlessly
local layer_list = {
  'BatchNormalization',
  'SpatialBatchNormalization',
  'SpatialColwolution',
  'SpatialCrossMapLRN',
  'SpatialFullColwolution',
  'SpatialMaxPooling',
  'SpatialAveragePooling',
  'ReLU',
  'Tanh',
  'Sigmoid',
  'SoftMax',
  'LogSoftMax',
  'VolumetricBatchNormalization',
  'VolumetricColwolution',
  'VolumetricMaxPooling',
  'VolumetricAveragePooling',
}

-- goes over a given net and colwerts all layers to dst backend
-- for example: net = lwdnn.colwert(net, lwdnn)
function lwdnn.colwert(net, dst, exclusion_fn)
  return net:replace(function(x)
    if torch.type(x) == 'nn.gModule' then
      io.stderr:write('Warning: lwdnn.colwert does not work with nngraph yet. Ignoring nn.gModule')
      return x
    end
    local y = 0
    local src = dst == nn and lwdnn or nn
    local src_prefix = src == nn and 'nn.' or 'lwdnn.'
    local dst_prefix = dst == nn and 'nn.' or 'lwdnn.'

    local function colwert(v)
      local y = {}
      torch.setmetatable(y, dst_prefix..v)
      if v == 'ReLU' then y = dst.ReLU() end -- because parameters
      for k,u in pairs(x) do y[k] = u end
      if src == lwdnn and x.clearDesc then x.clearDesc(y) end
      if src == lwdnn and v == 'SpatialAveragePooling' then
        y.divide = true
        y.count_include_pad = v.mode == 'LWDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING'
      end
      return y
    end

    if exclusion_fn and exclusion_fn(x) then
      return x
    end
    local t = torch.typename(x)
    if t == 'nn.SpatialColwolutionMM' then
      y = colwert('SpatialColwolution')
    elseif t == 'inn.SpatialCrossResponseNormalization' then
      y = colwert('SpatialCrossMapLRN')
    else
      for i,v in ipairs(layer_list) do
        if torch.typename(x) == src_prefix..v then
          y = colwert(v)
        end
      end
    end
    return y == 0 and x or y
  end)
end
