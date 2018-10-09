require 'nn'
require 'cunn'
require 'cudnn'
require 'nngraph'
----------------------------------------------------------------------
print '==> creating the model' -- creating the model
----------------------------------------------------------------------------
    model = nn.Sequential()

    my_net = nn.Sequential()
    my_net:add( nn.JoinTable(1) )

    my_net:add( cudnn.SpatialConvolution( num_features , 64, 1, 5, 1, 1, 0, 2) )
    my_net:add( cudnn.SpatialBatchNormalization( 64 ) )
    my_net:add( cudnn.ReLU(true) )
    my_net:add( cudnn.SpatialAveragePooling( 1, 4) )

    my_net:add( cudnn.SpatialConvolution( 64 , 128, 1, 5, 1, 1, 0, 2) )
    my_net:add( cudnn.SpatialBatchNormalization( 128 ) )
    my_net:add( cudnn.ReLU(true) )
    my_net:add( cudnn.SpatialAveragePooling( 1, 4) )

    my_net:add( cudnn.SpatialConvolution( 128 , 256, 1, 5, 1, 1, 0, 2) )
    my_net:add( cudnn.SpatialBatchNormalization( 256 ) )
    my_net:add( cudnn.ReLU(true) )
    my_net:add( cudnn.SpatialAveragePooling( 1, 4) )

    my_net:add( cudnn.SpatialConvolution( 256 , 512, 1, 5, 1, 1, 0, 2) )
    my_net:add( cudnn.SpatialBatchNormalization( 512 ) )
    my_net:add( cudnn.ReLU(true) )

    my_net:add( cudnn.SpatialAveragePooling( 1, opt.cropLength/64) )
    my_net:add( nn.Reshape(512) )

    outH = 512
    lin_net1 = nn.Sequential()
    lin_net1:add( nn.Linear( outH, outH) )
    lin_net1:add( nn.BatchNormalization( outH,1e-5,0.1,false ) )
    lin_net1:add( nn.ReLU(true))

    lin_net2 = nn.Sequential()
    lin_net2:add( nn.Linear( outH, outH) )
    lin_net2:add( nn.BatchNormalization( outH,1e-5,0.1,false ) )
    lin_net2:add( nn.ReLU(true))

    lin_net1c = lin_net1:clone('weight','bias','gradWeight','gradBias')
    lin_net2c = lin_net2:clone('weight','bias','gradWeight','gradBias')
      
    rand_layer1 = nn.Sequential()
    rand_layer1:add( nn.ConcatTable():add( lin_net1 ):add( lin_net2 ) )
    rand_layer1:add( nn.JoinTable(2) )
    rand_layer1:add( nn.Reshape(1024) )
  
    rand_layer2 = nn.Sequential()
    rand_layer2:add( nn.ConcatTable():add( lin_net2c ):add( lin_net1c ) )
    rand_layer2:add( nn.JoinTable(2) )
    rand_layer2:add( nn.Reshape(1024) )

    model:add( nn.MapTable():add( my_net ) )

    lin1 = nn.Linear( 512, 1);
    lin2 = lin1:clone('weight','bias','gradWeight','gradBias')

    model:add( nn.ParallelTable():add( rand_layer1  ):add( rand_layer2 ) )
    model:add( nn.CMulTable() )
    model:add( nn.View( 1024 ) )
    model:add( nn.Linear( 1024, 1 )) 
    model:add( nn.View(1) )    

if num_outputs == 1 then
    model:add( nn.Sigmoid() )
end


print(model)
----------------------------------------------------------------------------
-- Initialization
----------------------------------------------------------------------------
local function ConvInit(name)
   for k,v in pairs(model:findModules(name)) do
      local n = v.kW*v.kH*v.nOutputPlane
      v.weight:normal(0,math.sqrt(2/n))
      if cudnn.version >= 4000 then
         v.bias = nil
         v.gradBias = nil
      else
         v.bias:zero()
      end
   end
end

local function BNInit(name)
   for k,v in pairs(model:findModules(name)) do
      v.weight:fill(1)
      v.bias:zero()
   end
end

ConvInit('cudnn.SpatialConvolution')
ConvInit('nn.SpatialConvolution')
BNInit('fbnn.SpatialBatchNormalization')
for k,v in pairs(model:findModules('nn.Linear')) do
   v.bias:zero()
end

----------------------------------------------------------------------------
-- loss function
----------------------------------------------------------------------------
 criterion = nn.BCECriterion()

----------------------------------------------------------------------------
-- switch to Cuda
----------------------------------------------------------------------------
model:cuda()
criterion:cuda()
