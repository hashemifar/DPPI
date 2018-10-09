require 'xlua'    -- xlua provides useful tools, like progress bars
require 'optim'   -- an optimization package, for online and batch methods
require 'math'
----------------------------------------------------------------------
print '==> defining training procedure'
----------------------------------------------------------------------------
-- choosing the optimization method and hyper-parameters
----------------------------------------------------------------------------
if opt.optimization == 'SGD' then
   optimState = {
      learningRate = opt.learningRate,
      weightDecay = opt.weightDecay,
      momentum = opt.momentum,
      learningRateDecay = 0
   }
   optimMethod = optim.sgd
elseif opt.optimization =='ADAM' then
   optimState = {
      learningRate = opt.learningRate,
      weightDecay = opt.weightDecay,
   }
   optimMethod = optim.adam
else
   error('unknown optimization method')
end

function train()

  parameters,gradParameters = model:getParameters()
  epoch = epoch or 1

  if epoch == 300 or epoch == 400 then
    opt.learningRate = opt.learningRate / 10
  end

  local time = sys.clock()

  model:training()
  
  print('==> doing epoch on training data:')
  print("==> online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')

  local train_obj = 0
  local shuffle = torch.randperm( trainData.size )
  local num_batch = math.floor( trainData.size / opt.batchSize ) 

  for t = 1, num_batch*opt.batchSize, opt.batchSize do

    xlua.progress(t, num_batch*opt.batchSize )
    
    load_batch( trainData, shuffle[{{t,t+opt.batchSize-1}}] )
    model:zeroGradParameters()

    ---------------------------------------------
 
    local feval = function(x)

      if x ~= parameters then
        parameters:copy(x)
      end

      gradParameters:zero()
 
      local output = model:forward( inputs )


      local f = criterion:forward(output, targets)   
      
      train_obj = train_obj + f


      local df_do = criterion:backward(output, targets)

      df_do:cmul(weights)
      model:backward(inputs, df_do)

      --Values of RP are fixed
      model.modules[2].modules[1].modules[1].modules[1].modules[1].gradWeight:zero()
      model.modules[2].modules[1].modules[1].modules[1].modules[1].gradBias:zero()
      model.modules[2].modules[1].modules[1].modules[2].modules[1].gradWeight:zero()
      model.modules[2].modules[1].modules[1].modules[2].modules[1].gradBias:zero()
          
      model.modules[2].modules[2].modules[1].modules[1].modules[1].gradWeight:zero()
      model.modules[2].modules[2].modules[1].modules[1].modules[1].gradBias:zero()
      model.modules[2].modules[2].modules[1].modules[2].modules[1].gradWeight:zero()
      model.modules[2].modules[2].modules[1].modules[2].modules[1].gradBias:zero()

      return f,gradParameters
    
    end
    
    optimMethod(feval, parameters, optimState)
  
  end

    time = sys.clock() - time
    time = time / ( num_batch * opt.batchSize )
  print("\n==> time to learn 1 sample = " .. (time*1000) .. 'ms')

  epoch = epoch + 1

  train_obj = train_obj / ( num_batch * opt.batchSize )
  
  print('Objective: '.. train_obj )

  parameters, gradParameters = nil, nil

  collectgarbage()
  collectgarbage()
end
