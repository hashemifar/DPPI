require 'torch'
require 'cutorch'
require 'paths'
----------------------------------------------------------------------
workDir = '$Home/DPPI' --please change $Home to your own home directory (i.e. path to DPPI folder)
dataDir = '$Home/DPPI/' --please change $Home to your own home directory (i.e. path to DPPI folder)

print '==> Options'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Options:')

-- global:
cmd:option('-device', 2, 'set GPU Device')
cmd:option('-string', 'A', 'suffix to log files')
cmd:option('-saveModel', false, 'saves the model if true')
cmd:option('-seed', 1, 'manual seed')

-- data:
cmd:option('-dataset','A','data to use for training')

-- training:
cmd:option('-string', 'A', 'suffix to log files')
cmd:option('-optimization', 'SGD', 'optimization method: SGD | ADAM')
cmd:option('-learningRate', 0.01, 'learning rate at t=0')
cmd:option('-batchSize', 100, 'mini-batch size (1 = pure stochastic)')
cmd:option('-weightDecay', 0, 'weight decay (SGD only)')
cmd:option('-momentum', 0, 'momentum (SGD only)')
cmd:option('-epochs', 100, 'number of epochs')
cmd:option('-epochID', 1, 'staring epoch -used for resuming the run on servers')
cmd:option('-less_eval', false, 'evaluate every 10 epochs')
cmd:option('-crop', true, 'crop the sequence if true')
cmd:option('-cropLength', 512, 'length of the cropped sequence')
cmd:text()
opt = cmd:parse(arg or {})
print(opt)
----------------------------------------------------------------------
cutorch.setDevice(opt.device)

-- The string used to save the model and logs
saveName = string.format("%s-%s_crop%d-%s-rate_%g", opt.string, opt.dataset, opt.cropLength, opt.optimization, opt.learningRate)

dofile (workDir..'/load_data.lua')
dofile (workDir..'/model.lua')
dofile (workDir..'/train.lua')
dofile (workDir..'/evaluate.lua')

-- creating or loading the log tensor that stores evaluations
if opt.epochID == 1 then
  mylog = torch.Tensor(opt.epochs+1,20):zero()
  prlog = {}
else
  mylog = torch.load( dataDir..'results/'..saveName..'.t7', mylog )
end

epoch = opt.epochID

for counter=opt.epochID,opt.epochs do

  train()

  if opt.less_eval then
    if counter % 10 == 0 then
      evaluateAll( epoch )
    end
  else
    evaluateAll( epoch )
  end

  paths.mkdir(dataDir..'Result/')
  paths.mkdir(dataDir..'Model/')
  torch.save( dataDir..'Result/'..saveName..'.t7', mylog )
  torch.save( dataDir..'Result/pr_'..saveName..'.t7', prlog )
  if opt.saveModel then
    torch.save( dataDir..'/Model/'..saveName..'.t7', model )
  end

end
