require 'torch'
require 'cutorch'

----------------------------------------------------------------------
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
cmd:option('-dataset','mouse_linear','data to use for training')

-- model:
--cmd:option('-model', 'linear', 'type of model')
cmd:option('-top_rand', false, 'random layer on top')
cmd:option('-loss' , 'nll', 'type of loss' )
-- training:
cmd:option('-string', 'A', 'suffix to log files')
cmd:option('-optimization', 'SGD', 'optimization method: SGD | ADAM')
cmd:option('-learningRate', 0.01, 'learning rate at t=0')
cmd:option('-batchSize', 100, 'mini-batch size (1 = pure stochastic)')
cmd:option('-weightDecay', 0, 'weight decay (SGD only)')
cmd:option('-momentum', 0, 'momentum (SGD only)')
cmd:option('-epochs', 50000, 'number of epochs')
cmd:option('-epochID', 1, 'staring epoch -used for resuming the run on servers')
cmd:option('-preprocess', 'nothing', 'preprocessing')
cmd:option('-less_eval', false, 'evaluate every 10 epochs')
cmd:option('-crop', true, 'crop the sequence if true')
cmd:option('-cropLength', 512, 'length of the cropped sequence')
cmd:option('-splitNum', 0, 'the split id')
cmd:option('-cvNum', 0, 'the cross validation id')
cmd:text()
opt = cmd:parse(arg or {})
print(opt)
----------------------------------------------------------------------
--os.exit()

cutorch.setDevice(opt.device)
--torch.manualSeed(opt.seed)


workDir = '/home/hashemifar/ppi_predict/final'
dataDir = '/home/hashemifar' --os.getenv("DATA_DIR")

-- The string used to save the model and logs
saveName = string.format("%s-%s_crop%d_cv%d_split%d-%s-%s-rate_%g", opt.string, opt.dataset, opt.cropLength, opt.cvNum, opt.splitNum, opt.model, opt.optimization, opt.learningRate )

dofile (workDir..'/load_data.lua')
dofile (workDir..'/model.lua')
dofile (workDir..'/train.lua')
dofile (workDir..'/evaluate.lua')

-- creating or loading the log tensor that stores evaluations
if opt.epochID == 1 then
  mylog = torch.Tensor(opt.epochs+1,20):zero()
  prlog = {}
else
  mylog = torch.load( dataDir..'/ppi_predict/results/'..saveName..'.t7', mylog )
end

epoch = opt.epochID

--evaluateAll( epoch )

for counter=opt.epochID,opt.epochs do

  train()

  if opt.less_eval then
    if counter % 10 == 0 then
      evaluateAll( epoch )
    end
  else
    evaluateAll( epoch )
  end

  torch.save( dataDir..'/ppi_predict/results/'..saveName..'.t7', mylog )
  torch.save( dataDir..'/ppi_predict/results/pr_'..saveName..'.t7', prlog )
  if opt.saveModel then
    torch.save( dataDir..'/ppi_predict/models/'..saveName..'.t7', model )
  end

end
