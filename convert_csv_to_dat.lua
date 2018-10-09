require 'torch'
require 'cutorch'
require 'nn'
dofile 'Csv.lua'

-- data:
cmd = torch.CmdLine()
cmd:option('-dataset','A','data to convert')
cmd:text()
opt = cmd:parse(arg or {})

local ppiFile = Csv(opt.dataset..'.csv', "r")
local ppi = ppiFile:readall()
ppiFile:close()
torch.save( opt.dataset..'_labels.dat', ppi)
