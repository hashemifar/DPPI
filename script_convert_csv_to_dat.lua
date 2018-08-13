require 'torch'
require 'cutorch'
require 'nn'

cutorch.setDevice(2)
dofile 'Csv.lua'

local ppiFile = Csv("/home/hashemifar/ppi_predict/data/human.dimer", "r")
local ppi = ppiFile:readall()
ppiFile:close()
torch.save( '/home/hashemifar/ppi_predict/data/humanCV-dimer_labels.dat', ppi)
