----------------------------------------------------------------------
print '==> loading data'

function pair_crop_load(file, pos_weight, pNumber)


  local original_data = torch.load(file)

  local crop_data = {}

  counter = 1
  for i=1,#original_data do

   for j=1,pNumber[ original_data[i][1] ] do
      for k=1,pNumber[ original_data[i][2] ] do
	crop_data[ counter ] = {}
        crop_data[ counter ][1] = original_data[i][1]..'-sub'..j
        crop_data[ counter ][2] = original_data[i][2]..'-sub'..k
        crop_data[ counter ][3] = original_data[i][3]
        counter = counter + 1
      end
    end
  end
  local Data = {
        data = crop_data,
        org_data = original_data,
        pNum = pNumber,
        size = 0,
      }
  Data.size = #Data.data
  Data.pos_weight = pos_weight
  return Data
end

-- Tensor of labels that will be used during training
targets = torch.CudaTensor(opt.batchSize)

-- Tensor of weights that will be used during training
weights = torch.CudaTensor(opt.batchSize)

function pair_seq_load_batch( Data, index )

  inputs = {}
  inputs[1]={}
  inputs[2]={}
  for i=1, index:size(1) do

    local firstInd = 1
    local secondInd = 2
    if math.random() > 0.5 then
      firstInd = 2
      secondInd = 1
    end
    inputs[1][i] = feature[ Data.data[ index[i] ][ firstInd ] ]
    inputs[2][i] = feature[ Data.data[ index[i] ][ secondInd ] ]
   
    if inputs[1][i] == nil then
      print( Data.data[ index[i] ][ firstInd ] )
    end
    if inputs[2][i] == nil then
      print( Data.data[ index[i] ][ secondInd ] )
    end
      
    if num_outputs == 1 then
      targets[i] = 2 * Data.data[ index[i] ][ 3 ] - 1
    else
      targets[i] = Data.data[ index[i] ][ 3 ]
    end
  
    if targets[i] == 1 then
      weights[i] = Data.pos_weight
    else
      weights[i] = 1
    end

  end
end

-- List of validation sets and their names
valNames = {}
valData = {}

-- loading the dataset
if opt.dataset then 
  if opt.crop then
    pNumber = torch.load( dataDir..opt.dataset..'_number_crop_'..opt.cropLength..'.t7' )
    trainData = pair_crop_load( dataDir..opt.dataset..'_labels.dat',10, pNumber )

    valNames[1] = 'validation'
    valData[1] = pair_crop_load( dataDir..opt.dataset..'_valid_labels.dat',10, pNumber )
    print(valData[1].size)

    feature = torch.load( dataDir..opt.dataset..'_profile_crop_'..opt.cropLength..'.t7' )
    num_features = feature[ trainData.data[1][1] ]:size(2)
    num_outputs = 1

    load_batch = pair_seq_load_batch

  end
else
  error( 'Unknown dataset!')
end
