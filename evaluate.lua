require 'xlua'    -- xlua provides useful tools, like progress bars
require 'optim'   -- an optimization package, for online and batch methods
require 'math'
----------------------------------------------------------------------
print '==> defining evaluation procedure'

-- evaluate the model on training set and all validation sets
function evaluateAll( epoch )

  print('################## EVALUATION ##################')
 
  prlog[epoch] = {}
  prlog[epoch][1] = {}
  prRank = false
  mylog[epoch][1], mylog[epoch][2],prlog[epoch][1][1],prlog[epoch][1][2], prlog[epoch][1][3] = evaluate( trainData )
  print(' Training MAP and Objective ('..trainData.size..') : '.. mylog[epoch][2] )

  local tmp_obj = 0 

  for i=1,#valData do

    prlog[epoch][1+i] = {}    
    if (i==1) then
    	prRank = false
    else
	prRank = true
    end
    tmp_obj, mylog[epoch][2+i],prlog[epoch][1+i][1],prlog[epoch][1+i][2], prlog[epoch][1+i][3] = evaluate( valData[i] )
    print(' Validation MAP ('..valNames[i]..' - '..valData[i].size..') : '.. mylog[epoch][2+i] )
    
  end

end

-- evaluate the model on the given data
function evaluate( Data )

  parameters,gradParameters = model:getParameters()
  
  model:evaluate()

  local val_obj = 0
  local counter = 0
  local shuffle = torch.randperm( Data.size )
  for i=1,Data.size do
    shuffle[i]=i
  end

  local num_batch = math.floor( Data.size / opt.batchSize )
  
  local val_score = torch.Tensor( num_batch*opt.batchSize, 1)
  local val_labels = torch.Tensor( num_batch*opt.batchSize, 1) 

  for t = 1, num_batch*opt.batchSize, opt.batchSize do

    load_batch( Data, shuffle[{{t,t+opt.batchSize-1}}] )

    local output = model:forward( inputs )

    for k=1,output:size(1) do
   
      val_score[counter*opt.batchSize+k][1] = output[k][1]
      val_labels[counter*opt.batchSize+k][1]=targets[k]
    end

    counter = counter + 1
  end 

  val_obj = val_obj / counter

  if opt.crop then
    val_score, val_labels = recalculate_crop( val_score, val_labels, Data )
  end

  local val_map, precision, recall, specificity  = MAP( val_score, val_labels )

  parameters,gradParameters = nil, nil

  collectgarbage()
  collectgarbage()

  return val_obj, val_map, precision, recall, specificity

end

function recalculate_crop( v_score, v_labels, Data )

  new_score = torch.Tensor( #Data.org_data , 1):zero()
  new_labels = torch.Tensor( #Data.org_data , 1):zero()

  counter = 1

  for i=1, #Data.org_data do
    myscore1 = 0
    myscore2 = 0
    for j=1, Data.pNum[ Data.org_data[i][1] ] do
      for k=1, Data.pNum[ Data.org_data[i][2] ] do
        new_labels[i][1] = Data.org_data[i][3]
        if counter <= v_score:size(1) then
          myscore1 =  math.max( myscore1, v_score[counter][1] )
          myscore2 = myscore2 + v_score[counter][1]
          counter = counter + 1
        end
      end
    end
    new_score[i][1] = myscore1
  end
  
  return new_score, new_labels
end



-- Calculates the Mean Average Precision (MAP)
function MAP (score, truth)
  local x,ind,map,P,TP,FP,N
  x, ind = torch.sort(score, 1, true)

  if num_outputs == 1 then
    truth:add(1):div(2)
  end
  
  P = torch.sum( truth,1 )
  local precision = torch.Tensor(score:size(1),1)
  local recall = torch.Tensor(score:size(1),1)
  local specificity = torch.Tensor(score:size(1),1)

  my_error = 0
  map = 0
  for c=1, score:size(2) do
    TP = 0
    FP = 0
    FN = 0

    N = score:size(1) - P[1][c] 
    for i=1, score:size(1) do  
      TP = TP + truth[ind[i][c]][c]
      FP = FP + (1 - truth[ind[i][c]][c] )      

      precision[i][1] = TP / (FP + TP)
      recall[i][1] = TP / P[1][c] 
      specificity[i][1] = FP / N  
      
      map = map + ( truth[ind[i][c]][c] * TP / ( P[1][c] * ( FP + TP ) ) )
    end
  end


  print(my_error)

  map = map / ( score:size(2) ) 

  return map, precision, recall, specificity

end

