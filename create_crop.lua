require 'torch'   -- torch
require 'cutorch'

----------------------------------------------------------------------
cmd = torch.CmdLine()
cmd:text()
cmd:text('Options:')
-- global:
cmd:option('-dataset', 'A', 'suffix to log files')

opt = cmd:parse(arg or {})

crop_size = 512;
NumFeatures = 20
local C = 0.8

function file_exists(name)
   local f=io.open(name,"r")
   if f~=nil then io.close(f) return true else return false end
end

dofile 'Csv.lua'

background = {}

background[1]=0.0799912015849807; --A
background[2]=0.0484482507611578; --R
background[3]=0.044293531582512; --N
background[4]=0.0578891399707563; --D
background[5]=0.0171846021407367; --C
background[6]=0.0380578923048682; --Q
background[7]=0.0638169929675978; --E
background[8]=0.0760659374742852; --G
background[9]=0.0223465499452473; --H
background[10]=0.0550905793661343; --I
background[11]=0.0866897071203864; --L
background[12]=0.060458245507428; --K
background[13]=0.0215379186368154; --M
background[14]=0.0396348024787477; --F
background[15]=0.0465746314476874; --P
background[16]=0.0630028230885602; --S
background[17]=0.0580394726014824; --T
background[18]=0.0144991866213453; --W
background[19]=0.03635438623143; --Y
background[20]=0.0700241481678408; --V

local proteinFile = Csv(opt.dataset..".node","r")
local proteinString = proteinFile:readall()

ppFeature = {}
pNumber = {}


for i=1, #proteinString do
  
  local fileName = opt.dataset..'/'..proteinString[i][1]

  if file_exists( fileName ) then

    local proFile = Csv( fileName, 'r', '\t')
    local profile = proFile:readall()
   

    pNumber[ proteinString[i][1] ] = math.ceil( 2 * #profile / crop_size  - 1)

    if pNumber[ proteinString[i][1] ] < 1 then
      pNumber[ proteinString[i][1] ] = 1
    end
   
    for c = 1, pNumber[ proteinString[i][1] ] do

      start = math.min( (c-1) * crop_size / 2 + 1, #profile - crop_size + 1 )

      ppFeature[ proteinString[i][1]..'-sub'..c ] = torch.Tensor(1, 20, crop_size, 1):zero() 
 
      
      if start > 0 then
        
        for j=start,start+crop_size-1 do
          for k=1,20 do
            ppFeature[ proteinString[i][1]..'-sub'..c ][1][k][j-start+1] = math.log(C*( tonumber(profile[j][k+20]) / 100 )+(1-C)*background[k])  
          end
        end
     
      else
        
        for j=1,#profile do
          for k=1,20 do
            ppFeature[ proteinString[i][1]..'-sub'..c ][1][k][j-math.floor(start/2)] = math.log(C*( tonumber(profile[j][k+20]) / 100 )+(1-C)*background[k])
          end
        end
      
      end
    
    end 
    proFile:close()
  end

end

proteinFile:close()
collectgarbage()
torch.save(opt.dataset..'_profile_crop_'..crop_size..'.t7', ppFeature )
torch.save(opt.dataset..'_number_crop_'..crop_size..'.t7', pNumber )
