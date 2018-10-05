# DPPI
A convolutional neural network to predict PPI interactions


Command:
th main.lua -dataset myTrain  -learningRate 0.01 -momentum 0.9  -string dimer  -device 1  -preprocess nothing -top_rand -batchSize 10

Input parameters: 
-dataset: nickname of the training data (e.g. myTrain)
-string: a suffix that is added to the result file

Necessary input files before running the command:

Training data: It is in dat format. The name of this file should be the name of your training data followed by ‘_labels’ (e.g myTrain_labels.dat).

The dat file can be build by ???. Training data contains three column where first and second columns are two proteins and third column is either 1 or 0 indicating if the two proteins interact or not. 

Validation data: Same as Training data. The name of this file should be the name of your training data followed by ‘_valid_labels’ (e.g myTrain_valid_labels.dat). 

Similar to Training data, you can make the dat file using script_convert_csv_to_dat.lua

Cropped profiles of proteins: It is in t7 format. This file is made using a script called ‘create_crop.lua’.

Numbers of cropped per profile: It is in t7 format. This file is made using a script called ‘create_crop.lua’.

==========================

Creat_crop.lua

Command:
th creat_crop.lua -dataset myTrain  

Input parameters:
-dataset: name of your training data (e.g. ‘myTrain’).

Necessary input files before running the command:
You should have a 1)file and a 2)folder with names the same as the input name:
1)The suffix of the file is ‘.node’ (e.g myTrain.node). This file has a one column which contains names of all proteins in the train data. 

2) The folder that contains profiles of proteins (e.g myTrain). The name of the profiles inside this folder is the same as the protein names in ‘.node’ file 

Output:

1) cropped profiles: It is in t7 format. The name of this file is the input name followed by “_number_crop_512” (e.g. myTrain_profile_crop_512.t7)

2) numbers of cropped per profile: It is in t7 format. The name of this file is the input name followed by “_number_crop_512” (e.g. myTrain_number_crop_512.t7) 

Please remember you need to change the Work directory in the lua files.


