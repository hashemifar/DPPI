# DPPI
A convolutional neural network to predict PPI interactions.


Main Command:
th main.lua -dataset myTrain  -learningRate 0.01 -momentum 0.9  -string first-run  -device 1 -top_rand -batchSize 2 -saveModel

==> Input parameters: 

        -dataset: Name of the training data (e.g. myTrain)

        -string: A suffix that is added to the result file
        
        -device: GPU number
        
              
==> Necessary input files before running the command:

        -Training data: It is in dat format. 
         The name of this file should be the name of your training data followed by ‘_labels’ (e.g myTrain_labels.dat).

         The dat file is made using a script called 'convert_csv_to_dat.lua' . 

        -Validation data: Same as Training data. Name of this file is the same as training data followed by '_valid'
         (e.g myTrain_valid_labels.dat). 

        Similar to Training data, you can make the dat file using convert_csv_to_dat.lua

        -Cropped profiles of proteins: It is in t7 format. This file is made using a script called ‘create_crop.lua’.

        -Numbers of cropped per profile: It is in t7 format. This file is made using a script called ‘create_crop.lua’.

====================================================

convert_csv_to_dat.lua: This script converts a csv file to dat file.

Command:

th convert_csv_to_dat.lua -dataset myTrain
th convert_csv_to_dat.lua -dataset myTrain_valid

==> Input parameters: 

        -dataset: name of the dataset in csv format without suffix (e.g. myTrain).
        
        This file contains three column where first and second columns are two proteins and third column is either 1 or 0
        
        indicating if the two proteins interact or not (e.g. myTrain.csv and myTrain_valid.csv). 
        
==> Output:

        dataset in dat format (e.g. myTrain_labels.dat or myTrain_valid_labels.dat) 

====================================================

creat_crop.lua: This scripts makes the cropped profiles

Command:

th creat_crop.lua -dataset myTrain  

==> Input parameters:

        -dataset: name of your training data (e.g. ‘myTrain’).
        
==> Output:

        1) cropped profiles: It is in t7 format. The name of this file is the input name followed by “_number_crop_512” (e.g. myTrain_profile_crop_512.t7)

        2) numbers of cropped per profile: It is in t7 format. The name of this file is the input name followed by “_number_crop_512” (e.g. myTrain_number_crop_512.t7) 


==> Necessary input files before running the command:

        -You should have a 1)file and a 2)folder with names the same as the -dataset:
        
        1)The suffix of the file is ‘.node’ (e.g myTrain.node). This file has one column which contains names of all proteins in the training and validation data. 

        2) Profile folder with the same name as -dataset (e.g myTrain). This folder contains profiles of all proteins in training and validation data. 
        
        The name of the profiles inside this folder is the same as the protein names in ‘.node’ file 

====================================================

Please remember befor running the Main Command you need to change the data directory and work directory 

in main.lua file at lines 5 and 6. You need to replace '$HOME' with your own data directory and work directory. 


