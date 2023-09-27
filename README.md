# MIDI_to_Motion
Demo webpage: https://snowmint.github.io/Music-to-motion_generation_with_symbolic_surrogates/index.html

Colab training: https://drive.google.com/file/d/1nPChmQX_tWrZpCSx0Wa2OsGraaggrl14/view?usp=sharing

## 1. data_preprocess_<...>.py 
(There is no need for you to do it. Unless there are any customized code modifications you made, I have already run it and saved the pickle file in the specified directory.)

`python data_preprocess_<...>.py`

### Next, update the directory of preprocessed files in data_list_<...>.txt

`ls ./preprocessed_data_save_<...>/midi/ | awk '{print "preprocessed_data_save_<...>/midi/" $0 }' >> data_list_<...>.txt`

For MIDI data <br/>
`ls ./preprocessed_data_save_cross/midi/ | awk '{print "preprocessed_data_save_cross/midi/" $0 }' >> midi_list_symbolic_cross.txt`

For audio data <br/>
`ls ./preprocessed_data_save_cross_aud/audio/ | awk '{print "preprocessed_data_save_cross_aud/audio/" $0 }' >> data_list_symbolic_cross_audio.txt`

For the combination of MIDI and audio data <br/>
`ls ./preprocessed_data_save_cross_both/all/ | awk '{print "preprocessed_data_save_cross_both/all/" $0 }' >> data_list_symbolic_cross_both.txt`


## 2-1. train_pure_LSTM_symbolic_cross_validation_<...>.ipynb
包含訓練和測試的程式，超參數也定義於此。在此改成單純使用 LSTM 進行訓練。使用 new loss 作為 loss function。 <br/>
This file contains the training and testing program and defines the hyperparameters here. Modify it to use only LSTM for training and utilize 'new loss' as the loss function.

## 2-2. train_pure_LSTM_symbolic_cross_validation_mse_<...>.ipynb
包含訓練和測試的程式，超參數也定義於此。在此改成單純使用 LSTM 進行訓練。使用 MSE loss 作為 loss function。 <br/>
This file contains the training and testing program and defines the hyperparameters here. Modify it to use only LSTM for training and utilize 'MSE loss' as the loss function.

## 3. model.py
這份程式實作了 LSTM encoder-decoder 模型。雖然訓練沒有成效，但保留做參照。 <br/>
This code implements an LSTM encoder-decoder model. Although the training was not practical, it is kept for reference.

## 4. data_loader.py
定義 PyTorch Data loader 如何取用訓練資料。 <br/>
Define how the PyTorch data-loader to access the training data.

Dataset: <br/>

`init function`:  <br/>
只給予訓練資料 pickle file 的路徑。 <br/>
Only provide the path to the training data's pickle file. <br/>


`get_item function`:  <br/>
在指定的訓練資料路徑讀取 pickle file，並隨機在歌曲內挑選長度為 512 的片段。 <br/>
Read the pickle file from the specified training data path and randomly select segments of length 512 within the pieces. <br/>
  
`len function`:  <br/>
由於目前一首歌曲算一筆資料，透過設定 dataset 需要 100 倍的資料量，便可以使每一首歌都會隨機取用 100 個隨機片段。 <br/>
Since currently one song is considered one data entry, by setting the dataset to require 100 times the data amount, each piece will randomly utilize 100 segments. <br/>


## 5. dataset, preprocessed training data, saved model 
### 需要下載的檔案列於下方： <br/>
### The files that need to be downloaded are listed below: <br/>

original dataset: https://drive.google.com/drive/folders/1pSNXDOfcki7-Iw7w_Ws15lzc2cdJ5z7H?usp=drive_link

[MIDI preprossed data] https://drive.google.com/drive/folders/1LvVZC5pI3wnX1bKNrIp4DDRQyhTwgtxR?usp=drive_link <br/>
[Audio preprossed data] https://drive.google.com/drive/folders/13TQJBi7HOxOt2sKmx8ldwHRfe4jxZuZe?usp=drive_link <br/>
[MIDI+Audio preprossed data] https://drive.google.com/drive/folders/1OZ8lCXvU-tA6ikzkV-o9PSvVbf9a8mer?usp=drive_link <br/>

[1.5x faster motion preprossed data] https://drive.google.com/drive/folders/1WHRu9WMj9MkW79JHpEqzOyLNBvv0SxOA?usp=drive_link <br/>

[without annotation training model save] https://drive.google.com/drive/folders/1w74s7XUKmm9xd5qLQS5smMEWApb9Ykic?usp=drive_link <br/>
[with annotation training model save] https://drive.google.com/drive/folders/1I7v3dIdYWISuL-GcFRlO7RgcZpJss8-V?usp=drive_link

## 6. test_metric folder

執行 predict_draw.ipynb 程式可以使用預先儲存好的 model (存在 model_save 下) 進行預測並繪製影片，這份程式會將 evaluation 所需的預測資料儲存至 output_eval 資料夾中。 <br/>
Executing the 'predict_draw.ipynb' program allows you to make predictions and generate videos using a pre-saved model (located in the 'model_save' directory). This program will store the predicted data required for evaluation in the 'output_eval' folder.

將儲存於 output_eval 內的 pkl 複製到 test_metric 目錄下，test_metric 中有 output_eval 資料夾，其中細分為 MIDI, Audio 和 Both 的資料夾，將相對應的 pkl 檔案複製到相應的目錄，即可執行以下程式，進行 evaluation。 <br/>
Copy the .pkl files stored in the 'output_eval' directory to the 'test_metric' directory. Inside 'test_metric,' there are subdirectories for 'MIDI,' 'Audio,' and 'Both.' Copy the corresponding .pkl files to their respective directories, and you can then execute the following program for evaluation.

`python test.py`

`python test_faster.py`

## 7. BWV1001 folder
用於 testing 階段的額外測試資料。 <br/>
Additional test data for the testing phase.

## 8. Survey_20230908 folder
問卷調查的網頁與收集到的評分數據。 <br/>
The webpage for the questionnaire survey and the collected rating data.

## 9. 完整目錄所需的資料夾與檔案
![picture of directory view](https://github.com/snowmint/Music-to-motion_generation_with_symbolic_surrogates/blob/main/directory_view.png?raw=true)
