# MIDI_to_Motion
Demo webpage: https://snowmint.github.io/Music-to-motion_generation_with_symbolic_surrogates/index.html

Colab training: https://drive.google.com/file/d/1nPChmQX_tWrZpCSx0Wa2OsGraaggrl14/view?usp=sharing

## 1. data_preprocess_<...>.py 
(原則上不用執行。除非有修改 code ，否則我已經運行過，並儲存 pickle file 在指定目錄下了。)

`python data_preprocess_<...>.py`

接著更新前處理後檔案目錄至 data_list_<...>.txt 中：

`ls ./preprocessed_data_save_<...>/midi/ | awk '{print "preprocessed_data_save_<...>/midi/" $0 }' >> data_list_<...>.txt`

`ls ./preprocessed_data_save_cross/midi/ | awk '{print "preprocessed_data_save_cross/midi/" $0 }' >> midi_list_symbolic_cross.txt`
`ls ./preprocessed_data_save_cross_aud/audio/ | awk '{print "preprocessed_data_save_cross_aud/audio/" $0 }' >> data_list_symbolic_cross_audio.txt`
`ls ./data_preprocess_symbolic_cross_both/all/ | awk '{print "data_preprocess_symbolic_cross_both/all/" $0 }' >> data_list_symbolic_cross_both.txt`


## 2-1. train_pure_LSTM_symbolic_cross_validation_<...>.ipynb
包含訓練和測試的程式，超參數也定義於此。在此改成單純使用 LSTM 進行訓練。使用 new loss 作為 loss function。

## 2-2. train_pure_LSTM_symbolic_cross_validation_mse_<...>.ipynb
包含訓練和測試的程式，超參數也定義於此。在此改成單純使用 LSTM 進行訓練。使用 MSE loss 作為 loss function。

## 3. model.py
這份程式實作了 LSTM encoder-decoder 模型。訓練沒有成效，保留做參照。

## 4. data_loader.py
定義 PyTorch Data loader 如何取用訓練資料。

Dataset:

`init function`: 只給予訓練資料 pickle file 的路徑。

`get_item function`: 在指定的訓練資料路徑讀取 pickle file，並隨機在歌曲內挑選長度為 512 的片段。只在樂曲的開頭加上全為 0 的 <start-of-token>，以及在樂曲結尾加上全為 1 的 <end-of-token> 。
  
`len function`: 由於目前一首歌曲算一筆資料，透過設定 dataset 需要 100 倍的資料量，便可以使每一首歌都會隨機取用 100 個隨機片段。

## 5. dataset, preprocessed training data, saved model 
需要下載的檔案列於下方：

original dataset: https://drive.google.com/drive/folders/1pSNXDOfcki7-Iw7w_Ws15lzc2cdJ5z7H?usp=drive_link

[MIDI preprossed data] https://drive.google.com/drive/folders/1LvVZC5pI3wnX1bKNrIp4DDRQyhTwgtxR?usp=drive_link
[Audio preprossed data] https://drive.google.com/drive/folders/13TQJBi7HOxOt2sKmx8ldwHRfe4jxZuZe?usp=drive_link
[MIDI+Audio preprossed data] https://drive.google.com/drive/folders/1OZ8lCXvU-tA6ikzkV-o9PSvVbf9a8mer?usp=drive_link

[1.5x faster motion preprossed data] https://drive.google.com/drive/folders/1WHRu9WMj9MkW79JHpEqzOyLNBvv0SxOA?usp=drive_link

[without annotation training model save] https://drive.google.com/drive/folders/1w74s7XUKmm9xd5qLQS5smMEWApb9Ykic?usp=drive_link
[with annotation training model save] https://drive.google.com/drive/folders/1I7v3dIdYWISuL-GcFRlO7RgcZpJss8-V?usp=drive_link

## 6. test_metric folder
將儲存於 output_eval 內的 pkl 執行以下程式，進行 evaluation。

`python test.py`

`python test_faster.py`

## 7. BWV1001 folder
用於 testing 階段的額外資料。

## 8. Survey_20230908 folder
問卷調查的網頁與收集到的評分數據。

## 9. 完整目錄所需的資料夾與檔案
![picture of directory view](https://github.com/snowmint/Music-to-motion_generation_with_symbolic_surrogates/blob/main/directory_view.png?raw=true)