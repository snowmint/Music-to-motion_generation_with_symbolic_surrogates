# MIDI_to_Motion

## 1. data_preprocess.py
(原則上不用執行。除非有修改 code ，否則我已經運行過，並儲存 pickle file 在指定目錄下了。)

`python data_preprocess.py`

接著更新前處理後檔案目錄至 midi_list.txt 中：

`ls ./preprocessed_data_save/midi/ | awk '{print "preprocessed_data_save/midi/" $0 }' >> midi_list.txt`
`ls ./preprocessed_data_save_new/midi/ | awk '{print "preprocessed_data_save/midi/" $0 }' >> midi_list_symbolic.txt`
## 2-1. train.ipynb
包含訓練和測試的程式，超參數也定義於此。

## 2-2. train_pure_LSTM.ipynb
包含訓練和測試的程式，超參數也定義於此。在此改成單純使用 LSTM 進行訓練。

## 3. model.py
這份程式實作了 LSTM encoder-decoder 模型。

## 4. data_loader.py
定義 PyTorch Data loader 如何取用訓練資料。

Dataset:

`init function`: 只給予訓練資料 pickle file 的路徑。

`get_item function`: 在指定的訓練資料路徑讀取 pickle file，並隨機在歌曲內挑選長度為 512 的片段。只在樂曲的開頭加上全為 0 的 <start-of-token>，以及在樂曲結尾加上全為 1 的 <end-of-token> 。
  
`len function`: 由於目前一首歌曲算一筆資料，透過設定 dataset 需要 100 倍的資料量，便可以使每一首歌都會隨機取用 100 個隨機片段。

## 5. test result

### Only LSTM (Best result is been store at ./test_result folder)
100 epoch 訓練後的測試結果：（66首對齊資料）
[37min]100epoch_66align_data_MSEloss
https://drive.google.com/drive/folders/163lnPcq9v-q_RQ-iciJbmeOvHUMCfrQq?usp=drive_link

500 epoch 訓練後的測試結果：（66首對齊資料）
[187min]500epoch_66align_data_MSEloss
https://drive.google.com/drive/folders/1KamwniIZHCUTggkZypHPqkTxIy9CAN6T?usp=drive_link

### Enc-Dec
100 epoch 訓練後的測試結果：（22首對齊資料）
[22min]100epoch_(custom_loss)random_pick_2200_datasample_per_epoch
https://drive.google.com/drive/folders/1SobWLwwDAmP6CrF-iHaoQJWU0ozB4pq6?usp=drive_link

100 epoch 訓練後的測試結果：（66首對齊資料）
[39min]100epoch_66align_data_custom_loss
https://drive.google.com/drive/folders/1x2qSkkf6GJuF_zAtrE4dYrkBRmIAH58w?usp=drive_link

100 epoch 訓練後的測試結果：（110首對齊資料）
<待補上>
