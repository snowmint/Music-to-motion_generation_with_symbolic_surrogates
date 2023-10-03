# Music_to_Motion
Demo webpage for more details: https://snowmint.github.io/Music-to-motion_generation_with_symbolic_surrogates/index.html

[New-loss][Validation set] Elgar salut d'amour

https://github.com/snowmint/Music-to-motion_generation_with_symbolic_surrogates/assets/7868828/833968ce-3390-4c86-a647-0bf33ee0375b

[New-loss][Validation set]  雨夜花
 
https://github.com/snowmint/Music-to-motion_generation_with_symbolic_surrogates/assets/7868828/f0c7a05e-14d3-4fc3-b596-7ef6ec6a703c

[New-loss][Test set] BWV1001 Adagio

https://github.com/snowmint/Music-to-motion_generation_with_symbolic_surrogates/assets/7868828/90595460-af1a-457f-9c33-a25d0339280e

[New-loss][Test set] BWV1001 Adagio

https://github.com/snowmint/Music-to-motion_generation_with_symbolic_surrogates/assets/7868828/20dfbcfa-d0b4-4129-bca8-aa9fe1618479

## Colab demonstration(English)
Colab prediction: https://colab.research.google.com/drive/1lXHWYrx2NjMudjsHTDrorBGwBZTtJPw1

1. Run the first cell of the program to download the required packages. Occasionally, this part may encounter exceptions, leading to an AttributeError: 'Path3DCollection' object has no attribute '_offset_zordered' error when plotting. I've tried using `!pip3 install matplotlib==3.7.3` first and then restarting the runtime. After pressing 'Run All,' it works appropriately. <br/>

2. Download the model files you need (You can start off by choosing one model first. The more models you add, the longer the waiting time): https://drive.google.com/drive/folders/1w74s7XUKmm9xd5qLQS5smMEWApb9Ykic?usp=drive_link <br/>
And download  the MIDI and WAV files of the music you want to test. You can use BWV1001 as a sample test : https://drive.google.com/drive/folders/1q6zQKzlsde77v2FfOYFHnci12tE6bdaW?usp=drive_link <br/>

3. In the current Colab directory, running the first cell of the program will automatically create two new folders: `no_anno_model_save` and `test_file`. Please upload and place the files downloaded in the previous step (select the necessary files to reduce waiting time) in the corresponding folders. Put the model files in the `no_anno_model_save` folder, and audio files in the `test_file` folder as shown in the diagram below. Ensure that the `test_file` folder contains both MIDI and WAV files with consistent filenames. If you want to use your own additional test files, you can convert MIDI files to WAV using software like `apt install fluidsynth` or “MuseScore”. <br/>
   ![資料擺放位置](https://github.com/snowmint/Music-to-motion_generation_with_symbolic_surrogates/assets/7868828/f6764b80-c679-45b0-ba5b-0bb6ef09d800)

4. Currently it takes 9 minutes to generate a video with 400 frames for testing. The waiting time for video generation will increase depending on the number of models being tested. For example, using three models and one piece of music would require a waiting time of about 27 minutes. Please try to keep the duration reasonable to avoid interruptions in the Colab runtime. <br/>
5. The generated MP4 video file will appear in the file panel on the left-hand side of the Colab page, as shown in the diagram below. Double-click the video file to download and view the results: <br/>
   ![生成出的mp4會在左方檔案夾當前目錄下顯示](https://github.com/snowmint/Music-to-motion_generation_with_symbolic_surrogates/assets/7868828/11b51709-50a7-488a-9f8c-a103eb96c6fc)


## Colab 範例(中文)
Colab prediction: https://colab.research.google.com/drive/1lXHWYrx2NjMudjsHTDrorBGwBZTtJPw1

1. 運行第一格程式下載需要的套件，這個部分偶爾會遇到例外情況導致最後在畫圖的時候出現 AttributeError: 'Path3DCollection' object has no attribute '_offset_zordered'，嘗試過先使用 !pip3 install matplotlib==3.7.3 再**重新啟動執行階段**，按下全部執行就可以正常使用。<br/>
2. 需要先下載想要使用的模型檔案：https://drive.google.com/drive/folders/1w74s7XUKmm9xd5qLQS5smMEWApb9Ykic?usp=drive_link<br/>
   以及想要測試之樂曲的 mid 和 wav 檔案，可以以 BWV1001 做為測試：https://drive.google.com/drive/folders/1q6zQKzlsde77v2FfOYFHnci12tE6bdaW?usp=drive_link<br/>
3. 在 Colab 的當前目錄下執行第一格程式將自動建立新的資料夾 `no_anno_model_save` 以及 `test_file`，請將前一步驟下載的檔案(取需要的部分檔案即可減少等待時間)放置在相對應的資料夾下，模型檔案放置在 no_anno_model_save，而音訊放置在 test_file 資料下，如下圖所示。**test_file 資料夾下請確認有放置 mid 和 wav 檔案，且檔名需一致**。若欲使用自己額外的測試檔案， mid 檔案轉 wav 可以使用 apt install fluidsynth 或 musescore 等軟體進行轉檔。<br/>
   ![資料擺放位置](https://github.com/snowmint/Music-to-motion_generation_with_symbolic_surrogates/assets/7868828/f6764b80-c679-45b0-ba5b-0bb6ef09d800)
4. 目前測試繪製 400 frame 的影片需要 9 分鐘的時間，依想測試的模型數量等待影片生成的時間會延長，像是放了三個模型與一首樂曲就需等待 27 分鐘。盡量不要過長，以免 colab 中斷執行階段。<br/>
5. 生成的影片 mp4 檔案將出現在 colab 頁面左方檔案欄位，如下圖所示，雙擊影片檔案即可下載查看結果：
   ![生成出的mp4會在左方檔案夾當前目錄下顯示](https://github.com/snowmint/Music-to-motion_generation_with_symbolic_surrogates/assets/7868828/11b51709-50a7-488a-9f8c-a103eb96c6fc)

## ========== The following steps need to be executed only for local implementation. ==========
## ================================ 以下步驟為在本機實作才需執行 ================================
## 0. set up environment

`pip install -r requirement.txt`<br/>
`pip install midi2audio`<br/>
`pip install py-midi`<br/>
`pip install ffmpeg-python`<br/>
`sudo apt install ffmpeg` (conda install -c conda-forge ffmpeg)<br/>
`sudo apt-get install fluidsynth`<br/>

## 1. data_preprocess_<...>.py 
(There is no need for you to do it. Unless there are any customized code modifications you made, I have already run it and saved it as a pickle file. The download link can be found in step 5 below.)
(原則上不用執行。除非有修改 code ，否則我已經運行過，並儲存成 pickle file ，下載連結在 step 5。) <br/>

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
This file contains the training and testing program and defines the hyperparameters here. Modify it to use only LSTM for training and utilize 'new loss' as the loss function.
包含訓練和測試的程式，超參數也定義於此。在此改成單純使用 LSTM 進行訓練。使用 new loss 作為 loss function。 <br/>

## 2-2. train_pure_LSTM_symbolic_cross_validation_<...>_mse.ipynb
This file contains the training and testing program and defines the hyperparameters here. Modify it to use only LSTM for training and utilize 'MSE loss' as the loss function.
包含訓練和測試的程式，超參數也定義於此。在此改成單純使用 LSTM 進行訓練。使用 MSE loss 作為 loss function。 <br/>

## 3. model.py
This code implements an LSTM encoder-decoder model. It is kept for reference.
這份程式實作了 LSTM encoder-decoder 模型。保留做參照。 <br/>

## 4. data_loader.py
Define how to access the training data for the PyTorch data-loader.
定義 PyTorch Data loader 如何取用訓練資料。 <br/>
Dataset access define: <br/>

`init function`:  <br/>
This function provides only the path to the training data's pickle file as a measure to avoid loading all training and validation data directly into GPU memory which causes Out-of-Memory issues. The actual loading of the training data is performed in the get_item function.  <br/>
只給予訓練資料 pickle file 的路徑。為了避免在 init function 直接 load 全部的訓練與驗證資料進 GPU 記憶體導致 Out-of-Memory，因此在此只給予路徑，直到 get_item function 時才真正讀入訓練資料。 <br/>

`get_item function`:  <br/>
This function reads the pickle file from the specified training data path and randomly selects segments of length 512 within the pieces. <br/>
在指定的訓練資料路徑讀取 pickle file，並隨機在歌曲內挑選長度為 512 的片段。 <br/>

`len function`:  <br/>
Since one song is considered one data entry, using only a single 512-length segment from a song is not cost-effective. Therefore, by setting the dataset length to require data equivalent to 100 random segments from each song, every song will have 100 random segments selected, enhancing the richness of the training data.
由於目前一首歌曲算一筆資料，如果一首歌只取用一個 512 長度的片段，將會大幅浪費資料。因此透過設定 dataset 長度為需要每一首歌提供 100 個隨機片段的資料量，便可以使每一首歌都會隨機取用 100 個隨機片段，增加訓練資料的豐富性。 <br/>

## 5. dataset, preprocessed training data, saved model 
The required files are listed below: <br/>
需要下載的檔案列於下方： <br/>

Original dataset: https://drive.google.com/drive/folders/1pSNXDOfcki7-Iw7w_Ws15lzc2cdJ5z7H?usp=drive_link

[MIDI preprocessed data] https://drive.google.com/drive/folders/1LvVZC5pI3wnX1bKNrIp4DDRQyhTwgtxR?usp=drive_link <br/>
[Audio preprocessed data] https://drive.google.com/drive/folders/13TQJBi7HOxOt2sKmx8ldwHRfe4jxZuZe?usp=drive_link <br/>
[MIDI+Audio preprocessed data] https://drive.google.com/drive/folders/1OZ8lCXvU-tA6ikzkV-o9PSvVbf9a8mer?usp=drive_link <br/>

[1.5x faster motion preprossed data] https://drive.google.com/drive/folders/1WHRu9WMj9MkW79JHpEqzOyLNBvv0SxOA?usp=drive_link <br/>

[The saved training model(without annotation)] https://drive.google.com/drive/folders/1w74s7XUKmm9xd5qLQS5smMEWApb9Ykic?usp=drive_link <br/>
[The saved training model(with annotation)] https://drive.google.com/drive/folders/1I7v3dIdYWISuL-GcFRlO7RgcZpJss8-V?usp=drive_link

## 6. test_metric folder
Executing the 'predict_draw.ipynb' file allows you to make predictions and generate videos using a pre-saved model (located in the 'model_save' directory). This stores the predicted data required for evaluation in the 'output_eval' folder. <br/>
執行 predict_draw.ipynb 程式可以使用預先儲存好的 model (存在 model_save 下) 進行預測並繪製影片，這份程式會將 evaluation 所需的預測資料儲存至 output_eval 資料夾中。 <br/>

Copy the .pkl files stored in the 'output_eval' directory to the 'test_metric' directory. The 'test_metric' directory contains the 'MIDI,' 'Audio,' and 'Both' subdirectories. Copy the corresponding .pkl files to their respective directories, and you can then execute the command below to begin evaluations. <br/>
將儲存於 output_eval 內的 pkl 檔案複製到 test_metric 目錄下，test_metric 中有 output_eval 資料夾，其中細分為 MIDI, Audio 和 Both 的資料夾，將相對應的 pkl 檔案複製到相應的目錄，即可執行以下程式，進行 evaluation。 <br/>

`python test.py`

## 7. BWV1001 folder
Additional test data for the testing phase.
用於 testing 階段的額外測試資料。 <br/>

## 8. Survey_20230908 folder
The webpage for the questionnaire survey and the collected rating data.
問卷調查的網頁與收集到的評分數據。 <br/>

## 9. The required folders and files as seen in the screenshot below.
Make sure the files are put in the correct directory, then you can run the .ipynb for training and prediction.
<img src="https://github.com/snowmint/Music-to-motion_generation_with_symbolic_surrogates/blob/main/directory_view.png?raw=true" width="600">
