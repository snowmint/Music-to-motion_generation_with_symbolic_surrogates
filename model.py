import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from data_loader import *
from torch import optim
import datetime

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, p):
        super(Encoder, self).__init__()
        self.dropout = nn.Dropout(p)
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # self.embedding = nn.Embedding(input_size, embedding_size)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=p)

    def forward(self, x):
        outputs, (hidden, cell) = self.lstm(x)
        return hidden, cell

class Decoder(nn.Module):
    def __init__(
        self, input_size, hidden_size, output_size, num_layers, p
    ):
        super(Decoder, self).__init__()
        self.dropout = nn.Dropout(p)
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # self.embedding = nn.Embedding(input_size, embedding_size)
        # bugfix: batch first? => fixed
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=p)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden, cell):
        # <sos>, x1, x2, ..., xn-1
        # x shape: (N) where N is for batch size, we want it to be (1, N), seq_length
        # is 1 here because we are sending in a single word and not a sentence
        outputs, (hidden, cell) = self.lstm(x, (hidden, cell)) #, (hidden, cell)

        predictions = self.fc(outputs)
        predictions = predictions.squeeze(0)

        return predictions, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, source, target, teacher_force_ratio):
        # if teacher_force_ratio > 0:
        #     target = F.pad(target, (0,0,1,0), value = 0) #<sos>
        
        source = source.permute(1, 0, 2)
        target = target.permute(1, 0, 2)

        batch_size = source.shape[1]
        target_len = target.shape[0]
        target_dim = 102#102 #24#102
        
        outputs = torch.zeros(target_len, batch_size, target_dim).to(device) #target_vocab_size

        hidden, cell = self.encoder(source)

        # Grab the first input to the Decoder which will be <SOS> token
        # inference: target -> (513, 1, 102)
        # (513, 102) seq_len on 0 axis
        
        # x -> (1, 1, 102) (seq, bz, dim)
        
        # (b, s, d) <sos>, x, x, x, <eos>
        # print("<SOS>:", x)  <sos>, x0 ..... xn-1 <sos> full zero tensor
        x = target[:1,]
        for t in range(1, target_len):
            output, hidden, cell = self.decoder(x, hidden, cell)
            outputs[t] = output

            x = target[t][None, :] if random.random() < teacher_force_ratio else output[None, :] # (1, bz, dim)

        return outputs.transpose(0, 1)

def customized_mse_loss(output, target, prev_output, midi_array): #[512, 1, 102] [512, 1, 129]
    # target = target.transpose(0, 1)
    # print("output", output)
    mse_loss = F.mse_loss(output, target)
    # print("mse_loss:", mse_loss)

    var_diff = torch.var(torch.squeeze(output), dim=1, keepdim=True)
    mean_diff = torch.mean(var_diff)
    
    # Condition 1: Penalize if output is similar to previous output
    if mean_diff < 1e-4: #threshold
        #output [512, 1, 102] => [102] <-> [102] <-> [102] <-> ... <-> [102]
        mse_loss *= 1000

    # Condition 2: Stop movement if input is all zeros
    midi_transpose = midi_array.transpose(0, 1)
    midi_sum_row = torch.sum(midi_transpose, dim=-1)
    mask = midi_sum_row == 0
    mask = mask.unsqueeze(-1)
    mask = mask.to(device)
    # according to recorded index, make a mask [0, 1, 1, 0, ..., 0], true part will be omit(set value to 0).
    # before compute mse, use mask first to tensor, then caculate MES loss
    masked_output = output.masked_fill(mask, 0) #inplace function
    masked_target = target.masked_fill(mask, 0)
    mse_loss += F.mse_loss(masked_output, masked_target) * 1000 #output 和 previous output 不像的話，增大 loss

    # Condition 3: Penalize if right-hand movement is too different between outputs
    # if output.shape[-1] == 21:  # Assumes hand joints are the last 21 dimensions
    #     rh_indices = [i for i in range(12, 21)]  # Right-hand joint indices
    #     rh_output = output[..., rh_indices]
    #     rh_prev_output = prev_output[..., rh_indices]
    #     rh_loss = nn.functional.mse_loss(rh_output, rh_prev_output)
    #     if rh_loss > 0.1:
    #         mse_loss *= 1000

    return mse_loss

val_loss_list = []

def validate(model, val_dataloader):
    model.eval()
    print('Validation')
    valid_running_loss = 0.0
    counter = 0
    previous_output = torch.zeros(513, 0, 102).to(device)
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(val_dataloader): #tqdm(enumerate(val_dataloader), total=len(val_dataloader))
            if i > 10:
                break
            counter += 1

            inputs = inputs.to(device)
            targets = targets.to(device)
            targets_padding = F.pad(targets, (0,0,0,1), value = 0)
            outputs = model(inputs[None, :], targets_padding[None, :], 0)

            loss = customized_mse_loss(outputs, targets_padding[None, :], previous_output, inputs[None, :])
            valid_running_loss += loss.cpu().item()
            previous_output = outputs

    epoch_val_loss = valid_running_loss / counter
    return epoch_val_loss

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    
    learning_rate = 0.001

    input_size_encoder = 128 #129 #128
    input_size_decoder = 102 #102 #24
    output_size = 102#102 #24

    # encoder_embedding_size = 300
    # decoder_embedding_size = 300
    hidden_size = 256#1024 # Needs to be the same for both RNN's
    num_layers = 1 #3
    enc_dropout = 0.5
    dec_dropout = 0.
    step = 0
    
    dataset_name_path = f"./midi_list.txt"
    dataloader = get_dataloader(dataset_name_path, batch_size=20)
    
    encoder_net = Encoder(input_size_encoder, hidden_size, num_layers, enc_dropout).to(device)

    decoder_net = Decoder(
        input_size_decoder,
        hidden_size,
        output_size,
        num_layers,
        dec_dropout,
    ).to(device)
    
    model = Seq2Seq(encoder_net, decoder_net).to('cuda')
    model.train()
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    num_epochs = 100 #10
    avg_loss_list = []
    all_loss_list = []

    # train the model
    for epoch in range(num_epochs):
        losses = []
        previous_output = torch.zeros(1,512, 102).to(device)

        for i, (midi_batch, motion_batch) in enumerate(dataloader):
            model.train()
            
            midi_batch = midi_batch.to(device).float()
            motion_batch = motion_batch.to(device).float()

            optimizer.zero_grad()
            output = model(midi_batch, motion_batch, 0.5)

            motion_ground_truth_padding = F.pad(motion_batch, (0,0,0,1), value = 1) #<eot>
            # loss = customized_mse_loss(output, motion_ground_truth_padding.unsqueeze(0), previous_output, midi_batch[None, :])
            loss =  F.mse_loss(output, motion_ground_truth_padding)

            # losses 累計lose
            losses.append(loss.cpu().item())
            all_loss_list.append(loss.cpu().item())
            loss.backward()

            optimizer.step()
            mean_loss = sum(losses)/len(losses)

            print(f"Epoch {epoch}, batch {i}: loss = {loss.cpu().item():.4f}")
            previous_output = output

            
            loc_dt = datetime.datetime.today()
            loc_dt_format = loc_dt.strftime("%Y-%m-%d_%H-%M-%S")
            # save_best_model(
            #     val_loss, epoch, model, optimizer, loss, loc_dt_format, mean_loss
            # )

        
        # val_loss = validate(model, val_dataloader) #CUDA out of memory
        # val_loss_list.append(val_loss)
        # save_best_model(
        #         val_loss, epoch, model, optimizer, loss, loc_dt_format, mean_loss
        #     )
        avg_loss_list.append(mean_loss)
        loc_dt = datetime.datetime.today()
        loc_dt_format = loc_dt.strftime("%Y-%m-%d_%H-%M-%S")
        print(loc_dt_format)
        print(avg_loss_list)
        if epoch%10 == 0:
            torch.save({
                'epoch':epoch,
                'model_state_dict':model.state_dict(),
                'optimizer_state_dict':optimizer.state_dict(),
                'loss':loss
            }, "./model_save/[100epoch]LSTM_1LSTMenc_1LSTMdec_save_epoch_" + str(epoch)+ "_"+ str(loc_dt_format) + "_avg_loss_" + str(mean_loss) +".tar")