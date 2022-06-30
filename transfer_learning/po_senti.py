import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from simple_model import BiLSTM_attention

import numpy as np
import re

print("------import finished --------")
class Trans_BiLSTM_Attn(nn.Module):
    def __init__(self, pre_model, glove_embeddings, embedding_dim, n_hidden):
        self.n_hidden = n_hidden

        super(Trans_BiLSTM_Attn, self).__init__()
        self.pretrain_model = pre_model
        self.embedding = nn.Embedding.from_pretrained(glove_embeddings, freeze=True)
        self.lstm = nn.LSTM(embedding_dim, n_hidden, bidirectional=True)
        self.out = nn.Linear(n_hidden*4, 3)

    def pretrain_forward(self, X):
        for param in self.pretrain_model.parameters():
            param.requires_grad = False
        output, final_hidden_state, p_output, p_hidden_state = self.pretrain_model.forward(X)
        return output, final_hidden_state, p_output, p_hidden_state

    def attention_net(self, lstm_output, final_state):
        hidden = final_state.view(-1, self.n_hidden*4, 1)
        attn_weights = torch.bmm(lstm_output, hidden).squeeze(2)
        soft_attn_weights = F.softmax(attn_weights)
        context = torch.bmm(lstm_output.transpose(1,2), soft_attn_weights.unsqueeze(2)).squeeze(2)
        return context, soft_attn_weights.data.numpy()

    def forward(self, t_X):
        _, _, p_output, p_final_hidden_state = self.pretrain_forward(t_X)

        t_input = self.embedding(t_X)
        t_input = t_input.permute(1,0,2)
        t_w = torch.empty(size=[1*2, len(t_X), self.n_hidden])
        t_hidden_state = Variable(nn.init.normal(t_w))
        t_cell_state = Variable(nn.init.normal(t_w))

        t_output, (t_final_hidden_state, t_final_cell_state) = self.lstm(t_input, (t_hidden_state, t_cell_state))
        t_output = t_output.permute(1,0,2)
        output = torch.cat((p_output, t_output), 2)
        final_hidden_state = torch.cat((p_final_hidden_state, t_final_hidden_state), 2)
        attn_output, attn = self.attention_net(output, final_hidden_state)
        return self.out(attn_output), attn


def train_ft_model(model, learning_rate, epochs, training_data, dev_data, val_data, output_path, record_path):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(epochs):
        optimizer.zero_grad()
        for (training_batch, target_batch) in training_data:
            output, attention = model.forward(training_batch)
            loss = criterion(output, target_batch)

            loss.backward()
            optimizer.step()

        if (epoch+1) % 1 == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))
            dev_a_count = 0
            for (dev_input_batch, dev_target_batch) in dev_data:
                predict, _ = model.forward(dev_input_batch)
                predict = predict.data.max(1, keepdim=True)[1]
                if predict[0][0] == dev_target_batch:
                    dev_a_count += 1
            dev_acc = dev_a_count/len(dev_data)
            with open(record_path, "a") as record_output:
                record_output.write("Epoch: ")
                record_output.write(str(epoch+1))
                record_output.write(" cost =")
                record_output.write(str(loss))
                record_output.write(" dev accuracy= ")
                record_output.write(str(dev_acc))
                record_output.write("\n")


    val_a_count = 0
    for (val_input_batch, val_target_batch) in val_data:
        predict, _ = model.forward(val_input_batch)
        predict = predict.data.max(1, keepdim=True)[1]
        if predict[0][0] == val_target_batch:
            val_a_count += 1
    val_acc = val_a_count/len(val_data)
    with open(record_path, "a") as record_output_v:
        record_output_v.write("val accuracy= ")
        record_output_v.write(str(val_acc))
        record_output_v.write("\n")

    torch.save(model.state_dict(), output_path)

def preprocess (input_data, glove_data):
    with open(input_data, "r") as data_inputs:
        data_lines = data_inputs.readlines()

    with open(glove_data, "r") as glove_inputs:
        glove_lines = glove_inputs.readlines()

    embeddings = []
    vocabs = []
    for g_line in glove_lines:
        tokens = g_line.rstrip().split(' ')
        vocabs.append(tokens[0])
        embeddings.append(np.array([float(val) for val in tokens[1:]]))

    vocabs.insert(0, "<UNK>")
    vocabs.append("<PAD>")
    embeddings.insert(0, np.random.randn(100))
    embeddings.append(np.random.randn(100))
    dictionary = {w: i for i, w in enumerate(vocabs)}
    embeddings = torch.FloatTensor(embeddings)

    dataset = []
    for d_line in data_lines:
        tokens = d_line.rstrip().lower().split("\t")
        text = re.findall("[\w']+|[^A-Za-z0-9\\s]", tokens[0])
        if len(text) < 128:
            for p in range(len(text),128):
                text.append("<PAD>")
        elif len(text) > 128:
            text = text[:128]
        target = Variable(torch.LongTensor(np.asarray([float(tokens[1])])))
        input = []
        for t in text:
            if t in vocabs:
                input.append(dictionary[t])
            else:
                input.append(dictionary["<UNK>"])
        dataset.append((torch.LongTensor([input]), target))

    np.random.shuffle(dataset)
    train_data = dataset[:int(len(dataset)*0.8)]
    dev_data = dataset[int(len(dataset)*0.8):int(len(dataset)*0.9)]
    val_data = dataset[int(len(dataset)*0.9):]

    return train_data, dev_data, val_data, embeddings, vocabs


pretrained_model_path = ""
input_file = ""
glove_file = ""
output_file = ""
record_file = ""

print("---------------building data---------------------------")
train_data, dev_data, val_data, embeddings, vocabs = preprocess(input_file, glove_file)
print("---------------loading model -------------------------")
pre_model = BiLSTM_attention(embeddings, len(vocabs), _, _, _)
pre_model.load_state_dict(torch.load(pretrained_model_path))
pre_model.eval()
print("----------------start training -----------------------")
model = Trans_BiLSTM_Attn(pre_model, embeddings, _, _)
train_ft_model(model, _, _, train_data, dev_data, val_data, output_file, record_file)
print("----------------training finished---------------------")

