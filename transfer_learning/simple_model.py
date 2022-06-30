import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import re



class BiLSTM_attention (nn.Module):
    def __init__(self, glove_embedding, vocab_size, embedding_dim, n_hidden, num_class):
        self.n_hidden = n_hidden
        self.num_class = num_class
        self.vocab_size = vocab_size

        super(BiLSTM_attention, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(glove_embedding, freeze=True)
        #self.embedding = nn.Embedding(self.vocab_size, 50)
        self.lstm = nn.LSTM(embedding_dim, n_hidden, bidirectional=True)
        self.out = nn.Linear(n_hidden*2, num_class)

    def attention_net(self, lstm_output, final_state):
        hidden = final_state.view(-1, self.n_hidden*2, 1)
        attn_weights = torch.bmm(lstm_output, hidden).squeeze(2)
        soft_attn_weights = F.softmax(attn_weights)
        context = torch.bmm(lstm_output.transpose(1,2), soft_attn_weights.unsqueeze(2)).squeeze(2)
        return context, soft_attn_weights.data.numpy()

    def forward(self, X):
        input = self.embedding(X)
        input = input.permute(1,0,2)

        w = torch.empty(size=[1*2, len(X), self.n_hidden])
        hidden_state = Variable(nn.init.normal(w))
        cell_state = Variable(nn.init.normal(w))

        output, (final_hidden_state, final_cell_state) = self.lstm(input, (hidden_state, cell_state))
        output = output.permute(1,0,2)
        attn_output, attention = self.attention_net(output, final_hidden_state)
        return self.out(attn_output), attention, output, final_hidden_state


def train_model(model, learning_rate, epochs, training_data, dev_data, val_data, output_path, record_path):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(epochs):
        optimizer.zero_grad()
        for (training_batch, target_batch) in training_data:
            output, attention, _, _ = model.forward(training_batch)
            loss = criterion(output, target_batch)

            loss.backward()
            optimizer.step()

        if (epoch+1) % 2 == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))
            dev_a_count = 0
            for (dev_input_batch, dev_target_batch) in dev_data:
                predict, _, _,_ = model.forward(dev_input_batch)
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
        predict, _, _, _ = model.forward(val_input_batch)
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
    embeddings.insert(0, np.random.randn(50))
    embeddings.append(np.random.randn(50))
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

