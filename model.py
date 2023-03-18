import torch
import preproces as pr 
import emotions as Emotion
import json
import numpy as np
import os

class LSTM(torch.nn.Module):
    
    def __init__(self, input_dim, embedding_dim, hidden_dim,output_dim):
        super().__init__()

        self.embedding = torch.nn.Embedding(input_dim, embedding_dim)
        self.lstm = torch.nn.LSTM(embedding_dim,
                                 hidden_dim,
                                  batch_first = True)

        self.fc1 = torch.nn.Linear(8, 1)
        self.relu = torch.nn.ReLU(inplace=True)
        self.fc2 = torch.nn.Linear(hidden_dim+1, hidden_dim+1)
        self.fc = torch.nn.Linear(hidden_dim+1, output_dim)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, text,emo):
        
        embedded = self.embedding(text)
        output, (hidden, cell) = self.lstm(embedded)
        hidden.squeeze_(0)
        
        fc1 = self.fc1(emo)
        fc1 = self.relu(fc1)
        concatenated = torch.cat((hidden,fc1))
        
        fc2 = self.fc2(concatenated)
        fc2 = self.relu(fc2)
        output = self.fc(fc2)
        outp = self.sigmoid(output)

        return outp

savedmodel = LSTM(input_dim=10002,
            embedding_dim=256,
            hidden_dim=256,
            output_dim=1)

# savedmodel.load_state_dict(torch.load('D:\Skripsi\Demo-app\99cobastatedict.pth'))

def predict(text):
    current_path = os.getcwd()
    PATH = os.path.join(current_path, '99cobastatedict.pth')

    device = torch.device("cpu")
    mod = savedmodel 
    mod.load_state_dict(torch.load(PATH, map_location=device))

    emo_path = os.path.join(current_path, 'NRC.txt')
    
    cleaned = pr.Cleaning(text)
    cfold = pr.caseFolding(cleaned)
    token = pr.Tokenization(cfold)
    text = pr.stopWordRemoval(token)

    emo = Emotion.Emotion(text)
    numeric_emotion = emo.getScoreMatrix()

    index =[]
    with open('vocab.json') as json_file:
        data = json.load(json_file)
        for i in text :
            if i in data :
                index.append(data[i])
            else :
                index.append(data['<UNK>'])
    # print(index)
    
    mod.eval()
    with torch.no_grad():
        index = torch.LongTensor(index)
        text = index.to(device)
        numeric = torch.Tensor(numeric_emotion)
        emotion = numeric.to(device)
        y_pred = mod(text,emotion)
        pred_label= np.round(y_pred.cpu()).flatten()

    return int(pred_label.item())

# coba = predict("Hi Firda, We're so grateful for all the love and support you've shown our products these past few seasons. A new season is on it's way, so we thought it would be the perfect time to ask for your feedback on how we can improve and create better products and resources for you! We’re excited to get your feedback on: Artist of Life Workbook (+ vote on next year’s color!) Weekly Reset Planner tbh deck Future products for the Lavendaire shop! Feel free to skip the parts of the survey involving products if you do not have them")
# print("label terprediksi : ", coba)

# current_path = os.getcwd()
# print(current_path)
