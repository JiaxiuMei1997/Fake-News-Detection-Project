import torch.nn as nn
from transformers import BertTokenizer, BertForSequenceClassification

class BertCNN(nn.Module):
    def __init__(self, seqLen):
        super().__init__()
        self.bert = BertForSequenceClassification.from_pretrained("bert-base-uncased")
        self.conv1D = nn.Conv1d(seqLen, seqLen // 2, 5) #Kernel size = 5
        self.maxPool1D = nn.MaxPool1d(seqLen//2, seqLen//4)
    
    def forward(self, tokenizeInputs):
        op = self.bert(**tokenizeInputs, output_hidden_states = True)
        #Taking last hidden state as input to Conv1d
        convOutput = self.conv1D(op.hidden_states[-1])
        return self.maxPool1D(convOutput)

