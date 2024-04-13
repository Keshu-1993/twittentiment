import torch
import torch.nn as nn
import torch.optim as optim
from feat_engg import X_train
from data_prep import train_data

# Define neural network model
class SentimentClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SentimentClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        output = self.softmax(x)
        return output

# Initialize model
input_dim = X_train.shape[1]
hidden_dim = 128
output_dim = len(train_data['Sentiment'].unique())
model = SentimentClassifier(input_dim, hidden_dim, output_dim)

# Define loss function and optimizer
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
