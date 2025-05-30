import torch
import torch.nn as nn

class AttentionLayer(nn.Module):
    def __init__(self, input_dim):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, inputs):
        weights = self.attention(inputs)
        weights = torch.softmax(weights, dim=1)
        weighted = (inputs * weights).sum(dim=1)
        return weighted

class GoldPricePredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2):
        super(GoldPricePredictor, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.attention = AttentionLayer(hidden_dim * 2)
        self.fc = nn.Linear(hidden_dim * 2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attn_out = self.attention(lstm_out)
        out = self.fc(attn_out)
        return self.sigmoid(out)


def train_model(features, labels, seq_len=10, epochs=15):
    from torch.utils.data import TensorDataset, DataLoader
    from sklearn.model_selection import train_test_split
    import numpy as np

    X = []
    y = []
    for i in range(len(features) - seq_len):
        X.append(features.iloc[i:i+seq_len].values)
        y.append(labels.iloc[i+seq_len])
    X = np.array(X)
    y = np.array(y)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    train_data = TensorDataset(torch.tensor(X_train).float(), torch.tensor(y_train).float())
    val_data = TensorDataset(torch.tensor(X_val).float(), torch.tensor(y_val).float())

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32)

    model = GoldPricePredictor(input_dim=features.shape[1], hidden_dim=64)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            preds = model(xb).squeeze()
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")

    return model