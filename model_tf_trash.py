import pandas as pd
import numpy as np
from tqdm import tqdm

from torch import nn
import torch
from sklearn.model_selection import train_test_split

# Load the data
df = pd.read_parquet('train_fieit5.parquet')
df.drop('name', axis=1, inplace=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

window_size = 90
num_layers = 6
dropout = 0.1
output_size = 15


class TimeSeriesTransformer(nn.Module):
    def __init__(self, num_features, d_model, nhead=8, num_layers=2, dim_feedforward=370, dropout=0.1):
        super(TimeSeriesTransformer, self).__init__()
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout),
            num_layers)
        self.linear_out = nn.Linear(d_model, output_size)

    def forward(self, x):
        x = x.permute(1, 0, 2)
        out = self.transformer_encoder(x)
        out = self.linear_out(out[-1, :, :])
        return out


def create_sequences(data, window_size, output_size):
    sequences = []
    labels = []
    for i in range(len(data) - window_size - output_size + 1):
        sequences.append(data.iloc[i: i + window_size].values)
        labels.append(data['close'].iloc[i + window_size: i + window_size + output_size].values)
    return sequences, labels


def to_tensor(data, device):
    if isinstance(data, list):
        if len(data) > 0:
            data = np.array(data)
        else:
            data = np.array([], dtype=np.float32)
    return torch.tensor(data, dtype=torch.float).to(device)


def process_ticker(df, ticker, window_size=60, output_size=15):
    ticker_df = df[df['ticker'] == ticker].copy()
    ticker_df['date'] = pd.to_datetime(ticker_df['date'])
    base_date = ticker_df['date'].min()
    ticker_df['date'] = (ticker_df['date'] - base_date).dt.days

    X, y = create_sequences(ticker_df.drop(columns='ticker'), window_size, output_size)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)
    return X_train, X_val, y_train, y_val


tickers = df['ticker'].unique()

# Define model
num_features = 185
model = TimeSeriesTransformer(num_features, num_features, nhead=37, num_layers=num_layers,
                              dim_feedforward=num_features * 2, dropout=dropout).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0007)

# Train model
num_epochs = 100
model.train()

pbar = tqdm(total=len(tickers) * num_epochs)

for epoch in range(num_epochs):
    total_loss = 0
    num_batches = 0

    for ticker in tickers:
        X_train, _, y_train, _ = process_ticker(df, ticker)

        inputs = to_tensor(X_train, device)
        labels = to_tensor(y_train, device)

        optimizer.zero_grad()
        outputs = model(inputs)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        pbar.set_description(f'Epoch: {epoch}, Ticker: {ticker}, Loss: {loss.item():.4f}')
        pbar.update()

    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Avg Loss: {total_loss / num_batches:.4f}')

pbar.close()
print('Finished Training')

