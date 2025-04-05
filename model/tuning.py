import torch
from trainer import Trainer

tickers = ['AAPL', 'AMZN', 'META', 'GOOGL', 'NFLX', 'MSFT']
epochs = 50
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 32


hidden_sizes = [50, 100, 200]
num_layers = [1, 2]
dropouts = [0.2, 0.3]

trainer = Trainer(epochs=epochs, device=device, batch_size=batch_size)

for ticker in tickers:
    trainer.hyperparameter_tuning(ticker, hidden_sizes, num_layers, dropouts)
