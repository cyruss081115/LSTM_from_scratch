import torch
import yfinance as yf
import matplotlib.pyplot as plt

from tqdm import tqdm
from source.model import Model

def test():
    model = Model(num_features=5, input_length=30, output_length=1)
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()
    
    AAPL = yf.download('AAPL', start='2021-01-01', end='2021-12-31')
    AAPL_tensor = torch.from_numpy(AAPL.to_numpy()).float()
    
    start_idx = -60
    end_idx = -30

    predictions = []
    input_seq = AAPL_tensor[start_idx:end_idx].unsqueeze(0)
    for i in tqdm(range(30)):
        x_min, _ = torch.min(input_seq, dim=1, keepdim=True)
        x_max, _ = torch.max(input_seq, dim=1, keepdim=True)
        X = (input_seq - x_min) / (x_max - x_min)
        y_pred = model(X)
        y_pred = y_pred * (x_max[0, 0, 0] - x_min[0, 0, 0]) + x_min[0, 0, 0]
        predictions.append(y_pred.item())

        input_seq = AAPL_tensor[start_idx+i:end_idx+i].unsqueeze(0)
        input_seq[0, -1, 0] = y_pred.item()
    print(AAPL['Close'])
    plt.plot(predictions, label='Predictions')
    plt.plot(AAPL_tensor[start_idx:end_idx, 0].cpu().numpy(), label='True')
    plt.legend()
    plt.xlabel('Days')
    plt.ylabel('Price')
    plt.show()

if __name__ == '__main__':
    test()

