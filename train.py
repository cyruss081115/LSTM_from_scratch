import torch
from torch.utils.data import DataLoader

from source.dataset import StockDataset
from source.model import Model
from datetime import datetime


def train_model():
    stock_abbv = "AAPL"
    start = datetime(2010, 1, 1)
    end = datetime(2020, 1, 1)
    sequence_length = 30
    sequence_step = 5
    train_test_ratio = 0.8
    seed = 42
    batch_size = 32
    epochs = 100
    device = (
        'cuda' if torch.cuda.is_available() else
        'mps' if torch.backends.mps.is_available() else
        'cpu'
    )

    train_dataset = StockDataset(
        stock_abbv=stock_abbv,
        start=start,
        end=end,
        sequence_length=sequence_length,
        sequence_step=sequence_step,
        train_test_ratio=train_test_ratio,
        seed=seed,
    )
    test_dataset = StockDataset(
        stock_abbv=stock_abbv,
        start=start,
        end=end,
        sequence_length=sequence_length,
        sequence_step=sequence_step,
        train_test_ratio=train_test_ratio,
        seed=seed,
    )
    train_dataset.set_train()
    test_dataset.set_eval()
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = Model(num_features=train_dataset.X[0].shape[1], input_length=30, output_length=1).to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, verbose=True)

    model.train()

    training_losses = []
    testing_losses = []
    best_loss = float('inf')

    for epoch in range(epochs):
        total_training_loss = 0
        total_testing_loss = 0
        
        model.train()
        for i, (X, y) in enumerate(train_dataloader):
            X = X.float().to(device)
            y = y.float().to(device)

            optimizer.zero_grad()
            y_pred = model(X)
            loss = criterion(y_pred.squeeze(), y.squeeze())
            loss.backward()
            optimizer.step()

            total_training_loss += loss.item()

        avg_training_loss = total_training_loss / len(train_dataloader)
        training_losses.append(avg_training_loss)

        model.eval()
        for i, (X, y) in enumerate(test_dataloader):
            X = X.float().to(device)
            y = y.float().to(device)

            y_pred = model(X)
            loss = criterion(y_pred.squeeze(), y.squeeze())

            total_testing_loss += loss.item()

        avg_testing_loss = total_testing_loss / len(test_dataloader)
        testing_losses.append(avg_testing_loss)

        scheduler.step(avg_testing_loss)

        if avg_testing_loss < best_loss:
            best_loss = avg_testing_loss
            torch.save(model.state_dict(), "best_model.pth")

        print(f"Epoch {epoch+1}/{epochs} - Training Loss: {avg_training_loss:.4f}, Testing Loss: {avg_testing_loss:.4f}")

    return training_losses, testing_losses


def plot_losses(training_losses, testing_losses):
    import matplotlib.pyplot as plt
    plt.plot(training_losses, label="Training Loss")
    plt.plot(testing_losses, label="Testing Loss")
    plt.legend()
    plt.savefig("losses.png")


if __name__ == "__main__":
    training_losses, testing_losses = train_model()
    plot_losses(training_losses, testing_losses)

