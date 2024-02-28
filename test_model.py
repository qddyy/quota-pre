import torch
import os
from pathlib import Path
from data.lstm_datloader import lstm_test_data
from model.vgg_lstm import VGG_LSTM

model_path = Path(__file__).parent / "vgg_lstm_model.pth"
test_data = lstm_test_data("IC.CFX", 64, 50)
criterion = torch.nn.MSELoss(reduction="sum")
model = VGG_LSTM(5, 20, 50, 100)
model.load_state_dict(torch.load(model_path)())
model.eval()
test_loss = 0
correct = 0
with torch.no_grad():
    for data, target in test_data:
        y_pred = model(data)
        pred = torch.zeros_like(y_pred)
        batch_num = y_pred.size(0)
        test_loss += criterion(y_pred, target).item()  # 累加测试集上的损失
        for i in range(batch_num):
            max_ind = y_pred[i].argmax()
            pred[i][max_ind] = 1
        correct += (pred.data == target.data).all(dim=1).sum()

test_loss /= len(test_data.dataset)  # 平均损失
accuracy = 100.0 * correct / len(test_data.dataset)  # 计算准确率
print(
    "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
        test_loss, correct, len(test_data.dataset), accuracy
    )
)
