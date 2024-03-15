import torch
import os
from pathlib import Path
from data.lstm_datloader import lstm_test_data
from model.vgg_lstm import VGG_LSTM
from train_model import CustomLoss
from utils import read_env

env_path = Path(__file__).parent / "env_vars.txt"
os.environ.update(read_env(env_path))
seq_len = int(os.environ["SEQ_LEN"])
class_num = int(os.environ["CLASS_NUM"])
input_dim = int(os.environ["INPUT_DIM"])
batch_size = int(os.environ["BATCH_SIZE"])
hidden_dim = int(os.environ["HIDDEN_DIM"])
code = os.environ["CODE"]

model_path = Path(__file__).parent / f"vgg_lstm_model_{code}.pth"
test_data = lstm_test_data(code, batch_size, seq_len)
# criterion = torch.nn.MSELoss(reduction="sum")
criterion = CustomLoss()
model = VGG_LSTM(class_num, input_dim, seq_len, hidden_dim)
model.load_state_dict(torch.load(model_path))
model.eval()
test_loss = 0
correct = 0
n = 0
with torch.no_grad():
    for data, target in test_data:
        n += 1
        y_pred = model(data)
        pred = torch.zeros_like(y_pred)
        batch_num = y_pred.size(0)
        test_loss += criterion(y_pred, target).item()  # 累加测试集上的损失
        for i in range(batch_num):
            max_ind = y_pred[i].argmax()
            pred[i][max_ind] = 1
        correct += (pred.data == target.data).all(dim=1).sum()

test_loss /= n  # 平均损失
accuracy = 100.0 * correct / len(test_data.dataset)  # 计算准确率
print(
    "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
        test_loss, correct, len(test_data.dataset), accuracy
    )
)
