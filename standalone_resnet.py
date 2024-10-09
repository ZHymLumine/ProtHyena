import torch
import torch.nn as nn
from torch.nn import LayerNorm, ModuleList, functional as F
from torch.nn.utils.weight_norm import weight_norm
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from tqdm import tqdm
import logging
import os 


def train_model(model, train_dataloader, valid_dataloader, test_dataloader, optimizer, loss_function, device, epochs=50):
    model.train()  # Set the model to training mode
    for epoch in range(epochs):
        running_loss = 0.0
        progress_bar = tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{epochs}', leave=False)
        for sequences, labels in progress_bar:
            sequences = sequences.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()  
            outputs = model(sequences)  
            loss = loss_function(outputs.squeeze(), labels) 
            loss.backward()  
            optimizer.step()  
            running_loss += loss.item() * sequences.size(0)
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

        epoch_loss = running_loss / len(train_dataloader.dataset)
        print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}')
        logger.info(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}')
        # Evaluate the model
        evaluate_model(model, valid_dataloader, 'Validation', epoch)
        evaluate_model(model, test_dataloader, 'Testing', epoch)


def student_t(outs, y, nu=1.0):
    preds, y = outs.detach().cpu(), y.detach().cpu()
    diff = y - preds  
    # 计算 Student-T 损失
    loss = torch.log(1 + (diff ** 2) / nu).mean()  # 使用 .mean() 来计算批次的平均损失
    
    return loss


def evaluate_model(model, dataloader, phase, epoch):
    model.eval()  
    predictions, actuals = [], []
    progress_bar = tqdm(dataloader, desc=f'{phase} Evaluation', leave=False)
    with torch.no_grad():
        for sequences, labels in progress_bar:
            sequences = sequences.to(device)
            labels = labels.to(device)
            outputs = model(sequences)
            predictions.extend(outputs.squeeze().tolist())
            actuals.extend(labels.tolist())

    # Compute the Spearman's rho, MSE, MAE, and Student t loss
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    mse = ((predictions - actuals) ** 2).mean()
    mae = np.abs(predictions - actuals).mean()
    rho, _ = spearmanr(predictions, actuals)

    diff = actuals - predictions
    nu = 1.0
    # student_t_loss = np.sum((predictions - actuals)**2 / (1 + predictions**2 + actuals**2))
    student_t_loss = np.log(1 + (diff ** 2) / nu).mean()
    
    # TODO
    save_path = '/home/lr/zym/research/bio-transformers/test/logs/CNN/stability'
    dataset_name = 'stability'
    result_path = os.path.join(save_path, f'{phase}_{dataset_name}_{epoch}_predictions_labels.csv')
    df = pd.DataFrame({'Prediction': predictions.tolist(), 'Label': actuals.tolist()})
    df.to_csv(result_path, index=False)
    print(f'Predictions and labels saved to {result_path}')

    print(f'Epoch {epoch+1}: {phase} - Spearman\'s rho: {rho:.4f}, MSE: {mse:.4f}, MAE: {mae:.4f}, Student T Loss: {student_t_loss:.4f}')
    logger.info(f'Epoch {epoch+1}: {phase} - Spearman\'s rho: {rho:.4f}, MSE: {mse:.4f}, MAE: {mae:.4f}, Student T Loss: {student_t_loss:.4f}')
    progress_bar.set_postfix({'mse': f'{mse:.4f}', 'mae': f'{mae:.4f}', 'rho': f'{rho:.4f}', 'student_t_loss': f'{student_t_loss:.4f}'})
    

class ProteinDataset(Dataset):
    def __init__(self, dataframe, max_len):
        self.sequences = dataframe['seq'].apply(self.encode_sequence, max_len=max_len).values
        self.labels = dataframe['label'].values
        self.max_len = max_len

    def encode_sequence(self, sequence, max_len):
        # Map each amino acid to an integer
        aa_to_int = {aa: i for i, aa in enumerate('ARNDCQEGHILKMFPSTWYV')}
        encoded = [aa_to_int.get(aa, 0) for aa in sequence]
        # Pad or truncate the sequence
        padded_sequence = encoded[:max_len] if len(encoded) >= max_len else encoded + [0] * (max_len - len(encoded))
        return np.array(padded_sequence)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sequence = torch.tensor(self.sequences[idx], dtype=torch.long)
        label = torch.tensor(self.labels[idx], dtype=torch.float)
        return sequence, label


class ProteinConfig:
    """Basic configuration class for protein models."""
    def __init__(self, vocab_size=20, hidden_size=256, num_hidden_layers=35, hidden_act="relu",
                 hidden_dropout_prob=0.1, initializer_range=0.02, layer_norm_eps=1e-12):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps

def get_activation_fn(activation):
    """Return activation function given string"""
    if activation == "gelu":
        return torch.nn.functional.gelu
    elif activation == "relu":
        return torch.nn.functional.relu
    else:
        raise RuntimeError("Activation function not implemented")

class SimpleMLP(nn.Module):
    def __init__(self, in_dim: int, hid_dim: int, out_dim: int, dropout: float = 0.):
        super().__init__()
        self.main = nn.Sequential(
            weight_norm(nn.Linear(in_dim, hid_dim), dim=None),
            nn.ReLU(),
            nn.Dropout(dropout, inplace=True),
            weight_norm(nn.Linear(hid_dim, out_dim), dim=None))

    def forward(self, x):
        return self.main(x)

class MaskedConv1d(nn.Conv1d):
    def forward(self, x, input_mask=None):
        if input_mask is not None:
            x = x * input_mask
        return super().forward(x)

class ProteinResNetLayerNorm(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm = LayerNorm(config.hidden_size)

    def forward(self, x):
        return self.norm(x.transpose(1, 2)).transpose(1, 2)

class ProteinResNetBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        padding = 8
        self.conv1 = MaskedConv1d(config.hidden_size, config.hidden_size, kernel_size=9, padding=padding, dilation=2, bias=False)
        self.bn1 = ProteinResNetLayerNorm(config)
        self.conv2 = MaskedConv1d(config.hidden_size, config.hidden_size, kernel_size=9, padding=padding, dilation=2, bias=False)
        self.bn2 = ProteinResNetLayerNorm(config)
        self.activation_fn = get_activation_fn(config.hidden_act)

    def forward(self, x, input_mask=None):
        identity = x
        out = self.conv1(x, input_mask)
        out = self.bn1(out)
        out = self.activation_fn(out)
        out = self.conv2(out, input_mask)
        out = self.bn2(out)
        out += identity
        return self.activation_fn(out)

class ResNetEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer = ModuleList([ProteinResNetBlock(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, input_mask=None):
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, input_mask)
        return hidden_states

class ProteinResNetModel(nn.Module):
    """ResNet model for protein sequence processing."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.encoder = ResNetEncoder(config)
        self.pool = nn.AdaptiveAvgPool1d(1)  
        self.mlp = SimpleMLP(config.hidden_size, 512, 1)

    def forward(self, x, input_mask=None):
        x = self.embedding(x)
        x = x.transpose(1, 2)
        x = self.encoder(x, input_mask)
        x = self.pool(x).squeeze(-1)
        return self.mlp(x)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        handlers=[
                            logging.FileHandler("training_stability.log"),  # 日志文件输出
                            logging.StreamHandler()  # 控制台输出
                        ])
    logger = logging.getLogger()

    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("Using GPU:", torch.cuda.get_device_name(device))
    else:
        device = torch.device('cpu')
        print("Using CPU")
        
    config = ProteinConfig()
    model = ProteinResNetModel(config).to(device)

    print(model)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total trainable parameters: {num_params/1000000} M")


    # TODO
    train_data_path = '/raid_elmo/home/lr/zym/data/protein_data/fine_tuning/stability/train.csv'
    valid_data_path = '/raid_elmo/home/lr/zym/data/protein_data/fine_tuning/stability/valid.csv'
    test_data_path = '/raid_elmo/home/lr/zym/data/protein_data/fine_tuning/stability/test.csv'
    train_data = pd.read_csv(train_data_path)
    train_dataset = ProteinDataset(train_data, max_len=1024)
    train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)

    valid_data = pd.read_csv(valid_data_path)
    valid_dataset = ProteinDataset(valid_data, max_len=1024)
    valid_dataloader = DataLoader(valid_dataset, batch_size=64, shuffle=False)

    test_data = pd.read_csv(test_data_path)
    test_dataset = ProteinDataset(test_data, max_len=1024)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_function = nn.MSELoss()

    train_model(model, train_dataloader, valid_dataloader, test_dataloader, optimizer, loss_function, device)
