
import numpy as np
import torch
import random
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim

# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
set_seed(42)

# Load only good SINDy models and profiles
good_data = np.load('CSTR_sindy_data.npz')
Q_profiles = good_data['Q_profiles']
CA_profiles = good_data['CA_profiles']
T_profiles = good_data['T_profiles']
best_coeff_CA = good_data['best_coeff_CA']
best_coeff_T = good_data['best_coeff_T']

# Normalization
Q_min, Q_max = Q_profiles.min(), Q_profiles.max()
CA_min, CA_max = CA_profiles.min(), CA_profiles.max()
T_min, T_max = T_profiles.min(), T_profiles.max()

Q_profiles_norm = (Q_profiles - Q_min) / (Q_max - Q_min)
CA_profiles_norm = (CA_profiles - CA_min) / (CA_max - CA_min)
T_profiles_norm = (T_profiles - T_min) / (T_max - T_min)

global_norm_params = {
    'CA_min': CA_min, 'CA_max': CA_max,
    'T_min': T_min, 'T_max': T_max,
    'Q_min': Q_min, 'Q_max': Q_max
}
np.save('CSTR data profiles/global_norm_params.npy', global_norm_params)

# Coefficient normalization
coeff_CA_min = best_coeff_CA.min(axis=0)
coeff_CA_max = best_coeff_CA.max(axis=0)
coeff_T_min = best_coeff_T.min(axis=0)
coeff_T_max = best_coeff_T.max(axis=0)

best_coeff_CA_norm = (best_coeff_CA - coeff_CA_min) / (coeff_CA_max - coeff_CA_min + 1e-12)
best_coeff_T_norm = (best_coeff_T - coeff_T_min) / (coeff_T_max - coeff_T_min + 1e-12)

coeff_norm_params = {
    'coeff_CA_min': coeff_CA_min, 'coeff_CA_max': coeff_CA_max,
    'coeff_T_min': coeff_T_min, 'coeff_T_max': coeff_T_max
}
np.save('CSTR data profiles/coeff_norm_params.npy', coeff_norm_params)

# Prepare features and targets for GRU (multi-horizon)
N = 10  # input sequence length
k = 10  # number of future steps to predict
num_features = 3  # Q, CA, T
CA_dim = best_coeff_CA.shape[1]
T_dim = best_coeff_T.shape[1]
out_dim = CA_dim + T_dim

X_seq = []
Y_seq = []
for i in range(Q_profiles.shape[0]):
    traj_len = Q_profiles.shape[1]
    for start in range(traj_len - N - k + 1):
        # Input window
        seq = np.stack([
            Q_profiles_norm[i, start:start+N],
            CA_profiles_norm[i, start:start+N],
            T_profiles_norm[i, start:start+N]
        ], axis=1)  # (N, 3)
        X_seq.append(seq)
        # Multi-horizon targets
        y_future = []
        for j in range(k):
            # Use the same coefficients for all k steps (if only one set per trajectory)
            coeff_ca = best_coeff_CA_norm[i]
            coeff_t = best_coeff_T_norm[i]
            y_future.append(np.concatenate([coeff_ca, coeff_t]))
        Y_seq.append(np.stack(y_future, axis=0))  # (k, out_dim)
X_seq = np.array(X_seq)  # (num_samples, N, 3)
Y_seq = np.array(Y_seq)  # (num_samples, k, out_dim)

# After creating X_seq and Y_seq
print(f"X_seq shape: {X_seq.shape}")  # (num_samples, N, 3)
print(f"Y_seq shape: {Y_seq.shape}")  # (num_samples, k, out_dim)

# Train/val/test split
X_train, X_test, Y_train, Y_test = train_test_split(X_seq, Y_seq, test_size=0.2, random_state=42)
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.1, random_state=42)

# DataLoaders
batch_size = 64  # Increased batch size for more stable training
train_loader = DataLoader(TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(Y_train, dtype=torch.float32)), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(Y_val, dtype=torch.float32)), batch_size=batch_size)
test_loader = DataLoader(TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(Y_test, dtype=torch.float32)), batch_size=batch_size)

# GRU-only multi-horizon regressor
class GRUMultiHorizon(nn.Module):
    def __init__(self, seq_len, in_dim, out_dim, k, d_model=128, n_layers=4, dropout=0):
        super().__init__()
        self.gru = nn.GRU(
            input_size=in_dim,
            hidden_size=d_model,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0
        )
        self.head = nn.Linear(d_model, out_dim * k)
        self.k = k
        self.out_dim = out_dim

    def forward(self, x):
        # x: (batch, seq_len, in_dim)
        y, _ = self.gru(x)  # (batch, seq_len, d_model)
        y = y[:, -1, :]     # last hidden state
        out = self.head(y)  # (batch, out_dim*k)
        out = out.view(-1, self.k, self.out_dim)  # (batch, k, out_dim)
        return out

# Training setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GRUMultiHorizon(N, num_features, out_dim, k, d_model=128, n_layers=4, dropout=0).to(device)  # Adjusted model size
optimizer = optim.Adam(model.parameters(), lr=1e-4,  weight_decay=1e-4)
# Add a learning rate scheduler for adaptive learning
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
criterion = nn.L1Loss()  # MAE loss for robust regression

# After creating the model
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total trainable model parameters: {total_params}")

num_epochs = 500
patience = 50
best_val_loss = float('inf')
patience_counter = 0
train_losses, val_losses = [], []

for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * xb.size(0)
    train_loss /= len(train_loader.dataset)
    train_losses.append(train_loss)

    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            val_loss += loss.item() * xb.size(0)
    val_loss /= len(val_loader.dataset)
    val_losses.append(val_loss)

    # Step the scheduler
    scheduler.step(val_loss)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), 'gru_sindy_model.pth')
    else:
        patience_counter += 1

    print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.6f} - Val Loss: {val_loss:.6f} - Best Val: {best_val_loss:.6f}")
    if best_val_loss < 1e-4:
        print('Early stopping: Validation loss below 1e-4')
        break
    if patience_counter >= patience:
        print('Early stopping: No improvement for {} epochs'.format(patience))
        break

# Plot loss curves
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('MAE Loss')
plt.title('GRU Multi-Horizon Training and Validation Loss')
plt.savefig('save model/gru_loss_curve.png')
plt.legend()
plt.show()
