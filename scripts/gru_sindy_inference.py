
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import torch
import torch.nn as nn

# --- Load normalization parameters ---
global_norm_params = np.load('global_norm_params.npy', allow_pickle=True).item()
coeff_norm_params = np.load('coeff_norm_params.npy', allow_pickle=True).item()
CA_min, CA_max = global_norm_params['CA_min'], global_norm_params['CA_max']
T_min, T_max = global_norm_params['T_min'], global_norm_params['T_max']
Q_min, Q_max = global_norm_params['Q_min'], global_norm_params['Q_max']
coeff_CA_min, coeff_CA_max = coeff_norm_params['coeff_CA_min'], coeff_norm_params['coeff_CA_max']
coeff_T_min, coeff_T_max = coeff_norm_params['coeff_T_min'], coeff_norm_params['coeff_T_max']

# --- SINDy physical constants ---
E = 5e4
R = 8.314

# --- Load test profiles ---
good_data = np.load('CSTR_sindy_data.npz')
Q_profiles = good_data['Q_profiles']
CA_profiles = good_data['CA_profiles']
T_profiles = good_data['T_profiles']
best_coeff_CA = good_data['best_coeff_CA']
best_coeff_T = good_data['best_coeff_T']

# --- Model parameters ---
N = 10  # input sequence length
k = 10  # number of horizons
num_features = 3
CA_dim = best_coeff_CA.shape[1]
T_dim = best_coeff_T.shape[1]
out_dim = CA_dim + T_dim

time_eval = np.linspace(0, 1, 100)

# --- GRU multi-horizon regressor ---
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
        y, _ = self.gru(x)
        y = y[:, -1, :]
        out = self.head(y)
        out = out.view(-1, self.k, self.out_dim)
        return out

def denormalize_coeffs(coeffs_norm, coeff_min, coeff_max):
    min_vals = np.asarray(coeff_min).reshape(1, -1)
    max_vals = np.asarray(coeff_max).reshape(1, -1)
    return coeffs_norm * (max_vals - min_vals + 1e-12) + min_vals
# --- Autoregressive multi-horizon GRU-SINDy prediction ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GRUMultiHorizon(N, num_features, out_dim, k, d_model=128, n_layers=4, dropout=0).to(device)
model.load_state_dict(torch.load('save model/gru_sindy_model.pth', map_location=device))
model.eval()

test_traj = 2  # Change this index for other test trajectories
CA_true = CA_profiles[test_traj]
T_true = T_profiles[test_traj]
Q_true = Q_profiles[test_traj]

total_steps = len(time_eval)
CA_pred = np.zeros(total_steps)
T_pred = np.zeros(total_steps)
# Initialize first N values with ground truth for proper sequence construction
CA_pred[:N] = CA_true[:N]
T_pred[:N] = T_true[:N]

delta_t = time_eval[1] - time_eval[0]

# Measurement update interval - change this to any value (1, 10, 20, 40, etc.)
MEAS_UPDATE_INTERVAL = 10 # Change this value to control measurement update frequency

# Prediction horizon - change this to predict 1, 5, or 10 steps ahead
PREDICTION_HORIZON = 10  # Change this value: 1 for 1-step, 5 for 5-step, 10 for 10-step ahead

update_steps = []  # Store steps where ground truth measurement update occurs

# Multi-step prediction using all k horizons based on PREDICTION_HORIZON
for start_t in range(N, total_steps, PREDICTION_HORIZON):
    # Create input window using predictions except where ground truth is available
    input_start = start_t - N
    input_end = start_t
    
    Q_seq = (Q_true[input_start:input_end] - Q_min) / (Q_max - Q_min)
    
    CA_window = np.zeros(N)
    T_window = np.zeros(N)
    
    for i, idx in enumerate(range(input_start, input_end)):
        if idx % MEAS_UPDATE_INTERVAL == 0:  # Ground truth available at measurement updates
            CA_window[i] = CA_true[idx]
            T_window[i] = T_true[idx]
        else:  # Use predictions
            CA_window[i] = CA_pred[idx]
            T_window[i] = T_pred[idx]
    
    CA_seq = (CA_window - CA_min) / (CA_max - CA_min)
    T_seq = (T_window - T_min) / (T_max - T_min)
    seq = np.stack([Q_seq, CA_seq, T_seq], axis=1)  # (N, 3)
    seq_tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(device)  # (1, N, 3)
    
    with torch.no_grad():
        pred = model(seq_tensor)  # (1, k, out_dim) - Gets ALL k horizon predictions
    pred = pred.cpu().numpy()[0]  # (k, out_dim)
    
    # Use all available horizons up to PREDICTION_HORIZON
    num_steps_to_predict = min(PREDICTION_HORIZON, total_steps - start_t, k)
    
    # Initial state for predictions
    ca_current = CA_window[-1]
    t_current = T_window[-1]
    
    # Use each horizon to predict the corresponding step ahead
    for h in range(num_steps_to_predict):
        current_t = start_t + h
        
        # Use horizon h to predict step h+1 ahead
        CA_coeff = denormalize_coeffs(pred[h, :CA_dim], coeff_CA_min, coeff_CA_max).flatten()
        T_coeff = denormalize_coeffs(pred[h, CA_dim:], coeff_T_min, coeff_T_max).flatten()
        
        # Use current state for prediction
        prev_Q = Q_true[current_t-1] if current_t > 0 else Q_true[0]
        
        Theta_row = [
            1,
            ca_current,
            t_current,
            prev_Q,
            ca_current**2,
            t_current**2,
            ca_current**3,
            t_current**3,
            np.exp(-E/(R*t_current))*ca_current**2
        ]
        
        dCA = np.dot(Theta_row, CA_coeff)
        dT = np.dot(Theta_row, T_coeff)
        
        CA_pred[current_t] = ca_current + dCA * delta_t
        T_pred[current_t] = t_current + dT * delta_t
        
        # Apply measurement update if needed
        if current_t % MEAS_UPDATE_INTERVAL == 0:
            CA_pred[current_t] = CA_true[current_t]
            T_pred[current_t] = T_true[current_t]
            update_steps.append(current_t)
        
        # Update state for next prediction in this sequence
        ca_current = CA_pred[current_t]
        t_current = T_pred[current_t]

# --- Calculate relative errors ---
# Individual component relative errors (for comparison)
CA_rel_error = np.abs(CA_pred - CA_true) / (np.abs(CA_true) + epsilon)
T_rel_error = np.abs(T_pred - T_true) / (np.abs(T_true) + epsilon)

# Correct relative error using L2-norm: RE(t) = ||x(t) - x̂(t)||₂ / ||x(t)||₂
# where x(t) = [CA(t), T(t)] is the state vector
x_true = np.column_stack([CA_true, T_true])  # True state vectors (n_timesteps, 2)
x_pred = np.column_stack([CA_pred, T_pred])  # Predicted state vectors (n_timesteps, 2)

# Calculate L2-norm of error and true state for each time step
error_norm = np.linalg.norm(x_pred - x_true, axis=1)  # ||x(t) - x̂(t)||₂
true_norm = np.linalg.norm(x_true, axis=1)  # ||x(t)||₂

# Relative error using L2-norm formula
combined_error = error_norm / (true_norm + epsilon)

# --- Plot CA results ---
plt.figure(figsize=(7, 5))
plt.plot(time_eval, CA_true, label='True $C_{A}$', linestyle='--', color='black',linewidth=2)
plt.plot(time_eval, CA_pred, label='GRU-SINDy $C_{A}$', linestyle='-', color='green',linewidth=1.5)
plt.xlabel('Time (h)', fontsize=14)
plt.ylabel('Concentration (Kmol/m³)', fontsize=14)
plt.xlim(0, 1)
plt.legend()
plt.tight_layout()
plt.show()

# --- Plot T results ---
plt.figure(figsize=(7, 5))
plt.plot(time_eval, T_true, label='True T', linestyle='--', color='black',linewidth=2)
plt.plot(time_eval, T_pred, label='GRU-SINDy T', linestyle='-', color='green',linewidth=1.5)
plt.xlabel('Time (h)', fontsize=14)
plt.ylabel('Temperature (K)', fontsize=14)
plt.xlim(0, 1)
plt.legend()
plt.tight_layout()
plt.show()

# --- Plot Relative Error ---
plt.figure(figsize=(7, 5))
plt.plot(time_eval, combined_error, label='Relative Error', linestyle='-', color='purple', linewidth=1.5)
plt.xlabel('Time (h)', fontsize=14)
plt.ylabel('Relative Error', fontsize=14)
plt.xlim(0, 1)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
