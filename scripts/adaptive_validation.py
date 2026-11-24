import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import torch
import torch.nn as nn
from keras. models import load_model
import joblib
from sklearn. preprocessing import StandardScaler
import time

# Load normalization parameters
global_norm_params = np.load('CSTR data profiles/global_norm_params.npy', allow_pickle=True).item()
coeff_norm_params = np.load('CSTR data profiles/coeff_norm_params.npy', allow_pickle=True).item()

CA_min, CA_max = global_norm_params['CA_min'], global_norm_params['CA_max']
T_min, T_max = global_norm_params['T_min'], global_norm_params['T_max']
Q_min, Q_max = global_norm_params['Q_min'], global_norm_params['Q_max']

coeff_CA_min, coeff_CA_max = coeff_norm_params['coeff_CA_min'], coeff_norm_params['coeff_CA_max']
coeff_T_min, coeff_T_max = coeff_norm_params['coeff_T_min'], coeff_norm_params['coeff_T_max']

# Load OASIS (DNN) models and scalers
dnn_CA_loaded = load_model('dnn_CA_model.h5')
dnn_T_loaded = load_model('dnn_T_model.h5')
scaler_X = joblib.load('scaler_X1.joblib')
scaler_Y_CA = joblib.load('scaler_Y_CA1.joblib')
scaler_Y_T = joblib.load('scaler_Y_T1.joblib')

# --- Metrics calculation functions ---
def calculate_r2(y_true, y_pred):
    "Calculate R-squared (coefficient of determination)"
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

def calculate_rmse(y_true, y_pred):
    """Calculate Root Mean Square Error"""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

# --- CSTR physical parameters ---
F = 5.0
V_r = 1.0
R = 8.314
K0 = 8.46e6
E = 5e4
T_in = 300
C_in = 3.5
DH = -1.16e4
rho = 1000
Cp = 0.231

# --- Time and initial conditions ---
time_eval = np.linspace(0, 1, 100)
initial_conditions = [1.2, 600]

# --- Step input profile for Q ---
Q_step = np.zeros(len(time_eval))
step_duration = int(0.2 * len(time_eval))
Q_step[0:step_duration] = -4.25e5
Q_step[step_duration:2*step_duration] = -3.25e5
Q_step[2*step_duration:3*step_duration] = -2.25e5
Q_step[3*step_duration:4*step_duration] = -1.25e5
Q_step[4*step_duration:] = -0.25e5

# --- CSTR ODE model for ground truth ---
def reaction_rate(CA, T):
    K = K0 * np.exp(-E / (R * T))
    return K * CA**2

def cstr_model_step(t, y, Q, time_eval):
    CA, T = y
    Q_val = np.interp(t, time_eval, Q)
    dCA_dt = (F/V_r) * (C_in - CA) - reaction_rate(CA, T)
    dT_dt = (F/V_r) * (T_in - T) - (DH / (rho * Cp)) * reaction_rate(CA, T) + Q_val / (rho * Cp * V_r)
    return [dCA_dt, dT_dt]

def simulate_cstr(Q_profile):
    sol = solve_ivp(cstr_model_step, [time_eval[0], time_eval[-1]], initial_conditions,
                    args=(Q_profile, time_eval), t_eval=time_eval, method='RK45')
    return sol.y[0], sol.y[1]

# --- Simulate true CSTR profiles for the new input ---
CA_true, T_true = simulate_cstr(Q_step)

# --- Autoregressive multi-horizon GRU-SINDy prediction ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
N = 10  # input sequence length
k = len(time_eval) - N  # predict for all remaining steps
num_features = 3
CA_dim = len(coeff_CA_min)
T_dim = len(coeff_T_min)
out_dim = CA_dim + T_dim

def denormalize_coeffs(coeffs_norm, coeff_min, coeff_max):
    min_vals = np.asarray(coeff_min).reshape(1, -1)
    max_vals = np.asarray(coeff_max).reshape(1, -1)
    return coeffs_norm * (max_vals - min_vals + 1e-12) + min_vals

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


model = GRUMultiHorizon(N, num_features, out_dim, k, d_model=128, n_layers=4, dropout=0).to(device)
model.load_state_dict(torch.load('save model/gru_sindy_model.pth', map_location=device))
model.eval()

# Pad initial conditions for first N points

# Pad CA_pred and T_pred with the true trajectory for the first N points
CA_pred = np.zeros(len(time_eval))
T_pred = np.zeros(len(time_eval))
CA_pred[:N] = CA_true[:N]
T_pred[:N] = T_true[:N]

# Measurement update interval - change this to any value (1, 10, 20, 40, etc.)
MEAS_UPDATE_INTERVAL = 10  # Change this value to control measurement update frequency

# Prediction horizon - change this to predict 1, 5, or 10 steps ahead
PREDICTION_HORIZON = 10  # Change this value: 1 for 1-step, 5 for 5-step, 10 for 10-step ahead

update_steps = []  # Store steps where ground truth measurement update occurs

# Start timing for GRU-SINDy prediction
gru_start_time = time.time()

# Multi-step prediction using all k horizons based on PREDICTION_HORIZON
for start_t in range(N, len(time_eval), PREDICTION_HORIZON):
    # Create input window using predictions except where ground truth is available
    input_start = start_t - N
    input_end = start_t
    
    Q_seq = (Q_step[input_start:input_end] - Q_min) / (Q_max - Q_min)
    
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
    num_steps_to_predict = min(PREDICTION_HORIZON, len(time_eval) - start_t, k)
    
    # For multi-step prediction, we need to integrate step by step
    ca_current = CA_window[-1]
    t_current = T_window[-1]
    
    # Use each horizon to predict the corresponding step ahead
    for h in range(num_steps_to_predict):
        current_t = start_t + h
        
        # Use horizon h to predict step h+1 ahead
        CA_coeff = denormalize_coeffs(pred[h, :CA_dim], coeff_CA_min, coeff_CA_max).flatten()
        T_coeff = denormalize_coeffs(pred[h, CA_dim:], coeff_T_min, coeff_T_max).flatten()
        
        # Use current state for this prediction step
        prev_Q = Q_step[current_t-1] if current_t > 0 else Q_step[0]
        
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
        
        dt = time_eval[1] - time_eval[0]
        CA_pred[current_t] = ca_current + dCA * dt
        T_pred[current_t] = t_current + dT * dt
        
        # Apply measurement update if needed
        if current_t % MEAS_UPDATE_INTERVAL == 0:
            CA_pred[current_t] = CA_true[current_t]
            T_pred[current_t] = T_true[current_t]
            update_steps.append(current_t)
        
        # Update current state for next step in this prediction sequence
        ca_current = CA_pred[current_t]
        t_current = T_pred[current_t]

# End timing for GRU-SINDy prediction
gru_end_time = time.time()
gru_computation_time = gru_end_time - gru_start_time

# --- OASIS (DNN-SINDy) Prediction ---
def sindy_model_oasis(t, y, Q, coeffs_Ca, coeffs_T):
    Ca, T = y
    Theta_t = [1, Ca, T, Q, Ca**2, T**2, Ca**3, T**3, np.exp(-E/(R*T))*Ca**2] 
    dCa_dt_up = np.dot(coeffs_Ca, Theta_t)
    dT_dt_up = np.dot(coeffs_T, Theta_t)
    return [dCa_dt_up, dT_dt_up]

# Initialize OASIS prediction arrays
CA_oasis = np.zeros(len(time_eval))
T_oasis = np.zeros(len(time_eval))
CA_oasis[0] = CA_true[0]
T_oasis[0] = T_true[0]

# OASIS prediction with measurement updates
oasis_update_steps = []

# Start timing for OASIS prediction
oasis_start_time = time.time()

for i in range(1, len(time_eval)):
    # Prepare input for DNN prediction
    # Use ground truth only immediately after measurement updates, otherwise use predictions
    if i % MEAS_UPDATE_INTERVAL == 0:
        # This is a measurement update step - use ground truth as input
        CA_input = CA_true[i-1]
        T_input = T_true[i-1]
        oasis_update_steps.append(i-1)
    else:
        # Use previous OASIS predictions as input
        CA_input = CA_oasis[i-1]
        T_input = T_oasis[i-1]
    
    X_input = np.array([[CA_input, T_input, Q_step[i-1]]])
    X_input_scaled = scaler_X.transform(X_input)
    
    # Get DNN predictions for coefficients
    coeffs_Ca_scaled = dnn_CA_loaded.predict(X_input_scaled, verbose=0)[0]
    coeffs_T_scaled = dnn_T_loaded.predict(X_input_scaled, verbose=0)[0]
    
    # Inverse transform the predicted coefficients
    coeffs_Ca = scaler_Y_CA.inverse_transform(coeffs_Ca_scaled.reshape(1, -1)).flatten()
    coeffs_T = scaler_Y_T.inverse_transform(coeffs_T_scaled.reshape(1, -1)).flatten()
    
    # Set up and solve the SINDy model for one step
    y0 = [CA_input, T_input]
    t_span = [time_eval[i-1], time_eval[i]]
    
    sol = solve_ivp(
        sindy_model_oasis,
        t_span,
        y0,
        args=(Q_step[i-1], coeffs_Ca, coeffs_T),
        method='RK45',
        t_eval=[t_span[1]]
    )
    
    CA_oasis[i] = sol.y[0, -1]
    T_oasis[i] = sol.y[1, -1]
    
    # Apply measurement update at specified intervals
    if i % MEAS_UPDATE_INTERVAL == 0:
        CA_oasis[i] = CA_true[i]
        T_oasis[i] = T_true[i]
        oasis_update_steps.append(i)

# End timing for OASIS prediction
oasis_end_time = time.time()
oasis_computation_time = oasis_end_time - oasis_start_time

# Calculate performance metrics
ca_gru_r2 = calculate_r2(CA_true, CA_pred)
ca_gru_rmse = calculate_rmse(CA_true, CA_pred)
ca_oasis_r2 = calculate_r2(CA_true, CA_oasis)
ca_oasis_rmse = calculate_rmse(CA_true, CA_oasis)

t_gru_r2 = calculate_r2(T_true, T_pred)
t_gru_rmse = calculate_rmse(T_true, T_pred)
t_oasis_r2 = calculate_r2(T_true, T_oasis)
t_oasis_rmse = calculate_rmse(T_true, T_oasis)

# Print computation times
print(".4f")
print(".4f")
print(".4f")

# --- Plot results ---
plt.figure(figsize=(16, 5))

plt.subplot(1, 3, 1)
plt.plot(time_eval, Q_step, 'g')
plt.xlim(0, 1)
plt.xlabel('Time (h)', fontsize=14)
plt.ylabel('Heat Input (KJ/h)', fontsize=14)
#plt.title('Step Input - Q')

plt.subplot(1, 3, 2)
plt.plot(time_eval, CA_true, label='True $C_{A}$', linestyle='--', color='black', linewidth=2)
plt.plot(time_eval, CA_pred, label='GRU-SINDy $C_{A}$', linestyle='-', color='green', linewidth=1.5)
#plt.plot(time_eval, CA_oasis, label='OASIS $C_{A}$', linestyle='--', color='orange', linewidth=1.5)

# Add markers where measurement updates occur (ground truth corrections)
gt_update_marker = dict(marker='o', color='red', markersize=3, linestyle='None', zorder=5, label='Measurement Update')
gt_update_marker_nolegend = dict(marker='o', color='red', markersize=3, linestyle='None', zorder=5)
for i, step in enumerate(update_steps):
    if i == 0:
        plt.plot(time_eval[step], CA_pred[step], **gt_update_marker)
    else:
        plt.plot(time_eval[step], CA_pred[step], **gt_update_marker_nolegend)

plt.xlim(0, 1)
plt.xlabel('Time (h)', fontsize=14)
plt.ylabel('Concentration (Kmol/m³)', fontsize=14)
#plt.title(f'Step Output - CA ({PREDICTION_HORIZON}-step GRU vs 1-step OASIS, Meas: {MEAS_UPDATE_INTERVAL}) \n GRU-SINDy: R²={ca_gru_r2:.3f}, RMSE={ca_gru_rmse:.3f}, Time: {gru_computation_time:.4f}s \n OASIS: R²={ca_oasis_r2:.3f}, RMSE={ca_oasis_rmse:.3f}, Time: {oasis_computation_time:.4f}s')
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(time_eval, T_true, label='True T', linestyle='--', color='black', linewidth=2)
plt.plot(time_eval, T_pred, label='GRU-SINDy T', linestyle='-', color='green', linewidth=1.5)
#plt.plot(time_eval, T_oasis, label='OASIS T', linestyle='--', color='orange', linewidth=1.5)

# Add markers where measurement updates occur (ground truth corrections)
for i, step in enumerate(update_steps):
    if i == 0:
        plt.plot(time_eval[step], T_pred[step], **gt_update_marker)
    else:
        plt.plot(time_eval[step], T_pred[step], **gt_update_marker_nolegend)

plt.xlim(0, 1)
plt.xlabel('Time (h)', fontsize=14)
plt.ylabel('Temperature (K)', fontsize=14)
#plt.title(f'Step T ({PREDICTION_HORIZON}-step GRU vs 1-step OASIS, Meas: {MEAS_UPDATE_INTERVAL}) \n GRU-SINDy: R²={t_gru_r2:.3f}, RMSE={t_gru_rmse:.3f}, Time: {gru_computation_time:.4f}s \n OASIS: R²={t_oasis_r2:.3f}, RMSE={t_oasis_rmse:.3f}, Time: {oasis_computation_time:.4f}s')
plt.legend()

plt.tight_layout()
plt.show()
