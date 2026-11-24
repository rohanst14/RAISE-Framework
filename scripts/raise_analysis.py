import numpy as np
from scipy.optimize import minimize
from scipy.integrate import solve_ivp
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import time
import json
import scipy.stats as stats
import pickle
from scipy.stats import multivariate_normal
from scipy.stats import norm

# --- Physical and MPC Constants ---
F = 5.0
V_r = 1.0
R = 8.314
K0 = 8.46e6
E = 5e4
T_in = 300
C_in = 3.5
delta_H = -1.16e4
rho = 1000
Cp = 0.231

# Economic loss parameters
EML_equipment = 30000   # $/hr
EML_production = 15000  # $/hr
EML_quality = 5000      # $/hr
# Safety thresholds
alarm_threshold = 20000
esh_threshold = 30000

# Load distribution parameters
with open('distribution_parameters.json') as f:
    dist_params = json.load(f)

CA_dist_name = dist_params['CA']['dist_name']
CA_params = dist_params['CA']['params']
T_dist_name = dist_params['T']['dist_name']
T_params = dist_params['T']['params']

CA_dist = getattr(stats, CA_dist_name)(*CA_params)
T_dist = getattr(stats, T_dist_name)(*T_params)

# Load best copula
with open('copula_fitted_parameters.json', 'rb') as f:
    best_copula = json.load(f)

# Define setpoints and operational bounds based on distribution stats
setpoints = {
    'CA_sp': 3.06,  
    'T_sp': 345.29
}
# Use 90th percentile bounds (5th to 95th percentiles) as normal operational bounds
operational_bounds = {
    'A_CA': [dist_params['CA']['lower_bound_90'], dist_params['CA']['upper_bound_90']],
    'A_T': [dist_params['T']['lower_bound_90'], dist_params['T']['upper_bound_90']]
}

# Target values (setpoints) and operational bounds
T_CA = setpoints['CA_sp']
T_T = setpoints['T_sp']
lower_CA = operational_bounds['A_CA'][0]
upper_CA = operational_bounds['A_CA'][1]
lower_T = operational_bounds['A_T'][0]
upper_T = operational_bounds['A_T'][1]


# Inverted Gamma Loss Function (IGLF) for Gamma Distribution
def calculate_iglf(x, mu, lower_bound, upper_bound, shape, loc, scale, EML):
    # Calculate CDF values
    cdf_value = stats.gamma.cdf(x, a=shape, loc=loc, scale=scale)
    cdf_mu = stats.gamma.cdf(mu, a=shape, loc=loc, scale=scale)
    
    # Deviation from setpoint in CDF space
    deviation_cdf = abs(cdf_value - cdf_mu)
    
    # If within bounds, apply reduced penalty
    if lower_bound <= x <= upper_bound:
        # Within bounds: loss scales with distance from setpoint
        # Normalize by the CDF range within bounds
        cdf_lower = stats.gamma.cdf(lower_bound, a=shape, loc=loc, scale=scale)
        cdf_upper = stats.gamma.cdf(upper_bound, a=shape, loc=loc, scale=scale)
        cdf_range = cdf_upper - cdf_lower
        
        # Loss is proportional to deviation relative to operational range
        loss = EML * (deviation_cdf / cdf_range) if cdf_range > 0 else 0
    else:
        # Outside bounds: higher penalty
        # Calculate additional penalty for being outside bounds
        if x < lower_bound:
            cdf_bound = stats.gamma.cdf(lower_bound, a=shape, loc=loc, scale=scale)
            extra_penalty = abs(cdf_value - cdf_bound)
        else:  # x > upper_bound
            cdf_bound = stats.gamma.cdf(upper_bound, a=shape, loc=loc, scale=scale)
            extra_penalty = abs(cdf_value - cdf_bound)
        
        # Total loss includes both deviation from setpoint and penalty for crossing bounds
        loss = EML * (deviation_cdf + extra_penalty)
    # Ensure non-negative and bounded by EML
    return max(0, min(loss, EML))

# Inverted Log-Gamma Loss Function (ILGLF) for Log-Gamma Distribution
def calculate_ilglf(x, mu, lower_bound, upper_bound, c, loc, scale, EML):
    # Calculate CDF values
    cdf_value = stats.loggamma.cdf(x, c, loc=loc, scale=scale)
    cdf_mu = stats.loggamma.cdf(mu, c, loc=loc, scale=scale)
    
    # Deviation from setpoint in CDF space
    deviation_cdf = abs(cdf_value - cdf_mu)
    
    # If within bounds, apply reduced penalty
    if lower_bound <= x <= upper_bound:
        # Within bounds: loss scales with distance from setpoint
        # Normalize by the CDF range within bounds
        cdf_lower = stats.loggamma.cdf(lower_bound, c, loc=loc, scale=scale)
        cdf_upper = stats.loggamma.cdf(upper_bound, c, loc=loc, scale=scale)
        cdf_range = cdf_upper - cdf_lower
        
        # Loss is proportional to deviation relative to operational range
        loss = EML * (deviation_cdf / cdf_range) if cdf_range > 0 else 0
    else:
        # Outside bounds: higher penalty
        # Calculate additional penalty for being outside bounds
        if x < lower_bound:
            cdf_bound = stats.loggamma.cdf(lower_bound, c, loc=loc, scale=scale)
            extra_penalty = abs(cdf_value - cdf_bound)
        else:  # x > upper_bound
            cdf_bound = stats.loggamma.cdf(upper_bound, c, loc=loc, scale=scale)
            extra_penalty = abs(cdf_value - cdf_bound)
        
        # Total loss includes both deviation from setpoint and penalty for crossing bounds
        loss = EML * (deviation_cdf + extra_penalty)
    
    # Ensure non-negative and bounded by EML
    return max(0, min(loss, EML))

# Loss function parameters for CA and T from fitted distributions
# CA: loggamma distribution - params = [c (shape), loc, scale]
CA_c = CA_params[0]       # Shape parameter c from loggamma
CA_loc = CA_params[1]     # Location parameter from loggamma
CA_scale = CA_params[2]   # Scale parameter from loggamma

# T: gamma distribution - params = [shape, loc, scale]
T_shape = T_params[0]     # Shape parameter from gamma
T_loc = T_params[1]       # Location parameter from gamma
T_scale = T_params[2]     # Scale parameter from gamma

# Copula-based joint probability calculation using Mahalanobis distance
def calculate_joint_probability_copula(ca_value, t_value, ca_dist, t_dist, ca_sp, t_sp, copula_params):
    # --- Step 1: Transform to Uniform [0,1] space ---
    u_ca = ca_dist.cdf(ca_value)
    u_t = t_dist.cdf(t_value)
    u_ca_sp = ca_dist.cdf(ca_sp)
    u_t_sp = t_dist.cdf(t_sp)

    # --- Step 2: Transform to Standard Normal (Z-space) ---
    # This step is the same for Gaussian and Independence copulas
    z_ca = norm.ppf(u_ca)
    z_t = norm.ppf(u_t)
    z_ca_sp = norm.ppf(u_ca_sp)
    z_t_sp = norm.ppf(u_t_sp)
    
    # Handle potential infinite values if u=0 or u=1
    z_vec = np.nan_to_num(np.array([z_ca, z_t]), neginf=-10.0, posinf=10.0)
    z_sp_vec = np.nan_to_num(np.array([z_ca_sp, z_t_sp]), neginf=-10.0, posinf=10.0)

    # Deviation vector in Z-space
    z_diff = z_vec - z_sp_vec
    
    # --- Step 3: Calculate Mahalanobis Distance Squared (D^2) ---
    copula_type = copula_params.get('copula_type', 'gaussian')
    
    if copula_type == 'gaussian':
        # --- Case 1: Correlated (Gaussian Copula) ---
        rho = copula_params['copula_params'][0]
        rho_sq = rho**2
        inv_cov_matrix = np.array([[1, -rho], [-rho, 1]]) / (1 - rho_sq)
        mahal_dist_sq = z_diff @ inv_cov_matrix @ z_diff
        
    else:
        # Mahalanobis distance simplifies to squared Euclidean distance in Z-space
        mahal_dist_sq = z_diff[0]**2 + z_diff[1]**2
    prob = 1.0 - chi2.sf(mahal_dist_sq, df=2)
    return prob

def calculate_fault_probability(ca_value, t_value, ca_dist, t_dist, ca_sp, t_sp, ca_bounds, t_bounds):
   
    return calculate_joint_probability_copula(ca_value, t_value, ca_dist, t_dist,
                                              ca_sp, t_sp, best_copula)

# Severity calculation using ILGLF for CA (loggamma) and IGLF for T (gamma)
def calculate_severity(CA, T):
    # CA uses Inverted Log-Gamma Loss Function (loggamma distribution)
    # lower_CA and upper_CA are 90th percentile bounds from fitted distribution
    ilglf_CA = calculate_ilglf(CA, T_CA, lower_CA, upper_CA, CA_c, CA_loc, CA_scale, EML_quality)
    
    # T uses Inverted Gamma Loss Function (gamma distribution)
    # lower_T and upper_T are 90th percentile bounds from fitted distribution
    iglf_T = calculate_iglf(T, T_T, lower_T, upper_T, T_shape, T_loc, T_scale, (EML_equipment + EML_production))
    # Total severity is sum of individual loss functions
    severity = ilglf_CA + iglf_T
    return severity


Qc = np.diag([1e5, 5e4])  # Stronger state penalty
Rc = 1e-6  # Stronger input penalty
ts = 0.01
N = 10  # Prediction horizon (number of steps to predict ahead)
M = 10   # Control horizon (number of control moves to optimize)
umax = 1e5

# --- Load normalization and model parameters ---
global_norm_params = np.load('global_norm_params.npy', allow_pickle=True).item()
coeff_norm_params = np.load('coeff_norm_params.npy', allow_pickle=True).item()
CA_min, CA_max = global_norm_params['CA_min'], global_norm_params['CA_max']
T_min, T_max = global_norm_params['T_min'], global_norm_params['T_max']
Q_min, Q_max = global_norm_params['Q_min'], global_norm_params['Q_max']
coeff_CA_min, coeff_CA_max = coeff_norm_params['coeff_CA_min'], coeff_norm_params['coeff_CA_max']
coeff_T_min, coeff_T_max = coeff_norm_params['coeff_T_min'], coeff_norm_params['coeff_T_max']

# --- Standard GRU multi-horizon regressor (matches GRU_inference.py) ---
class GRUMultiHorizon(nn.Module):
    def __init__(self, seq_len, in_dim, out_dim, k, d_model=128, n_layers=4, dropout=0.1):
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

def simulate_cstr_gru_sindy(x, u, coeffs_CA, coeffs_T, ts):
    CA, T = x
    theta = np.array([
        1, CA, T, u, CA**2, T**2, CA**3, T**3, np.exp(-E/(R*T))*CA**2
    ])
    dCA_dt = np.dot(theta, coeffs_CA)
    dT_dt = np.dot(theta, coeffs_T)
    CA_next = CA + dCA_dt * ts
    T_next = T + dT_dt * ts
    return np.array([CA_next, T_next])

def predict_sindy_coeffs_gru(model, CA_hist, T_hist, Q_hist, device, N, k, coeff_CA_min, coeff_CA_max, coeff_T_min, coeff_T_max):
    # Normalize histories
    CA_norm = (CA_hist - CA_min) / (CA_max - CA_min + 1e-12)
    T_norm = (T_hist - T_min) / (T_max - T_min + 1e-12)
    Q_norm = (Q_hist - Q_min) / (Q_max - Q_min + 1e-12)
    x_seq = np.stack([Q_norm, CA_norm, T_norm], axis=1)[None, ...]  # (1, N, 3)
    x_seq_tensor = torch.tensor(x_seq, dtype=torch.float32).to(device)
    model.eval()
    with torch.no_grad():
        y_pred_norm = model(x_seq_tensor).cpu().numpy()[0]  # (k, coeff_dim)
    CA_dim = len(coeff_CA_min)
    T_dim = len(coeff_T_min)
    coeffs_CA = np.array([denormalize_coeffs(y_pred_norm[j][:CA_dim], coeff_CA_min, coeff_CA_max).flatten() for j in range(k)])
    coeffs_T = np.array([denormalize_coeffs(y_pred_norm[j][CA_dim:], coeff_T_min, coeff_T_max).flatten() for j in range(k)])
    return coeffs_CA, coeffs_T  # Return all k sets

def cstr_true(t, y, Q):
    CA, T = y
    k = K0 * np.exp(-E / (R * T))
    r = k * CA**2
    dCA_dt = (F / V_r) * (C_in - CA) - r
    dT_dt = (F / V_r) * (T_in - T) + (Q / (rho * Cp * V_r)) + (-delta_H / (rho * Cp)) * r
    return [dCA_dt, dT_dt]
                         
    # --- Main simulation loop ---
if __name__ == "__main__":
    xs = np.array([3.06, 345.29])
    x0 = np.array([2.5, 380.0])
    time_steps = int(1 / ts)
    X_bgru = np.zeros((time_steps + 1, 2))  # True plant state
    X_est = np.zeros((time_steps + 1, 2))   # EKF estimated state (used by controller)
    U_bgru = np.zeros(time_steps)
    X_bgru[0] = x0
    X_est[0] = x0  # Initial estimate equals true state
    Q_guess = 80000.0
    P = np.diag([0.01, 0.01])

    # --- Safety System Variables ---
    risk_instant_values = np.zeros(time_steps)  # Process risk (h=0, R_total in risk.py)
    risk_max_values = np.zeros(time_steps)  # Proactive risk (max over horizon, R_max in risk.py)
    severity_values = np.zeros(time_steps)
    probability_values = np.zeros(time_steps)
    esh_active = np.zeros(time_steps, dtype=bool)
    alarm_active = np.zeros(time_steps, dtype=bool)
    process_shutdown = False

    # --- Measurement update interval variable ---
    MEAS_UPDATE_INTERVAL = 10  # Update model with plant measurement every N steps (set to 1, 5, 10, etc.)

    # History arrays for BGRU input
    CA_hist = np.full(N, x0[0])
    T_hist = np.full(N, x0[1])
    Q_hist = np.full(N, Q_guess)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_dim = len(coeff_CA_min) + len(coeff_T_min)
    model = GRUMultiHorizon(N, 3, out_dim, N, d_model=256, n_layers=3, dropout=0.1).to(device)
    model.load_state_dict(torch.load('save model/gru_good_sindy1.pth', map_location=device))

    # --- Model-predicted state for controller (initially true state) ---
    x_ctrl = x0.copy()

    # Start timing the simulation
    simulation_start_time = time.time()
    print(f"Starting simulation at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(simulation_start_time))}")
    print("=" * 60)

    for i in range(time_steps):
        # Get current controller state (EKF estimate)
        x_ctrl = X_est[i].copy()
        
        # 1. Define MPC cost function (true multi-horizon MPC with prediction horizon N and control horizon M)
        def mpc_cost(U):
            ca_hist = CA_hist.copy()
            t_hist = T_hist.copy()
            q_hist = Q_hist.copy()
            x_sim = x_ctrl.copy()
            
            # Create control sequence: optimize M moves, hold last move for remaining N-M steps
            U_horizon = np.zeros(N)
            U_horizon[:M] = U[:M]  # First M control moves from optimization
            U_horizon[M:] = U[M-1]  # Hold last control move for remaining steps
            
            # Predict N-step-ahead SINDy coefficients using current window
            coeffs_CA_N, coeffs_T_N = predict_sindy_coeffs_gru(
                model, ca_hist, t_hist, q_hist, device, N, N,
                coeff_CA_min, coeff_CA_max, coeff_T_min, coeff_T_max
            )
            cost = 0
            for step in range(N):
                u = U_horizon[step]
                coeff_CA = coeffs_CA_N[step]
                coeff_T = coeffs_T_N[step]
                x_sim = simulate_cstr_gru_sindy(x_sim, u, coeff_CA, coeff_T, ts)
                
                # State penalty for all prediction steps
                cost += (x_sim - xs).T @ Qc @ (x_sim - xs)
                
                # Input penalty only for control moves (first M steps)
                if step < M:
                    cost += Rc * u**2
            return cost

        # Optimization variables: only M control moves (multi-horizon MPC)
        U0 = np.full(M, 0.0)
        bounds = [(-umax, umax)] * M

        # --- DEBUG: Print cost for all-zero and all-umax input sequences ---
        U_zero = np.zeros(M)
        U_max = np.full(M, umax)
        cost_zero = mpc_cost(U_zero)
        cost_max = mpc_cost(U_max)
        print(f"Step {i+1}: Cost(U=0)={cost_zero:.2e}, Cost(U=umax)={cost_max:.2e}")

        res = minimize(mpc_cost, U0, method='SLSQP', bounds=bounds, options={'maxiter': 500})
        print(f"Step {i+1}: Optimizer success={res.success}, message={res.message}, min cost={res.fun:.2e}")
        
        # --- Safety System Control Logic ---
        # Check if process should be shut down due to ESH activation
        if process_shutdown:
            # ESH is active - process is shut down, no heat input
            u = 0.0  # Emergency shutdown - turn off heating
            print(f"Step {i+1}: *** PROCESS SHUTDOWN ACTIVE - ESH TRIGGERED *** (u=0)")
        else:
            # Normal MPC operation - check optimization results
            if res.success:
                # Extract optimal control sequence
                U_optimal = res.x
                u = U_optimal[0]  # Apply first control move (receding horizon principle)
                # Debug: show control sequence
                U_full_horizon = np.zeros(N)
                U_full_horizon[:M] = U_optimal
                U_full_horizon[M:] = U_optimal[-1]  # Hold last control move
                print(f"Step {i+1}: Control sequence U_opt = {U_optimal} (applied u={u:.2f})")
            else:
                u = Q_guess
                print(f"Step {i+1}: Optimization failed, using previous control u={u:.2f}")
            
        U_bgru[i] = u
        Q_guess = u

        # 2. Simulate true plant forward one step (true CSTR dynamics)
        sol = solve_ivp(cstr_true, [0, ts], X_bgru[i], args=(u,), method='RK45', max_step=ts/10)
        plant_next = sol.y[:, -1]
        X_bgru[i+1] = plant_next  # Store true plant state
        
        # 3. Predict estimated state forward (before measurement update)
        # Start from current estimate, simulate one step with true model
        sol_est = solve_ivp(cstr_true, [0, ts], X_est[i], args=(u,), method='RK45', max_step=ts/10)
        X_est[i+1] = sol_est.y[:, -1]  # Predicted estimate

        # --- DEBUG: Compare model (GRU-SINDy) and plant one-step-ahead prediction ---
        ca_hist_dbg = CA_hist.copy()
        t_hist_dbg = T_hist.copy()
        q_hist_dbg = Q_hist.copy()
        # Predict 1-step-ahead SINDy coefficients (for debug)
        coeffs_CA_1, coeffs_T_1 = predict_sindy_coeffs_gru(
            model, ca_hist_dbg, t_hist_dbg, q_hist_dbg, device, N, 1,
            coeff_CA_min, coeff_CA_max, coeff_T_min, coeff_T_max
        )
        model_next = simulate_cstr_gru_sindy(x_ctrl, u, coeffs_CA_1[0], coeffs_T_1[0], ts)
        print(f"Step {i+1}: Plant next=[{plant_next[0]:.4f}, {plant_next[1]:.2f}], Estimate next=[{X_est[i+1,0]:.4f}, {X_est[i+1,1]:.2f}], Model next=[{model_next[0]:.4f}, {model_next[1]:.2f}], u={u:.2f}")
        plant_error = X_bgru[i+1] - xs
        est_error = X_est[i+1] - xs
        print(f"Step {i+1}: Plant error = [{plant_error[0]:.4f}, {plant_error[1]:.2f}], Estimate error = [{est_error[0]:.4f}, {est_error[1]:.2f}]")

        # --- Risk Assessment (matching risk_bound_copula.py methodology) ---
        # Calculate risk using current state (always calculate, not just after N-1 steps)
        if i >= N-1:  # Multi-horizon risk calculation when enough history available
            # Build normalized window for risk calculation INCLUDING current plant state
            # Window should be [i-N+2, i-N+3, ..., i, i+1] (N points ending at current state)
            CA_hist_risk = np.roll(CA_hist, -1); CA_hist_risk[-1] = X_bgru[i+1, 0]
            T_hist_risk = np.roll(T_hist, -1); T_hist_risk[-1] = X_bgru[i+1, 1]
            Q_hist_risk = np.roll(Q_hist, -1); Q_hist_risk[-1] = u
            
            Q_seq_norm = (Q_hist_risk - Q_min) / (Q_max - Q_min + 1e-12)
            CA_seq_norm = (CA_hist_risk - CA_min) / (CA_max - CA_min + 1e-12)
            T_seq_norm = (T_hist_risk - T_min) / (T_max - T_min + 1e-12)
            seq_window = np.stack([Q_seq_norm, CA_seq_norm, T_seq_norm], axis=1)
            
            # Predict multi-horizon coefficients for risk assessment
            seq_tensor = torch.tensor(seq_window, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                pred = model(seq_tensor).cpu().numpy()[0]
            
            # Prepare rollout from current plant state (matching risk_bound_copula.py)
            ca_prev = X_bgru[i+1, 0]  # Use current plant state
            T_prev = X_bgru[i+1, 1]
            horizon_risks = []
            
            for h in range(N):
                # Get horizon-h coefficients
                coeffs_h = pred[h]
                CA_coeff = denormalize_coeffs(coeffs_h[:len(coeff_CA_min)][None, :], coeff_CA_min, coeff_CA_max).flatten()
                T_coeff = denormalize_coeffs(coeffs_h[len(coeff_CA_min):][None, :], coeff_T_min, coeff_T_max).flatten()
                
                # Use control input for horizon h (matching risk_bound_copula.py approach)
                # For MPC: assume control stays constant at current value u for all horizons
                Q_val = u
                
                # SINDy Theta (matching risk_bound_copula.py exactly)
                Theta_row = np.array([
                    1.0, ca_prev, T_prev, Q_val, ca_prev**2, T_prev**2,
                    ca_prev**3, T_prev**3, np.exp(-E/(R*T_prev)) * ca_prev**2
                ], dtype=float)
                
                dCA = float(np.dot(Theta_row, CA_coeff))
                dT = float(np.dot(Theta_row, T_coeff))
                ca_next = ca_prev + dCA * ts
                T_next = T_prev + dT * ts
                
                # Calculate dynamic probability and severity using statistical distance approach
                # (exactly matching risk_bound_copula.py)
                fault_prob = calculate_fault_probability(ca_next, T_next, CA_dist, T_dist, 
                                             T_CA, T_T, [lower_CA, upper_CA], [lower_T, upper_T])
                severity = calculate_severity(ca_next, T_next)
                risk_h = fault_prob * severity
                
                # Store instantaneous values (h=0) - matching risk.py
                if h == 0:
                    probability_values[i] = fault_prob
                    severity_values[i] = severity
                    risk_instant_values[i] = risk_h  # This is R_total in risk.py (h=0 prediction)
                
                horizon_risks.append(risk_h)
                ca_prev, T_prev = ca_next, T_next
            
            # Store risk values - matching risk.py exactly
            # risk_instant_values[i] already set to horizon_risks[0] above
            risk_max_values[i] = np.max(horizon_risks)  # R_max in risk.py
            
            # Safety system status based on max risk
            alarm_active[i] = risk_max_values[i] > alarm_threshold
            esh_active[i] = risk_max_values[i] > esh_threshold
            
            print(f"Step {i+1}: Process Risk={risk_instant_values[i]:.2e}, Proactive Risk={risk_max_values[i]:.2e}, Severity={severity_values[i]:.2e}")
            if alarm_active[i]:
                print(f"Step {i+1}: *** ALARM TRIGGERED *** (Risk={risk_max_values[i]:.2e} > {alarm_threshold:.2e})")
            if esh_active[i]:
                print(f"Step {i+1}: *** ESH ACTIVATED *** (Risk={risk_max_values[i]:.2e} > {esh_threshold:.2e})")
                if not process_shutdown:
                    process_shutdown = True
                    print(f"Step {i+1}: *** EMERGENCY SHUTDOWN INITIATED ***")
        else:
            # For early time steps, calculate risk based on current state only (no multi-horizon)
            current_CA = X_bgru[i+1, 0]
            current_T = X_bgru[i+1, 1]
            
            # Calculate current state risk
            severity_values[i] = calculate_severity(current_CA, current_T)
            probability_values[i] = calculate_fault_probability(current_CA, current_T, CA_dist, T_dist, 
                                                              T_CA, T_T, [lower_CA, upper_CA], [lower_T, upper_T])
            risk_instant_values[i] = probability_values[i] * severity_values[i]  # Process risk
            risk_max_values[i] = risk_instant_values[i]  # Same as process risk for early steps (no multi-horizon)
            
            # Safety system status
            alarm_active[i] = risk_max_values[i] > alarm_threshold
            esh_active[i] = risk_max_values[i] > esh_threshold
            
            if esh_active[i]:
                print(f"Step {i+1}: *** ESH ACTIVATED (Early) *** (Risk={risk_max_values[i]:.2e} > {esh_threshold:.2e})")
                if not process_shutdown:
                    process_shutdown = True
                    print(f"Step {i+1}: *** EMERGENCY SHUTDOWN INITIATED ***")

        # 4. EKF Update (only updates the state estimate, not true plant)
        # Update EKF only at measurement update intervals
        if (i+1) % MEAS_UPDATE_INTERVAL == 0:
            # EKF corrects the estimated state using actual T measurement
            measured_T = plant_next[1]  # Temperature from plant (actual value)
            X_est[i+1], P = ekf_update(X_est[i+1], P, u, measured_T)
        
        # 5. Update controller history arrays
        # When measurement is available: feed EKF-estimated CA and measured T to model
        # Otherwise: use previous GRU-SINDy model predictions (not EKF predictions)
        CA_hist = np.roll(CA_hist, -1)
        T_hist = np.roll(T_hist, -1)
        if (i+1) % MEAS_UPDATE_INTERVAL == 0:
            # Measurement available: use EKF-corrected CA and actual measured T
            CA_hist[-1] = X_est[i+1, 0]
            T_hist[-1] = plant_next[1]
        else:
            # No measurement: use previous GRU-SINDy model prediction
            CA_hist[-1] = model_next[0]
            T_hist[-1] = model_next[1]
        Q_hist = np.roll(Q_hist, -1); Q_hist[-1] = u

        # Calculate current simulation time and elapsed real time
        current_sim_time = (i+1) * ts
        elapsed_real_time = time.time() - simulation_start_time
        print(f"Step {i+1}/{time_steps}: t={current_sim_time:.3f}s, u={u:.2f}, CA_plant={X_bgru[i+1,0]:.3f}, T_plant={X_bgru[i+1,1]:.2f}, CA_est={X_est[i+1,0]:.3f}, T_est={X_est[i+1,1]:.2f} | Elapsed: {elapsed_real_time:.2f}s")

    # Fill the last risk value (since loop stops at time_steps-1)
    risk_instant_values[-1] = risk_instant_values[-2]
    risk_max_values[-1] = risk_max_values[-2]
    severity_values[-1] = severity_values[-2]
    probability_values[-1] = probability_values[-2]

    # End timing
    simulation_end_time = time.time()
    total_elapsed_time = simulation_end_time - simulation_start_time
    print("=" * 60)
    print(f"Simulation completed at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(simulation_end_time))}")
    print(f"Total simulation time: {total_elapsed_time:.2f} seconds")
 
    print(f"Real-time factor: {(time_steps * ts * 3600) / total_elapsed_time:.2f}x")
    print("=" * 60)

    # --- Plot results with Safety System ---

    plt.figure(figsize=(12, 10))
  
    # Indices where model is updated with plant measurement
    update_indices = np.arange(MEAS_UPDATE_INTERVAL-1, len(X_bgru), MEAS_UPDATE_INTERVAL)
    t_all = np.arange(0, 1 + ts, ts)
    t_input = np.arange(0, 1, ts)
    
    # ESH and alarm indices for plotting
    esh_indices = np.where(esh_active)[0]
    alarm_indices = np.where(alarm_active)[0]

    # Top row - Process variables
    plt.subplot(2, 2, 1)
    plt.step(t_input, U_bgru, "r", label='RAISE Q', linewidth=2, where="post")
    plt.ylabel("Heat Input (KJ/h)", fontsize=14)
    plt.xlabel("Time (h)", fontsize=14)
    plt.xlim(0, 1)
    plt.legend(fontsize=10)
    plt.text(0.5, -0.25, '(a)', transform=plt.gca().transAxes, fontsize=16, fontweight='bold', ha='center')
   

    plt.subplot(2, 2, 2)
    plt.plot(t_all, X_bgru[:, 0], "b", label='RAISE $C_{A}$', linewidth=2)
    plt.scatter(t_all[update_indices], X_bgru[update_indices, 0], c='k',  s=30, marker='o', label='Measurement update')
    plt.axhline(xs[0], color='r', linestyle="--", linewidth=1.5, label=f'Setpoint')
    plt.axhline(lower_CA, color='grey', linestyle='--', linewidth=1.5, label=f'Bounds')
    plt.axhline(upper_CA, color='grey', linestyle='--', linewidth=1.5)
    if len(esh_indices) > 0:
        plt.scatter(t_all[esh_indices+1], X_bgru[esh_indices+1, 0], c='red',  s=50, marker='x', label='ESH Active', linewidths=3)
    plt.ylabel("Concentration (Kmol/m³)", fontsize=14)
    plt.xlabel("Time (h)", fontsize=14)
    plt.xlim(0, 1)
    plt.legend(fontsize=10)
    plt.text(0.5, -0.25, '(b)', transform=plt.gca().transAxes, fontsize=16, fontweight='bold', ha='center')


    plt.subplot(2, 2, 3)
    plt.plot(t_all, X_bgru[:, 1], "g", label='RAISE T', linewidth=2)
    plt.scatter(t_all[update_indices], X_bgru[update_indices, 1], c='k', s=30, marker='o', label='Measurement update')
    plt.axhline(xs[1], color='r', linestyle="--", linewidth=1.5, label=f'Setpoint')
    plt.axhline(lower_T, color='grey', linestyle='--', linewidth=1.5, label=f'Bounds')
    plt.axhline(upper_T, color='grey', linestyle='--', linewidth=1.5)
    if len(esh_indices) > 0:
        plt.scatter(t_all[esh_indices+1], X_bgru[esh_indices+1, 1], c='red', s=50, marker='x', label='ESH Active', linewidths=3)
    plt.ylabel("Temperature (K)", fontsize=14)
    plt.xlabel("Time (h)", fontsize=14)
    plt.xlim(0, 1)
    plt.legend(fontsize=10)
    plt.text(0.5, -0.25, '(c)', transform=plt.gca().transAxes, fontsize=16, fontweight='bold', ha='center')


    plt.subplot(2, 2, 4)
    plt.plot(t_input, risk_instant_values, "blue", label='Process Risk', linewidth=2)
    plt.plot(t_input, risk_max_values, "red", label='Proactive Risk', linewidth=2)
    plt.axhline(alarm_threshold, color='orange', linestyle='--', linewidth=1.5, label=f'Alarm')
    plt.axhline(esh_threshold, color='red', linestyle='--', linewidth=1.5, label=f'ESH')
    if len(esh_indices) > 0:
        plt.scatter(t_input[esh_indices], risk_max_values[esh_indices], c='red',  s=50, marker='x', label='ESH Triggered', linewidths=3)
    plt.ylabel("Risk ($/h)", fontsize=14)
    plt.xlabel("Time (h)", fontsize=14)
    plt.xlim(0, 1)
    plt.legend(fontsize=10)
    plt.text(0.5, -0.25, '(d)', transform=plt.gca().transAxes, fontsize=16, fontweight='bold', ha='center')

    plt.tight_layout()
    
    # Print safety system summary
    print("\n" + "="*60)
    print("SAFETY SYSTEM SUMMARY")
    print("="*60)
    print(f"Total simulation steps: {time_steps}")
    print(f"Alarm activations: {np.sum(alarm_active)} steps ({np.sum(alarm_active)/time_steps*100:.1f}%)")
    print(f"ESH activations: {np.sum(esh_active)} steps ({np.sum(esh_active)/time_steps*100:.1f}%)")
    print(f"Process shutdown occurred: {'Yes' if process_shutdown else 'No'}")
    if process_shutdown:
        first_esh_step = np.where(esh_active)[0][0] if len(np.where(esh_active)[0]) > 0 else -1
        if first_esh_step >= 0:
            print(f"First ESH activation at step {first_esh_step+1} (t={first_esh_step*ts:.3f}s)")
    print(f"Maximum risk encountered: {np.max(risk_max_values):.2e} $/hr")
    print(f"Maximum severity encountered: {np.max(severity_values):.2e} $/hr")
    print("="*60)
    plt.show()
