
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.linalg import lstsq

# Define CSTR parameters
F = 5.0  # Flow rate (m^3/h)
V_r = 1.0  # Reactor volume (m^3)
R = 8.314  # Gas constant (kJ/Kmol-K)
K0 = 8.46e6  # Arrhenius pre-exponential factor (1/h)
E = 5e4  # Activation energy (kJ/Kmol)
T_in = 300  # Inlet temperature (K)
C_in = 3.5  # Inlet concentration (Kmol/m^3)
delta_H = -1.16e4  # Enthalpy change (kJ/Kmol)
rho = 1000  # Fluid density (kg/m^3)
Cp = 0.231  # Specific heat (kJ/kg-K)

# Time span for the simulation
time_eval = np.linspace(0, 1, 100)  # Time from 0 to 1 hour
initial_conditions = [2, 400]   # Initial concentration and temperature

# Define the reaction rate function
def reaction_rate(CA, T):
    K = K0 * np.exp(-E / (R * T))
    return K * CA ** 2

# Define the CSTR model with time-varying Q input
def cstr_model(t, y, Q_profile, time_eval):
    CA, T = y
    Q = np.interp(t, time_eval, Q_profile)  # Interpolate Q at the current time
    dCA_dt = (F / V_r) * (C_in - CA) - reaction_rate(CA, T)
    dT_dt = (F / V_r) * (T_in - T) - (delta_H / (rho * Cp)) * reaction_rate(CA, T) + Q / (rho * Cp * V_r)
    return [dCA_dt, dT_dt]

# Function to simulate and return profiles for a given Q input profile
def simulate_cstr(Q_profile):
    sol = solve_ivp(cstr_model, [time_eval[0], time_eval[-1]], initial_conditions,
                    args=(Q_profile, time_eval), t_eval=time_eval, method='RK45')
    return sol.y[0], sol.y[1]  # CA_data, T_data

# Generate synthetic data
num_trajectories = 100
CA_profiles = []
T_profiles = []
Q_profiles = []

for _ in range(num_trajectories):
    Q_profile = np.random.uniform(-2e5, 2e5, len(time_eval))  # Increased variability
    Q_profiles.append(Q_profile)

    CA_data, T_data = simulate_cstr(Q_profile)
    CA_profiles.append(CA_data)
    T_profiles.append(T_data)

CA_profiles = np.array(CA_profiles)
T_profiles = np.array(T_profiles)
Q_profiles = np.array(Q_profiles)

# Plotting the Results for first few trajectories
plt.figure(figsize=(20, 15))

# Plot a subset of random inputs
plt.subplot(3, 1, 1)
for i in range(min(5, num_trajectories)):  # Plot first 5 profiles
    plt.plot(time_eval, Q_profiles[i, :], label=f'Trajectory {i + 1}')
plt.xlabel('Time (h)')
plt.ylabel('Heat Input (kJ/h)')
plt.title('Sample Random Input - Q')
plt.legend()

# Plot Concentration Data for sample trajectories
plt.subplot(3, 1, 2)
for i in range(min(5, num_trajectories)):
    plt.plot(time_eval, CA_profiles[i, :], label=f'Trajectory {i + 1}')
plt.xlabel('Time (h)')
plt.ylabel('Concentration (Kmol/m^3)')
plt.title('Concentration Trajectories - CA')
plt.legend()

# Plot Temperature Data for sample trajectories
plt.subplot(3, 1, 3)
for i in range(min(5, num_trajectories)):
    plt.plot(time_eval, T_profiles[i, :], label=f'Trajectory {i + 1}')
plt.xlabel('Time (h)')
plt.ylabel('Temperature (K)')
plt.title('Temperature Trajectories - T')
plt.legend()

plt.tight_layout()
plt.show()
