import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from scipy.optimize import fsolve
from scipy.interpolate import CubicSpline

# Load bond data from Excel file
file_path = "Selected Bonds.xlsx"  # Update to your file name
df = pd.read_excel(file_path, sheet_name="Sheet1", engine="openpyxl")
frequency = 2

# Fixed last coupon date for all bonds
last_coupon = pd.to_datetime("9/1/2024", format='%m/%d/%Y')
next_coupon = pd.to_datetime("3/1/2025", format='%m/%d/%Y')
days_in_coupon_period = (next_coupon - last_coupon).days

def calculate_dirty_price(clean_price, coupon_rate, settlement_date, frequency=2):
    """Calculate dirty price of a bond."""
    days_since_last_coupon = (settlement_date - last_coupon).days
    face_value = 100
    coupon_payment = coupon_rate * face_value / frequency
    accrued_interest = coupon_payment * (days_since_last_coupon / days_in_coupon_period)
    return clean_price + accrued_interest

def bond_ytm(price, face_value, coupon_rate, years_to_maturity, frequency=2):
    """Solve for Yield to Maturity (YTM) using numerical root finding."""
    coupon_payment = coupon_rate * face_value / frequency

    def equation(r):
        return sum([coupon_payment / (1 + r / frequency) ** (years_to_maturity * frequency - t)
                    for t in range(math.ceil(years_to_maturity * frequency))]) + \
            face_value / (1 + r / frequency) ** (frequency * years_to_maturity) - price

    ytm_guess = 0.03  # Initial guess for YTM (3%)
    ytm_solution = fsolve(equation, ytm_guess)
    return ytm_solution[0] * 100

def compute_forward_rates(spot_rates, maturities):
    """Compute forward rates from spot rates."""
    forward_rates = []
    for i in range(1, len(maturities)):
        t1 = maturities[i - 1]
        t2 = maturities[i]
        s1 = spot_rates[i - 1] / 100  # Convert to decimal
        s2 = spot_rates[i] / 100  # Convert to decimal
        fwd_rate = ((1 + s2) ** t2 / (1 + s1) ** t1) ** (1 / (t2 - t1)) - 1
        forward_rates.append(fwd_rate * 100)  # Convert to percentage
    return forward_rates

def compute_log_returns(data):
    """Compute log-returns for a given time series."""
    return np.log(data.iloc[1:] / data.iloc[:-1].values)

def compute_covariance_matrix(log_returns):
    """Compute the covariance matrix of log-returns."""
    return log_returns.cov()

# Convert maturity dates
df["Maturity date"] = pd.to_datetime(df["Maturity date"], format='%m/%d/%Y')

ytm_table = pd.DataFrame()
forward_rate_table = pd.DataFrame()
date_columns = df.columns[4:]

forward_rate_matrices = {}

for date in date_columns:
    settlement_date = pd.to_datetime(date, format='%m/%d/%Y')
    ytm_values = []
    maturities = []

    for _, row in df.iterrows():
        clean_price = row[date]
        coupon_rate = float(row["Coupon Rate (%)"])
        maturity_date = row["Maturity date"]
        year_difference = (maturity_date - next_coupon).days / 365

        years_to_next_coupon = ((next_coupon - settlement_date).days / days_in_coupon_period) / 2
        years_to_maturity = years_to_next_coupon + round(year_difference * 2) / 2

        dirty_price = calculate_dirty_price(clean_price, coupon_rate, settlement_date)
        ytm = bond_ytm(dirty_price, 100, coupon_rate, years_to_maturity)

        ytm_values.append(float(f"{ytm:.2f}"))  # Convert to float
        maturities.append(years_to_maturity)

    # Compute spot rates (used for forward rates)
    spot_rates = [ytm_values[0]]  # Assume first YTM is the spot rate for the shortest maturity
    for i in range(1, len(maturities)):
        maturity = maturities[i]
        ytm = ytm_values[i] / 100  # Convert to decimal
        coupon = ytm * 100 / frequency
        price = 100  # Assume par bond
        sum_discounted = sum([coupon / (1 + spot_rates[j] / 100 / frequency) ** (maturities[j] * frequency)
                              for j in range(i)])
        spot_rate = ((price - sum_discounted) / (100 + coupon)) ** (-1 / (maturity * frequency)) - 1
        spot_rates.append(spot_rate * 100)  # Convert to percentage

    # Compute forward rates
    forward_rates = compute_forward_rates(spot_rates, maturities)
    forward_rate_table[date] = forward_rates
    forward_rate_matrices[date] = forward_rates

# Transpose forward rate data for log-returns computation
forward_rate_df = pd.DataFrame(forward_rate_matrices).T

# Compute daily log-returns for forward rates
log_returns_forward = compute_log_returns(forward_rate_df)

# Compute covariance matrix for forward rate log-returns
cov_matrix_forward = compute_covariance_matrix(log_returns_forward)

# Update the covariance matrix to start labels from 1 instead of 0
cov_matrix_forward.index = [f"1yr-{i+1}yr" for i in range(len(cov_matrix_forward.index))]
cov_matrix_forward.columns = [f"1yr-{i+1}yr" for i in range(len(cov_matrix_forward.columns))]

# Print the covariance matrix as a plain table
print("Covariance Matrix for Forward Rate Log-Returns:")
print(cov_matrix_forward.to_string(index=True, header=True))

# Plot Forward Rate Curves
plt.figure(figsize=(12, 6))
for date in forward_rate_matrices.keys():
    forward_rates = forward_rate_matrices[date]
    labels = [f"1yr-{i+1}yr" for i in range(len(forward_rates))]
    plt.plot(range(1, len(forward_rates) + 1), forward_rates, label=f"{str(date)[:10]}")

plt.xlabel("Forward Rate Periods (1yr-xyr)")
plt.ylabel("Forward Rate (%)")
plt.title("Forward Rate Curves Over Time")
plt.legend(title="Settlement Date", loc='upper left', fontsize=8)
plt.grid(True)
plt.show()
