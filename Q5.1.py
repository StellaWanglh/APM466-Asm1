import numpy as np
import pandas as pd
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

def compute_log_returns(data):
    """Compute log-returns for a given time series."""
    return np.log(data.iloc[1:] / data.iloc[:-1].values)

def compute_covariance_matrix(log_returns):
    """Compute the covariance matrix of log-returns."""
    return log_returns.cov()

# Convert maturity dates
df["Maturity date"] = pd.to_datetime(df["Maturity date"], format='%m/%d/%Y')

ytms_table = pd.DataFrame()
date_columns = df.columns[4:]

yield_matrix = pd.DataFrame()

for date in date_columns:
    settlement_date = pd.to_datetime(date, format='%m/%d/%Y')
    ytm_values = []

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

    yield_matrix[date] = ytm_values

# Compute log-returns and covariance matrix
log_returns_yield = compute_log_returns(yield_matrix.T)
cov_matrix_yield = compute_covariance_matrix(log_returns_yield)

# Update the covariance matrix to start labels from 1 instead of 0
cov_matrix_yield.index = cov_matrix_yield.index + 1
cov_matrix_yield.columns = cov_matrix_yield.columns + 1

# Print the covariance matrix as a plain table
print("Covariance Matrix for Yield Log-Returns:")
print(cov_matrix_yield.to_string(index=True, header=True))
