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
    days_since_last_coupon = (settlement_date - last_coupon).days
    face_value = 100
    coupon_payment = coupon_rate * face_value / frequency
    accrued_interest = coupon_payment * (days_since_last_coupon / days_in_coupon_period)
    return clean_price + accrued_interest

def bond_ytm(price, face_value, coupon_rate, years_to_maturity, frequency=2):
    """Solve for Yield to Maturity (YTM) using numerical root finding."""
    coupon_payment = coupon_rate * face_value / frequency

    def equation(r):
        return sum([coupon_payment / (1 + r / frequency) ** (years_to_next_coupon * 2 + t)
                    for t in range(0, math.ceil(years_to_maturity * frequency))]) + \
            face_value / (1 + r / frequency) ** (frequency * years_to_maturity) - price

    ytm_guess = 0.03  # Initial guess for YTM (3%)
    ytm_solution = fsolve(equation, ytm_guess)
    return ytm_solution[0] * 100

def bootstrap_spot_rates(maturities, ytms):
    """Compute spot rates using the bootstrapping method."""
    spot_rates = []
    for i in range(len(maturities)):
        maturity = maturities[i]
        ytm = ytms[i] / 100  # Convert to decimal
        coupon = ytm * 100 / frequency
        price = 100  # Assume par bond

        if i == 0:  # First bond (zero-coupon)
            spot_rate = ytm
        else:
            sum_discounted = sum([coupon / (1 + spot_rates[j] / frequency) ** (maturities[j] * frequency)
                                  for j in range(i)])
            spot_rate = ((price - sum_discounted) / (100 + coupon)) ** (-1 / (maturity * frequency)) - 1

        spot_rates.append(spot_rate * 100)  # Convert to percentage

    return spot_rates

# Convert maturity dates
df["Maturity date"] = pd.to_datetime(df["Maturity date"], format='%m/%d/%Y')

ytm_table = pd.DataFrame()
spot_table = pd.DataFrame()
date_columns = df.columns[4:]

yield_curves = {}
spot_curves = {}

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

    yield_curves[date] = (maturities, ytm_values)

    # Compute spot rates
    spot_rates = bootstrap_spot_rates(maturities, ytm_values)
    spot_curves[date] = (maturities, spot_rates)

    ytm_table[date] = [f"{ytm:.2f}%" for ytm in ytm_values]
    spot_table[date] = [f"{sr:.2f}%" for sr in spot_rates]

# Output YTM and Spot Rate tables
ytm_table.index = [f"Bond {i+1}" for i in range(len(df))]
spot_table.index = [f"Bond {i+1}" for i in range(len(df))]

print("Yield to Maturity (YTM) Table:")
print(ytm_table)
print("\nSpot Rate Curve Table:")
print(spot_table)

# Plot Yield Curves
plt.figure(figsize=(12, 6))
for date in yield_curves.keys():
    maturities, yields = yield_curves[date]
    cs_yield = CubicSpline(maturities, yields)

    smooth_maturities = np.linspace(min(maturities), max(maturities), 100)

    plt.plot(smooth_maturities, cs_yield(smooth_maturities), label=f"YTM {str(date)[:10]}", linestyle='dashed')

plt.xlabel("Years to Maturity")
plt.ylabel("Yield to Maturity (%)")
plt.title("Yield Curve Over Time")
plt.legend(title="Settlement Date", loc='upper left', fontsize=8)
plt.grid(True)
plt.show()

# Plot Spot Rate Curves (Separate Graph)
plt.figure(figsize=(12, 6))
for date in spot_curves.keys():
    spot_maturities, spot_rates = spot_curves[date]
    cs_spot = CubicSpline(spot_maturities, spot_rates)

    smooth_maturities = np.linspace(min(spot_maturities), max(spot_maturities), 100)

    plt.plot(smooth_maturities, cs_spot(smooth_maturities), label=f"Spot {str(date)[:10]}")

plt.xlabel("Years to Maturity")
plt.ylabel("Spot Rate (%)")
plt.title("Spot Rate Curve Over Time")
plt.legend(title="Settlement Date", loc='upper left', fontsize=8)
plt.grid(True)
plt.show()
