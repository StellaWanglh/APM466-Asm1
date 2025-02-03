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
    dirty_price = clean_price + accrued_interest
    return dirty_price

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

# Convert maturity dates
df["Maturity date"] = pd.to_datetime(df["Maturity date"], format='%m/%d/%Y')

ytm_table = pd.DataFrame()
date_columns = df.columns[4:]

yield_curves = {}

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

        ytm_values.append(f"{ytm:.2f}%")
        maturities.append(years_to_maturity)

    ytm_table[date] = ytm_values
    yield_curves[date] = (maturities, [float(y.strip('%')) for y in ytm_values])

# Output the table
ytm_table.index = [f"Bond {i+1}" for i in range(len(df))]
print(ytm_table)

# Plot yield curves for each date
plt.figure(figsize=(10, 6))

for date, (maturities, yields) in yield_curves.items():
    # Spline interpolation to smooth the yield curve
    cs = CubicSpline(maturities, yields)
    smooth_maturities = np.linspace(min(maturities), max(maturities), 100)
    plt.plot(smooth_maturities, cs(smooth_maturities), label=str(date)[:10])

plt.xlabel("Years to Maturity")
plt.ylabel("Yield to Maturity (%)")
plt.title("5-Year Yield Curve Over Time")
plt.legend(title="Settlement Date", loc='upper left')
plt.grid(True)
plt.show()
