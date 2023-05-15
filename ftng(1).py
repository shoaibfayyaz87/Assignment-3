import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import errors as err
import scipy.optimize as opt

#https://datahub.io/core/gdp-uk/r/0.html

def exponential(t, n0, g):
   """Calculates exponential function with scale factor n0 and growth rate g."""
   t = t - 1990
   f = n0 * np.exp(g*t)
   return f

def logistic(t, n0, g, t0):
   """Calculates the logistic function with scale factor n0 and growth rate g"""
   f = n0 / (1 + np.exp(-g*(t - t0)))
   return f

def poly(x, a, b, c, d, e):
   """ Calulates polynominal"""
   x = x - 1990
   f = a + b*x + c*x**2 + d*x**3 + e*x**4
   return f

def linear(x, a, b):
   """ Simple linear function calculating a + b*x. """
   f = a + b*x
   return f



df_gdp = pd.read_csv("UK_GDP.csv")
#mask = df_gdp['Country Name'] == 'United Kingdom'
df_gdp.plot("date", "GDP")
plt.show()

print(type(df_gdp["date"].iloc[1]))
df_gdp["date"] = pd.to_numeric(df_gdp["date"])
print(type(df_gdp["date"].iloc[1]))
param, covar = opt.curve_fit(exponential, df_gdp["date"], df_gdp["GDP"], p0=(1.212, 0.03))
print("GDP 1990", param[0]/1e9)
print("growth rate", param[1])

plt.figure()
plt.plot(df_gdp["date"], exponential(df_gdp["date"], 1.2e12, 0.03), label = "trial fit")
plt.plot(df_gdp["date"], df_gdp["GDP"])
plt.xlabel("date")
plt.legend()
plt.show()

df_gdp["fit"] = exponential(df_gdp["date"], *param)
df_gdp.plot("date", ["GDP", "fit"])
plt.show()

param, covar = opt.curve_fit(logistic, df_gdp["date"], df_gdp["GDP"],p0=(1.2e12, 0.03, 1990.0))


sigma = np.sqrt(np.diag(covar))
df_gdp["fit"] = logistic(df_gdp["date"], *param)
df_gdp.plot("date", ["GDP", "fit"])
plt.show()
print("turning point", param[2], "+/-", sigma[2])
print("GDP at turning point", param[0]/1e9, "+/-", sigma[0]/1e9)
print("growth rate", param[1], "+/-", sigma[1])

df_gdp["trial"] = logistic(df_gdp["date"], 3e12, 0.10, 1990)
df_gdp.plot("date", ["GDP", "trial"])
plt.show()



date = np.arange(1960, 2031)
forecast = logistic(date, *param)

plt.figure()
plt.plot(df_gdp["date"], df_gdp["GDP"], label="GDP")
plt.plot(date, forecast, label="forecast")
plt.xlabel("date")
plt.ylabel("GDP")
plt.legend()
plt.show()

low, up = err.err_ranges(date, logistic, param, sigma)
plt.figure()
plt.plot(df_gdp["date"], df_gdp["GDP"], label="GDP")
plt.plot(date, forecast, label="forecast")
plt.fill_between(date, low, up, color="yellow", alpha=0.7)
plt.xlabel("date")
plt.ylabel("GDP")
plt.legend()
plt.show()

print(logistic(2030, *param)/1e9)
print(err.err_ranges(2030, logistic, param, sigma))
# assuming symmetrie estimate sigma
gdp2030 = logistic(2030, *param)/1e9
low, up = err.err_ranges(2030, logistic, param, sigma)
sig = np.abs(up-low)/(2.0 * 1e9)
print()
print("GDP 2030", gdp2030, "+/-", sig)



param, covar = opt.curve_fit(poly, df_gdp["date"], df_gdp["GDP"])
sigma = np.sqrt(np.diag(covar))
print(sigma)
year = np.arange(1960, 2031)
forecast = poly(date, *param)
low, up = err.err_ranges(date, poly, param, sigma)
df_gdp["fit"] = poly(df_gdp["date"], *param)
plt.figure()
plt.plot(df_gdp["date"], df_gdp["GDP"], label="GDP")
plt.plot(year, forecast, label="forecast")
plt.fill_between(date, low, up, color="yellow", alpha=0.7)
plt.xlabel("date")
plt.ylabel("GDP")
plt.legend()
plt.show()


# create a few points with normal distributed random errors
xarr = np.linspace(0.0, 10.0, 21)
yarr = linear(xarr, 1.0, 0.2)
ymeasure = yarr + np.random.normal(0.0, 0.10, len(yarr))
# create a list of x-y pairs for conversion to a dataframe
lin_list = []
for x, ym in zip(xarr, ymeasure):
   lin_list.append([x, ym])
# do the conversion
df_lin = pd.DataFrame(lin_list, columns=["x", "measure"])
print(df_lin)



# fit the data
param, covar = opt.curve_fit(linear, df_lin["x"], df_lin["measure"])
# plot the result
plt.figure()
plt.plot(df_lin["x"], df_lin["measure"], "go", label="measurements")
plt.plot(df_lin["x"], linear(df_lin["x"], *param), label="fit")
plt.plot(df_lin["x"], linear(df_lin["x"], 1.0, 0.2), label="a = 1.0+0.2*x")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()



df_lin.iloc[-3, 1] = 7.0
# fit the data
param, covar = opt.curve_fit(linear, df_lin["x"], df_lin["measure"])
# plot the result
plt.figure()
plt.plot(df_lin["x"], df_lin["measure"], "go", label="measurements")
plt.plot(df_lin["x"], linear(df_lin["x"], *param), label="fit")
plt.plot(df_lin["x"], linear(df_lin["x"], 1.0, 0.2), label="a = 1.0+0.2*x")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()


# create column with fit values
df_lin["fit"] = linear(df_lin["x"], *param)
df_lin["diff"] = df_lin["measure"] - df_lin["fit"]
sigma = df_lin["diff"].std()
print(sigma)
df_lin["z"] = abs(df_lin["diff"] / sigma)
print(df_lin)


df_lin = df_lin[df_lin["z"]<3.0].copy()
print(df_lin)


sigma = df_lin["diff"].std()
print(sigma)
df_lin["z"] = abs(df_lin["diff"] / sigma)
print(df_lin)

param, covar = opt.curve_fit(linear, df_lin["x"], df_lin["measure"])
df_lin["fit"] = linear(df_lin["x"], *param)
plt.figure()
plt.plot(df_lin["x"], linear(df_lin["x"], 1.0, 0.2), label="1.0 + 0.2*x")
plt.plot(df_lin["x"], df_lin["measure"], "go", label="measurements")
plt.plot(df_lin["x"], df_lin["fit"], label="fit")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()