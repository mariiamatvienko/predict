import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import chart_studio.plotly as py
from plotly.graph_objs import *
import plotly.tools as tls
import chart_studio.tools as ctls
from fbprophet import Prophet
import logging
from scipy import stats
import statsmodels.api as sm
import seaborn as sns
import glob
import os
import plotly.express as px


#sns.set()
#pd.set_option('display.max_rows', None)

PZ_dayahead_price = pd.read_csv("E:\\electricity\csv\RU_Electricity_Market_PZ_dayahead_price_volume.csv")
UES_dayahead_price = pd.read_csv("E:\\electricity\csv\RU_Electricity_Market_UES_dayahead_price.csv")
UES_intraday_price = pd.read_csv("E:\\electricity\csv\RU_Electricity_Market_UES_intraday_price.csv")


# графики для PZ_dayahead_price

PZ_dayahead_price.head()

plt.figure(figsize = (29,15))

ax1 = plt.subplot(4, 1, 1)
PZ_dayahead_price['consumption_eur'].plot()
ax1.set_ylabel("consumption_eur")

ax2 = plt.subplot(4, 1, 2)
PZ_dayahead_price['consumption_sib'].plot()
ax2.set_ylabel("consumption_sib")

ax3 = plt.subplot(4, 1, 3)
PZ_dayahead_price['price_eur'].plot()
ax3.set_ylabel("price_eur")

ax4 = plt.subplot(4, 1, 4)
PZ_dayahead_price['price_sib'].plot()
ax4.set_ylabel("price_sib")

plt.show()

#графики для UES_dayahead_price

UES_dayahead_price.head()


plt.figure(figsize = (29,15))

ax1 = plt.subplot(6, 1, 1)
UES_dayahead_price['UES_Northwest'].plot()
ax1.set_ylabel("UES_Northwest")

ax2 = plt.subplot(6, 1, 2)
UES_dayahead_price['UES_Siberia'].plot()
ax2.set_ylabel("UES_Siberia")

ax3 = plt.subplot(6, 1, 3)
UES_dayahead_price['UES_Middle_Volga'].plot()
ax3.set_ylabel("UES_Middle_Volga")

ax4 = plt.subplot(6, 1, 4)
UES_dayahead_price['UES_Urals'].plot()
ax4.set_ylabel("UES_Urals")

ax5 = plt.subplot(6, 1, 5)
UES_dayahead_price['UES_Center'].plot()
ax5.set_ylabel("UES_Center")

ax6 = plt.subplot(6, 1, 6)
UES_dayahead_price['UES_South'].plot()
ax4.set_ylabel("UES_South")

plt.show()

#графики для UES_intraday_price

UES_intraday_price.head()

plt.figure(figsize = (29,15))

ax1 = plt.subplot(3, 1, 1)
UES_intraday_price['UES_Northwest'].plot()
ax1.set_ylabel("UES_Northwest")

ax2 = plt.subplot(3, 1, 2)
UES_intraday_price['UES_Siberia'].plot()
ax2.set_ylabel("UES_Siberia")

ax3 = plt.subplot(3, 1, 3)
UES_intraday_price['UES_Center'].plot()
ax3.set_ylabel("UES_Center")

plt.show()

"""PREDICT"""


def resample_weekly(data):
	data.reset_index(drop = True, inplace = True)
	data["timestep"] = pd.to_datetime(data["timestep"])
	data.drop("timestep", axis = 1)

	global weekly_data
	daily_data = data.groupby("timestep").mean()
	weekly_data = daily_data.resample("W").mean()

	return weekly_data


def plot_data(data):
	for col in data.columns:
		plt.figure(figsize=(17, 8))
		plt.plot(data[col])
		plt.xlabel('timestep')
		plt.ylabel(col)
		plt.title(f"Weekly Data for {col}")
		plt.grid(True)
		plt.show()



def reset(data):
	global wd_cons
	global lst
	lst = []
	for col in data:
		wd_con = pd.DataFrame(data[col], index=data.index)
		wd_cons = wd_con.reset_index()
		wd_cons.columns = ["ds", "y"]
		lst.append(wd_cons)

	return lst


def split(lst):
	global my_model
	global fut
	global split_data
	fcst_size = 20
	split_data = []
	for dt in lst:
		data = dt.copy()
		splited = data[: - fcst_size]
		split_data.append(splited)

	return split_data


def train_predict(split_data):
	global model_lst
	global forcast_lst
	global fcst_size
	fcst_size = 20
	forcast_lst = []
	model_lst = []
	for tr in split_data:
		my_model = Prophet().fit(tr)
		model_lst.append(my_model)
		fut = my_model.make_future_dataframe(periods=fcst_size, freq="W")
		forcast = my_model.predict(fut)
		forcast_lst.append(forcast)

	return forcast_lst


def plot_prediction(pred_lst, weekly_data):
	for fcst, mod, col in zip(forcast_lst, model_lst, weekly_data):
		mod.plot(fcst)
		plt.title(f"Prediction plot for {col}")
		pred_table(fcst)


def plot_pred_comp(pred_lst, weekly_data):
	for fcst, mod, col in zip(forcast_lst, model_lst, weekly_data):
		mod.plot_components(pd.DataFrame(fcst))
		plt.title(f"Prediction Components plot for {col}")


def evaluate(historical, predicted):
	eval_data = predicted.set_index("ds")[["yhat", "yhat_lower", "yhat_upper"]].join(historical.set_index("ds"))
	return eval_data


def pred_errors(df, prediction_size):
	# df = df.copy()

	df['e'] = df['y'] - df['yhat']
	df['p'] = 100 * df['e'] / df['y']

	predicted_part = df[- fcst_size:]

	error_mean = lambda error_name: np.mean(np.abs(predicted_part[error_name]))

	return {'MAPE': error_mean('p'), 'MAE': error_mean('e')}

def pred_table(data):
	global table
	table = pd.DataFrame(data, columns = ["y", "yhat", "yhat_lower", "yhat_upper"], index = data.index)
	table.columns = ["Actul Value", "Prediction", "Pred_lower", "Pred_Upper"]
	return table.tail(10)



resample_weekly(PZ_dayahead_price)
plot_data(weekly_data)

reset(weekly_data)
split(lst)
train_predict(split_data)
plot_prediction(forcast_lst, weekly_data)
plot_pred_comp(forcast_lst, weekly_data)
#
eval_data = list(map(lambda X: evaluate(X[0], X[1]), list(zip(lst, forcast_lst))))

for it in eval_data:
	pred_error = pred_errors(it, fcst_size)
	print(pred_error)

count = 0
for item, ls, colu in zip(forcast_lst, lst, weekly_data):
    item = item[["ds", "yhat", "yhat_lower", "yhat_upper"]]
    item["y"] = ls[["y"]]
    item = item.sort_index(axis = 1)
    df = item
    l = locals()
    cols_plt = ["y", "yhat", "yhat_lower", "yhat_upper"]
    l["fig_" + str(count)] = px.line(df, x= "ds", y= ["y", "yhat", "yhat_lower", "yhat_upper"],
                  hover_data={"ds": "|%B %d, %Y"},
                  title= (f'Time series Prediction Plot for {colu} UES_intraday_price' ))
    l["fig_" + str(count)].update_xaxes(
        dtick="M1",
        tickformat="%b\n%Y",
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1d", step="day", stepmode="todate"),
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="YTD", step="year", stepmode="todate"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(step="all")])))
    l["fig_" + str(count)].show()
    count += 1
    print(pd.DataFrame(pred_table(item)))



# ----------------------------- UES_intraday_price-------------------------
UES_intraday_price.tail()
intraday_price = UES_intraday_price.copy()

resample_weekly(intraday_price)
plot_data(weekly_data)
reset(weekly_data)
split(lst)
train_predict(split_data)
plot_prediction(forcast_lst, weekly_data)
plot_pred_comp(forcast_lst, weekly_data)

eval_data = list(map(lambda X: evaluate(X[0], X[1]), list(zip(lst, forcast_lst))))
eval_data

for it in eval_data:
	pred_error = pred_errors(it, fcst_size)
	print(pred_error)

count = 0
for item, ls, colu in zip(forcast_lst, lst, weekly_data):
	item = item[["ds", "yhat", "yhat_lower", "yhat_upper"]]
	item["y"] = ls[["y"]]
	item = item.sort_index(axis=1)
	df = item
	l = locals()
	cols_plt = ["y", "yhat", "yhat_lower", "yhat_upper"]
	l["fig_" + str(count)] = px.line(df, x="ds", y=["y", "yhat", "yhat_lower", "yhat_upper"],
									 hover_data={"ds": "|%B %d, %Y"},
									 title=(f'Time series Prediction Plot for {colu} UES_intraday_price'))
	l["fig_" + str(count)].update_xaxes(
		dtick="M1",
		tickformat="%b\n%Y",
		rangeselector=dict(
			buttons=list([
				dict(count=1, label="1d", step="day", stepmode="todate"),
				dict(count=1, label="1m", step="month", stepmode="backward"),
				dict(count=6, label="6m", step="month", stepmode="backward"),
				dict(count=1, label="YTD", step="year", stepmode="todate"),
				dict(count=1, label="1y", step="year", stepmode="backward"),
				dict(step="all")])))
	l["fig_" + str(count)].show()
	count += 1
	print(pd.DataFrame(pred_table(item)))


# --------- UES_DAYAHEAD_PRICE ----------------------

resample_weekly(UES_dayahead_price)

plot_data(weekly_data)

reset(weekly_data)
split(lst)

train_predict(split_data)
plot_prediction(forcast_lst, weekly_data)
plot_pred_comp(forcast_lst, weekly_data)

eval_data = list(map(lambda X: evaluate(X[0], X[1]), list(zip(lst, forcast_lst))))
eval_data

for it in eval_data:
	pred_error = pred_errors(it, fcst_size)
	print(pred_error)

# з верхнйою та нижньою межею
count = 0
for item, ls, colu in zip(forcast_lst, lst, weekly_data):
	item = item[["ds", "yhat", "yhat_lower", "yhat_upper"]]
	item["y"] = ls[["y"]]
	item = item.sort_index(axis=1)
	df = item
	l = locals()
	cols_plt = ["y", "yhat", "yhat_lower", "yhat_upper"]
	l["fig_" + str(count)] = px.line(df, x="ds", y=["y", "yhat", "yhat_lower", "yhat_upper"],
									 hover_data={"ds": "|%B %d, %Y"},
									 title=(f'Time series Prediction Plot for {colu} UES_intraday_price'))
	l["fig_" + str(count)].update_xaxes(
		dtick="M1",
		tickformat="%b\n%Y",
		rangeselector=dict(
			buttons=list([
				dict(count=1, label="1d", step="day", stepmode="todate"),
				dict(count=1, label="1m", step="month", stepmode="backward"),
				dict(count=6, label="6m", step="month", stepmode="backward"),
				dict(count=1, label="YTD", step="year", stepmode="todate"),
				dict(count=1, label="1y", step="year", stepmode="backward"),
				dict(step="all")])))
	l["fig_" + str(count)].show()
	count += 1
	print(pd.DataFrame(pred_table(item)))


