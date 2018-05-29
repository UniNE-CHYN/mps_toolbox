import mps

nx = 550
ny = 500
image = mps.loadtxt("ti_categoricalSoilCracks.txt", nx, ny)
categories = image.categories_list()
n_lags = 40
max_lag = 150
lag_classes = image.generate_lag_classes(n_lags, max_lag)

for category in categories:
    connectivity = mps.connectivity_function(image, lag_classes , category)
    connectivity.plot()
