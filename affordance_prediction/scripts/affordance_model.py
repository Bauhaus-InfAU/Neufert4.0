

import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
from pickle import dump, load

import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Input, Add
from tensorflow.keras.regularizers import L2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model

import sklearn as sc
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import keras.backend as K

n_embedded_features = 64

os.getcwd()
p = os.path.normpath(os.getcwd() + os.sep + os.pardir + os.sep + "data")

y_df = pd.read_csv(os.path.join(p, "output", 'output.csv'))
if "bld_flr_apt" in y_df.columns:
    y_df = y_df.rename(columns={"bld_flr_apt": "id"})
y_df = y_df.drop(['id', 'has_second_bathroom', 'has_balcony'], axis = 1)
y_df.info(verbose = True)

y_df.number_of_rooms = y_df.number_of_rooms.astype('float32')

rows_to_drop = np.argwhere(np.isnan(y_df['largest_room_kitchen_distance']).to_numpy()).reshape(-1)
y_df = y_df.drop(rows_to_drop)

edge_features_df = pd.read_csv(os.path.join(p, "input", f'edge_{n_embedded_features}.csv'))
shape_features_df = pd.read_csv(os.path.join(p, "input", 'shape.csv'))
vertex_features_df = pd.read_csv(os.path.join(p, "input", f'vertex_{n_embedded_features}.csv'))

X_df = pd.concat([edge_features_df, shape_features_df, vertex_features_df], axis=1)

if "bld_flr_apt" in X_df.columns:
    X_df = X_df.drop('bld_flr_apt', axis = 1)
elif "id" in X_df.columns:  
    X_df = X_df.drop('id', axis = 1)
#drop columns that are not of a float64 type
X_df = X_df.loc[:, X_df.dtypes == 'float64']

X_df = X_df.drop(rows_to_drop)


# split data into train dev and test sets

X_train, X_test, Y_train, Y_test = train_test_split(X_df, y_df, test_size=0.2, random_state=1)

X_train, X_dev, Y_train, Y_dev = train_test_split(X_train, Y_train, test_size=0.25, random_state=1)

scaler_input = StandardScaler()
scaler_input.fit(X_train)

X_train = scaler_input.transform(X_train)
X_dev = scaler_input.transform(X_dev)
X_test = scaler_input.transform(X_test)
scaler_output = ColumnTransformer(
    [('myscaler', StandardScaler(), ['corridor_area_ratio','largest_room_sunlight', 'largest_room_noise', 'largest_room_kitchen_distance'])],
    remainder='passthrough',
    verbose_feature_names_out=False
    ).set_output(transform="pandas")

scaler_output.fit(Y_train)

Y_train = scaler_output.transform(Y_train)
Y_dev = scaler_output.transform(Y_dev)
Y_test = scaler_output.transform(Y_test)




# Compute class weights for both output variables
class_weights_loggia = compute_class_weight('balanced', classes=np.unique(Y_train["has_loggia"]), y=Y_train["has_loggia"])
class_weights_bathroom = compute_class_weight('balanced', classes=np.unique(Y_train["bathroom_has_window"]), y=Y_train["bathroom_has_window"])


### Save normalising parameters
with open('../models/scaler_input.pkl', 'wb') as f:
    dump(scaler_input, f)
with open('../models/scaler_output.pkl', 'wb') as f:
    dump(scaler_output, f)
with open('../models/scaler_output.pkl', 'rb') as f:
    test = load(f)

N_FEATURES = X_train.shape[1]

hparams = {
    "HP_N_HIDDEN": 5,
    "HP_N_UNITS": 256,
    "HP_DROPOUT": 0.3,
    "HP_L2_LAMBDA": .0001,
    }

BINARY_METRICS = {
    "loggia": [
        tf.keras.metrics.TruePositives(name='loggia_tp'),
        tf.keras.metrics.FalsePositives(name='loggia_fp'),
        tf.keras.metrics.TrueNegatives(name='loggia_tn'),
        tf.keras.metrics.FalseNegatives(name='loggia_fn'), 
        tf.keras.metrics.BinaryAccuracy(name='loggia_accuracy'),
        tf.keras.metrics.Precision(name='loggia_precision'),
        tf.keras.metrics.Recall(name='loggia_recall'),
        tf.keras.metrics.AUC(name='loggia_auc'),
        tf.keras.metrics.AUC(name='loggia_prc', curve='PR'), # precision-recall curve
    ],
    "bathroom": [
        tf.keras.metrics.TruePositives(name='bathroom_tp'),
        tf.keras.metrics.FalsePositives(name='bathroom_fp'),
        tf.keras.metrics.TrueNegatives(name='bathroom_tn'),
        tf.keras.metrics.FalseNegatives(name='bathroom_fn'), 
        tf.keras.metrics.BinaryAccuracy(name='bathroom_accuracy'),
        tf.keras.metrics.Precision(name='bathroom_precision'),
        tf.keras.metrics.Recall(name='bathroom_recall'),
        tf.keras.metrics.AUC(name='bathroom_auc'),
        tf.keras.metrics.AUC(name='bathroom_prc', curve='PR'), # precision-recall curve
    ],
}


# get data into the right format
def preproc_df(df, cols, dict_names):
    out = dict(df[cols])
    for i, c in enumerate(cols):
        out[dict_names[i]] =  out.pop(c)
    return out

def weighted_binary_crossentropy(pos_weight=1.0, neg_weight=1.0):
    def loss_function(y_true, y_pred):
        y_true = tf.dtypes.cast(y_true, tf.float64)  # Ensure that y_true is in float64
        y_pred = tf.dtypes.cast(y_pred, tf.float64)  # Ensure that y_pred is in float64
        
        loss_pos = -pos_weight * y_true * tf.math.log(y_pred + 1e-10)
        loss_neg = -neg_weight * (1 - y_true) * tf.math.log(1 - y_pred + 1e-10)
        total_loss = tf.reduce_mean(loss_pos + loss_neg)
        return total_loss
    return loss_function

class CustomAccuracy(tf.keras.metrics.Metric):
    def __init__(self, name = 'accuracy', **kwargs):
        super(CustomAccuracy, self).__init__(**kwargs)
        self.hits = self.add_weight('acc', initializer = 'zeros')
        self.total = self.add_weight('acc', initializer = 'zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.int32)
        y_pred = tf.cast(tf.math.round(y_pred), tf.int32)     
        n_hits = tf.equal(y_true, y_pred)
        self.hits.assign_add(tf.reduce_sum(tf.cast(n_hits, self.dtype)))        
        self.total.assign_add(tf.cast(tf.size(n_hits), self.dtype))

    def reset_state(self):
        self.hits.assign(0)
        self.total.assign(0)

    def result(self):
        return self.hits / self.total


input_layer = Input(shape = (N_FEATURES,), name="Input")
x = input_layer
### Hidden layers ###
for n in range(hparams["HP_N_HIDDEN"]):
    initializer = tf.keras.initializers.GlorotNormal()

    x_prev = x
    x = Dense(hparams["HP_N_UNITS"], "relu", kernel_regularizer = L2(l2 = hparams["HP_L2_LAMBDA"]), name = f"Dense_{str(n)}")(x)
    x = Dropout(rate = hparams["HP_DROPOUT"], name = f"Dropout_{str(n)}")(x)
    if n > 0:
        x = Add(name=f"Skip_add_{str(n)}")([x_prev, x])
    x  = tf.keras.layers.BatchNormalization()(x)

# reduce n units in last two layers
x = Dense(hparams["HP_N_UNITS"]/2, "relu", kernel_regularizer = L2(l2 = hparams["HP_L2_LAMBDA"]), name="Reduce_1")(x)
x = Dropout(rate = hparams["HP_DROPOUT"], name=f"Dropout_{str(n+1)}")(x)
x = tf.keras.layers.BatchNormalization()(x)

x = Dense(hparams["HP_N_UNITS"]/4, "relu", kernel_regularizer = L2(l2 = hparams["HP_L2_LAMBDA"]), name="Reduce_2")(x)
x = Dropout(rate = hparams["HP_DROPOUT"], name=f"Dropout_{str(n+2)}")(x)
x = tf.keras.layers.BatchNormalization()(x)

### Output layers ###
out_efficiency = Dense(1, name = "efficiency_output")(x)
out_kitchen_dist = Dense(1, name = "kitchen_dist")(x)
out_sunlight = Dense(1, name = "sunlight_output")(x)
out_noise = Dense(1, name = "noise_output")(x)
out_kitchen_sunlight = Dense(1, name = "kitchen_sunlight")(x)
out_loggia = Dense(1, activation = "sigmoid", name = "loggia_output")(x)
out_bathroom = Dense(1, activation = "sigmoid", name = "outer_bathroom")(x)
out_rooms = Dense(1, name = "room_output")(x)

### weighted binary x-entropy functions to use in the model
loggia_loss = weighted_binary_crossentropy(
    pos_weight=class_weights_loggia[1],
    neg_weight=class_weights_loggia[0])
bathroom_loss = weighted_binary_crossentropy(
    pos_weight=class_weights_bathroom[1],
    neg_weight=class_weights_bathroom[0])

### build model ###
model = Model(inputs=input_layer, outputs= [
    out_rooms,
    out_efficiency,
    out_kitchen_dist,
    out_sunlight,
    out_noise,
    out_kitchen_sunlight,
    out_loggia, out_bathroom
    ])

# compile the keras model
losses = {
    "room_output": "mse",
    "efficiency_output": "mse",
    "kitchen_dist": "mse",
    "sunlight_output": "mse",
    "noise_output": "mse",
    "kitchen_sunlight": "mse",
    "loggia_output": loggia_loss,
    "outer_bathroom": bathroom_loss
}
lossWeights = {"room_output": 1.0,
            "efficiency_output": 1.0,
            "kitchen_dist": 1.0,
            "sunlight_output": 1.0,
            "noise_output": 1.0,
            "kitchen_sunlight": 1.0,
            "loggia_output": 1.0,
            "outer_bathroom": 1.0}

model.compile(loss=losses,
            loss_weights = lossWeights,
            optimizer=Adam(
                learning_rate = 0.0001
                ),
            metrics={
                "room_output": [CustomAccuracy()],
                "efficiency_output": "mae",
                "kitchen_dist": "mae",
                "sunlight_output": "mae",
                "noise_output": "mae",
                "kitchen_sunlight": "mae",
                "loggia_output": BINARY_METRICS["loggia"],
                "outer_bathroom": BINARY_METRICS["bathroom"]
                })

cols = [
    'number_of_rooms',
    'corridor_area_ratio',
    'largest_room_kitchen_distance',
    'largest_room_sunlight',
    'largest_room_noise',
    'kitchen_sunlight',
    'has_loggia',
    'bathroom_has_window']
loss_names = [k for k in losses.keys()]

hist = model.fit(
    X_train,
    preproc_df(Y_train, cols, loss_names),
    batch_size=256,
    epochs=500,
    validation_data=(X_dev, preproc_df(Y_dev, cols, loss_names)),
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            mode="min",
            min_delta=1e-3,
            patience=20,
            verbose=1,
            restore_best_weights=True
        )
    ],
)

# for i, c in enumerate(cols):
#     plt.figure()
#     plt.scatter(x=model.predict(X_test)[i], y=Y_test[c], alpha = 1/8)
#     plt.title(c)
#     plt.show()


# Save the entire model to a HDF5 file.
# The '.h5' extension indicates that the model should be saved to HDF5.
model.save('../models/affordance.keras')

model_eval = model.evaluate(x=X_test, y=preproc_df(Y_test, cols, loss_names))



