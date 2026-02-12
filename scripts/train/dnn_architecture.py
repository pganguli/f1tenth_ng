from keras import models, layers, regularizers
import os
import numpy as np
import pandas as pd
import json

def load_lidar_dataset(train_csv_path, test_csv_path):
    # Load CSVs
    train_df = pd.read_csv(train_csv_path)
    test_df = pd.read_csv(test_csv_path)

    # -------- Inputs --------
    xTrain = train_df.iloc[:, :1080].values.astype(np.float32)
    xTest  = test_df.iloc[:, :1080].values.astype(np.float32)

    # -------- Targets --------
    yTrain1 = train_df["left_wall_dist"].values.astype(np.float32)
    yTest1  = test_df["left_wall_dist"].values.astype(np.float32)

    yTrain2 = train_df["right_wall_dist"].values.astype(np.float32)
    yTest2  = test_df["right_wall_dist"].values.astype(np.float32)

    yTrain3 = train_df["heading_error"].values.astype(np.float32)
    yTest3  = test_df["heading_error"].values.astype(np.float32)

    # -------- Normalize Inputs --------
    max_val = np.max(xTrain)
    if max_val > 0:
        xTrain = xTrain / max_val
        xTest = xTest / max_val

    return (
        xTrain, xTest,
        yTrain1, yTest1,
        yTrain2, yTest2,
        yTrain3, yTest3
    )

# Create and save models
def architecture(x):
    if x not in [1, 2, 3, 4, 5, 6, 7]:
        raise ValueError("Input parameter must be 1, 2, or 3.")
    # 24 K-bytes
    if x == 1:
        model= models.Sequential([
            layers.Reshape((1080, 1)),

            layers.Conv1D(1, kernel_size=3, padding='valid'),
            layers.ELU(),
            layers.MaxPooling1D(pool_size=8), 

            layers.Flatten(),
            layers.Dense(1, kernel_regularizer=regularizers.l2(0.0001)), 
            #layers.Dense(1)
            ])
        return model
    # 66 K_bytes
    elif x == 2:
        model= models.Sequential([
            layers.Reshape((1080, 1,)),

            layers.Conv1D(1, kernel_size=3, padding='valid'),
            layers.ELU(),
            layers.MaxPooling1D(pool_size=4),

            layers.Conv1D(1, kernel_size=3, padding='valid'),
            layers.ELU(),
            layers.MaxPooling1D(pool_size=2),

            layers.Conv1D(1, kernel_size=3, padding='valid'),
            layers.ELU(),
            layers.MaxPooling1D(pool_size=4),

            layers.Flatten(),
            layers.Dense(8, kernel_regularizer=regularizers.l2(0.0001)),
            layers.ELU(), 
            layers.Dense(1)
        ])
        return model
    # 192 K_bytes
    elif x == 3:
        model= models.Sequential([
            layers.Reshape((1080, 1,)),

            layers.Conv1D(1, kernel_size=3, padding='valid'),
            layers.ELU(),
            layers.MaxPooling1D(pool_size=2),

            layers.Conv1D(8, kernel_size=3, padding='valid'),
            layers.ELU(),
            layers.MaxPooling1D(pool_size=4),

            layers.Flatten(),
            layers.Dense(32, kernel_regularizer=regularizers.l2(0.0001)),
            layers.ELU(),

            layers.Dense(1)
        ])
        return model
    # 326 K_bytes
    elif x == 4:
        model= models.Sequential([
            layers.Reshape((1080, 1,)),

            layers.Conv1D(1, kernel_size=3, padding='valid'),
            layers.ELU(),
            layers.MaxPooling1D(pool_size=2),

            layers.Conv1D(16, kernel_size=3, padding='valid'),
            layers.ELU(),
            layers.MaxPooling1D(pool_size=4),

            layers.Flatten(),
            layers.Dense(32, kernel_regularizer=regularizers.l2(0.0001)),
            layers.ELU(),

            layers.Dense(1)
        ])
        return model
    # 596 K_bytes
    elif x == 5:
        model= models.Sequential([
            layers.Reshape((1080, 1,)),

            layers.Conv1D(8, kernel_size=3, padding='valid'),
            layers.ELU(),
            layers.MaxPooling1D(pool_size=2),

            layers.Conv1D(16, kernel_size=3, padding='valid'),
            layers.ELU(),
            layers.MaxPooling1D(pool_size=4),

            layers.Flatten(),
            layers.Dense(64, kernel_regularizer=regularizers.l2(0.0001)),
            layers.ELU(),

            layers.Dense(1)
        ])
        return model
    # 1112.54 k_bytes
    elif x == 6:
        model= models.Sequential([
            layers.Reshape((1080, 1,)),

            layers.Conv1D(8, kernel_size=3, padding='valid'),
            layers.ELU(),
            layers.MaxPooling1D(pool_size=2),

            layers.Conv1D(32, kernel_size=3, padding='valid'),
            layers.ELU(),
            layers.MaxPooling1D(pool_size=4),

            layers.Flatten(),
            layers.Dense(64, kernel_regularizer=regularizers.l2(0.0001)),
            layers.ELU(),

            layers.Dense(1)
        ])
        return model
    # 2193.90 k_bytes
    elif x == 7:
        model= models.Sequential([
            layers.Reshape((1080, 1,)),

            layers.Conv1D(16, kernel_size=3, padding='valid'),
            layers.ELU(),
            layers.MaxPooling1D(pool_size=2),

            layers.Conv1D(32, kernel_size=3, padding='valid'),
            layers.ELU(),
            layers.MaxPooling1D(pool_size=4),

            layers.Flatten(),
            layers.Dense(128, kernel_regularizer=regularizers.l2(0.0001)),
            layers.ELU(),

            layers.Dense(1)
        ])
        return model
    
    else:
        raise ValueError("Input parameter must be 1, 2, or 3.")

store_mse = {f'model{i}_{j}': 0 for i in range(1, 4) for j in range(1, 8)}



def train_models(train_csv_path, test_csv_path, num_state, num_model, trials):

    os.makedirs("./models", exist_ok=True)

    xTrain, xTest, yTrain1, yTest1, yTrain2, yTest2, yTrain3, yTest3 = \
        load_lidar_dataset(train_csv_path, test_csv_path)

    store_mse = {f"model{i}_{j}": 0.0 for i in range(1,num_state+1) for j in range(1,num_model+1)}


    for arch_id in range(1, num_model+1):          # architectures
        for trial in range(trials):

            # -------- Model 1 (left wall) --------
            model1 = architecture(arch_id)
            model1.compile(optimizer='adam',
                           loss='mean_squared_error',
                           metrics=['mse'])

            model1.fit(xTrain, yTrain1,
                       epochs=13,
                       batch_size=32,
                       verbose=1)

            _, mse = model1.evaluate(xTest, yTest1, verbose=0)
            store_mse[f"model1_{arch_id}"] += mse
            model1.save(f"./models/lidar_model1_{arch_id}.keras")


            # -------- Model 2 (right wall) --------
            model2 = architecture(arch_id)
            model2.compile(optimizer='adam',
                           loss='mean_squared_error',
                           metrics=['mse'])

            model2.fit(xTrain, yTrain2,
                       epochs=13,
                       batch_size=32,
                       verbose=1)

            _, mse = model2.evaluate(xTest, yTest2, verbose=0)
            store_mse[f"model2_{arch_id}"] += mse
            model2.save(f"./models/lidar_model2_{arch_id}.keras")


            # -------- Model 3 (heading error) --------
            model3 = architecture(arch_id)
            model3.compile(optimizer='adam',
                           loss='mean_squared_error',
                           metrics=['mse'])

            model3.fit(xTrain, yTrain3,
                       epochs=13,
                       batch_size=32,
                       verbose=1)

            _, mse = model3.evaluate(xTest, yTest3, verbose=0)
            store_mse[f"model3_{arch_id}"] += mse
            model3.save(f"./models/lidar_model3_{arch_id}.keras")

        print(f"Finished architecture {arch_id}")
        
    average_mse = {}

    for key in store_mse:
        average_mse[key] = store_mse[key] / trials
        print(f"Average MSE for {key}: {average_mse[key]}")

    # Print model sizes
    for i in range(1,num_state+1):
        for j in range(1,num_model+1):
            file_path = f'./models/lidar_model{i}_{j}.keras'
            if os.path.exists(file_path):
                file_size = os.path.getsize(file_path)
                print(f"Model {i}_{j} file size:",
                      round(file_size / 1024, 2), "kB")
    
    with open('./dnn_data/mse_data.json', 'w') as f:
        json.dump(average_mse, f)
                
    print("Training complete.")
    return average_mse


