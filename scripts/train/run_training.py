from dnn_architecture import train_models


def main():

    train_csv = "./dnn_data/training_data.csv"
    test_csv  = "./dnn_data/testing_data.csv"

    num_state = 3      # number of architectures
    num_model = 2      # number of output models
    trials= 2

    results = train_models(
        train_csv_path=train_csv,
        test_csv_path=test_csv,
        num_state=num_state,
        num_model=num_model,
        trials=trials
    )

    print("\nFinal Averaged MSE Results:")
    for k, v in results.items():
        print(f"{k}: {v:.6f}")


if __name__ == "__main__":
    main()
