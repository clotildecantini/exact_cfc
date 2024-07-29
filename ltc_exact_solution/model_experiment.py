from ltc_exact_solution.experiment_mnist import *
import pandas as pd
import numpy as np
import argparse

def experiment_mnist(num_epochs=10):
    """Launch the comparison of model training for different ltc layers."""
    loss = {}
    accuracy = {}
    train_dataset, val_dataset, test_dataset = load_mnist()
    for ltc_layer in ['ode', 'approx', 'exact']:
        print('Experimenting with', ltc_layer)  
        model = create_model(ltc_layer=ltc_layer)
        history = model.fit(train_dataset, epochs=num_epochs, validation_data=val_dataset)
        test_loss_approx, test_acc_approx = model.evaluate(test_dataset)
        # Save the model
        model.save(f'./model_{ltc_layer}.h5')   
        # Save the history
        np.save(f'./history_{ltc_layer}', history.history)
        # Save the test loss and accuracy
        loss[ltc_layer] = test_loss_approx
        accuracy[ltc_layer] = test_acc_approx
    df = pd.DataFrame({'LTC layer': ['ode', 'approx', 'exact'], 'Test loss': [loss['ode'], loss['approx'], loss['exact']], 'Test accuracy': [accuracy['ode'], accuracy['approx'], accuracy['exact']]})
    df.to_csv('./results_training.csv')



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A script that takes an integer argument.")
    parser.add_argument('num_epochs', type=int, help="An integer number")
    args = parser.parse_args()
    num_epochs = args.num_epochs
    experiment_mnist(num_epochs=num_epochs)

    