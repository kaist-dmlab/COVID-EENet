#####################################################################
# Title  : train function
# Date   : 2020/11/19
# Update : 2021/02/08
# Author : Hyangsuk
#####################################################################
#%%
import tensorflow as tf
import numpy as np
import os, logging, pickle, datetime
from datetime import date
import matplotlib.pyplot as plt

from utils import Dataloader, loadData, split_data
from models import COVIDEENet
from Config import Config
config = Config()
tf.keras.backend.set_floatx('float32')

#%%
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # 텐서플로가 첫 번째 GPU만 사용하도록 제한
  try:
    tf.config.experimental.set_visible_devices(gpus[config.gpu_num], 'GPU')
    tf.config.experimental.set_memory_growth(gpus[config.gpu_num], True)
  except RuntimeError as e:
    # 프로그램 시작시에 접근 가능한 장치가 설정되어야만 합니다
    print(e)

#%%
#######################################################################
# Utils
#######################################################################
def Callback_ModelCheckpoint(epoch, model, pred_tests, outputs_tests, config):
    # save the weights only
    model.save_weights(os.path.join(config.checkpoint_filepath, 
                        config.checkpoint_model)+"_"+str(epoch), save_format='tf', overwrite=True)
    pickle.dump((pred_tests , outputs_tests ), open(os.path.join(config.checkpoint_filepath, config.checkpoint_outputs)+"_test_" +str(epoch)+".pkl","wb"))
    pickle.dump(model.weights, open(os.path.join(config.checkpoint_filepath, config.checkpoint_outputs)+"_weights_"+str(epoch)+".pkl","wb"))
    
    # LOG & PRINT
    print("Save weights at {}".format(os.path.join(config.checkpoint_filepath, 
                        config.checkpoint_model)+"_"+str(epoch)))
    logging.info("Save weights at {}".format(os.path.join(config.checkpoint_filepath, 
                        config.checkpoint_model)+"_"+str(epoch)))
    print("Save test output at {}".format(os.path.join(config.checkpoint_filepath, config.checkpoint_outputs)+"_test_" +str(epoch)+".pkl"))
    logging.info("Save test output at {}".format(os.path.join(config.checkpoint_filepath, config.checkpoint_outputs)+"_test_" +str(epoch)+".pkl"))
    print("Save weights in pickle at {}".format(os.path.join(config.checkpoint_filepath, config.checkpoint_outputs)+"_weights_"+str(epoch)+".pkl"))
    logging.info("Save weights in pickle at {}".format(os.path.join(config.checkpoint_filepath, config.checkpoint_outputs)+"_weights_"+str(epoch)+".pkl"))

def order_batch(inputs, targets, lossMask, metadata, n_steps, batch_size):
    st = (n_steps-1)*batch_size
    ed = st + batch_size
    if ((inputs.shape[0] // batch_size) == n_steps) & ((inputs.shape[0] % batch_size) != 0 ):
        X = inputs[st:]
        X = X.transpose((2,1,0))
        dataset = []
        for i in range(X.shape[0]):
            dataset.append(np.array(X[i].tolist()).astype('float32'))
        dataset = tuple(dataset)
        return dataset, targets[st:].astype('float32'), lossMask[st:].astype('float32'), metadata[st:]
    else :
        X = inputs[st:ed]
        X = X.transpose((2,1,0))
        dataset = []
        for i in range(X.shape[0]):
            dataset.append(np.array(X[i].tolist()).astype('float32'))
        dataset = tuple(dataset)
        return dataset, targets[st:ed].astype('float32'), lossMask[st:ed].astype('float32'), metadata[st:ed]

def print_status_bar(logging, iteration, total, train_loss, metric = None):
    metrics = " - ".join(["{}: {:4f}".format(m.name, m.result())\
                        for m in [train_loss] + ([metric] or [])])
    end = "" if iteration < total else "\n"
    print("\r{}/{} - ".format(iteration, total) + metrics,
            end=end)
    logging.info("\r{}/{} - ".format(iteration, total) + metrics)

#%%
#######################################################################################
# Initialize Data / Model / Loss / Optimizer
#######################################################################################
dataloader = Dataloader(config)
datasets = loadData(config, dataloader)
train_datasets, test_datasets = split_data(config, datasets)

#%%
#######################################################################################
# Initialize weights / Model / Loss / Optimizer
#######################################################################################
model = COVIDEENet(config)
epoch_st = 1

scheduler = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate= config.lr, decay_steps = config.batch_size*2, decay_rate= config.weight_decay)
optimizer = tf.keras.optimizers.Adam(learning_rate = scheduler)

loss_fn = tf.keras.losses.MeanSquaredError(reduction = tf.losses.Reduction.NONE)

train_loss = tf.keras.metrics.Mean(name = 'train_loss')
train_mae_ = tf.keras.losses.MeanAbsoluteError(reduction = tf.losses.Reduction.NONE)
train_mae = tf.keras.metrics.Mean(name = "train_mae")

test_loss = tf.keras.metrics.Mean(name = 'test_loss')
test_mae_ = tf.keras.losses.MeanAbsoluteError(reduction = tf.losses.Reduction.NONE)
test_mae = tf.keras.metrics.Mean(name = "test_mae")

@tf.function
def train_step(inputs, targets, lossMask):
    with tf.GradientTape() as tape:
        predictions, OutputStorage = model(inputs = inputs, training = True)
        loss = loss_fn(targets, predictions)          # [None, config.n]
        # mask
        loss = tf.math.sqrt(tf.math.reduce_mean(tf.math.multiply(loss, lossMask)))
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss(loss)
    mae = tf.math.reduce_mean(tf.math.multiply(train_mae_(targets, predictions), lossMask))
    train_mae(mae)
    return predictions, OutputStorage

@tf.function
def test_step(inputs, targets, lossMask):
    predictions, OutputStorage = model(inputs = inputs, training = False)
    loss = loss_fn(targets, predictions)
    loss = tf.math.sqrt(tf.math.reduce_mean(tf.math.multiply(loss, lossMask)))
    test_loss(loss)
    mae = tf.math.reduce_mean(tf.math.multiply(test_mae_(targets, predictions), lossMask))
    test_mae(mae)
    return predictions, OutputStorage

#######################################################################################
# Main
#######################################################################################
def main(config, train_datasets, test_datasets) :
    fileSavePath = config.checkpoint_filepath
    try :
        os.mkdir(fileSavePath)
    except OSError:
        print ("Creation of the directory %s failed" % fileSavePath)
    else :
        print ("Successfully created the directory %s " % fileSavePath)
    ###################################################################################
    # Test data changed
    ###################################################################################
    n_epochs = config.epochs
    batch_size = config.batch_size
    regions = config.r

    X_train, y_train, lossMask_train, metadata_train = train_datasets
    X_test, y_test, lossMask_test, metadata_test = test_datasets

    n_steps_train = X_train.shape[0] // batch_size
    n_steps_test = X_test.shape[0] // regions

    logging.basicConfig(filename=os.path.join(config.log_filepath, config.log_name), level=logging.INFO)
    logging.info("Config {} \n".format(str(config.__dict__)))
    print("Log file at {}".format(os.path.join(config.log_filepath, config.log_name)))
    print("Config {} \n".format(str(config.__dict__)))

    best_mse_loss = float('inf')
    is_best = False

    for epoch in range(epoch_st, n_epochs):
        epoch_start = datetime.datetime.now()
        print("Epoch {}/{} at {}".format(epoch, n_epochs, epoch_start))
        logging.info("Epoch {}/{} at {}".format(epoch, n_epochs, epoch_start))

        print("Optimizer learning rate : {:4f} ".format(optimizer._decayed_lr('float32').numpy()))
        logging.info("Optimizer learning rate : {:4f} ".format(optimizer._decayed_lr('float32').numpy()))
        ###############################
        # TRAINING
        ###############################
        print("Training...")
        for step_train in range(1, n_steps_train + 1) :
            X_train_batch, y_train_batch, l_train_batch, metadata_train_batch = order_batch(X_train, y_train, lossMask_train, metadata_train, step_train, batch_size) # X_batch : [batch_size, # of mass_infection_cases, # of features, None, None]
            pred_train, outputStorage_train = train_step(X_train_batch, y_train_batch, l_train_batch)
            print_status_bar(logging, step_train * batch_size, len(y_train), train_loss, train_mae)
        print_status_bar(logging, len(y_train), len(y_train), train_loss, train_mae)
        print((datetime.datetime.now() - epoch_start).total_seconds() / 60)            
        ###############################
        # Validation
        ###############################
        print("Validating...")
        pred_tests = []
        outputStorage_tests = []
        for step_test in range(1, n_steps_test + 1) :
            X_test_batch, y_test_batch, l_test_batch, metadata_test_batch = order_batch(X_test, y_test, lossMask_test, metadata_test, step_test, regions)
            pred_test, outputStorage_test = test_step(X_test_batch, y_test_batch, l_test_batch)
            pred_tests.append(pred_test)
            outputStorage_tests.append(outputStorage_test)
            print_status_bar(logging, step_test * regions, len(y_test), test_loss, test_mae)
        print_status_bar(logging, len(y_test), len(y_test), test_loss, test_mae)

        ###############################
        # Model Save
        ###############################
        if best_mse_loss > test_loss.result():
            best_mse_loss = test_loss.result()
            print("Best model with loss {} at iteration {}/{}".format(best_mse_loss, epoch, n_epochs))
            logging.info("Best model with loss {} at iteration {}/{}".format(best_mse_loss, epoch, n_epochs))
            is_best = True
        else :
            is_best = False

        if is_best : 
            print("BEST Saving the model weight at iteration {} test loss {:4f}".format(epoch, best_mse_loss))
            print("BEST Saved at {}".format(os.path.join(config.checkpoint_filepath, config.checkpoint_model)+"_"+str(epoch)))
            logging.info("BEST Saving the model weight at iteration {} test loss {:4f}".format(epoch, best_mse_loss))
            logging.info("BEST Saved at {}".format(os.path.join(config.checkpoint_filepath, config.checkpoint_model)+"_"+str(epoch)))
            Callback_ModelCheckpoint(
                epoch,
                model = model,
                pred_tests = pred_tests,
                outputs_tests = outputStorage_tests,
                config = config
                )

        for metric in [train_loss, train_mae, test_loss, test_mae]:
            metric.reset_states()
#%%
if __name__ == '__main__' :
    main(config, train_datasets, test_datasets)
