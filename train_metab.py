#!/usr/bin/python3

####################################################
#                                                  #
# 31P-SPAWNN                                       #
# reference paper doi: 10.1002/mrm.29446           #
# url: gitlab.unige.ch/Julien.Songeon/31P-SPAWNN   #
#      See READE.me for documentation              #
# last modification: 09.11.2022                    #
#                                                  #
####################################################


#### Python imports
import sys, os
import argparse
import csv
import datetime
import h5py
import importlib
import json
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from numpy.random import rand, randn, seed
import pandas as pd
from pathlib import Path
import seaborn as sns
import time
from tqdm import tqdm

#### Self functions imports
sys.path.append('../Tools')
import modulable_CNN
import tools
import visual_callbacks

#### Keras imports
import keras
from keras import backend as K
from keras.utils.vis_utils import plot_model
from keras.callbacks import CSVLogger

#### Tensorflow import
import tensorflow as tf


def main():
    time_initial = time.time()

#### Retrieving arguments
    parser = argparse.ArgumentParser(description='Generate dataset.')
    parser.add_argument('-o', '--output', dest='model_name',required=True)	# output folder target for data
    parser.add_argument('-i', '--intput', dest='filename',required=True)	# .h5 file generated with Generation.py

    parser.add_argument('--Nbatch', dest='Nbatch',default = 800)        # Nb of samples propagated through cnn at single time
    parser.add_argument('--Nepochs', dest='Nepochs',default = 80)       # Nb of iteration on all dataset
    parser.add_argument('--nNeuron', dest='nNeuron',default = 50)       # Nb of neurons in dense layers
    parser.add_argument('--nFilters', dest='nFilters',default = 16)     # Nb of filters in convolution layers
    parser.add_argument('--GPUpartition', dest='GPUpartition',default = 0)      # GPU management
    parser.add_argument('--dropOut', dest='dropOut',default = 0.2)      # Dropout value for drop layers
    parser.add_argument('--optimizer', dest='optimizer',default = 'adam')       # Algo of function optimization
    parser.add_argument('--tLayer', dest='tLayer',default = 'ReLU')     # Activation function
    parser.add_argument('--regularizer', dest='regularizer',default = False)    # Reweighting of layers during optimization
    parser.add_argument('--nevent_test', dest='nevent_test',default = 2500)     #Nb of event tested
    parser.add_argument('-v', dest='verbose', action='store_true', help='Verbose output') # Use if you want print output


    args = parser.parse_args()
    model_name = args.model_name
    filename = args.filename

    Nbatch      = int(args.Nbatch)
    Nepochs     = int(args.Nepochs)
    nNeuron     = int(args.nNeuron)
    nFilters    = int(args.nFilters)
    GPUpartition = float(args.GPUpartition)
    dropOut     = float(args.dropOut)
    optimizer   = args.optimizer
    tLayer      = args.tLayer
    regularizer = args.regularizer
    nevent_test = int(args.nevent_test)
    verbose       = args.verbose

    
    if not os.path.exists(model_name):
        os.makedirs(model_name)


#### Parametrize the GPU
    from keras.backend.tensorflow_backend import set_session
    config = tf.ConfigProto()

    if GPUpartition == 0 :
        config.gpu_options.allow_growth = True                                      #To allow GPU to manadge alone its memory
    else:
        config.gpu_options.per_process_gpu_memory_fraction = GPUpartition           #To fraction manually the GPU memory

    check = True
    while check:
        #Control if tensor can allocate request memory
        try:
            set_session(tf.Session(config=config))
            check = False
        except:
            print("Error during memory allocation")
            time.sleep(5)
            check = True
   

#### Loading the training set
    data_train = tools.load_h5_file(filename,
                                    load=('train/spectra',
                                          'train/amplitudes',
                                          'train/index',
                                          'train/metab_spectra',
                                          'train/phshift',
                                          'train/acqdelay',
                                          'train/Lorwidth',
                                          'train/Gauwidth',
                                          'train/SNR',
                                          'train/freqshift'),
                                    verbose = verbose)

    # Constantes
    Fs           = data_train['train/spectra/Fs']
    Npt          = data_train['train/spectra/Npt']
    NMRFreq      = data_train['train/spectra/NMRFreq']
    WINDOW_START = data_train['train/spectra/WINDOW_START']
    WINDOW_END   = data_train['train/spectra/WINDOW_END']
    N1           = data_train['train/spectra/N1']
    N2           = data_train['train/spectra/N2']
    index        = data_train['train/index']
    names        = data_train['train/names']
    Basis_Metab  = data_train['train/metab_spectra']

    # Variables 
    spectra     = data_train['train/spectra'][:]
    amplitudes  = data_train['train/amplitudes'][:]

    snr = data_train['train/SNR'][:]


    # Data permutation
    seed()
    perm = np.random.permutation(spectra.shape[0])

    spectra     = spectra[perm,:]
    amplitudes  = amplitudes[perm,:]
    snr = snr[perm]

    # Stacking and normalizing input
    spectra_stack  = np.stack((np.real(spectra), np.imag(spectra)), axis=-1)
    spectra_energy = np.sum(np.abs(spectra)**2, axis=1).mean()
    spectra_stack *= np.sqrt(1.0/spectra_energy)
    y_in =  spectra_stack
    y_in = np.pad(y_in, ((0,0),(5, 5),(1,0)), 'wrap')   #padding with add 5 up/down in the spectral dimension and replicate the imaginary part
    y_in = np.expand_dims(y_in, axis = -1)

    # Stacking and normalizing output
    y_out = amplitudes 
    y_min = y_out.min(axis=0)
    y_span = np.zeros(y_out.shape[1])
    for ii in range(y_span.shape[0]):
            y_span[ii] = np.percentile(y_out[:,ii],99) - y_out[:,ii].min()
    y_out = (y_out - y_min) / y_span



#### Train the model
    model = modulable_CNN.SPAWNN_Q(input_shape=(y_in.shape[1],y_in.shape[2],1,),
                                   output_shape=y_out.shape[1],
                                   regularizer=regularizer,
                                   nFilters=nFilters,
                                   nNeuron=nNeuron,
                                   drop=dropOut,
                                   endReLU=True)

    V_splits = 0.01
    Tloss = 'mse'

    if optimizer == 'adam':
            opimizer = keras.optimizers.adam()
    elif optimizer == 'RMSprop':
        opimizer = keras.optimizers.RMSprop(lr=lr)

    model.compile(loss=Tloss,
                  optimizer= optimizer,
                  metrics=[tools.R2])
    model.summary()
    plot_model(model,
               to_file="%smodel_plot.png"%(model_name),
               show_shapes=True,
               show_layer_names=True)
    plotter = visual_callbacks.trainPlotter(graphs=['loss','R2'],
                                            save_graph=True,
                                            names = "All Metab",
                                            name_model= "%s/graf_train.png"%(model_name))   #Initialization of display
    csv_logger = CSVLogger("%s/epoch_log.csv"%(model_name),
                           separator=',',
                           append=False)
    callbacks = [plotter,
                 keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=50, verbose=1),
                 csv_logger]

    t = time.time()
    train_history = model.fit(y_in,
                              y_out,
                              epochs=Nepochs,
                              validation_split=V_splits,
                              batch_size=Nbatch,
                              callbacks= callbacks,
                              verbose=1
                              )


    elapsed_train = np.round(time.time() - t, decimals=4)             #Time to train the model
    print("Time to train : %s [s]"%(elapsed_train))
    train_history = train_history.history

    out_predict = model.predict(y_in)
    r2 = tools.np_rscore(y_out,out_predict)
    r2 = r2*r2
    print("R2 score : %s" %(train_history['val_R2'][-1]))

    for param in range(len(names)):
        print("R2 score of %s : %s" %(names[param], r2[param]) )


#### Saving model
    Dico_value = {"Model name :" : model_name,
                  "Data loaded :": filename,
                  "Time to train [s] :" : elapsed_train,
                  "Mini-batch size :" : Nbatch,
                  "Number of Epochs :" : Nepochs,
                  "Train set size :" : y_in.shape[0] * (1-V_splits),
                  "Valid set size :" : y_in.shape[0] * V_splits,
                  "Optimizer :" : opimizer,
                  "Loss function :" : Tloss,
                  "Number of parameters :" : model.count_params(),
                  "Last Valida loss value :" : train_history['val_loss'][-1],
                  "Last train loss value :" : train_history['loss'][-1],
                  "R2 :" : r2,
                  "Output names :" : names,
#                  "Learning rate Value :" : lr,
                  "DropOut Value :" : dropOut}

    with open("%s/Values.txt" %(model_name), "w") as log:
        log.write("INTERESTING VALUES \n")
        for keys in Dico_value.keys():
            log.write('%s %s \n' % ( keys , Dico_value.get(keys) ))

    # Save weights
    model.save_weights("%s/weights.h5" %(model_name), overwrite=True)

    #Save Model Parameters
    with h5py.File("%s/params.h5" %(model_name), 'w', libver='earliest') as f:
        train = f.create_group('Model')
        train.create_dataset('y_span', data=y_span)
        train.create_dataset('y_min', data=y_min)
        train.create_dataset('Fs', data=Fs)
        train.create_dataset('Npt', data=Npt)
        train.create_dataset('spectra_energy', data=spectra_energy)
        train.create_dataset('NMRFreq', data=NMRFreq)
        train.create_dataset('WINDOW_START', data=WINDOW_START)
        train.create_dataset('WINDOW_END', data=WINDOW_END)
        train.create_dataset('N1', data=N1)
        train.create_dataset('N2', data=N2)
        train.create_dataset('index', data=index)
        train.create_dataset('Basis_Metab', data=Basis_Metab)
        asciinames = [n.encode("ascii", "ignore") for n in names]
        train.create_dataset('names', data=asciinames)

    #model
    model_json = model.to_json()
    with open("%s/model.json" %(model_name), "w") as json_file :
        json_file.write(model_json)
    #Image of layer
    plot_model(model, to_file= "%s/Structur.png" %(model_name),
                show_shapes=True, show_layer_names=True)
    #Training History

    print("Saved model to disk in folder %s" %(model_name))
    print("Script to train all metab model is Over!")



#### Evaluate on training set
    print('Plotting R^2')
    out_predict = (out_predict*y_span) + y_min
    out_predict = tools.no_negative_concentration(out_predict,names,y_span,y_min)
    y_out = (y_out*y_span) + y_min

    #SNR vs R2 plot training
    mask_name = ["SNR>5.5","5.5>SNR>4.5","4.5>SNR>3.5","3.5>SNR>2.5","2.5>SNR>1.5","1.5>SNR"]
    mask = []
    mask.append(np.where(snr[:,-1]>=5.5)[0])
    mask.append(np.where((snr[:,-1]<5.5) & (snr[:,-1]>=4.5))[0])
    mask.append(np.where((snr[:,-1]<4.5) & (snr[:,-1]>=3.5))[0])
    mask.append(np.where((snr[:,-1]<3.5) & (snr[:,-1]>=2.5))[0])
    mask.append(np.where((snr[:,-1]<2.5) & (snr[:,-1]>=1.5))[0])
    mask.append(np.where(snr[:,-1]<1.5)[0])
    snrvr2 = []
    for ii in range(len(mask)):
        snrvr2.append(tools.np_rscore(y_out[mask[ii],:],out_predict[mask[ii],:]))
    snrvr2 = np.array(snrvr2)
    snrvr2 = snrvr2*snrvr2

    t = time.time()
    pdf = PdfPages("%s/heatmap_SNRvR2_training.pdf" %(model_name))
    fig, ax = plt.subplots()
    fig.set_size_inches(20, 20)
    plt.subplots_adjust(left=0.15, right=0.85)
    plt.title('R2 in function of SNR', fontsize=30)
    cmap = matplotlib.cm.magma
    cmap.set_over('w')
    snrvr2 = snrvr2.transpose((1,0))
    im = ax.imshow(snrvr2,vmin=0., vmax=1.,cmap=cmap)
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.tick_params(labelsize=30)
    cbar.ax.set_ylabel('R2', rotation=-90, va="bottom", fontsize=30)
    for jj in range(snrvr2.shape[0]):
        for kk in range(snrvr2.shape[1]):
            r2text = ax.text(kk,jj,str(np.round(snrvr2[jj,kk],3)),ha="center", va="center", color="k", fontsize=20)
    ax.set_yticks(np.arange(snrvr2.shape[0]))
    ax.set_yticklabels(names[:], fontsize=30)
    ax.set_xticks(np.arange(snrvr2.shape[1]))
    ax.set_xticklabels(mask_name, fontsize=30)
    ax.grid(False)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",rotation_mode="anchor")
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close()
    pdf.close()
    elapsed_snr = np.round(time.time() - t, decimals=4)
    print("Time to plot training snr vs r2: %s [s]"%(elapsed_snr))


#### Evaluate on testing set
    print('\n\nEvaluating the model with the test set')
    data_test = tools.load_h5_file(filename,
                                  load=('test/spectra',
                                        'test/amplitudes',
                                        'test/index',
                                        'test/metab_spectra',
                                        'test/phshift',
                                        'test/acqdelay',
                                        'test/Lorwidth',
                                        'test/Gauwidth',
                                        'test/SNR',
                                        'test/freqshift'),
                                    verbose = verbose)

    # Variables 
    test_spectra     = data_test['test/spectra'][:]
    test_amplitudes  = data_test['test/amplitudes'][:]
    test_snr = data_test['test/SNR'][:]

    test_spectra_stack  = np.stack((np.real(test_spectra), np.imag(test_spectra)), axis=-1)
    test_spectra_energy = np.sum(np.abs(test_spectra)**2, axis=1).mean()
    test_spectra_stack *= np.sqrt(1.0/test_spectra_energy)
    y_in_test =  test_spectra_stack
    y_in_test = np.pad(y_in_test, ((0,0),(5, 5),(1,0)), 'wrap')
    y_in_test = np.expand_dims(y_in_test, axis = -1)

    y_out_test = test_amplitudes

    # Prediction
    out_predict = model.predict(y_in_test)
    out_predict = (out_predict*y_span) + y_min
    out_predict = tools.no_negative_concentration(out_predict,names,y_span,y_min)


    #SNR vs R2 plot testing
    mask = []
    mask.append(np.where(test_snr[:]>=4.5)[0])
    mask.append(np.where((test_snr[:]<5.5) & (test_snr[:,-1]>=4.5))[0])
    mask.append(np.where((test_snr[:]<4.5) & (test_snr[:,-1]>=3.5))[0])
    mask.append(np.where((test_snr[:]<3.5) & (test_snr[:,-1]>=2.5))[0])
    mask.append(np.where((test_snr[:]<2.5) & (test_snr[:,-1]>=1.5))[0])
    mask.append(np.where(test_snr[:]<1.5)[0])
    snrvr2 = []
    for ii in range(len(mask)):
        snrvr2.append(tools.np_rscore(y_out_test[mask[ii],:],out_predict[mask[ii],:]))
    snrvr2 = np.array(snrvr2)
    snrvr2 = snrvr2*snrvr2

    t = time.time()
    pdf = PdfPages("%s/heatmap_SNRvR2_testing.pdf" %(model_name))
    fig, ax = plt.subplots()
    fig.set_size_inches(20, 20)
    plt.subplots_adjust(left=0.15, right=0.85)
    plt.title('R2 in function of SNR', fontsize=30)
    cmap = matplotlib.cm.magma
    cmap.set_over('w')
    snrvr2 = snrvr2.transpose((1,0))
    im = ax.imshow(snrvr2,vmin=0., vmax=1.,cmap=cmap)
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.tick_params(labelsize=30)
    cbar.ax.set_ylabel('R2', rotation=-90, va="bottom", fontsize=30)
    for jj in range(snrvr2.shape[0]):
        for kk in range(snrvr2.shape[1]):
            r2text = ax.text(kk,jj,str(np.round(snrvr2[jj,kk],3)),ha="center", va="center", color="k", fontsize=20)
    ax.set_yticks(np.arange(snrvr2.shape[0]))
    ax.set_yticklabels(names[:], fontsize=30)
    ax.set_xticks(np.arange(snrvr2.shape[1]))
    ax.set_xticklabels(mask_name, fontsize=30)
    ax.grid(False)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",rotation_mode="anchor")
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close()
    pdf.close()
    elapsed_snr = np.round(time.time() - t, decimals=4)
    print("Time to plot testing snr vs r2: %s [s]"%(elapsed_snr))


#### Sending news on telegram
    time_tot = np.round(time.time() - time_initial, decimals=4)

    Dico_msg = {1: "Model name: "+model_name.split('/')[-2],
                2: "File name: "+filename.split('/')[-1],
                3: "Total time of running script [s]: "+str(datetime.timedelta(seconds=time_tot)),
                4: "Time to train CNN [s]: "+str(datetime.timedelta(seconds=elapsed_train)),
                5: "R2 score: %s" %(train_history['val_R2'][-1])}

    Dico_msg_2 = {1: "Nbatch :"+str(Nbatch),
                  2: "Nepochs :"+str(Nepochs),
                  3: "nNeuron :"+str(nNeuron),
                  4: "nFilters :"+str(nFilters),
                  5: "GPUpartition :"+str(GPUpartition),
                  6: "dropOut :"+str(dropOut),
                  7: "optimizer :"+str(optimizer),
                  8: "tLayer :"+str(tLayer),
                  9: "regularizer :"+str(regularizer),
                  10: "nevent_test :"+str(nevent_test)}

    msg_to_send = ''
    msg_to_send += 'Script train_Metab.py has finished. \n \n'
    msg_to_send += 'Informations: \n'
    for keys in Dico_msg.keys():
        msg_to_send +='  %s \n'%(Dico_msg.get(keys))
    msg_to_send += '\n\n'
    msg_to_send += 'Options of creation: \n'
    for keys in Dico_msg_2.keys():
        msg_to_send +='  %s \n'%(Dico_msg_2.get(keys))

    print('\n\n'+msg_to_send)
    tools.appreciation()





if __name__ == '__main__':
    main()
