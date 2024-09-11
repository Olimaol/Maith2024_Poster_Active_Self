# Poster Active Self

This is the code repository for the poster "A Neurocomputational Model of Basal Ganglia-Intralaminar Nuclei Interactions: Implications for Attentional Orienting and Sense of Agency".
[View Poster](./poster_files/pdf/poster_active_self_BGprediction.pdf)

Authors:
* Oliver Maith (oliver.maith@informatik.tu-chemnitz.de)
* Erik Syniawa (erik.syniawa@informatik.tu-chemnitz.de)


# How to use the simulation Code

## Prerequirements
* Linux
* Within conda environment
* Python>=3.10
* [CompNeuroPy](https://olimaol.github.io/CompNeuroPy/installation/)==1.0.4
* [ANNarchy](https://annarchy.github.io/Installation.html)==4.8.1

## Quick Overview
* 'core_parameters': all global parameters
* 'core_model': how the ANNarchy model is created (Poster content is related to the 'prediction_model')
* 'core_experiment': how the experimental paradigm is implemented
* 'simulation_run', 'simulation', and 'simulation_analyze': scripts for running the task with the model + creating figures

## Simulate and Analyze
Simulating and analyzing is started using 'simulation_run.py'. You can start both together or separately.

### Simulate / create the date:
Make sure in the file 'core_paramters': `SIMULATION_SIMULATE = True` and `SIMULATION_OPTIMIZE = False`.
Then run the pyhton file 'simulation_run.py'.

### Analyze / create figures:
Make sure in the file 'core_paramters': `SIMULATION_ANALYZE = True`, `SIMULATION_PF_RESPONSE_PLOT = True`, and `SIMULATION_SINGLE_TRIAL_PLOT = True`.
Set `SIMULATION_SIMULATE = False` if you already created data / dont want to simulate again.
Then run the python file 'simulation_run.py'.

Data and figures are stored in the 'simulation_data/' folder (as defined in the 'core_paramters' file).
