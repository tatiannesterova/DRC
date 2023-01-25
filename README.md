# DRC

*****APD Dynamic Restitution Curve*****

This script simulates and plots the APD restitution curve according to a protocol with gradually decreasing stimulation period (dynamic protocol) in rat ventricular cardiomyocyte models in control (1C-4C) and under leptin (1L-4L).


***Dependencies Installation***

The code requires:
 - Python 3 interpreter with packages:
     - numpy
     - pandas
     - matplotlib
     - seaborn
 - Myokit Python-based software package (installation instructions and documentation are available on the developer's website: http://myokit.org/)


***Run Variants***

1. run the script in the model "updated_Gattoni2016_model_1C.mmt" through Myokit IDE (button "Run embedded script")
2. run the python script: python3 dynamic_protocol_calculation.py

The result of both scripts is the figure "rc.png" similar to Figure 2 from the article, which shows the APD90 restitution curves in the Models 1–4 in the control (C, blue) and under leptin (L, red).


***Model***

Section [[model]] of the model "updated_Gattoni2016_model_1C.mmt" contains right hand side descriptions of the 1C model for a 1 Hz stimulation frequency. This model is a modification of the Gattoni2016 ionic model of ventricular rat cardiomyocytes. The original Gattoni2016 model was converted into Myokit format from the CellML file (https://models.cellml.org/workspace/285) provided by the model authors in their article (https://doi.org/10.1113/JP272011).

The steady-state variables of models 1C-4C, 1L-4L are given in the table "./model_parameters/steady_states.csv"

The modified parameter values of models 1C-4C and 1L-4L at each stimulation period from 1000 to 90 ms are presented in the tables "./model_parameters/modelXX_parameters.csv", where XX is the model label.

The steady-state variables and parameter values are passed to the "run_dynamic_0D_protocol" function, where they replace the corresponding parameter values from the model description in [[model]]. 


***Script Parameters***

The script parameters are after the comment "# calculation parameters" are: 
 - on the "Embedded script" tab when running through the Myokit IDE
 - at the end of the module "dynamic_protocol_calculation.py" when running via python

Parameters:
 - is_datalog_saver - True or False, if True - datalog is saved in npz. Datalog is dict: {model_label : list containing myokit datalog simulations for different stimulation periods, 
                     ordered according to the parameter table indexes}
 - model_ids - list of model numbers involved in the calculations (for example, [1,2,3,4], [1,4], [3]) 
 - model_types - ['C', 'L'], ['C'] or ['L'] - model types involved in the calculations (C - control, L - under application of leptin)
 - max_step -- maximum step size for CVODE. (In the article the 0.005 max_step was used and the computations took some time. The default value in Myokit is None which here stands for infinity; the computations are much faster but the results may differ)
 - n_stimulus -- the number of consecutive calculated stimuli without changing the stimulation period (set to 40)
 - log -- variables for which the result should be written


***Implementation of a Dynamic Stimulation Protocol***

see the documentation lines in the module dynamic_protocol_calculation
