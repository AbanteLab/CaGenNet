###############################################################################################
# dependancies
###############################################################################################
#%%

# Load general deps
import os
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split

# load torch deps
from torch import from_numpy

###############################################################################################
# arguments
###############################################################################################
#%%
# Create the parser
parser = argparse.ArgumentParser(description="""
    This script partitions a dataset given a random seed. It reads and processes ROI and trace 
    data, normalizes the data, and splits it into training and validation sets. The resulting 
    partitions are saved to a specified output directory.
    
    Arguments:
        -s, --seed (int, optional): RNG seed (default: 0)
""")

# required arguments
parser.add_argument(
    "-s",
    "--seed", 
    type=int, 
    required=False, 
    default=0,
    help="RNG seed (default: 42)"
)

# Parse the arguments
args = parser.parse_args()

# Access the arguments
fluo_noise = 0.0
seed = args.seed            # seed=1

###############################################################################################
# functions
###############################################################################################
#%%
def superprint(message):
    # Get the current date and time
    now = datetime.now()
    # Format the date and time
    timestamp = now.strftime("[%Y-%m-%d %H:%M:%S]")
    # Print the message with the timestamp
    print(f"{timestamp} {message}")

def read_roi(path,sample,group):
    
    # read in 
    x = pd.read_csv(path)
    
    # assign sample and group
    x["sample"] = sample
    x["group"] = group
    
    return(x)

def read_trace(path,thin=False):
    
    # read in 
    x = pd.read_csv(path,header=None,sep="\t")
    x = x.iloc[:, :30000]
    
    # thinning
    if thin:
        x = x.iloc[:, ::2]
    
    return(x)

###############################################################################################
# IO
###############################################################################################
#%%
# dirs
data_dir = "/pool01/data/private/soriano_lab/simulations/set_3"
rs_dir = f"{data_dir}/Flat_RS/L_0.1/EI_ratio_0.8/Iz_param_noise_0.05/dt_0.001/fluo_noise_{fluo_noise}"
ch_dir = f"{data_dir}/Flat_CH/L_0.1/EI_ratio_0.8/Iz_param_noise_0.05/dt_0.001/fluo_noise_{fluo_noise}"
ib_dir = f"{data_dir}/Flat_IB/L_0.1/EI_ratio_0.8/Iz_param_noise_0.05/dt_0.001/fluo_noise_{fluo_noise}"

# output dirs
bse_dir = "/pool01/projects/abante_lab/snDGM/eccb2025/simulations/set_3/AVAE_2_lstm_lin"
part_dir = f"{bse_dir}/partitions/fluo_noise_{fluo_noise}/"

# make output dir
os.makedirs(part_dir, exist_ok=True)

# output files
suff = f'_seed_{seed}'
data_file = f'{part_dir}/partition_{suff}.npz'

###############################################################################################
# ROI
###############################################################################################
#%%
superprint("Loading data")

# ROI files
roi_chat_1_files = [f"{ch_dir}/Flat_CH_L_0.1_rois_{i}.txt" for i in range(1,6)]

# read in
roi_chat_1 = [read_roi(roi_chat_1_files[i],f"{i+1}","CH") for i in range(0,5)]

# merge dataframes
roi_chat_1 = pd.concat(roi_chat_1, axis=0)

# ROI files
roi_int_1_files = [f"{ib_dir}/Flat_IB_L_0.1_rois_{i}.txt" for i in range(1,6)]

# read in
roi_int_1 = [read_roi(roi_int_1_files[i],f"{i+1}","IB") for i in range(0,5)]

# merge dataframes
roi_int_1 = pd.concat(roi_int_1, axis=0)

# ROI files
roi_reg_1_files = [f"{rs_dir}/Flat_RS_L_0.1_rois_{i}.txt" for i in range(1,6)]

# read in
roi_reg_1 = [read_roi(roi_reg_1_files[i],f"{i+1}","RS") for i in range(0,5)]

# merge dataframes
roi_reg_1 = pd.concat(roi_reg_1, axis=0)

# concatenate
roi_df = pd.concat([roi_chat_1,roi_int_1,roi_reg_1])

###############################################################################################
# Traces
###############################################################################################
#%%
# files
trace_chat_1_files = [f"{ch_dir}/Flat_CH_L_0.1_calcium_{i}.txt.gz" for i in range(1,6)]

# read in
trace_chat_1 = [read_trace(trace_chat_1_files[i],thin=True) for i in range(0,5)]

# merge dataframes
trace_chat_1 = pd.concat(trace_chat_1, axis=0)

# files
trace_int_1_files = [f"{ib_dir}/Flat_IB_L_0.1_calcium_{i}.txt.gz" for i in range(1,6)]

# read in
trace_int_1 = [read_trace(trace_int_1_files[i],thin=True) for i in range(0,5)]

# merge dataframes
trace_int_1 = pd.concat(trace_int_1, axis=0)

# files
trace_reg_1_files = [f"{rs_dir}/Flat_RS_L_0.1_calcium_{i}.txt.gz" for i in range(1,6)]

# read in
trace_reg_1 = [read_trace(trace_reg_1_files[i],thin=True) for i in range(0,5)]

# merge dataframes
trace_reg_1 = pd.concat(trace_reg_1, axis=0)

# concatenate
trace_df = pd.concat([trace_chat_1,trace_int_1,trace_reg_1])

###############################################################################################
# prep data
###############################################################################################
#%%
superprint("Saving partitioned data")

# define numpy arrays
xdata = trace_df.to_numpy()
ydata = roi_df.to_numpy()

# work with float32
xdata = xdata.astype(np.float32)

# normalization
xdata = (xdata-xdata.min())/(xdata.max()-xdata.min()) - 0.5

# split the data into training and validation sets (seed was 42 in the original code)
xtrain, xval, ytrain, yval = train_test_split(xdata, ydata, test_size=0.20, random_state=seed)

# store data
np.savez(data_file, xtrain=xtrain, xval=from_numpy(xval), ytrain=ytrain, yval=yval)
