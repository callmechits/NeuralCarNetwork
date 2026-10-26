import os
import serial
import sys

N_DIST = 13                # Number of records every sweep
N_FIELDS = N_DIST + 2      # 1 class for rule + 1 class for nn
HEADER = ",".join([f"d{i}" for i in range(N_DIST)] + ["rule", "nn"])

def main():
    if len(sys.argv) < 2:
        sys.exit("usage: py capture.py <COM port> [output.csv]")        
# To find COM port, open arduino and look at the options allowed then select 
# the one which has the car connected to it
    port = sys.argv[1]
    here = os.path.dirname(os.path.abspath(__file__))
    out_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join(here, "..", "data", "sweeps.csv")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok = True)

    new_file = not os.path.exists(out_path) or os.path.getsize(out_path) == 0
