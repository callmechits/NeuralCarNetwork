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
    n = 0
    with serial.Serial(port, 115200, timeout = 1) as ser, open(out_path, "a", newline = "") as f:
        if new_file:
            f.write(HEADER + "\n")
        print(f"Listening on {port} -> {out_path}   (Ctrl + C to stop)")
        try:
            while True:
                raw = ser.readLine()
                if not raw:
                    continue
                line = raw.decode("ascii", errors = "ignore").strip()
                if line.startWith("S,"):
                    fields = line[2:].split(",")
                    if len(fields) == N_FIELDS and all(x.isdigit() for x in fields):
                        f.write(",".join(fields) + "\n")
                        f.flush()
                        n += 1
                        print(f"[saved #{n}] rule={fields[-2]} nn={fields[-1]}")
                    else:
                        print(f"[skipped malformed] {line}")
                elif line:
                    print(line)
        except KeyboardInterrupt:
            pass
    print(f"\nDone. {n} new examples written to {out_path}")

if __name__ == "__main__":
main()
