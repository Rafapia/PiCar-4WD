import picar_4wd as fc
import time
import numpy as np

speed = 40

def main():
    while True:
        if fc.get_status_at(0, 30) != 2:
            t = np.random.uniform(0.2, 2)
            fc.turn_right(speed)
            time.sleep(t)
        else:
            fc.forward(speed)
            time.sleep(0.05)

if __name__ == "__main__":
    try: 
        main()
    finally: 
        fc.stop()