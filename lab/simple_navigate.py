import picar_4wd as fc
import time
import numpy as np

speed = 10
dist = 20
turn_time = 4.7

def scan(min=-120, max=120, num_scans=9):
    step = (max - min)/(num_scans-1)
    dists = []
    for i in range(0, num_scans):
        x = fc.get_distance_at(min + i*step)
        dists.append(x if x >= 0 else 999)
        time.sleep(0.05)
    return dists

def main():
    fc.get_distance_at(0)
    while True:
        dists = scan()

        center_idx = len(dists)//2

        center_dist = min(dists[center_idx-2:center_idx+3])
        right_dist = sum(dists[:center_idx])/(len(dists)//2)
        left_dist = sum(dists[-center_idx:])/(len(dists)//2)
        print(f"L: {left_dist}, C: {center_dist}, R: {right_dist}" )
        
        if center_dist < 20:

            if right_dist < 20 and left_dist < 20:
                fc.backward(speed)
                print("BACKWARD")
                time.sleep(1.5)
                fc.stop()
            else:
                if right_dist > left_dist:
                    fc.turn_right(speed)
                    print("RIGHT")
                    time.sleep(0.125*turn_time)
                    fc.stop()
                else:
                    fc.turn_left(speed)
                    print("LEFT")
                    time.sleep(0.125*turn_time)
                    fc.stop()
        else:
            fc.forward(speed)
            print("FORWARD")
            time.sleep(0.5)
            fc.stop()

if __name__ == "__main__":
    try: 
        main()
    finally: 
        fc.stop()