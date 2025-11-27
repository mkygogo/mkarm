
from follower_mkarm import MKFollower, MKFollowerConfig
from leader_mkarm import MKLeader, MKLeaderConfig
import time


follower_config = MKFollowerConfig(
    port="/dev/ttyACM0",
    joint_velocity_scaling=1.0,
)

leader_config = MKLeaderConfig(
    port="/dev/ttyUSB0"
)

leader = MKLeader(leader_config)
leader.connect()

follower = MKFollower(follower_config)
follower.connect()

freq = 200 # Hz

def action_filter(action):
    new_action = {'joint_1.pos': action["joint_1.pos"], 
                'joint_2.pos': 0, 
                'joint_3.pos': 0, 
                'joint_4.pos': 0, 
                'joint_5.pos': 0, 
                'joint_6.pos': 0}

    return new_action

try:
    while True:
        action = leader.get_action()

        #action = action_filter(action)
        print(action)

        follower.send_action(action)    
        time.sleep(1/freq)
except KeyboardInterrupt:
    print("\nStopping teleop...")
    leader.disconnect()
    follower.disconnect()
