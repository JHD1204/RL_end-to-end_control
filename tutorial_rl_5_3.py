import glob
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
import random
import time
import numpy as np
import cv2
import math
from collections import deque

from tensorflow import keras
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Flatten, AveragePooling2D, Dense, Input

import tensorflow as tf
# import keras.backend.tensorflow_backend as backend  # error
# from threading import Thread

from tqdm import tqdm

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
import carla

SHOW_PREVIEN = False
IM_WIDTH = 640
IM_HEIGHT = 480
SECONDS_PER_EPISODE = 10
REPLAY_MEMORY_SIZE = 5000
MIN_REPLAY_MOMORY_SIZE = 1000
MINIBATCH_SIZE = 16
PREDICTION_BATCH_SIZE = 1
TRAINING_BATCH_SIZE = MINIBATCH_SIZE // 4
UPDATE_TARGET_EVERY = 5
MODEL_NAME = "CNN_Model"

MEMORY_FRACTION = 0.8
MIN_REWARD = -200

EPISODES = 10000

DISCOUNT = 0.99
epsilon = 1
EPSILON_DECAY = 0.999
MIN_EPSILON = 0.001

AGGREGATE_STATS_EVERY = 100

ACTION_DIM = 9

# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Own Tensorboard class
class ModifiedTensorBoard(TensorBoard):

    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.create_file_writer(self.log_dir)
        self._log_write_dir = self.log_dir

    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        # pass
        self.model = model
        self._train_dir = os.path.join(self._log_write_dir, 'train')
        self._train_step = self.model._train_counter
        self._val_dir = os.path.join(self._log_write_dir, 'validation')
        self._val_step = self.model._test_counter
        self._should_write_train_graph = False

    # Overrided, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Overrided
    # We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    # Overrided, so won't close writer
    def on_train_end(self, _):
        pass

    # Custom method for saving own metrics(指标)
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        self._write_logs(stats, self.step)

    def _write_logs(self, logs, index):
        with self.writer.as_default():
            for name, value in logs.items():
                tf.summary.scalar(name, value, step=index)
                self.step += 1
                self.writer.flush()


class CarEnv:
    SHOW_CAM = SHOW_PREVIEN
    THROTTLE = 1.0
    BRAKE = 0.6
    STEER_AMT = 0.5
    DESIRED_SPEED = 30

    im_width = IM_WIDTH
    im_height = IM_HEIGHT
    actor_list = []

    front_camera = None
    collision_list = []

    def __init__(self):
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(2.0)

        # Once we have a client we can retrieve the world that is currently running.
        self.world = self.client.get_world()
        # The world contains the list blueprints that we can use for adding new actors into the simulation.
        self.blueprint_library = self.world.get_blueprint_library()
        # Now let's filter all the blueprints of type 'vehicle' and choose one at random.
        # print(blueprint_library.filter('vehicle'))
        self.model_3 = self.blueprint_library.filter("model3")[0]

    def reset(self):
        self.collision_hist = []
        self.actor_list = []

        self.transform = random.choice(self.world.get_map().get_spawn_points())
        self.vehicle = self.world.spawn_actor(self.model_3, self.transform)
        self.actor_list.append(self.vehicle)

        self.rgb_cam = self.blueprint_library.find("sensor.camera.rgb")
        self.rgb_cam.set_attribute("image_size_x", "{}".format(self.im_width))
        self.rgb_cam.set_attribute("image_size_y", "{}".format(self.im_height))
        self.rgb_cam.set_attribute("fov", "110")

        transform = carla.Transform(carla.Location(x=2.5, z=0.7))
        self.sensor = self.world.spawn_actor(self.rgb_cam, transform, attach_to=self.vehicle)
        self.actor_list.append(self.sensor)
        self.sensor.listen(self.process_img)

        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
        # sleep to get things started and to not detect a collision when the car spawns/falls from sky.
        time.sleep(4)

        col_sensor = self.blueprint_library.find("sensor.other.collision")
        self.col_sensor = self.world.spawn_actor(col_sensor, transform, attach_to=self.vehicle)
        self.actor_list.append(self.col_sensor)
        self.col_sensor.listen(self.collision_data)

        while self.front_camera is None:
            time.sleep(0.01)

        self.episode_start = time.time()

        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))

        return self.front_camera

    def collision_data(self, event):
        self.collision_hist.append(event)

    def process_img(self, image):
        i = np.array(image.raw_data)
        # np.save("iout.npy", i)
        i2 = i.reshape((self.im_height, self.im_width, 4))
        i3 = i2[:, :, :3]
        if self.SHOW_CAM:
            cv2.imshow("", i3)
            cv2.waitKey(1)
        self.front_camera = i3

    def step(self, action):
        # set actions
        if action == 0:
            self.vehicle.apply_control(carla.VehicleControl(throttle=0, steer=0))
        elif action == 1:
            self.vehicle.apply_control(carla.VehicleControl(throttle=0, steer=self.STEER_AMT))
        elif action == 2:
            self.vehicle.apply_control(carla.VehicleControl(throttle=0, steer=-self.STEER_AMT))
        elif action == 3:
            self.vehicle.apply_control(carla.VehicleControl(throttle=self.THROTTLE, steer=0))
        elif action == 4:
            self.vehicle.apply_control(carla.VehicleControl(brake=self.BRAKE, steer=0))
        elif action == 5:
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.5*self.THROTTLE, steer=self.STEER_AMT))
        elif action == 6:
            self.vehicle.apply_control(carla.VehicleControl(brake=0.5*self.BRAKE, steer=self.STEER_AMT))
        elif action == 7:
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.5*self.THROTTLE, steer=-self.STEER_AMT))
        elif action == 8:
            self.vehicle.apply_control(carla.VehicleControl(brake=0.5*self.BRAKE, steer=-self.STEER_AMT))

        # set rewards and done
        done = False
        # reward for speed tracking
        v = self.vehicle.get_velocity()
        v_kmh = int(3.6 * math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2))
        r_speed = -abs(v_kmh - self.DESIRED_SPEED)

        # reward for collision
        r_collision = 0
        if len(self.collision_hist) != 0:
            done = True
            r_collision = -1

        # reward for steering
        r_steer = -self.vehicle.get_control().steer**2

        # reward for out of lane
        vehicle_location = self.vehicle.get_location()
        vehicle_x, vehicle_y = vehicle_location.x, vehicle_location.y
        way_point = self.world.get_map().get_waypoint(vehicle_location)
        way_x, way_y = way_point.transform.location.x, way_point.transform.location.y
        way_yaw = way_point.transform.rotation.yaw
        lane_width = way_point.lane_width

        # calculate distance from (x, y) to waypoints
        vec = np.array([vehicle_x - way_x, vehicle_y - way_y])
        lv = np.linalg.norm(np.array(vec))
        w = np.array([np.cos(way_yaw / 180 * np.pi), np.sin(way_yaw / 180 * np.pi)])
        cross = np.cross(w, vec / lv)
        distance = - lv * cross

        if distance < lane_width / 2:
            r_out = -distance / lane_width
        else:
            done = True
            r_out = -2

        reward = 200 * r_collision + 1 * r_speed + 5 * r_steer + 10 * r_out

        if self.episode_start + SECONDS_PER_EPISODE < time.time():
            done = True

        return self.front_camera, reward, done, None


class DQNAgent:
    def __init__(self):
        self.model = self.create_model()
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        self.log_dir = "logs/{}{}".format(MODEL_NAME, time.strftime('%Y-%m-%d-%H-%M-%S'))
        self.tensorboard = ModifiedTensorBoard(log_dir=self.log_dir)
        self.target_update_counter = 0

        self.terminate = False
        self.last_logged_episode = 0

    def create_model(self):

        input_image = Input(shape=(IM_HEIGHT, IM_WIDTH, 3))
        layer1 = Conv2D(64, (3, 3), padding='same', activation='relu')(input_image)
        layer2 = AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='same')(layer1)

        layer3 = Conv2D(64, (3, 3), padding='same', activation='relu')(layer2)
        layer4 = AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='same')(layer3)

        layer5 = Conv2D(64, (3, 3), padding='same', activation='relu')(layer4)
        layer6 = AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='same')(layer5)

        layer7 = Flatten()(layer6)
        output_q = Dense(ACTION_DIM, activation='linear')(layer7)

        model = Model(inputs=input_image, outputs=output_q)

        model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=["accuracy"])

        return model

    def update_replay_memory(self, transition):
        # transition = (current_state, action, reward, new_state, done)
        self.replay_memory.append(transition)

    def train(self):
        if len(self.replay_memory) < MIN_REPLAY_MOMORY_SIZE:
            return

        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        current_states = np.array([transition[0] for transition in minibatch]) / 255
        current_qs_list = self.model.predict(current_states, PREDICTION_BATCH_SIZE)

        new_current_states = np.array([transition[3] for transition in minibatch]) / 255
        future_qs_list = self.target_model.predict(new_current_states, PREDICTION_BATCH_SIZE)

        X = []
        y = []

        for index, (current_state, action, reward, new_state, done) in enumerate(minibatch):
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward

            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            X.append(current_state)
            y.append(current_qs)

        log_this_step = False
        if self.tensorboard.step > self.last_logged_episode:
            log_this_step = True
            self.last_logged_episode = self.tensorboard.step

        self.model.fit(np.array(X) / 255, np.array(y), batch_size=TRAINING_BATCH_SIZE,
                       verbose=0, shuffle=False, callbacks=[self.tensorboard] if log_this_step else None)

        if log_this_step:
            self.target_update_counter += 1

        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

    def get_qs(self, state):
        return self.model.predict(np.array(state).reshape(-1, *state.shape) / 255)[0]

    def get_qs_init(self, state):
        return self.model.predict(np.array(state).reshape(-1, *state.shape))[0]

if __name__ == "__main__":
    FPS = 60
    # For stats
    ep_rewards = [-200]

    # For more repetitive results
    random.seed(1)
    np.random.seed(1)
    tf.compat.v1.set_random_seed(1)

    # Memory fraction, used mostly when training multiple agents(8)
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = MEMORY_FRACTION
    session = tf.compat.v1.Session(config=config)
    tf.compat.v1.keras.backend.set_session(session)

    # Create models folder
    if not os.path.isdir("models"):
        os.makedirs("models")

    # Create agent and environment
    agent = DQNAgent()
    env = CarEnv()

    # Start training thread and wait for training to be initialized
    # trainer_thread = Thread(target=agent.train_in_loop, daemon=True)
    # trainer_thread.start()

    # Initialize predictions - forst prediction takes longer as of initialization that has to be done
    # It's better to do a first prediction then before we start iterating over episode steps
    agent.get_qs_init(np.ones((env.im_height, env.im_width, 3)))

    # Iterate over episodes
    for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit="episodes"):

        env.collision_hist = []

        # Update tensorboard step every episode
        agent.tensorboard.step = episode

        # Restarting episode - reset episode reward and step number
        episode_reward = 0
        step = 1

        # Reset environment and get initial state
        current_state = env.reset()

        # Reset flag and start iterating until episode ends
        done = False
        episode_start = time.time()

        # Play for given number of seconds only
        while True:

            # This part stays mostly the same, the change is to query a model for Q values
            if np.random.random() > epsilon:
                # Get action from Q table
                action = np.argmax(agent.get_qs(current_state))
            else:
                # Get random action
                action = np.random.randint(0, ACTION_DIM)
                # This takes no time, so we add a delay matching 60 FPS (prediction above takes longer)
                time.sleep(1 / FPS)

            new_state, reward, done, _ = env.step(action)

            # Transform new continous state to new discrete state and count reward
            episode_reward += reward

            agent.train()

            # Every step we update replay memory
            agent.update_replay_memory((current_state, action, reward, new_state, done))

            current_state = new_state
            step += 1

            if done:
                break

        # End of episode - destroy agents
        for actor in env.actor_list:
            actor.destroy()

        # Append episode reward to a list and log stats (every given number of episodes)
        ep_rewards.append(episode_reward)

        if not episode % AGGREGATE_STATS_EVERY or episode == 1:
            average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:]) / len(ep_rewards[-AGGREGATE_STATS_EVERY:])
            min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
            max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
            agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward,
                                           epsilon=epsilon)

            # Save model, but only when min reward is greater or equal a set value
            # if min_reward >= MIN_REWARD:
            #     agent.model.save('models/{}__{:_>7.2f}max_{:_>7.2f}avg_{:_>7.2f}min__{}.model'
            #                      .format(MODEL_NAME, max_reward, average_reward, min_reward, time.strftime('%Y-%m-%d %H:%M:%S')))

        # Decay epsilon
        if epsilon > MIN_EPSILON:
            epsilon *= EPSILON_DECAY
            epsilon = max(MIN_EPSILON, epsilon)

    # Set termination flag for training thread and wait for it to finish
    agent.terminate = True
    # trainer_thread.join()
    agent.model.save('models/{}_{}.model'.format(MODEL_NAME, time.strftime('%Y-%m-%d %H:%M:%S')))






