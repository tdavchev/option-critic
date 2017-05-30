
import os
import sys
import gym
from gym import wrappers

# ========================================
#   Utility Parameters
# ========================================
# Render gym env during training
RENDER_ENV = False
# Use Gym Monitor
GYM_MONITOR_EN = True
# Gym environment
ENV_NAME = 'Pong-v0'
# Directory for storing gym results
MONITOR_DIR = './results/gym_ddpg'
# Directory for storing tensorboard summary results
SUMMARY_DIR = './results/tf_ddpg'
# Seed
RANDOM_SEED = 1234

# ==========================
#   Training Parameters
# ==========================
# Update Frequency
update_freq = 4
# Max training steps
MAX_EPISODES = 8#000
# Max episode length
MAX_EP_STEPS = 2500#00
# Base learning rate for the Actor network
ACTOR_LEARNING_RATE = 0.00025
# Base learning rate for the Critic Network
CRITIC_LEARNING_RATE = 0.00025
# Contributes to the nitial random walk
MAX_START_ACTION_ATTEMPTS = 30
# Update params
FREEZE_INTERVAL = 10#000
# Discount factor
GAMMA = 0.99
# Soft target update param
TAU = 0.001
# Starting chance of random action
START_EPS = 1
# Final chance of random action
END_EPS = 0.05
# How many steps of training to reduce startE to endE.
ANNEALING = 1000000
# Number of options
OPTION_DIM = 8
# Pretrain steps
PRE_TRAIN_STEPS = 5#0000
# Size of replay buffer
BUFFER_SIZE = 100#0000
# Minibatch size
MINIBATCH_SIZE = 32

# ===========================
#   Tensorflow Summary Opself.model
# ===========================
def build_summaries():
    summary_ops = tf.Summary()
    episode_reward = tf.Variable(0.)
    tf.summary.scalar("DOCA/Reward", episode_reward)
    episode_ave_max_q = tf.Variable(0.)
    tf.summary.scalar("DOCA/Qmax Value", episode_ave_max_q)
    episode_termination_ratio = tf.Variable(0.)
    tf.summary.scalar("DOCA/Term Ratio", episode_termination_ratio)
    tot_reward = tf.Variable(0.)
    tf.summary.scalar("DOCA/Total Reward", tot_reward)
    cum_reward = tf.Variable(0.)
    tf.summary.scalar("DOCA/Cummulative Reward", tot_reward)
    rmng_frames = tf.Variable(0.)
    tf.summary.scalar("DOCA/Remaining Frames", rmng_frames)

    summary_vars = [episode_reward, episode_ave_max_q, episode_termination_ratio, tot_reward, cum_reward, rmng_frames]
    summary_ops = tf.summary.merge_all()

    return summary_ops, summary_vars

def get_reward(reward):
    if reward < 0:
        score = -1
    elif reward > 0:
        score = 1
    else:
        score = 0

    return score, reward

# ===========================
#   Agent Training
# ===========================
def train(sess, env, option_critic):#, critic):
    # Set up summary Ops
    summary_ops, summary_vars = build_summaries()
    np.random.seed(RANDOM_SEED)

    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter(SUMMARY_DIR, sess.graph)

    # Initialize target network weights
    option_critic.update_target_network()
    # critic.update_target_network()

    # State processor
    state_processor = StateProcessor()

    # Initialize replay memory
    replay_buffer = ReplayBuffer(84, 84, RANDOM_SEED, BUFFER_SIZE, 4)
    # Set the rate of random action decrease.
    eps = START_EPS
    stepDrop = (START_EPS - END_EPS)/ANNEALING

    total_steps = 0
    print_option_stats = False

    action_counter = [{j:0 for j in range(env.action_space.n)} for i in range(OPTION_DIM)]
    total_reward = 0
    for i in xrange(MAX_EPISODES):
        s = env.reset() # note I'm using only one step, original uses 4
        s = state_processor.process(sess, s)
        s = np.stack([s] * 4, axis=2)
        current_option = 0
        current_action = 0
        new_option = np.argmax(option_critic.predict(s))#+ (1./(1. + i)) # state has more than 3 features in pong
        done = False
        termination = True
        ep_reward = 0
        ep_ave_max_q = 0
        termination_counter = 0
        since_last_term = 1

        for j in xrange(MAX_EP_STEPS):
            # if RENDER_ENV:
            #     env.render()

            if termination:
                if print_option_stats:
                    print "terminated -------", since_last_term,

                termination_counter += 1
                since_last_term = 1
                current_option = np.random.randint(OPTION_DIM) if np.random.rand() < eps else new_option
            else:
                if print_option_stats:
                    print "keep going"

                since_last_term += 1

            action_probs = option_critic.predict_action([s], np.reshape(current_option, [1,1]))[0]
            current_action = np.argmax(np.random.multinomial(1, action_probs))
            if print_option_stats:
                 print current_option
                 if True:
                    action_counter[current_option][current_action] += 1
                    data_table = []
                    option_count = []
                    for ii, aa in enumerate(action_counter):
                        s3 = sum([aa[a] for a in aa])
                        if s3 < 1:
                            continue

                        print ii, aa, s3
                        option_count.append(s3)
                        print [str(float(aa[a])/s3)[:5] for a in aa]
                        data_table.append([float(aa[a])/s3 for a in aa])
                        print

                    print

            s2, reward, done, info = env.step(current_action)
            s2 = state_processor.process(sess, s2)
            s2 = np.append(s[:,:,1:], np.expand_dims(s2, 2), axis=2)
            score, reward = get_reward(reward)
            total_steps += 1

            replay_buffer.add_sample(s[:,:,-1], current_option, score, done)

            term = option_critic.predict_termination([s2], [[current_option]])
            option_term_ps, Q = term[0], term[1]
            ep_ave_max_q += np.max(Q)
            new_option = np.argmax(Q)
            randomize = np.random.uniform(size=np.asarray([0]).shape)
            termination = option_term_ps if termination>=randomize else randomize
            if total_steps < PRE_TRAIN_STEPS:
               termination = 1
            
            if total_steps > PRE_TRAIN_STEPS:
                if eps > END_EPS:
                    eps -= stepDrop

                # done in the original paper, actor is trained on current data
                # critic trained on sampled one
                _ = option_critic.train_actor(
                        [s], [s2], np.reshape(current_option, [1, 1]), np.reshape(current_action, [1, 1]), np.reshape(score, [1, 1]), np.reshape(done+0, [1, 1]))

                if total_steps % (update_freq) == 0:
                    if RENDER_ENV:
                        env.render()

                    # Keep adding experience to the memory until
                    # there are at least minibatch size samples
                    if len(replay_buffer) > MINIBATCH_SIZE:
                        s_batch, o_batch, score_batch, s2_batch, done_batch = \
                            replay_buffer.random_batch(MINIBATCH_SIZE)

                        _ = option_critic.train_critic(
                                s_batch, s2_batch,  np.reshape(o_batch, [MINIBATCH_SIZE, 1]),  np.reshape(score_batch, [MINIBATCH_SIZE, 1]), np.reshape(done_batch+0, [MINIBATCH_SIZE, 1]))

                if total_steps % (FREEZE_INTERVAL) == 0:
                    # Update target networks
                    option_critic.update_target_network()

            s = s2
            ep_reward += reward
            total_reward += reward

            if done:
                summary_str = sess.run(summary_ops, feed_dict={
                    summary_vars[0]:ep_reward,
                    summary_vars[1]:ep_ave_max_q / float(j),
                    summary_vars[2]:float(termination_counter) / float(j),
                    summary_vars[3]:total_reward,
                    summary_vars[4]:total_reward/float(i+1),
                    summary_vars[5]:(MAX_EP_STEPS-j)
                })

                writer.add_summary(summary_str, i)
                writer.flush()

                break

        print '| Reward: %.2i' % int(ep_reward), " | Episode", i, \
            '| Qmax: %.4f' % (ep_ave_max_q / float(j)), ' | Cummulative Reward: %.1f' % (total_reward/(i+1)), \
            ' | %d Remaining Frames: ' %(MAX_EP_STEPS-j), \
            ' Epsilon: %.4f'%eps

def set_up_gym():
    env = gym.make(ENV_NAME)
    env.seed(RANDOM_SEED)

    if GYM_MONITOR_EN:
        if not RENDER_ENV:
            env = wrappers.Monitor(
                env, MONITOR_DIR, video_callable=None, force=True
            )
        else:
            env = wrappers.Monitor(env, MONITOR_DIR, force=True)

        # env.monitor.close()

    env.reset()
    #env.render()
    return env

def main(env):
    if not os.path.exists(MONITOR_DIR):
        os.makedirs(MONITOR_DIR)

    if not os.path.exists(SUMMARY_DIR):
        os.makedirs(SUMMARY_DIR)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    if state_dim == 210:
        # state_dim *= env.observation_space.shape[1] # for grey scale
        state_dim = 84 * 84 * 4
    # action_bound = env.action_space.high
    # Ensure action bound is symmetric
    # assert(env.action_space.high == -env.action_space.low)


    with tf.Session() as sess:
        tf.set_random_seed(RANDOM_SEED)
        # sess, h_size, temp, state_dim, action_dim, option_dim, action_bound, learning_rate, tau
        option_critic = OptionsNetwork(sess, 512, 1, state_dim, action_dim, 8, ACTOR_LEARNING_RATE, TAU, clip_delta=1)
        # critic = CriticNetwork(sess, state_dim, action_dim, CRITIC_LEARNING_RATE, TAU, actor.get_num_trainable_vars)

        train(sess, env, option_critic)#, critic)

    # if GYM_MONITOR_EN:
    #     env.monitor.close()

if __name__ == '__main__':
    env = set_up_gym()
    # circumvening TLS static error
    import tensorflow as tf

    execfile("models/option_critic_network.py")
    execfile("helper/buffer.py")
    execfile("helper/state_processor.py")

    main(env)
