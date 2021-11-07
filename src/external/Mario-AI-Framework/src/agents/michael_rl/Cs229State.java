package agents.michael_rl;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import engine.core.MarioForwardModel;
import engine.core.MarioSprite;
import engine.core.MarioWorld;
import engine.helper.SpriteType;

/**
 * Abstract representation of the game environment.
 * 
 * @author kunyi@stanford.edu (Kun Yi)
 */
public class Cs229State {
    public class LearningParams {

        public static int DEBUG = 0;

        /**
         * Total number of iterations to train for each configuration (mode/seed).
         */
        public static int NUM_TRAINING_ITERATIONS = 800;

        /**
         * Number of Mario modes to train.
         */
        public static int NUM_MODES_TO_TRAIN = 3;

        /**
         * Whether we should use a different random seed for training.
         */
        public static int NUM_SEEDS_TO_TRAIN = 1;

        /**
         * Number of evaluation iterations to run.
         */
        public static int NUM_EVAL_ITERATIONS = 10;

        /**
         * Whether we should use a different random seed for evaluation.
         */
        public static int EVAL_SEED = -1;

        /**
         * Exploration chance during evaluation.
         */
        public static final float EVAL_EXPLORATION_CHANCE = 0.01f;

        // E-GREEDY Q-LEARNING SPECIFIC VARIABLES
        /**
         * For e-greedy Q-learning, when taking an action a random number is checked
         * against the explorationChance variable: if the number is below the
         * explorationChance, then exploration takes place picking an action at random.
         * Note that the explorationChance is not a final because it is customary that
         * the exploration chance changes as the training goes on.
         */
        public static final float EXPLORATION_CHANCE = 0.3f;

        /**
         * The discount factor is saved as the gammaValue variable. The discount factor
         * determines the importance of future rewards. If the gammaValue is 0 then the
         * AI will only consider immediate rewards, while with a gammaValue near 1 (but
         * below 1) the AI will try to maximize the long-term reward even if it is many
         * moves away.
         */
        public static final float GAMMA_VALUE = 0.6f;
        // public static final float GAMMA_VALUE = 0.2f;

        /**
         * The learningRate determines how new information affects accumulated
         * information from previous instances. If the learningRate is 1, then the new
         * information completely overrides any previous information. Note that the
         * learningRate is not a final because it is customary that the exploration
         * chance changes as the training goes on.
         * 
         * The actual learning rate will decrease as the number of times the given state
         * and action are visited increase.
         */
        public static final float LEARNING_RATE = 0.8f;

        // Reward/state related params.
        /**
         * The minimum distance Mario must travel in a frame to receive distance reward,
         * and to indicate that Mario is moving instead of being stuck.
         */
        public static final int MIN_MOVE_DISTANCE = 2;

        /**
         * Mario will change to the stuck state if he has been stuck for
         * NUM_STUCK_FRAMES number of frames.
         */
        public static final int NUM_STUCK_FRAMES = 25;

        /**
         * Number of observation levels (window sizes).
         */
        public static final int NUM_OBSERVATION_LEVELS = 3;

        /**
         * Window size of each observation level.
         */
        public static final int[] OBSERVATION_SIZES = { 1, 3, 5 };

        /**
         * Scalers to apply to distance/elevation rewards when enemies are present in
         * the corresponding observation level.
         */
        public static final float[] ENEMIES_AROUND_REWARD_SCALER = { 0f, 0f, 0.15f };

        public static final class REWARD_PARAMS {
            public static final int distance = 2;
            public static final int elevation = 8;
            public static final int collision = -800;
            public static final int killedByFire = 60;
            public static final int killedByStomp = 60;
            public static final int stuck = -20;

            // Params below are not used.
            public static final int win = 0;
            public static final int mode = 0;
            public static final int coins = 0;
            public static final int flowerFire = 0;
            public static final int kills = 0;
            public static final int killedByShell = 0;
            public static final int mushroom = 0;
            public static final int timeLeft = 0;
            public static final int hiddenBlock = 0;
            public static final int greenMushroom = 0;
            public static final int stomp = 0;
        };

        // Logging related params.
        /**
         * Whether we should dump intermediate Q table values.
         */
        public static final boolean DUMP_INTERMEDIATE_QTABLE = false;

        /**
         * Whether we should load the final Q table trained from last time.
         */
        public static boolean LOAD_QTABLE = false;

        /**
         * The format of intermediate Q table dump filenames.
         */
        public static String QTABLE_NAME_FORMAT = "qt.%d.txt";

        /**
         * The filename of the final Q table dump.
         */
        public static String FINAL_QTABLE_NAME = "qt.final.txt";

        /**
         * The filename of scores dump.
         */
        public static String SCORES_NAME = "scores.txt";
    }

    private static final int MARIO_X = 9;
    private static final int MARIO_Y = 9;

    public final List<Field> fields = new ArrayList<Field>();

    // 0-small, 1-big, 2-fire.
    private Int marioMode = new Int("m", 2);

    // 0~8.
    private Int marioDirection = new Int("Dir", 8);
    private float marioX = 0;
    private float marioY = 0;

    private Int stuck = new Int("!!", 1);
    public static int stuckCount = 0;

    private Int onGround = new Int("g", 1);
    private Int canJump = new Int("j", 1);

    private Int collisionsWithCreatures = new Int("C", 1);
    private int lastCollisionsWithCreatures = 0;

    private BitArray[] enemies = new BitArray[LearningParams.NUM_OBSERVATION_LEVELS];

    // To keep track enemies killed in the observation window.
    private int[] enemiesCount = new int[LearningParams.NUM_OBSERVATION_LEVELS];

    private int totalEnemiesCount = 0;
    private int lastTotalEnemiesCount = 0;

    // Whether enemies are killed in this frame.
    private Int enemiesKilledByStomp = new Int("ks", 1);
    private Int enemiesKilledByFire = new Int("kf", 1);
    private int killsByFire = 0;
    private int killsByStomp = 0;

    // private Int marioXCoord = new Int("MarioX", 127);
    // private Int marioYCoord = new Int("MarioY", 15);
    // private Int marioXVel = new Int("MarioXVel", 7);
    // private Int marioYVel = new Int("MarioYVel", 3);

    // private Int isMarioSpedUp = new Int("MarioSpedUp", 1);

    // [4 bits] Whether obstacles are in front of Mario.
    // | 3
    // | 2
    // M | 1
    // | 0
    private BitArray obstacles = new BitArray("o", 3);

    // [2 bits] Whether there are gaps under or in front of Mario.
    // M |
    // ---|---
    // 0 | 1
    BitArray gaps;

    // private Int win = new Int("W", 1);

    private long stateNumber = 0;
    private MarioForwardModel environment;
    private int[][] scene;

    public static float dDistance = 0;
    private int dElevation = 0;
    private int lastElevation = 0;
    public static float lastDistance = 0;

    public Cs229State() {
        for (int i = 0; i < LearningParams.NUM_OBSERVATION_LEVELS; i++) {
            // Enemy directions: 0~7.
            enemies[i] = new BitArray("e" + i, 8);
        }
        gaps = new BitArray("gaps", 4);
        // lastDistance = 0;
    }

    /**
     * Updates the state with the given Environment.
     */
    public void update(MarioForwardModel environment) {

        this.environment = environment;
        // this.scene = environment.getMergedObservationZZ(1, 1);
        this.scene = environment.getMarioCompleteObservation(1, 1);

        // Update distance and elevation.
        float distance = environment.getCompletionPercentage(); // environment.getEvaluationInfo().distancePassedPhys;
        dDistance = distance - lastDistance;
        // System.out.println("dDistance = "+ dDistance + "start = " + lastDistance + " end = " + distance + "stuckCount " + stuckCount);;
        if (Math.abs(dDistance) <= 0.005) {
            dDistance = 0;
        }
        lastDistance = distance;

        int elevation = Math.max(0, getDistanceToGround(MARIO_X - 1) - getDistanceToGround(MARIO_X));
        dElevation = Math.max(0, elevation - lastElevation);
        lastElevation = elevation;

        // *******************************************************************
        // Update state params.
        marioMode.value = environment.getMarioMode();

        float[] pos = environment.getMarioFloatPos();
        marioDirection.value = getDirection(pos[0] - marioX, pos[1] - marioY);
        marioX = pos[0];
        marioY = pos[1];

        

        if (dDistance == 0) {
            stuckCount += 1;
        } else {
            stuckCount = 0;
            stuck.value = 0;
        }
        if (stuckCount >= LearningParams.NUM_STUCK_FRAMES) {
            // TODO does this help?
            // stuck.value = 1;
        }
        /*
        marioXCoord.value = (int)(marioX / 16.0);
        marioYCoord.value = (int)(marioY / 16.0);
        // environment.getMarioFloatVelocity()
        float[] vel = environment.getMarioFloatVelocity();
        if (Math.abs(vel[1]) <= 0.5f){
            marioYVel.value = 0;
        }else if (vel[1] < -0.5f){
            marioYVel.value = 1;
        }else if (vel[1] > 0.5){
            vel[1] = 2;
        }

        // clamp to between -3 and 3.
        // and quantise these values in 7 distinct bins, ranging from
        // -3 to 3.
        vel[0] = Math.max(Math.min(vel[0], 3.0f), -3.0f) + 3.0f;
        marioXVel.value = Math.round(vel[0]);
        assert 0 <= marioXVel.value && marioXVel.value <= 7;

        // test bad
        marioXCoord.value = 0;
        marioYCoord.value = 0;
        // marioXVel.value = 0;
        // marioYVel.value = 0;

        */
        // isMarioSpedUp.value = 0; //environment.get_is_mario_sped_up() ? 1 : 0;

        // collisionsWithCreatures.value =
        // environment.getEvaluationInfo().collisionsWithCreatures
        // - lastCollisionsWithCreatures;
        // lastCollisionsWithCreatures =
        // environment.getEvaluationInfo().collisionsWithCreatures;

        // Fill can jump.
        /// *
        canJump.value = (!environment.isMarioOnGround() || environment.mayMarioJump()) ? 1 : 0;
        // */

        onGround.value = environment.isMarioOnGround() ? 1 : 0;

        // Fill enemy info.
        /// *
        int maxSize = LearningParams.OBSERVATION_SIZES[enemies.length - 1];
        int startX = MARIO_X - maxSize;
        int endX = MARIO_X + maxSize;
        int startY = MARIO_Y - maxSize - getMarioHeight() + 1;
        int endY = MARIO_Y + maxSize;

        totalEnemiesCount = 0;
        for (int i = 0; i < enemiesCount.length; i++) {
            enemiesCount[i] = 0;
        }

        for (int i = 0; i < enemies.length; i++) {
            enemies[i].reset();
        }
        for (int y = startY; y <= endY; y++) {
            for (int x = startX; x <= endX; x++) {
                if (scene[y][x] == SpriteType.GOOMBA.getValue() || scene[y][x] == SpriteType.SPIKY.getValue()) {
                    int i = getObservationLevel(x, y);
                    int d = getDirection(x - MARIO_X, y - MARIO_Y);
                    if (i < 0 || d == Direction.NONE) {
                        continue;
                    }
                    enemies[i].value[d] = true;
                    enemiesCount[i]++;
                    totalEnemiesCount++;
                }
            }
        }

        // Fill killed info.
        enemiesKilledByStomp.value = environment.getKillsByStomp() - killsByStomp;

        // Only count killed by fire within our observation range.
        if (totalEnemiesCount < lastTotalEnemiesCount) {
            enemiesKilledByFire.value = environment.getKillsByFire() - killsByFire;
        } else {
            enemiesKilledByFire.value = 0;
        }

        lastTotalEnemiesCount = totalEnemiesCount;
        killsByFire = environment.getKillsByFire();
        killsByStomp = environment.getKillsByStomp();
        // */

        // Fill obstacle info.
        obstacles.reset();
        for (int y = 0; y < obstacles.value.length; y++) {
            if (isObstacle(MARIO_X + 1, MARIO_Y - y + 1)) {
                obstacles.value[y] = true;
            }
        }

        // Fill gap info.
        gaps.reset();
        for (int i = 0; i < gaps.value.length; i++) {
            gaps.value[i] = getDistanceToGround(MARIO_X + i) < 0;
        }

        this.computeStateNumber();
    }

    public float calculateReward() {
        float rewardScaler = 1f;
        for (int i = 0; i < enemiesCount.length; i++) {
            if (enemiesCount[i] > 0) {
                rewardScaler = LearningParams.ENEMIES_AROUND_REWARD_SCALER[i];
                break;
            }
        }

        float reward =
                // Penalty to help Mario get out of stuck.
                stuck.value * LearningParams.REWARD_PARAMS.stuck +
                // Reward for making forward and upward progress.
                        rewardScaler * dDistance * LearningParams.REWARD_PARAMS.distance
                        + rewardScaler * dElevation * LearningParams.REWARD_PARAMS.elevation +
                        // Reward for killing/avoiding enemies.
                        collisionsWithCreatures.value * LearningParams.REWARD_PARAMS.collision
                        + enemiesKilledByFire.value * LearningParams.REWARD_PARAMS.killedByFire
                        + enemiesKilledByStomp.value * LearningParams.REWARD_PARAMS.killedByStomp;

        // Logger.println(2, "D: " + dDistance);
        // Logger.println(2, "H:" + dElevation);
        // Logger.println(2, "Reward = " + reward);

        return reward;
    }

    public boolean canJump() {
        return environment.mayMarioJump();
    }

    /**
     * Returns a unique number to identify each different state.
     */
    public long getStateNumber() {
        return stateNumber;
    }

    private void computeStateNumber() {
        stateNumber = 0;
        int i = 0;
        for (Field field : fields) {
            stateNumber += field.getInt() << i;
            i += field.getNBits();
        }
        // System.out.println("Number of bits = " + i + " x coord " + marioXCoord.value + "y = " + marioYCoord.value);
        if (i >= Long.SIZE) {
            System.err.println("State number too large!! = " + i + "bits!!");
            System.exit(1);
        }
    }

    public static String printStateNumber(long state) {
        StringBuilder sb = new StringBuilder("[]");
        /*
         * int n = 0; for (Field field : FIELDS) { n += field.getNBits(); } for (Field
         * field : FIELDS) { n -= field.getNBits();
         * 
         * }
         */
        return sb.toString();
    }

    private int getMarioHeight() {
        return marioMode.value > 0 ? 2 : 1;
    }

    private int getObservationLevel(int x, int y) {
        for (int i = 0; i < LearningParams.OBSERVATION_SIZES.length; i++) {
            int size = LearningParams.OBSERVATION_SIZES[i];
            int dy = y >= MARIO_Y ? (y - MARIO_Y) : (MARIO_Y - getMarioHeight() - y + 1);
            if (Math.abs(x - MARIO_X) <= size && dy <= size) {
                return i;
            }
        }
        System.err.println("Bad observation level!! " + x + " " + y);
        return -1;
    }

    /**
     * Computes the distance from Mario to the ground. This method will return -1 if
     * there's no ground below Mario.
     */
    private int getDistanceToGround(int x) {
        if (x >= scene[0].length) return -1;
        for (int y = MARIO_Y + 1; y < scene.length; y++) {
            if (isGround(x, y)) {
                return Math.min(3, y - MARIO_Y - 1);
            }
        }
        return -1;
    }

    private boolean isObstacle(int x, int y) {
        switch (scene[y][x]) {
            case MarioForwardModel.OBS_BRICK:
            case MarioForwardModel.OBS_SOLID:
            case MarioForwardModel.OBS_CANNON:
                return true;
        }
        return false;
    }

    private boolean isGround(int x, int y) {
        return isObstacle(x, y);
    }

    public static class Direction {
        public static final int UP = 0;
        public static final int RIGHT = 1;
        public static final int DOWN = 2;
        public static final int LEFT = 3;
        public static final int UP_RIGHT = 4;
        public static final int DOWN_RIGHT = 5;
        public static final int DOWN_LEFT = 6;
        public static final int UP_LEFT = 7;
        public static final int NONE = 8;
    }

    private static final float DIRECTION_THRESHOLD = 0.8f;

    private int getDirection(float dx, float dy) {
        if (Math.abs(dx) < DIRECTION_THRESHOLD) {
            dx = 0;
        }
        if (Math.abs(dy) < DIRECTION_THRESHOLD) {
            dy = 0;
        }

        if (dx == 0 && dy > 0) {
            return Direction.UP;
        } else if (dx > 0 && dy > 0) {
            return Direction.UP_RIGHT;
        } else if (dx > 0 && dy == 0) {
            return Direction.RIGHT;
        } else if (dx > 0 && dy < 0) {
            return Direction.DOWN_RIGHT;
        } else if (dx == 0 && dy < 0) {
            return Direction.DOWN;
        } else if (dx < 0 && dy < 0) {
            return Direction.DOWN_LEFT;
        } else if (dx < 0 && dy == 0) {
            return Direction.LEFT;
        } else if (dx < 0 && dy > 0) {
            return Direction.UP_LEFT;
        }
        return Direction.NONE;
    }

    public abstract class Field {
        String name;

        public Field(String name) {
            this.name = name;
            fields.add(this);
        }

        @Override
        public String toString() {
            return String.format("%s: %s", name, getValueToString());
        }

        public abstract String getValueToString();

        public abstract int getNBits();

        public abstract int getInt();
    }

    public class BitArray extends Field {
        boolean[] value;

        public BitArray(String name, int n) {
            super(name);
            value = new boolean[n];
        }

        @Override
        public int getNBits() {
            return value.length;
        }

        @Override
        public int getInt() {
            int decInt = 0;
            for (int i = 0; i < value.length; i++) {
                decInt <<= 1;
                decInt += value[i] ? 1 : 0;
            }
            return decInt;
        }

        @Override
        public String getValueToString() {
            return "";
        }

        private void reset() {
            for (int i = 0; i < value.length; i++) {
                value[i] = false;
            }
        }
    }

    public class Int extends Field {
        int value;
        // Maximum possible value of this integer.
        private final int max;

        public Int(String name, int max) {
            super(name);
            this.max = max;
        }

        @Override
        public int getNBits() {
            return (int) Math.ceil(Math.log(max + 1) / Math.log(2));
        }

        @Override
        public int getInt() {
            value = Math.max(0, Math.min(max, value));
            return value;
        }

        @Override
        public String getValueToString() {
            return String.valueOf(value);
        }
    }

    public static void main(String[] argv) {
        // MarioState state = new MarioState();
        // state.marioMode.value = 0;
        // state.canJump.value = 1;
        // state.onGround.value = 1;
        // state.stuck.value = 1;
        // state.obstacles.value[0] = true;
        // state.obstacles.value[1] = true;
        // state.obstacles.value[2] = false;
        // state.computeStateNumber();
        // System.out.println(state.getStateNumber());
    }
}
