package agents.michael_rl;

import java.util.ArrayList;
import java.util.List;

import engine.core.MarioForwardModel;

public class RLState {
    public static boolean DEBUG = false;
    public final List<Field> fields = new ArrayList<Field>();

    // All of my fields
    private Int state_posx = new Int("posx", 200);
    private Int state_posy = new Int("posy", 200);

    // diff enemies detail size
    private BitArray[] enemies = new BitArray[3];

    private long stateNumber;
    Cs229State my_proper_state;

    RLState(Cs229State proper_state) {
        my_proper_state = proper_state;
    }

    public static RLState getStateFromModel(MarioForwardModel model) {
        // TODO
        Cs229State my_proper_state = new Cs229State();
        my_proper_state.update(model);
        RLState s = new RLState(my_proper_state);
        for (int i = 0; i < s.enemies.length; ++i) {
            s.enemies[i] = s.new BitArray("e" + i, 8);
        }

        for (int i = 0; i < s.enemies.length; i++) {
            s.enemies[i].reset();
        }

        s.state_posx.value = (int) (model.getMarioFloatPos()[0] / 16);
        s.state_posy.value = (int) (model.getMarioFloatPos()[1] / 16);

        int[][] A = model.getMarioEnemiesObservation();
        if (DEBUG) {
            System.out.println("--- x= " + s.state_posx.value + " y = " + s.state_posy.value + " "
                    + A[s.state_posy.value][s.state_posx.value] + " " + A[s.state_posx.value][s.state_posy.value]);
            System.out.println("Hi " + (int) model.getMarioFloatPos()[0] / 16);
            for (int y = 0; y < A.length; ++y) {
                for (int x = 0; x < A[0].length; ++x) {
                    // if (x == s.state_posx.value && y == s.state_posy.value){
                    if (x == 8 && y == 8) {
                        System.out.print("M");
                    } else
                        System.out.print(A[x][y]);
                }
                System.out.println("");
            }
            System.out.println("---");
        }
        // return s;
        for (int y = 8 - 2; y <= 8 + 2; y++) {
            for (int x = 8 - 2; x <= 8 + 2; x++) {
                if (y < 0 || y >= A.length || x < 0 || x >= A[0].length)
                    continue;
                int i = (A[y][x] > 0) ? 1 : 0;
                int d = s.getDirection(x - s.state_posx.value, y - s.state_posy.value);
                if (i <= 0 || d == Direction.NONE) {
                    continue;
                }
                // try{
                s.enemies[i].value[d] = true;
                if (DEBUG)
                    System.out.print("Have enemy");
                // }catch(Exception e){
                // System.out.println("I = " + i + " d = " + d);

                // }
                // enemiesCount[i]++;
                // totalEnemiesCount++;
            }
        }
        s.computeStateNumber();
        return s;
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

    private void computeStateNumber() {
        if (my_proper_state != null){
            stateNumber = my_proper_state.getStateNumber();
            return;
        }
        stateNumber = 0;
        int i = 0;
        for (Field field : fields) {
            stateNumber += field.getInt() << i;
            if (field.getInt() >= field.max) {
                System.err.println("Field " + field.name + " is bad with value " + field.getInt());
            }
            // System.out.println("Field " + field.name + " has " + field.getNBits() + "
            // bits");
            i += field.getNBits();
        }
        if (i >= Long.SIZE) {
            System.err.println("State number too large!! = " + i + "bits!!");
            System.exit(1);
        }
    }

    public long getStateNumber() {
        return stateNumber;
    }

    public static int getTotalNumberOfStates() {
        return 2000;
    }

    // From cs229
    public abstract class Field {
        String name;
        public int max;

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
            max = (int) Math.pow(2, n);
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

}
