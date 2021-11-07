package agents.michael_rl;

import java.util.HashMap;
import java.util.Random;

import engine.core.MarioAgent;
import engine.core.MarioForwardModel;
import engine.core.MarioTimer;
import engine.helper.MarioActions;

public class RLAgent {
    public static class FloatInt{
        int integer;
        float real;
        public FloatInt(int i, float f){
            integer =i;
            real = f;
        }
    }
    private boolean[] actions;
    int number_of_actions, number_of_states;
    // float[][] QTable;
    HashMap<Long, float[]> QTable;
    HashMap<Long, int[]> QNumberOfTimes;
    // todo qtable.
    // private 
    public float lr = 0.6f;
    public float gamma = 1f;
    public  float epsilon = 0f;//1f;//0.05f;
    private Random r;
    
    public RLAgent(){
        actions = new boolean[MarioActions.numberOfActions()];
        number_of_actions = (int)Math.pow(2, actions.length);
        number_of_states = RLState.getTotalNumberOfStates();
        // QTable = new float[number_of_states][number_of_actions];
        QTable = new HashMap<>();
        QNumberOfTimes = new HashMap<>();
        r = new Random();
    }
    
    private void check(RLState s){
        long snum = s.getStateNumber();
        if (!QTable.containsKey(snum)){
            QTable.put(snum, new float[number_of_actions]);
            QNumberOfTimes.put(snum, new int[number_of_actions]);
        }
    }

    private FloatInt Q(RLState s){
        check(s);
        long snum = s.getStateNumber();
        int best_i = 0;
        for (int i=1; i< number_of_actions; ++i){
            if (QTable.get(snum)[i] > QTable.get(snum)[best_i]){
                best_i = i;
            }
        }

        return new FloatInt(best_i, QTable.get(snum)[best_i]);
    }

    public void addSample(RLState s, int action, float reward, RLState next_state, int next_action, boolean is_terminal){
        // todo
        /*
                action = self.get_action(state)
                next_state, reward, done, info = env.step(action)
                Qsa = self.table[state, action]
                best_next = self.table[next_state].max() 
                # Since this is tabular, we always have that table[terminal, :] = 0, because:
                # - We initialised that to 0 initially
                # - we never update table[terminal, :], because if next_state is terminal, the loop ends,
                # and we don't update it.
                # Q learning update rule, from Sutton And Barto 2018, p131
                self.table[state, action] = Qsa + self.alpha * \
                                                (reward + self.gamma * best_next - Qsa)
        */
        // decay
        // epsilon *= 0.99f;
        long snum = s.getStateNumber();
        // int snext_num =
        float Qsa = QTable.get(snum)[action];

        float best_next = 0;
        if (next_state != null)
            best_next = this.Q(next_state).real;

        float alpha = this.lr; // / QNumberOfTimes.get(snum)[action];
        QTable.get(snum)[action] = Qsa + alpha * (reward + this.gamma * best_next - Qsa);
    }

    public boolean[] getActionFromInt(int action){
        // we want the integer action to become a boolean array. We can do that using its binary representation.
        actions = new boolean[MarioActions.numberOfActions()];
        for (int i=0; i < actions.length; ++i){
            if ((action & (1 << i)) > 0){
                // this bit is on
                actions[i] = true;
            }
        }
        return actions;
    }

    public int getActions(RLState state){
        int action_to_choose;
        check(state);
        if (r.nextDouble() < this.epsilon){
            action_to_choose = r.nextInt(number_of_actions);
        }else{
            action_to_choose = Q(state).integer;
        }
        QNumberOfTimes.get(state.getStateNumber())[action_to_choose]++;
        return action_to_choose;
    }
    int getStateNumber(){
        return 1;
    }
}
