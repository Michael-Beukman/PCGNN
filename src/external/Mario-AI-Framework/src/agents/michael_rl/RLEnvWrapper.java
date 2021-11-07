package agents.michael_rl;

import java.util.ArrayList;

import engine.core.MarioAgent;
import engine.core.MarioForwardModel;
import engine.core.MarioResult;
import engine.core.MarioTimer;
import engine.helper.GameStatus;
import engine.helper.MarioActions;

public class RLEnvWrapper implements MarioAgent{
    public static class RewardPair{
        float reward;
        boolean done;
        RLState state;
    }
    public static class Position{
        public float x, y;
        public Position(float _x, float _y){
            x = _x; y = _y;
        }
    }

    public ArrayList<Integer> actions = new ArrayList<>();
    public ArrayList<Long> states = new ArrayList<>();
    public ArrayList<Position> positions = new ArrayList<>();
    
    private RLState curr_state = null;
    private RLState prev_state = null;
    int prev_action = -1;
    private RLAgent agent;
    public boolean is_learning = true;
    public float total_rew = 0;
    float eps;


    float previous_x = 0;

    public RLEnvWrapper(RLAgent agent, boolean learn){
        is_learning = learn;
        this.agent = agent;
        total_rew = 0;
        eps = agent.epsilon;
    }

    public void endEpisode(MarioResult result){
        previous_x = 0;
        Cs229State.stuckCount = 0;
        Cs229State.lastDistance = 0;
        actions.clear();
        states.clear();
        positions.clear();

        // here we update based on the score
        prev_state = curr_state;
        float reward = -1000;
        boolean terminal = false;
        if (result.getGameStatus() == GameStatus.WIN){
            terminal = true;
            reward = 1000;
        }else if (result.getGameStatus() == GameStatus.LOSE){// || result.getGameStatus() == GameStatus.TIME_OUT){
            // so it isn't incentivized to immediately die.
            // reward -= result.getRemainingTime() / 30 * 2;
            terminal = true;
        }
        // reward = result.getCompletionPercentage() * 1;
        if (is_learning){
            agent.addSample(prev_state, prev_action, reward, null, -1, terminal); 
        }
        
        total_rew += reward;

    }

    @Override
    public void initialize(MarioForwardModel model, MarioTimer timer) {
        // System.out.println("INIT");
        prev_state = null;
        curr_state = null;
        prev_action = -1;

        curr_state = RLState.getStateFromModel(model);
        total_rew = 0;
        if (!is_learning)
            this.agent.epsilon = 0;
        else{
            this.agent.epsilon = eps;
        }
    }

    @Override
    public boolean[] getActions(MarioForwardModel model, MarioTimer timer) {
        prev_state = curr_state;
        float current_x = model.getMarioFloatPos()[0];

        float diff = current_x - previous_x;
        previous_x = current_x;
        float move_reward = 0;
        if (diff > 0)      move_reward = 1 * 0.5f;
        else if (!(diff > 0)) move_reward = -1;

        curr_state = RLState.getStateFromModel(model);
        float reward = -1f;// + move_reward;

        
        if (prev_state != null && prev_action != -1){
            // we have a state, so we can update
            
            // if we use speed, then it is better?
            if ((prev_action & (int)Math.pow(2, MarioActions.SPEED.getValue())) > 0 &&
                (prev_action & (int)Math.pow(2, MarioActions.RIGHT.getValue())) > 0){
                reward += 0.5;
            }
            if (is_learning)
                agent.addSample(prev_state, prev_action, reward, curr_state, -1, false);
        }
        total_rew += reward;
        // todo reward and such
        prev_action = agent.getActions(curr_state);
        
        actions.add(prev_action);
        states.add(curr_state.getStateNumber());
        positions.add(new Position(model.getMarioFloatPos()[0], model.getMarioFloatPos()[1]));

        return agent.getActionFromInt(prev_action);
    }

    @Override
    public String getAgentName() {
        return  "RLEnvWrapper";
    }
}
