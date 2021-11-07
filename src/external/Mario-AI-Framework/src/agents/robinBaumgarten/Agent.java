package agents.robinBaumgarten;

import java.util.ArrayList;

import engine.core.MarioAgent;
import engine.core.MarioForwardModel;
import engine.core.MarioTimer;
import engine.helper.MarioActions;

/**
 * @author RobinBaumgarten
 */
public class Agent implements MarioAgent {
    private boolean[] action;
    private AStarTree tree;
    public ArrayList<Integer> number_of_states_considered;

    @Override
    public void initialize(MarioForwardModel model, MarioTimer timer) {
        this.action = new boolean[MarioActions.numberOfActions()];
        this.tree = new AStarTree();
        number_of_states_considered = new ArrayList<>();
    }

    @Override
    public boolean[] getActions(MarioForwardModel model, MarioTimer timer) {
        this.tree.number_of_states_in_open_set = 0;
        this.tree.test = 0;
        this.tree.number_of_things_visited = 0;
        action = this.tree.optimise(model, timer);
        // int num_states = this.tree.number_of_states_in_open_set;
        // how many states have we actively considered, i.e. visited.
        // int num_states = this.tree.visitedStates.size() + this.tree.number_of_states_in_open_set;
        int num_states = this.tree.number_of_things_visited;
        number_of_states_considered.add(num_states);
        // System.out.println("-------------");
        // for (int[] T: this.tree.visitedStates){
        //     System.out.println(T[0] +"(" + T[0] / 16.0 + ")" + ", " + T[1] +"(" + T[1] / 16.0 + ")"+ ", " + T[2]);
        // }
        // System.out.println("-------------");
        return action;
    }

    @Override
    public String getAgentName() {
        return "RobinBaumgartenAgent";
    }

}
