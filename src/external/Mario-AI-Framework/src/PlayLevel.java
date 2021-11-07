import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;

import agents.michael_rl.RLAgent;
import agents.michael_rl.RLEnvWrapper;
import agents.michael_rl.RLEnvWrapper.Position;
import engine.core.MarioAgent;
import engine.core.MarioGame;
import engine.core.MarioResult;
import engine.helper.GameStatus;

public class PlayLevel {
    public static void printResults(MarioResult result) {
        System.out.println("****************************************************************");
        System.out.println("Game Status: " + result.getGameStatus().toString() + " Percentage Completion: "
                + result.getCompletionPercentage());
        System.out.println("Lives: " + result.getCurrentLives() + " Coins: " + result.getCurrentCoins()
                + " Remaining Time: " + (int) Math.ceil(result.getRemainingTime() / 1000f));
        System.out.println("Mario State: " + result.getMarioMode() + " (Mushrooms: " + result.getNumCollectedMushrooms()
                + " Fire Flowers: " + result.getNumCollectedFireflower() + ")");
        System.out.println("Total Kills: " + result.getKillsTotal() + " (Stomps: " + result.getKillsByStomp()
                + " Fireballs: " + result.getKillsByFire() + " Shells: " + result.getKillsByShell() + " Falls: "
                + result.getKillsByFall() + ")");
        System.out.println("Bricks: " + result.getNumDestroyedBricks() + " Jumps: " + result.getNumJumps()
                + " Max X Jump: " + result.getMaxXJump() + " Max Air Time: " + result.getMaxJumpAirTime());
        System.out.println("****************************************************************");
    }

    public static String getLevel(String filepath) {
        String content = "";
        try {
            content = new String(Files.readAllBytes(Paths.get(filepath)));
        } catch (IOException e) {
            System.out.println("HELLO AN ERROR" + e.getMessage());
        }
        return content;
    }

    /**
     * This runs the environment according to the arguments given, on the specified
     * levels and with the specified agents.
     * 
     * @param args
     */
    public static void run_properly(String[] args) {
        System.out.println("Arguments are: " + Arrays.toString(args));
        String what_mode = args[0];
        String what_level_filename = args[1];
        String level = getLevel(what_level_filename);
        int time_per_level = Integer.parseInt(args[2]);
        int number_of_episodes = Integer.parseInt(args[3]);
        boolean verbose = Boolean.parseBoolean(args[4]);

        boolean should_vis = false;
        int time_to_vis = 500;
        if (args.length >= 6) {
            should_vis = Boolean.parseBoolean(args[5]);
        }
        
        if (args.length >= 7) {
            time_to_vis = Integer.parseInt(args[6]);
        }

        MarioGame game = new MarioGame();
        RLAgent my_agent = new agents.michael_rl.RLAgent();
        RLEnvWrapper rl_agent = new RLEnvWrapper(my_agent, true);

        if (verbose){
            System.out.println("Running with params: Gamma = " + my_agent.gamma + " LR = " + my_agent.lr + " Epsilon = " +my_agent.epsilon);
        }
        if (what_mode.equals("RL_Diversity")) {
            // todo learn lots
            // then eval and somehow get trajectories.

            float[] rewards = new float[number_of_episodes];
            for (int i = 0; i < number_of_episodes; ++i) {
                MarioResult res = game.runGame(rl_agent, level, time_per_level, 0, false);
                rl_agent.endEpisode(res);
                rewards[i] = rl_agent.total_rew;
                if (verbose && i % 100 == 0 && i != 0) {
                    float ave = 0;
                    for (int j = i - 100; j < i; ++j) {
                        ave += rewards[j];
                    }
                    System.out.println("Ave over 100 eps at ep " + i + " = " + ave / 100);
                }
            }

            // eval
            rl_agent.is_learning = false;
            MarioResult res = game.runGame(rl_agent, level, time_per_level, 0, false);
            // now print out actions
            System.out.println("Result Status: " + ((res.getGameStatus() == GameStatus.WIN) ? "WIN" : "LOSE"));
            System.out.print("Result Actions: ");
            for (int action : rl_agent.actions) {
                System.out.print(action + " ");
            }
            System.out.print("\nResult States: ");
            for (long state : rl_agent.states) {
                System.out.print(state + " ");
            }
            System.out.print("\nResult Positions: ");
            for (Position pos : rl_agent.positions) {
                System.out.print((int) pos.x + "," + (int) pos.y + " ");
            }

        } else if (what_mode.equals("RL_Difficulty")) {
            int BS = 5;
            int steps = number_of_episodes / BS;
            float prev_r = -10000000;
            int step = 0;
            System.out.println("Steps = " + steps + " BS = " + BS + " eps " + number_of_episodes);
            float all_rewards = 0.0f;
            for (; step < steps; ++step) {
                // train a little bit
                rl_agent.is_learning = true;
                for (int i = 0; i < BS; ++i) {
                    MarioResult res = game.runGame(rl_agent, level, time_per_level, 0, false);
                    rl_agent.endEpisode(res);
                    all_rewards += rl_agent.total_rew;
                }
                if (verbose && step != 0 && (step * BS) % 100 == 0) {
                    System.out.println("Rewards at " + step + " = " + all_rewards / 100);
                    all_rewards = 0;
                }
                // then eval
                float mean_reward_over_5_eps = 0.0f;
                boolean has_solved = true;
                rl_agent.is_learning = false;
                for (int i = 0; i < 5; ++i) {
                    MarioResult res = game.runGame(rl_agent, level, time_per_level, 0, should_vis && step * BS > time_to_vis);
                    has_solved = has_solved && (res.getGameStatus() == GameStatus.WIN);
                    rl_agent.endEpisode(res);

                    mean_reward_over_5_eps += rl_agent.total_rew;
                }

                mean_reward_over_5_eps /= 5.0;

                if (has_solved && Math.abs(mean_reward_over_5_eps - prev_r) < 10) {
                    break;
                }
                prev_r = mean_reward_over_5_eps;
            }
            System.out.println("Step = " + step);
            float diff = (float) (step - 1) / (steps - 1);
            System.out.println("Result: " + diff);
        } else if (what_mode.equals("Astar_Solvability")) {
            // assert number_of_episodes == 1;
            MarioAgent astar_agent = new agents.robinBaumgarten.Agent();
            MarioResult res = game.runGame(astar_agent, level, time_per_level, 0, false);
            if (verbose){
                ArrayList<float[]> traj = res.getPositionsTrajectories();
                System.out.print("Trajectories:");
                for (float[] t: traj){
                    System.out.print(t[0]+","+t[1] + " ");
                }
                System.out.print("\n");

                ArrayList<boolean[]> acts = res.getActions();
                System.out.print("Actions:");
                for (boolean[] act: acts){
                    // convert to integer
                    int ans = 0;
                    for (int i=0; i < act.length; ++i){
                        ans += (act[i] ? 1 : 0) * (1 << i);
                    }
                    System.out.print(ans + " ");
                }
                System.out.print("\n");


                ArrayList<Integer> num_states = ((agents.robinBaumgarten.Agent)astar_agent).number_of_states_considered;
                System.out.print("NumberOfStatesExpanded:");
                int sum = 0;
                for (int t: num_states){
                    System.out.print(t + " ");
                    sum += t;
                }
                System.out.print("\n");
                System.out.println("Sum:" + sum);
            }
            System.out.println("Result: " + ((res.getGameStatus() == GameStatus.WIN) ? "WIN" : "LOSE"));
        } else if (what_mode.equals("Visual_Test")) {
            // assert number_of_episodes == 1;
            MarioAgent astar_agent = new agents.robinBaumgarten.Agent();
            MarioResult res = game.runGame(astar_agent, level, time_per_level, 0, true);

            System.out.println("Result: " + ((res.getGameStatus() == GameStatus.WIN) ? "WIN" : "LOSE"));
        } else if (what_mode.equals("Human_Play")){
            System.out.println("Playing level manually. Use arrow keys to move and s to jump");
            game.playGame(level, 200, 0);
        }else {
            System.out.println("Bad mode");
        }

    }

    public static void main(String[] args) {
        System.out.println("In main, Arguments are: " + Arrays.toString(args));
        if (args.length > 0 || args.length == 0) {
            run_properly(args);
            return;
        }
    }
}
