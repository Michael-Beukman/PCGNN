/**
 * Created by Matt Mancuso on 2/6/2015.
 *
 * Handles calls for mazes.
 */
import java.util.Arrays;
public class Handler {
    public static void main(String[] args) {
        Builder builder = new Builder();
        // System.out.println(Arrays.toString(args));
        int diff = Integer.parseInt(args[0]);

        Maze maze = builder
        .setSize(20, 20)
        // .setDifficulty(Maze.Difficulty.EASY)
        .setDifficulty(diff)
        // .setDifficulty(Maze.Difficulty.MEDIUM)
        // .setDifficulty(Maze.Difficulty.HARD)
        .saveCorrectPath(true)
        .build(); // .saveCorrectPath(true).build();

        String print = maze.getAsciiMaze("#", ".");

        System.out.print(print);
    }
}
