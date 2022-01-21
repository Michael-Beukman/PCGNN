import java.util.Arrays;
import java.util.Random;

/**
 * Created by Matt Mancuso on 1/29/2015.
 *
 * Maze class is the container for everything having to do with the actual maze.
 * Contains enum for difficulty assignment and numeric extraction (difficulty must
 * be between 1 and 10). No variable setters because Maze is  simply a static entity.
 *
 * This class contains no logic for creating or solving mazes; it simply houses the
 * maze data and methods for displaying the maze.
 */
public class Maze {
    /*
    Difficulty may be defined from 1 to 10 (inclusive). The difficulty is used by the Builder class to determine the
    minimum length of the correct path and the maximum length of the sub-paths. Assuming the maze is solved from start
    to finish, longer sub paths will yield harder solutions, since the likelihood of the solver hitting a dead-end far
    into the maze is higher.
     */
    public static enum Difficulty {
        EASY(1),
        MEDIUM(5),
        HARD(8),
        EXTREME(10);

        private int difficulty;

        Difficulty(int difficulty) {
            this.difficulty = difficulty;
        }

        public int getValue() {
            return difficulty;
        }
    }

    // Random number generation
    long seed = new Random().nextLong();
    Random random;

    // Maze parameters
    int difficulty = Difficulty.MEDIUM.getValue();
    int sizeX = 16;
    int sizeY = 16;

    // Maze data
    Integer[] startingCoord;
    Layout[][] layout;
    Integer[][] correctPath;

    /**
    Creates string of full maze using printChar as border, and correctChar (if given) as the correct path indicator.
     */
    public String getAsciiMaze(String printChar, String correctChar) {
        String[][] strings;
        StringBuffer buffer = new StringBuffer();
        int stringsX;
        int stringsY;

        // Requires 2x + 1 the maze size; one row/col for the borders and one for the actual path, plus one for the
        // top and left borders.
        strings = new String[sizeY*2+1][sizeX*2+1];

        // Initializes the string as empty
        for (int i=0; i<strings.length; i++) {
            Arrays.fill(strings[i], " ");
        }

        // Sets top-left cell to wall
        strings[0][0] = printChar;

        // Loops over each cell
        for (int row=0; row<sizeY; row++) {
            for (int col=0; col<sizeX; col++) {
                // We're only printing wall characters in every other cell
                stringsX = 1 + col*2;
                stringsY = 1 + row*2;

                // If we're along the top or left edge of the maze, we have to fill in every character to build a
                // complete boundary, unless the boundary is an entrance or exit
                if (row == 0) {
                    strings[0][stringsX+1] = printChar;
                    if (layout[row][col].borderIs(Layout.Border.TOP, Layout.Type.WALL)) {
                        strings[0][stringsX] = printChar;
                    }
                }
                if (col == 0) {
                    strings[stringsY+1][0] = printChar;
                    if (layout[row][col].borderIs(Layout.Border.LEFT, Layout.Type.WALL)) {
                        strings[stringsY][0] = printChar;
                    }
                }

                // Fills every corner
                strings[stringsY+1][stringsX+1] = printChar;

                // Fills remaining cells from left to right/top to bottom, depending on whether it is a path or wall
                if (layout[row][col].borderIs(Layout.Border.BOTTOM, Layout.Type.WALL)) {
                    strings[stringsY+1][stringsX] = printChar;
                }
                if (layout[row][col].borderIs(Layout.Border.RIGHT, Layout.Type.WALL)) {
                    strings[stringsY][stringsX+1] = printChar;
                }
            }
        }

        // Fills in correct path with correctChar if provided
        if (correctChar != null && correctPath != null) {
            for (Integer[] coord : correctPath) {
                strings[1 + coord[1] * 2][1 + coord[0] * 2] = correctChar;
            }
        }

        // Converts to string
        for (String[] row : strings) {
            for (String col : row) {
                buffer.append(col);
            }
            buffer.append("\n");
        }

        return buffer.toString();
    }

    public String getAsciiMaze(String printChar) {
        return getAsciiMaze(printChar, null);
    }

    public String getAsciiMaze() {
        return getAsciiMaze("#");
    }

    public int getDifficulty() {
        return difficulty;
    }

    public int[] getSize() {
        return new int[] {sizeX, sizeY};
    }

    public Integer[] getStartingCoord() {
        return startingCoord;
    }

    public long getSeed() {
        return seed;
    }
}