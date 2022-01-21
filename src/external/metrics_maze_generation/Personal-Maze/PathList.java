import java.util.ArrayList;

/**
 * Created by Matt Mancuso on 1/29/2015.
 *
 * PathList keeps track of maze paths and path connections.
 */
public class PathList extends ArrayList<ArrayList<Integer[]>> {
    private int currentPath = 0;
    private int currentCoordIndex = 0;
    private int currentSubPath;
    private int currentParentPath;
    private int subPathLevel = 0;
    private ArrayList<ArrayList<Integer[]>> connections = new ArrayList<ArrayList<Integer[]>>();

    public PathList addPath() {
        this.add(new ArrayList<Integer[]>());
        connections.add(new ArrayList<Integer[]>()); //0=path, 1=starting coord index, 2=sub path level
        currentPath = this.size() - 1;
        currentCoordIndex = 0;
        return this;
    }

    public PathList addCoord(Integer[] coord) {
        if (coord.length != 2) throw new IllegalArgumentException("Incorrect coordinate length.");

        this.get(currentPath).add(coord);
        currentCoordIndex = this.get(currentPath).size() - 1;

        return this;
    }

    public PathList addSubPath() {
        subPathLevel++;

        connections.get(currentPath).add(new Integer[] {this.size(), currentCoordIndex, subPathLevel});
        currentParentPath = currentPath;
        currentSubPath = connections.get(currentPath).size() - 1;
        addPath();

        return this;
    }

    public PathList setPath(int path) {
        currentPath = path;
        resetParentPath();
        return this;
    }

    public PathList setCoordIndex(int coord) {
        currentCoordIndex = coord;
        return this;
    }

    public PathList setSubPath (int parentPath, int subPath, int level) {
        subPathLevel = level;
        currentParentPath = parentPath;
        currentSubPath = subPath;
        return this;
    }

    public int getPath() {
        return currentPath;
    }

    public Integer[] getCoord(int coordIndex) {
        return this.get(currentPath).get(coordIndex);
    }

    public Integer[] getCoord() {
        return this.get(currentPath).get(currentCoordIndex);
    }

    public int getCoordIndex() {
        return currentCoordIndex;
    }

    public PathList getParentPath() {
        if (subPathLevel == 0) return this;

        currentPath = currentParentPath;
        currentCoordIndex = connections.get(currentParentPath).get(currentSubPath)[1] - 1;

        resetParentPath();

        return this;
    }

    public int getSubPathLevel() {
        return subPathLevel;
    }

    public PathList removePath() {
        this.remove(currentPath);
        currentPath = -1;

        return this;
    }

    public PathList removeCoord() {
        this.get(currentPath).remove(currentCoordIndex);

        if (currentCoordIndex == this.get(currentPath).size()) currentCoordIndex--;
        else currentCoordIndex = -1;

        return this;
    }

    public int numPaths() {
        return this.size();
    }

    public int numCoords() {
        return this.get(currentPath).size();
    }

    public int numSubPaths() {
        return this.get(currentParentPath).size();
    }

    public PathList nextPath() {
        if (currentPath + 1 == this.size()) currentPath = 0;
        else currentPath++;

        resetParentPath();

        return this;
    }

    public PathList nextPath(int testSubPathLevel) {
        int originalPath = currentPath;
        int testPath = currentPath + 1 < this.size() ? currentPath + 1 : 0;

        if (testSubPathLevel == 0) return this;

        while (testPath != originalPath) {
                for (int i = 0; i < connections.size(); i++) {
                    for (int n = 0; n < connections.get(i).size(); n++) {
                        if (connections.get(i).get(n)[0] == testPath
                                && connections.get(i).get(n)[2] == testSubPathLevel) {
                            currentPath = testPath;
                            currentCoordIndex = 0;
                            currentParentPath = i;
                            currentSubPath = n;
                            subPathLevel = testSubPathLevel;
                            return this;
                        }
                    }
                }
            testPath = testPath + 1 < this.size() ? testPath + 1 : 0;
        }

        return this;
    }

    public PathList nextCoord() {
        if (currentCoordIndex + 1 == this.get(currentPath).size()) currentCoordIndex = 0;
        else currentCoordIndex++;

        return this;
    }

    private void resetParentPath() {
        currentParentPath = 0;
        currentSubPath = 0;
        subPathLevel = 0;

        for (int i=0; i<this.size(); i++) {
            for (int n=0; n<connections.get(i).size(); n++) {
                if (connections.get(i).get(n)[0] == currentPath) {
                    currentParentPath = i;
                    currentSubPath = n;
                    subPathLevel = connections.get(i).get(n)[2];
                    return;
                }
            }
        }
    }
}
