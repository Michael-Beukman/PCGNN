/**
 * Created by Matt Mancuso on 1/30/15.
 *
 * Layout is an individual piece of a maze. I.e. a 20x20 maze has 400 layouts.
 * Each Layout maintains four Borders on which a border Type is stored. Border
 * enum is used primarily for referencing from other code.
 */
public class Layout {
    public static enum Border {
        TOP,
        BOTTOM,
        LEFT,
        RIGHT
    }
    public static enum Type {
        OPEN,
        ENTRANCE,
        EXIT,
        WALL
    }

    private Type top;
    private Type bottom;
    private Type left;
    private Type right;

    public Layout(Type top, Type bottom, Type left, Type right) {
        this.top = top;
        this.bottom = bottom;
        this.left = left;
        this.right = right;
    }

    public Layout() {
        top = Type.WALL;
        bottom = Type.WALL;
        left = Type.WALL;
        right = Type.WALL;
    }

    public Layout setBorder(Border border, Type type) {
        switch (border) {
            case TOP:
                top = type;
                break;
            case BOTTOM:
                bottom = type;
                break;
            case LEFT:
                left = type;
                break;
            case RIGHT:
                right = type;
                break;
        }
        return this;
    }

    public Type getBorder(Border border) {
        switch (border) {
            case TOP:
                return top;
            case BOTTOM:
                return bottom;
            case LEFT:
                return left;
            case RIGHT:
                return right;
        }

        return Type.OPEN;
    }

    public boolean borderIs(Border border, Type type) {
        switch (border) {
            case TOP:
                return top == type;
            case BOTTOM:
                return bottom == type;
            case LEFT:
                return left == type;
            case RIGHT:
                return right == type;
        }

        return false;
    }
}
