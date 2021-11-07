//don't touch these, just a bunch of variable declarations
int numPoints;
int maxX;
int maxY;
int minX;
int minY;
int numDataPoints;
float[] xData;
float[] yData;
int[][] bins;
ArrayList[][] binLevelNames;

int maxBinCount;
int minBinCount;

//drawing constants
int NORMAL_BORDER_SIZE = 75;
int NO_LABELS_BORDER_SIZE = 25;
int borderSize;
int graphSize = 500;

//data parsing
String[] strings;

//text stuff
PFont axisLabelFont;
PFont tickLabelFont;

color[] heatmapColors = { color(2, 21, 77), //deep blue - 0
                          color(35, 85, 95), //medium blue - 0.25
                          color(52, 229, 30), //green - 0.5
                          color(237, 245, 32), //yellow - 0.75
                          color(240, 15, 15) }; //red - 1.0

color cool = color(61, 160, 65, 50);
color warm = color(250, 53, 23);
