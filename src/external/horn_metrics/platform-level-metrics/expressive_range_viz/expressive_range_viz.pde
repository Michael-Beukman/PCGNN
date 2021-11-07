String generatorName; //name of the generator directory
String xAxisLabel; //x axis label for graph
String yAxisLabel;  //y axis label for graph
String outputFileName; //name of the file to save the image to
String title; //title of the graph

//which column (counting from 0) in combined metrics is the xAxis and yAxis contained in?
int xAxisIndex;
int yAxisIndex;

//how many bins should there be along X and Y axes?
int binSize = 50;

String[] generatorNames = {"GE", "hopper", "launchpad", "launchpad-rhythm", "notch", 
                           "notch param", "notch param rand", "ORE", "original", "pb count", 
                           "pb occurence", "pb weighted count"};
String[] axisLabels = {"density", "leniency", "linearity", "pattern density", "pattern variation"};
int[] axisIndices = {1, 2, 4, 5, 6};

//counters for going through all combinations of axes and generators
int currentGenerator;
int currentXAxis;
int currentYAxis;
boolean done;

//non-labeled mode?
boolean noLabelsMode = false;

void setup()
{
  borderSize = (noLabelsMode ? NO_LABELS_BORDER_SIZE : NORMAL_BORDER_SIZE);
  
  size(graphSize + 2*borderSize, graphSize + 2*borderSize);
  bins = new int[binSize][binSize];
  binLevelNames = new ArrayList[binSize][binSize];
  
  axisLabelFont = loadFont("Calibri-Bold-24.vlw");
  tickLabelFont = loadFont("Calibri-16.vlw");
  
  currentGenerator = 0;
  currentXAxis = 0;
  currentYAxis = 1;
  
  generatorName = generatorNames[currentGenerator];
  xAxisIndex = axisIndices[currentXAxis];
  yAxisIndex = axisIndices[currentYAxis];
  xAxisLabel = axisLabels[currentXAxis];
  yAxisLabel = axisLabels[currentYAxis];
  title = generatorName + ": " + xAxisLabel + " vs. " + yAxisLabel;
  outputFileName = generatorName + "/" + xAxisLabel + "-" + yAxisLabel;
  done = false;
  
  loadData();
   
  smooth();
  
}

void loadNextCombination()
{
  currentYAxis++;
  if (currentYAxis >= axisIndices.length) {
    currentXAxis++;
    currentYAxis = currentXAxis + 1;
    if (currentXAxis >= axisIndices.length - 1) {
      currentGenerator++;
      currentXAxis = 0;
      currentYAxis = 1;
    }
  }
  
  if (currentGenerator < generatorNames.length) {
    generatorName = generatorNames[currentGenerator];
    xAxisIndex = axisIndices[currentXAxis];
    yAxisIndex = axisIndices[currentYAxis];
    xAxisLabel = axisLabels[currentXAxis];
    yAxisLabel = axisLabels[currentYAxis];
    title = generatorName + ": " + xAxisLabel + " vs. " + yAxisLabel;
    outputFileName = generatorName + "/" + xAxisLabel + "-" + yAxisLabel;
    
    loadData();
  }
  else {
    done = true;
  }
}

//loads the data from the combined_metrics file into lists and calculates histogram
void loadData() {
  //initialize the arrays
  for (int i = 0; i < binSize; i++) {
    for (int j = 0; j < binSize; j++) {
      bins[i][j] = 0;
      binLevelNames[i][j] = new ArrayList();
    }
  }  
  maxX = 1;
  maxY = 1;
  minX = 0;
  minY = 0;
  
  //open up the combined metrics file
  strings = loadStrings("../metrics/" + generatorName + "/combined_metrics.csv");
  
  numDataPoints = strings.length - 1;
  String[] levelNames = new String[numDataPoints];
  xData = new float[numDataPoints];
  yData = new float[numDataPoints];
  for (int i = 1; i < numDataPoints; i++)
  {
    String[] data = split(strings[i], ',');
    levelNames[i] = data[0];
    xData[i] = float(data[xAxisIndex]);
    yData[i] = float(data[yAxisIndex]);
  }
  
  //now put each data point into a bin
  maxBinCount = (generatorName.equals("original") ? 5 : 40);
  minBinCount = 0;
  //print("Min x: " + min(xData) + "  max x: " + max(xData));
  for (int i = 0; i < numDataPoints; i++) {
    float binSizeX = (float)(maxX - minX)/binSize;
    float binSizeY = (float)(maxY - minY)/binSize;

    int xBin = floor(xData[i]/binSizeX);
    int yBin = floor(yData[i]/binSizeY);
    if (xBin >= binSize) xBin = binSize - 1;
    if (yBin >= binSize) yBin = binSize - 1;

//    println("x: " + xData[i] + " " + xBin + "  y: " + yData[i] + " " + yBin);

    bins[xBin][yBin]++;
    binLevelNames[xBin][yBin].add(levelNames[i]);
  }
}

void draw()
{
  background(0);
  
  fill(255);
  
  noStroke();
  rect(0, 0, width, borderSize);
  rect(0, height - borderSize, width, borderSize);
  rect(0, 0, borderSize, height);
  rect(width - borderSize, 0, borderSize, height);
  
  //title of the graph
  if (!noLabelsMode) {
    fill(0);
    textFont(axisLabelFont);
    textSize(32);
    textAlign(CENTER);
    text(title, 0, 15, width, 40); 
  }
  
  
  translate(borderSize, borderSize);
  for (int i = 0; i < binSize; i++) {
    for (int j = 0; j < binSize; j++) {
      if (bins[i][j] != 0) {
        float ratio = (bins[i][j] > maxBinCount) ? 1.0 : (float)bins[i][j]/maxBinCount;
        println(ratio);
//        if (ratio < 0.25) {
//          fill(lerpColor(heatmapColors[0], heatmapColors[1], (ratio - 0)/(0.25)));
//        }
//        else if (ratio < 0.5) {
//          fill(lerpColor(heatmapColors[1], heatmapColors[2], (ratio - 0.25)/(0.25)));
//        }
//        else if (ratio < 0.75) {
//          fill(lerpColor(heatmapColors[2], heatmapColors[3], (ratio - 0.5)/(0.25)));
//        }
//        else {
//          fill(lerpColor(heatmapColors[3], heatmapColors[4], (ratio - 0.75)/(0.25)));
//        }
        
        fill((bins[i][j] > maxBinCount) ? 255 : 200*bins[i][j]/maxBinCount + 50);
        
        //fill(lerpColor(cool, warm, ratio));
        noStroke();
        rect(i*graphSize/binSize, graphSize - (graphSize/binSize) - j*graphSize/binSize, graphSize/binSize, graphSize/binSize);
      }
      
    }
  }
  stroke(128, 128, 128, 50);
  line(0, graphSize/4, graphSize, graphSize/4);
  line(0, graphSize/2, graphSize, graphSize/2);
  line(0, 3*graphSize/4, graphSize, 3*graphSize/4);
  line(graphSize/4, 0, graphSize/4, graphSize);
  line(graphSize/2, 0, graphSize/2, graphSize);  
  line(3*graphSize/4, 0, 3*graphSize/4, graphSize);
  
  if (!noLabelsMode) {
    fill(0);
    //x axis labels
    textFont(tickLabelFont);
    textAlign(CENTER);
    text("0.25", graphSize/4 - 25, graphSize + 10, 50, 50);
    text("0.5", graphSize/2 - 25, graphSize + 10, 50, 50);
    text("0.75", 3*graphSize/4 - 25, graphSize + 10, 50, 50);
    
    //y axis labels
    textAlign(RIGHT);
    text("0.75", -60, graphSize/4 - 10, 50, 50);
    text("0.5", -60, graphSize/2 - 10, 50, 50);
    text("0.25", -60, 3*graphSize/4 - 10, 50, 50);
    
    //x axis label
    textAlign(CENTER);
    textFont(axisLabelFont);
    text(xAxisLabel, 0, graphSize + 30, graphSize, 50);
    
    //y axis label
    translate(-borderSize, -borderSize + height);
    rotate(3*PI/2);
    text(yAxisLabel, 0, 5, height, 50);
  }
  
  //save to a the output file name
  //println(dataPath(outputFileName) + ".png");
  save(dataPath(outputFileName) + ".png");
  
  if (!done) loadNextCombination();
  else noLoop();
}

void mouseClicked() {
  int graphXPos= mouseX - borderSize;
  int graphYPos = mouseY - borderSize;
  
  int binX = graphXPos/(graphSize/binSize);
  int binY = binSize - 1 - graphYPos/(graphSize/binSize);
  
  print(binX + ", " + binY + ":  ");
  for (int i = 0; i < binLevelNames[binX][binY].size(); i++) {
    print(binLevelNames[binX][binY].get(i) + " ");
  }
  println();
}
