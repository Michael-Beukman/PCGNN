color[] heatmapColors = { color(2, 21, 77), //deep blue - 0
                          color(35, 85, 95), //medium blue - 0.25
                          color(52, 229, 30), //green - 0.5
                          color(237, 245, 32), //yellow - 0.75
                          color(240, 15, 15) }; //red - 1.0

size(50, 600);

for (int i = 0; i < height; i += 5) {
  float ratio = 1 - (float)i/height;
  if (ratio < 0.25) {
    fill(lerpColor(heatmapColors[0], heatmapColors[1], (ratio - 0)/(0.25)));
  }
  else if (ratio < 0.5) {
    fill(lerpColor(heatmapColors[1], heatmapColors[2], (ratio - 0.25)/(0.25)));
  }
  else if (ratio < 0.75) {
    fill(lerpColor(heatmapColors[2], heatmapColors[3], (ratio - 0.5)/(0.25)));
  }
  else {
    fill(lerpColor(heatmapColors[3], heatmapColors[4], (ratio - 0.75)/(0.25)));
  }
  
  
  noStroke();
  rect(0, i, width, i+5);
}

save("gradientBar.png");
