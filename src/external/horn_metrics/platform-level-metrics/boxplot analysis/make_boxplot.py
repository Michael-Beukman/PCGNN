import sys
import os
import copy
from matplotlib.pyplot import plot, show, savefig, xlim, figure, \
                hold, ylim, legend, boxplot, setp, axes, axvline

# function for setting the colors of the box plots pairs
def setBoxColors(bp):
    colors = ['#ff0300', '#ffac00', '#25ff00', '#00eaff', '#0036ff', '#d100ff']
    for i in range(6):
        setp(bp['boxes'][i], color=colors[i])
        setp(bp['medians'][i], color=colors[i])

    for i in range(6):
        setp(bp['caps'][i*2], color=colors[i])
        setp(bp['caps'][i*2 + 1], color = colors[i])
        setp(bp['whiskers'][i*2], color= colors[i])
        setp(bp['whiskers'][i*2 + 1], color= colors[i])
        setp(bp['fliers'][i*2], color= colors[i])
        setp(bp['fliers'][i*2 + 1], color= colors[i])

def extract_single_metric(mf):
    lst = []
    for line in mf:
        #split on comma
        split_line = line.strip().split(',')
        lst.append(float(split_line[1]))
    return lst

def extract_paired_metric(mf):
    lst = []
    for line in mf:
        #split on comma
        split_line = line.strip().split(',')
        #ignore all empty slots and first non-zero entry
        last = ""
        for n in split_line:
            if n != "" and last != "":
                try:
                    val = float(n)
                    if val <= 1.0 and val >= 0.0:
                        lst.append(float(n))
                except:
                    pass
            last = n
    return lst

#Generate the data to plot
#should be a dictionary of generator name -> list of lists

metrics_directory = "../metrics"

file_list = os.listdir(metrics_directory)

all_gen = {}
#[GE, hopper, launchpad, launchpad_rhythm, notch, 
#notch_parameterized, notch_parameterized_random, ORE, original,
#pattern_based, pattern_based_2]

for f in file_list:
    #if it's not a directory, ignore it
    if os.path.isdir(metrics_directory + "/" + f):
        print "Gathering data from " + f + "\n"
        #ok, now pull out the metrics files as needed
        linearity = [0]
        #linearity
        if os.path.exists(metrics_directory + "/" + f + "/LinearityMetric.csv"):
            metricsfile = open(metrics_directory + "/" + f + "/LinearityMetric.csv", 'r')
            linearity = extract_single_metric(metricsfile)      
            metricsfile.close()

        #leniency
        leniency = [0]
        if os.path.exists(metrics_directory + "/" + f + "/LeniencyMetric.csv"):
            metricsfile = open(metrics_directory + "/" + f + "/LeniencyMetric.csv", 'r')
            leniency = extract_single_metric(metricsfile)
            metricsfile.close()
        
        #density
        density = [0]
        if os.path.exists(metrics_directory + "/" + f + "/density.csv"):
            metricsfile = open(metrics_directory + "/" + f + "/density.csv", 'rU')
            density = extract_single_metric(metricsfile)
            metricsfile.close()
        

        #patterndensity
        patterndensity = [0]
        if os.path.exists(metrics_directory + "/" + f + "/Patterndensity.csv"):
            metricsfile = open(metrics_directory + "/" + f + "/Patterndensity.csv", 'r')
            patterndensity = extract_single_metric(metricsfile)
            metricsfile.close()
        
        #patternvariation
        patternvariation = [0]
        if os.path.exists(metrics_directory + "/" + f + "/patternvariation.csv"):
            metricsfile = open(metrics_directory + "/" + f + "/patternvariation.csv", 'r')
            patternvariation = extract_single_metric(metricsfile)
            metricsfile.close()

        #compressiondistance
        compressiondistance = [0]
        if os.path.exists(metrics_directory + "/" + f + "/ncd.csv"):
            metricsfile = open(metrics_directory + "/" + f + "/ncd.csv", 'r')
            compressiondistance = extract_paired_metric(metricsfile)
            metricsfile.close()

        #order of lists: leniency, linearity, density, patterndensity, compressiondistance
        all_gen[f] = [copy.copy(leniency), copy.copy(linearity), copy.copy(density), copy.copy(patterndensity), copy.copy(patternvariation), copy.copy(compressiondistance)]


fig = figure(figsize=(17, 8), dpi=100)
ax = axes()
hold(True)

num_metrics = 6
position_counter = 1

generator_names = all_gen.keys()
generator_names.sort(key=lambda y: y.lower())

for gen in generator_names:
    bp = boxplot(all_gen[gen], positions = range(position_counter, position_counter + num_metrics), sym=',', whis=1.9, widths = 0.8)
    setBoxColors(bp)
    position_counter = position_counter + num_metrics + 2

# set axes limits and labels
xlim(0,96)
ylim(0,1)
ax.set_xticklabels(generator_names, rotation=20)
ax.set_xticks([x*8 + 3 for x in range(12)])

for i in [x*8 + 7 for x in range(11)]:
    axvline(x = i, color='0.8')


# draw temporary ['#ff0300', 'ff8600', '#a1ff00', '00eaff', '#0036ff', '#d100ff'] lines and use them to draw a legend
hR, = plot([1,1], color='#ff0300', linestyle='-', linewidth=3)
hO, = plot([1,1], color='#ffac00', linestyle='-', linewidth=3)
hY, = plot([1,1], color='#25ff00', linestyle='-', linewidth=3)
hG, = plot([1,1], color='#00eaff', linestyle='-', linewidth=3)
hB, = plot([1,1], color='#0036ff', linestyle='-', linewidth=3)
hM, = plot([1,1], color='#d100ff', linestyle='-', linewidth=3)
legend((hR, hO, hY, hG, hB, hM),('Leniency', 'Linearity', 'Density', 'Pattern Density', 'Pattern Variation', 'Compression Distance'), 
    bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=6, mode="expand", borderaxespad=0.)
hR.set_visible(False)
hO.set_visible(False)
hY.set_visible(False)
hG.set_visible(False)
hB.set_visible(False)
hM.set_visible(False)

savefig('boxcompare.png')
show()