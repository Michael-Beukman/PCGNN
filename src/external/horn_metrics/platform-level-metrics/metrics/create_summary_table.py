#/usr/bin/python

import sys
import os
import numpy
import scipy.stats
import matplotlib

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

def get_boxplot_data(arr):
	minimum = str(min(arr))
	firstquart = str(scipy.stats.scoreatpercentile(arr, 25))
	median = str(numpy.median(arr))
	thirdquart = str(scipy.stats.scoreatpercentile(arr,75))
	maximum = str(max(arr))

	return median + "," + firstquart + "," + maximum + "," + minimum + "," + thirdquart + "," + "\n"

metrics_directory = "."

file_list = os.listdir(metrics_directory)



#create a new files for the tables
avg_table = open(metrics_directory + '/avg_table.csv', 'w')
std_table = open(metrics_directory + '/std_table.csv', 'w')
latex_table = open(metrics_directory + "/latex_table.txt", 'w')

avg_table.write("generator, leniency, linearity, density, pattern density, pattern variation, compression distance\n")
std_table.write("generator, leniency, linearity, density, pattern density, pattern variation, compression distance\n")
latex_table.write("generator & leniency & linearity & density & pattern density & pattern variation & compression distance \\\\ \\hline \n")

#create new files for boxplots (one per metric, except level slope)
leniency_boxplot = open("../boxplot analysis/individual_metrics/leniency.csv", 'w')
linearity_boxplot = open("../boxplot analysis/individual_metrics/linearity.csv", 'w')
density_boxplot = open("../boxplot analysis/individual_metrics/density.csv", 'w')
patterndensity_boxplot = open("../boxplot analysis/individual_metrics/patterndensity.csv", 'w')
patternvariation_boxplot = open("../boxplot analysis/individual_metrics/patternvariation.csv", 'w')
compressiondistance_boxplot = open("../boxplot analysis/individual_metrics/compressiondistance.csv", 'w')

#firstquart + "," + maximum + "," + minimum + "," + thirdquart + "," + median + "\n"
boxplot_header = "generator,median,firstquartile,maximum,minimum,thirdquartile\n"
leniency_boxplot.write(boxplot_header)
linearity_boxplot.write(boxplot_header)
density_boxplot.write(boxplot_header)
patterndensity_boxplot.write(boxplot_header)
patternvariation_boxplot.write(boxplot_header)
compressiondistance_boxplot.write(boxplot_header)

for f in file_list:
	#if it's not a directory, ignore it
	if os.path.isdir(f):
		print "Gathering data from " + f + "\n"
		#ok, now pull out the metrics files as needed
		linearity = [0]
		#linearity
		if os.path.exists(f + "/LinearityMetric.csv"):
			metricsfile = open(f + "/LinearityMetric.csv", 'r')
			linearity = extract_single_metric(metricsfile)		
			metricsfile.close()

		#leniency
		leniency = [0]
		if os.path.exists(f + "/LeniencyMetric.csv"):
			metricsfile = open(f + "/LeniencyMetric.csv", 'r')
			leniency = extract_single_metric(metricsfile)
			metricsfile.close()

		#levelslope
		levelslope = [0]
		if os.path.exists(f + "/LevelSlopeMetric.csv"):
			metricsfile = open(f + "/LevelSlopeMetric.csv", 'r')
			levelslope = extract_single_metric(metricsfile)
			metricsfile.close()

		
		#density
		density = [0]
		if os.path.exists(f + "/density.csv"):
			metricsfile = open(f + "/density.csv", 'rU')
			density = extract_single_metric(metricsfile)
			metricsfile.close()
		

		#patterndensity
		patterndensity = [0]
		if os.path.exists(f + "/Patterndensity.csv"):
			metricsfile = open(f + "/Patterndensity.csv", 'r')
			patterndensity = extract_single_metric(metricsfile)
			metricsfile.close()

		#patternvariation
		patternvariation = [0]
		if os.path.exists(f + "/patternvariation.csv"):
			metricsfile = open(f + "/patternvariation.csv", 'r')
			patternvariation = extract_single_metric(metricsfile)
			metricsfile.close()
		

		#compressiondistance
		compressiondistance = [0]
		if os.path.exists(f + "/ncd.csv"):
			metricsfile = open(f + "/ncd.csv", 'r')
			compressiondistance = extract_paired_metric(metricsfile)
			metricsfile.close()
		
		
		#go through all lists and write to file
		#order: leniency, level slope, linearity, density, pattern density, pattern distance, compression distance
		m_leniency = 0 if leniency == [] else round(numpy.mean(leniency), 2)
		m_linearity = 0 if linearity == [] else round(numpy.mean(linearity), 2)
		m_density = 0 if density == [] else round(numpy.mean(density), 2)
		m_patterndensity = 0 if patterndensity == [] else round(numpy.mean(patterndensity), 2)
		m_patternvariation = 0 if patternvariation == [] else round(numpy.mean(patternvariation), 2)
		m_compressiondistance = 0 if compressiondistance == [] else round(numpy.mean(compressiondistance), 2)

		s_leniency = 0 if leniency == [] else round(numpy.std(leniency), 2)
		s_linearity = 0 if linearity == [] else round(numpy.std(linearity), 2)
		s_density = 0 if density == [] else round(numpy.std(density), 2)
		s_patterndensity = 0 if patterndensity == [] else round(numpy.std(patterndensity), 2)
		s_patternvariation = 0 if patternvariation == [] else round(numpy.std(patternvariation), 2)
		s_compressiondistance = 0 if compressiondistance == [] else round(numpy.std(compressiondistance), 2)


		avg_table.write(f + "," + str(m_leniency) + "," + str(m_linearity) + "," + str(m_density) + "," + str(m_patterndensity) + "," + str(m_patternvariation) + "," + str(m_compressiondistance) + "\n")
		std_table.write(f + "," + str(s_leniency) + "," + str(s_linearity) + "," + str(s_density) + "," + str(s_patterndensity) + "," + str(s_patternvariation) + "," + str(s_compressiondistance) + "\n")
		latex_table.write(f + " & " + str(m_leniency) + " (" + str(s_leniency) + ") & " +
			str(m_linearity) + " (" + str(s_linearity) + ") & " +
			str(m_density) + " (" + str(s_density) + ") & " +
			str(m_patterndensity) + " (" + str(s_patterndensity) + ") & " +
			str(m_patternvariation) + " (" + str(s_patternvariation) + ") & " +
			str(m_compressiondistance) + " (" + str(s_compressiondistance) + ") \\\\ \\hline \n")

		#generator,min,firstquartile,median,thirdquartile,max
		linearity_boxplot.write(f + "," + get_boxplot_data(linearity))
		leniency_boxplot.write(f + "," + get_boxplot_data(leniency))
		density_boxplot.write(f + "," + get_boxplot_data(density))
		patterndensity_boxplot.write(f + "," + get_boxplot_data(patterndensity))
		patternvariation_boxplot.write(f + "," + get_boxplot_data(patternvariation))
		compressiondistance_boxplot.write(f + "," + get_boxplot_data(compressiondistance))


avg_table.close()
std_table.close()
latex_table.close()

linearity_boxplot.close()
leniency_boxplot.close()
density_boxplot.close()
patterndensity_boxplot.close()
patternvariation_boxplot.close()
compressiondistance_boxplot.close()
