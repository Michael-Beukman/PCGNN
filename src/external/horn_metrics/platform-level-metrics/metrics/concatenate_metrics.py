#/usr/bin/python

import sys
import os

if len(sys.argv) == 1:
	print "USAGE: first and only argument to script should be name of directory containing metrics files."
	exit(1)

generator_directory = sys.argv[1]

if generator_directory == "" or not os.path.exists(generator_directory):
	print "USAGE: first and only argument to script should be name of directory containing metrics files."
	exit(1)

#open the directory and grab a list of all the files in it
file_list = os.listdir(generator_directory)
metrics = []
for f in file_list:
	if not f.startswith('.') and f.endswith('.csv') and not f == 'combined_metrics.csv' and not f == 'ncd.csv':
		#DEBUG
		print "Looking at file " + f

		#open the metrics file
		metric_file = open(generator_directory + "/" + f, 'rU')

		#create a new list for all the metrics, TITLE->metric type; levelname->metric
		metric_dict = {}
		metric_dict['TITLE'] = f

		#loop through all the entries in the csv file, adding them to the dictionary
		for line in metric_file:
			#split on comma
			split_line = line.strip().split(',')
			metric_dict[split_line[0]] = split_line[1]

		#assign the dictionary to the metrics list
		metrics.append(metric_dict.copy())

		#close the file
		metric_file.close()

#create a new file for combined metrics
combined_file = open(generator_directory + '/combined_metrics.csv', 'w')

#go through all the metrics and write them all to the file, starting with the title row
all_metrics = ""
for m in metrics:
	all_metrics += ',' + m['TITLE']
combined_file.write('Level' + all_metrics + '\n')

#now grab the list of level names
level_names = metrics[0].keys()
level_names.sort()

#loop through each level name, pulling out the metric associated with it for each file
for ln in level_names:
	if not ln == 'TITLE':
		new_line = ln
		for m in metrics:
			new_line += ',' + m[ln]
		combined_file.write(new_line + '\n')

#close the file now that writing is complete
combined_file.close()