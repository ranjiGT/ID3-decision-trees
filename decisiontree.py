import pandas as pd
import math
from xml.etree import ElementTree
from xml.etree.ElementTree import Element,SubElement,tostring
from xml.dom import minidom
import argparse


def beautify(elem):
    rough_string = ElementTree.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")


def dataframe_generator(filename):
    #print(filename)

    if filename == "breast-cancer.csv" or filename == "nursery.csv" or "car.csv":

        temp_data = pd.read_csv(filename, header=None)

        c_name = []
        for i in temp_data:
            c_name.append("att" + str(i))
        temp_data.columns = c_name

    else:
        temp_data = pd.read_csv(filename)

    return temp_data


def logbase_calc(filename):

    temp_data = dataframe_generator(filename)

    unique_target = list(temp_data[temp_data.columns[-1]].value_counts())
    logbase = len(unique_target)
    #print("logbase= {0}".format(logbase))
    return logbase


def overall_entropy(temp_data, logbase):

    unique_target = list(temp_data[temp_data.columns[-1]].value_counts())
    #print("Number of Unique target values are: " + str(unique_target))
    #logbase = len(unique_target)
    #logbase = 4

    # System Entropy calculation
    overallentropy = 0
    for i in unique_target:
        temp = (-1) * (i / (sum(unique_target))) * (math.log((i / (sum(unique_target))), logbase))
        overallentropy += temp
    #print("System's entropy is: " + str(overallentropy))

    return overallentropy, temp_data


def tree_generator(temp_data, logbase, parentnode):

    top = parentnode

    overallentropy, temp_data = overall_entropy(temp_data, logbase)
    derived_data = {}

    top.set("entropy",str(overallentropy))


    if overallentropy != 0:

        unique_dependent_vals = list(temp_data[temp_data.columns[-1]].unique())
        #print("List of unique target values are" + str(unique_dependent_vals))
        column_info_gain = {}

        independent_cols = list(temp_data.columns[0:-1])

        for cols in independent_cols:
            col_values = list(temp_data[cols].unique())
            #print("List of unique values in {0} is {1}".format(cols, col_values))
            columnentropy = 0

            for vals in col_values:
                t1 = temp_data[temp_data[cols] == vals]
                esum = 0

                for j in unique_dependent_vals:
                    t2 = t1[t1[t1.columns[-1]] == j]

                    if t2.shape[0] == 0:
                        e = 0
                    else:
                        e = (-1) * (t2.shape[0] / t1.shape[0]) * (math.log((t2.shape[0] / t1.shape[0]), logbase))
                    esum = esum + e

                columnentropy += esum * (t1.shape[0] / temp_data.shape[0])

            column_info_gain.update({cols: (overallentropy - columnentropy)})
        #print(column_info_gain)

        #print(max(column_info_gain.values()))
        att_ig = sorted(column_info_gain, key=column_info_gain.get, reverse=True)
        selected_node = att_ig[0]
        #print(selected_node)

        selected_node_unique_vals = list(temp_data[selected_node].unique())
        #print(selected_node_unique_vals)

        for _ in selected_node_unique_vals:
            __ = temp_data[temp_data[selected_node] == _].copy()

            __.drop(selected_node, axis=1, inplace=True)

            # path = "tempdata\\" + filename.rstrip(".csv") + "_" + _ + "_" + selected_node + ".csv"
            # print(path)

            child = SubElement(top, "node")
            child.set("feature", selected_node)
            child.set("value", _)

            tree_generator(__, logbase, child)

    elif overallentropy == 0:

        #print("this is pure node")
        only_target = str(temp_data[temp_data.columns[-1]].unique())
        #print(only_target)
        top.text = only_target.lstrip("['").rstrip("']")


parser = argparse.ArgumentParser(description='ID3 Decision Tree')
parser.add_argument('--data', required=True, help='Input Filename')
parser.add_argument('--output',required=True, help='Output Filename')
args = parser.parse_args()

temp_data = dataframe_generator(args.data)
logbase = logbase_calc(args.data)
a, b = overall_entropy(temp_data, logbase)
top = Element('tree')

tree_generator(temp_data,logbase,top)

#print(beautify(top))

open(args.output, "w").close()

with open(args.output, "a") as myfile:
    myfile.write(beautify(top))

#print("removing first line")

with open(args.output, 'r') as fin:
    data = fin.readlines()
with open(args.output, 'w') as fout:
    fout.writelines((data[1:]))

print("Output is available  in {}".format(args.output))
