import os
import sys
import errno
import math
import random
import argparse
import json
import platform
import plaidml
import colorsys
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib as mpl
from plaidml import plaidml_setup

class plotter(object):
    def getColor(self, hue, satur, val):
        rgb = colorsys.hsv_to_rgb(hue, satur, val)

        r = int(rgb[0] * 256)
        g = int(rgb[1] * 256)
        b = int(rgb[2] * 256)
        
        color= '#%02x%02x%02x' % (r, g, b)

        return color

    def generate_plot(self, df, args, isTrain):
        title_str = args.name
        path_str = args.path
        isProspector = args.include_comparisons
        set_style()

        # prepping data
        col_order = (list(set(df['model'])))
        max_time = (float(max(df['time per example (seconds)'])))

        exponent = np.floor(np.log10(np.abs(max_time))).astype(int)
        base_10 = 10
        if exponent > 0:
            base_10 = 1
        else:
            for number in range(1, np.abs(exponent)):
                base_10 = base_10 * 10
        max_time = ((math.ceil(base_10 * max_time)) / base_10)

        # creating graph
        palette = []
        palette_dict = {}
        gradient_step = .99 / (len(list(set(df['batch']))))
        num = -1
        golden_ratio = 0.618033988749895
        h = random.random()

        for x in df['model']:
            if x not in palette_dict:
                palette_dict[x] = h
                h += golden_ratio
                h = h % 1
        
        for x in palette_dict:
            num = palette_dict[x]
            gradient = gradient_step
            for y in list(set(df['batch'])):
                color = self.getColor(num, gradient, 1 - gradient)
                palette.append(color)
                gradient = gradient + gradient_step

        fig, axes = plot(df, "model", col_order, max_time)
        labels = (list(set(df['model'])))
        set_labels(fig, axes, labels, list(set(df['batch'])), len(labels))
        color_bars(axes, palette, len(df['model']), len(list(set(df['batch']))))

        # saving graph
        title = ''
        if title_str != '':
            title = title_str + '.png'
        else:
            title = time.strftime("plaidbench %Y-%m-%d-%H:%M.png")
        print("saving figure at '" + path_str + '/' + title + "'\n")
        fig.savefig(path_str + '/' + title)

    def find_golden_paths(self, df, isTrain):
        # importing golden npy files, not other golden file utilization though
        models = list(set(df['model']))
        batches = list(set(df['batch']))

        this_dir = os.path.dirname(os.path.abspath(__file__))
        golden_path = os.path.join(this_dir, 'golden')

        path_to_golden = ''
        GOLD = '\033[0;33m'
        BGOLD = '\033[1;33m'
        PURPLE = '\033[0;35m'
        ENDC = '\033[0m'

        print('Attempting to retrieve ' + BGOLD + 'Golden Files' + ENDC + '...\n')
        for model in models:
            path_to_golden = os.path.join(golden_path, model)
            for batch in batches:
                if isTrain == True:
                    filename = '{},bs-{}.npy'.format('train', batch)
                else:
                    filename = '{},bs-{}.npy'.format('infer', batch)
                path_to_golden = os.path.join(golden_path, model, filename)
                print(path_to_golden)
                if not os.path.exists(path_to_golden):
                    print(PURPLE + '- no file -\n' + ENDC)
                else:
                    print(GOLD + '- Found! -\n' + ENDC)  
                    #data = (np.load(path_to_golden))[0]


def plot(data, column, column_order, ymax):
    g = sns.FacetGrid(
        data,
        col=column,
        col_order = column_order,
        sharex=False,
        size = 6,
        aspect = .33
    )

    g.map(
        sns.barplot,
        "model", "time per example (seconds)", "batch",
        hue_order = list(set(data['batch'])).sort(),
        order = list(set(data['batch'])).sort()
    )

    if ymax == 0:
        ymax = 1
    else:
        plt.yticks(np.arange(0, ymax + (ymax * .1), ymax/10))

    axes = np.array(g.axes.flat)

    for ax in axes:
        #ax.hlines(.0003, -0.5, 0.5, linestyle='--', linewidth=1, color=getColor(hue_start, .6, .9))
        ax.set_ylim(0, ymax)

    return plt.gcf(), axes


def set_labels(fig, axes, labels, batch_list, model_count):
    for i, ax in enumerate(axes):
        increment = .75 / len(batch_list)
        label_positions = []
        
        if len(batch_list) % 2 == 0:        
            intial_position = increment / 2
            inverse_position = -1 * intial_position
            label_positions.append(intial_position)
            label_positions.append(inverse_position)

            for j in range((len(batch_list) - 2) / 2):
                intial_position = intial_position + increment
                inverse_position = -1 * intial_position
                label_positions.append(intial_position)
                label_positions.append(inverse_position)    
        else:
            label_positions.append(0)
            half_len = (len(batch_list) - 1) / 2

            for j in range(half_len):
                label_positions.append((increment + (increment * j)))
                label_positions.append(-1 * (increment + (increment * j)))
                
        label_positions.sort()
        ax.set_xticks(label_positions) 
        batch_list.sort()
        ax.set_xticklabels(batch_list)

        ax.get_yaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
        ax.grid(b=True, which='both', linewidth=.6)
        
        ax.set_xlabel(labels[i])
        ax.set_ylabel("")
        ax.set_title("")
    axes.flat[0].set_ylabel("Time (sec)")
    
    for x in range(model_count):
        sns.despine(ax=axes[x], left=True)
    
    fig.suptitle("Single example runtime\nby batch size", verticalalignment='top', fontsize=11, y='.99', horizontalalignment='center')
    plt.subplots_adjust(top=0.91)
    

def set_style():
    sns.set_style("whitegrid", {    
        "font.family": "serif",
        "font.serif": ["Times", "Palatino", "serif"]
    })


def color_bars(axes, colors, networks, batches):
    for i in range(networks/batches):
        for x in range(len(axes[i].patches)):
            patch = axes[i].patches[x]
            patch.set_color(colors[(i * batches) + x])
            patch.set_edgecolor('black')
            if len(axes[i].patches) == 1:
                patch.set_hatch('//') 
                patch.set_color('grey')
                patch.set_edgecolor('black')


def main():
    # set intial exit status
    exit_status = 0

    # intialize and run arguement parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default='/tmp/plaidbench_results', 
                        help='system path to blanket run output that is to be graphed')
    parser.add_argument('--name', default='report.json',
                        help='file name of output run (should end with .json)')
    parser.add_argument('--include_comparisons', action='store_true', default=False,
                        help='seek out golden paths')
    args = parser.parse_args()

    # intialize variables
    plot_maker = plotter()
    data = {}

    # open results file
    try:
        os.makedirs(args.path)
    except OSError as ex:
        if ex.errno != errno.EEXIST:
            printf(ex)
            return
    with open(os.path.join(args.path, args.name), 'r') as saved_file:
        for line in saved_file:
            data = json.loads(line)

    # creating dict with completed runs
    d = data
    runs = {}
    for key, values in data.items():
        if 'compile_duration' in values:
            runs[key] = values

    # sort runs dictionary
    models_list = []
    executions_list = []
    batch_list2 = []
    name = []

    for x, y in sorted(runs.items()):
        models_list.append(y['model'])
        executions_list.append( y['execution_duration'] / data['run_configuration']['example_size'] )
        batch_list2.append(y['batch_size'])
        name.append(y['model'] + " : " + str(y['batch_size']))

    # setting up data frame
    uber_list = pd.DataFrame()
    uber_list['model'] = models_list
    uber_list['time per example (seconds)'] = executions_list
    uber_list['batch'] = batch_list2
    uber_list['name'] = name
    isTrain = (data['run_configuration']['train'])

    # attempting to get info about users env
    userSys = platform.uname()
    userPyV = platform.python_version()
    machine_info = []
    for info in userSys:
        machine_info.append(info)
    machine_info.append(userPyV)
    ctx = plaidml.Context()
    devices, _ = plaidml.devices(ctx, limit=100, return_all=True)
    for dev in devices:      
        plt.suptitle(str(dev))
        machine_info.append(str(dev))
    
    # find golden paths
    if args.include_comparisons == True:
        plot_maker.find_golden_paths(uber_list, isTrain)

    # generate plot
    plot_maker.generate_plot(uber_list, args, isTrain)      

    # close program
    sys.exit(exit_status)

if __name__ == '__main__':
    main()
