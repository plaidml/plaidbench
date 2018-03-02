import os
import sys
import errno
import math
import random
import argparse
import json
import platform
import plaidml
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib as mpl
from plaidml import plaidml_setup

class plotter(object):
    def getColor(self, hue, satur, val):
        hue_i = int(hue * 6)
        r = -1
        g = -1
        b = -1
        f = hue * 6 - hue_i
        p = val * (1 - satur)
        q = val * (1 - f * satur)
        t = val * (1 - (1 - f) * satur)

        if (0 <= hue_i and hue_i < 1):
            r = val 
            g = t 
            b = p 
        elif (1 <= hue_i and hue_i < 2):
            r = q
            g = val 
            b = p 
        elif (2 <= hue_i and hue_i < 3):
            r = p 
            g = val 
            b = t 
        elif (3 <= hue_i and hue_i < 4):
            r = p 
            g = q 
            b = val 
        elif (4 <= hue_i and hue_i < 5):
            r = t 
            g = p 
            b = val 
        else:
            r = val 
            g = p 
            b = q 

        r = int(r * 256)
        g = int(g * 256)
        b = int(b * 256)

        color = 'rgb(' + str(r) + ', ' + str(g) + ', ' + str(b) + ')'
        color = '#%02x%02x%02x' % (r, g, b)

        return color

    def generate_plot(self, df, title_str, path_str):
        set_style()
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

        title = ''
        if title_str != '':
            title = title_str + '.png'
        else:
            title = time.strftime("plaidbench %Y-%m-%d-%H:%M.png")
        print("\nsaving figure at '" + path_str + '/' + title + "'")
        fig.savefig(path_str + '/' + title)


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
    #hue_start = random.random()
    for ax in axes:
        #ax.hlines(.0003, -0.5, 0.5, linestyle='--', linewidth=1, color=getColor(hue_start, .6, .9))
        ax.set_ylim(0, ymax)

    return plt.gcf(), axes


def set_labels(fig, axes, labels, batch_list, model_count):
    for i, ax in enumerate(axes):
        increment = .75 / len(batch_list)
        illusory = []
        
        if len(batch_list) % 2 == 0:        
            foo = increment / 2
            bar = -1 * foo
            illusory.append(foo)
            illusory.append(bar)

            for j in range((len(batch_list) - 2) / 2):
                foo = foo + increment
                bar = -1 * foo
                illusory.append(foo)
                illusory.append(bar)    
        else:
            illusory.append(0)
            half_len = (len(batch_list) - 1) / 2

            for j in range(half_len):
                illusory.append((increment + (increment * j)))
                illusory.append(-1 * (increment + (increment * j)))
                
        illusory.sort()
        ax.set_xticks(illusory) 
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
            illusory = axes[i].patches[x]
            illusory.set_color(colors[(i * batches) + x])
            illusory.set_edgecolor('black')
            if len(axes[i].patches) == 1:
                illusory.set_hatch('//') 
                illusory.set_color('grey')
                illusory.set_edgecolor('black')


def main():
    print('Hello Stranger...')

    # set intial exit status
    exit_status = 0

    # intialize and run arguement parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default='/tmp/plaidbench_results')
    parser.add_argument('--name', default='report.json')
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

    plot_maker.generate_plot(uber_list, args.name, args.path)     

    # close program
    print('So Long Stranger...')
    sys.exit(exit_status)

if __name__ == '__main__':
    main()
