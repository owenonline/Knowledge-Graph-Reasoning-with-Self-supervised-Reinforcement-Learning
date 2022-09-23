import enum
import re
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import argparse
import os
import seaborn as sns
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument("--experiment", default="", type=str)
parser.add_argument("--n", default=75, type=int)
parser.add_argument("--iterations_sl", default=0, type=int)
parser.add_argument("--iterations_rl", default=0, type=int)
parsed = vars(parser.parse_args())

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

################
# Get heatmaps #
################

num_rl = len([name for name in os.listdir(parsed['experiment']) if "checkpoint_sl_epoch_" in name])
num_sl = num_rl - 1

lbls = list(range(num_rl))

raw_data = []
zeroed_data = []
first_iter = []
with open(parsed['experiment'] + "/checkpoint_sl_epoch_0/scores.txt","r") as scores:
    chunk = scores.read().split("Final score:")[1]
    first_iter.append(float(chunk.split("Hits@1:  ")[1].split("\n")[0]))
    first_iter.append(float(chunk.split("\nHits@3:  ")[1].split("\n")[0]))
    first_iter.append(float(chunk.split("\nHits@5:  ")[1].split("\n")[0]))
    first_iter.append(float(chunk.split("\nHits@10:  ")[1].split("\n")[0]))
    first_iter.append(float(chunk.split("\nHits@20:  ")[1].split("\n")[0]))
    first_iter.append(float(chunk.split("\nauc:  ")[1].split("\n")[0]))
    
for i in range(num_rl):
    raw_ckpt = []
    zeroed_ckpt = []
    with open(parsed['experiment'] + "/checkpoint_sl_epoch_"+str(i)+"/scores.txt","r") as scores:
        chunk = scores.read().split("Final score:")[1]
        raw_ckpt.append(float(chunk.split("Hits@1:  ")[1].split("\n")[0]))
        raw_ckpt.append(float(chunk.split("\nHits@3:  ")[1].split("\n")[0]))
        raw_ckpt.append(float(chunk.split("\nHits@5:  ")[1].split("\n")[0]))
        raw_ckpt.append(float(chunk.split("\nHits@10:  ")[1].split("\n")[0]))
        raw_ckpt.append(float(chunk.split("\nHits@20:  ")[1].split("\n")[0]))
        raw_ckpt.append(float(chunk.split("\nauc:  ")[1].split("\n")[0]))
        
        zeroed_ckpt.append(float(chunk.split("Hits@1:  ")[1].split("\n")[0]) - first_iter[0])
        zeroed_ckpt.append(float(chunk.split("\nHits@3:  ")[1].split("\n")[0]) - first_iter[1])
        zeroed_ckpt.append(float(chunk.split("\nHits@5:  ")[1].split("\n")[0]) - first_iter[2])
        zeroed_ckpt.append(float(chunk.split("\nHits@10:  ")[1].split("\n")[0]) - first_iter[3])
        zeroed_ckpt.append(float(chunk.split("\nHits@20:  ")[1].split("\n")[0]) - first_iter[4])
        zeroed_ckpt.append(float(chunk.split("\nauc:  ")[1].split("\n")[0]) - first_iter[5])
    raw_data.append(raw_ckpt)
    zeroed_data.append(zeroed_ckpt)

raw_data = np.array(raw_data)
zeroed_data = np.array(zeroed_data)
f, ax = plt.subplots(figsize=(9, 6))
sns.heatmap(raw_data, annot=True, fmt='.3g', linewidths=.5, ax=ax, xticklabels=['Hits@1','Hits@3','Hits@5','Hits@10','Hits@20','MRR'], yticklabels=lbls).figure.savefig(parsed['experiment'] + "/raw_heatmap.png")
f, ax = plt.subplots(figsize=(9, 6))
sns.heatmap(zeroed_data, annot=True, fmt='.3g', linewidths=.5, ax=ax, xticklabels=['Hits@1','Hits@3','Hits@5','Hits@10','Hits@20','MRR'], yticklabels=lbls).figure.savefig(parsed['experiment'] + "/zeroed_heatmap.png")

#######################
# Get training curves #
#######################

plt.rcParams.update({'font.size': 30})
plt.figure(figsize=(25,12))

path = parsed['experiment'] + '/log.txt'
rl_len = parsed['iterations_rl']
sl_len = parsed['iterations_sl']
colors = ['b', 'g', 'c', 'm', 'y', 'k', 'saddlebrown', 'coral', 'slategrey']

with open(path, "r") as logfile:
    lftext = logfile.read()

# avg reward per batch
text = re.findall('.*\/.*avg. reward per batch  .*, num_ep_correct.*', lftext)
processed = []
for x in text:    
    x = re.sub(".*\/.*avg. reward per batch  ", '', x)
    x = re.sub(", num_ep_correct.*", '', x)
    processed.append(float(x))
    
sl = []
for i in range(1, num_sl + 1):
    sl += processed[(rl_len + sl_len)*i - sl_len:(rl_len + sl_len)*i]
num_iters = len(sl)//sl_len
sl = moving_average(sl, n=parsed['n'])
new_sl_len = len(sl)//num_iters
plt.plot(range(len(sl)), sl, 'r', label="SL")

for i in range(num_rl):
    rl = processed[(rl_len + sl_len)*i:(rl_len + sl_len)*i + rl_len]
    rl = moving_average(rl, n=parsed['n'])
    new_rl_len = len(rl)
    color = colors[i%9]
    plt.plot(range(new_sl_len*i, new_sl_len*i + new_rl_len), rl, color, label="{} steps SL + {} steps RL".format(i*sl_len, rl_len))

plt.ylabel("Average reward per batch")
plt.xlabel("Number of training batches")
plt.legend(loc="lower right", ncol=1)
plt.savefig(parsed['experiment'] + "/avg_reward_per_batch.png")
plt.clf()

#############################
# Get relation correlations #
#############################

relation_dist = {}
checkpoints = {}
plt.rcParams.update({'font.size': 24})

fig, ax = plt.subplots(1)
ax.set_ylabel("% correct out of total appearances")
ax.set_xlabel("% of dataset")
fig.tight_layout()
fig.set_size_inches(17.5, 16.5)
def getplots(lftext):
    text = re.findall('.*\/.*relation ckpt .*', lftext)
    for line in text:
        checkpoint_num = int(re.sub(' --.*', '', re.sub('.*\/.*relation ckpt ', '', line)))
        if not checkpoint_num in checkpoints:
            checkpoints[checkpoint_num] = {
                "total": 0,
                "relation_correct": defaultdict(list),
                "relation_appearances": defaultdict(list)
            }
        relation = int(re.sub('\: .*', '', re.sub('.*\/.* -- ', '', line)))
        correct = int(re.sub(' correct.*', '', re.sub('.*\/.*\: ', '', line)))
        total = int(re.sub(' appearances.*', '', re.sub('.*\/.*out of ', '', line)))
        ofset = float(re.sub('\% of.*', '', re.sub('.*\/.* \(', '', line)))/100

        checkpoints[checkpoint_num]['total'] += total
        checkpoints[checkpoint_num]['relation_correct'][relation] = correct
        checkpoints[checkpoint_num]['relation_appearances'][relation] = total
        relation_dist[relation] = ofset

    relation_x_plots = defaultdict(list)
    relation_y_plots = defaultdict(list)
    for checkpoint in checkpoints:
        for relation in checkpoints[checkpoint]['relation_correct']:
            relation_x_plots[relation].append(relation_dist[relation])
            relation_y_plots[relation].append(checkpoints[checkpoint]['relation_correct'][relation]/checkpoints[checkpoint]['relation_appearances'][relation])
    return relation_x_plots, relation_y_plots

relation_x_plots, relation_y_plots = getplots(lftext)
for count, relation in enumerate(relation_x_plots):
    ax.plot(relation_x_plots[relation], relation_y_plots[relation], colors[count%9])

plt.savefig(parsed['experiment'] + "/relation_score_distribution.png")
plt.clf()