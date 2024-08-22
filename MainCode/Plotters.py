import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as patches
import matplotlib.colors as colors
from matplotlib.gridspec import GridSpec


def PlotComparisons(var, data_frames, labels, colors, linestyles, suffix):

    binsize = 10

    vals = np.array([])
    for df in data_frames:
        vals = np.concatenate([vals, df[var].values])

    vals = np.unique(vals)
    num_plots = vals.shape[0]

    figs = []
    axes = []

    for i in range(num_plots):
        f, a = plt.subplots(1, 3, figsize=(12, 4))

        a[0].axis('off')
        a[2].set_xlabel('Episode')
        a[1].set_xlabel('Episode')
        a[1].set_ylabel('reward')
        a[2].set_ylabel('steps')
        a[2].set_title('Episode Length')
        a[1].set_title('Reward')
        a[1].set_xticks(np.arange(0, 1001, 250))
        a[2].set_xticks(np.arange(0, 1001, 250))

        a[1].spines['top'].set_visible(False)
        a[1].spines['right'].set_visible(False)
        a[2].spines['top'].set_visible(False)
        a[2].spines['right'].set_visible(False)

        a[1].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        a[2].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

        a[1].set_yscale('symlog')
        a[2].set_yscale('symlog')

        figs.append(f)
        axes.append(a)

    for df, label, color, linestyle in zip(data_frames, labels, colors, linestyles):

        length_results = [[] for i in range(num_plots)]
        reward_results = [[] for i in range(num_plots)]
        ideal_results = [[] for i in range(num_plots)]

        for v, rewards, lengths, maze in zip(df[var], df['rewards'], df['lengths'], df['maze']):

            p = np.where(vals == v)[0][0]

            # axes[p][0].set_title(var + ': ' + str(v))
            axes[p][0].imshow(maze)

            length_results[p].append(lengths)#np.cumsum(lengths))
            reward_results[p].append(rewards)#(np.cumsum(rewards))
            ideal_results[p].append(np.array(rewards) > np.array(lengths) * -0.05)


        for p in range(num_plots):

            if(length_results[p]):

                l = np.array(length_results[p])
                l = np.concatenate([np.zeros((l.shape[0],1)), l], axis=1)
                l = np.reshape(l, (l.shape[0], int(1000/binsize), binsize))
                l = np.mean(l, axis=-1)

                y = np.mean(l, axis=0)
                x = np.arange(binsize, 1001, binsize)
                error = np.std(l, axis=0)

                axes[p][2].plot(x, y, label=label, color=color, linestyle=linestyle, alpha=1.0)
                # axes[p][2].fill_between(x, y-error, y+error, color=color, alpha=.25)

                r = np.array(reward_results[p])
                r = np.concatenate([np.zeros((r.shape[0], 1)), r], axis=1)
                r = np.reshape(r, (r.shape[0], int(1000/binsize), binsize))
                r = np.mean(r, axis=-1)

                y = np.mean(r, axis=0)
                x = np.arange(binsize, 1001, binsize)
                error = np.std(r, axis=0)

                axes[p][1].plot(x, y, color=color, linestyle=linestyle, alpha=1.0)
                # axes[p][1].fill_between(x, y - error, y + error, color=color, alpha=.25)

    for f, a in zip(figs, axes):
        # a[2].legend(bbox_to_anchor=(.8, 0), loc="lower right",
        #                      bbox_transform=f.transFigure, ncol=3, frameon=False,
        #                      fontsize='small')

        a[2].legend(ncol=1, frameon=False, fontsize='medium')

    for i, f in enumerate(figs):
        f.tight_layout()
        f.savefig('Plots/ComparisonPlot' + str(i) + suffix + '.pdf')
        plt.close(f)

    return


def PlotPairwiseComparison(df1, df2, labels):

    vals = np.array([])
    vals = np.concatenate([vals, df1['random_seed'].values])
    vals = np.concatenate([vals, df2['random_seed'].values])

    vals = np.unique(vals)
    num_points = vals.shape[0]

    reward_results = [[] for i in range(num_points)]
    ideal_results = [[] for i in range(num_points)]

    for seed, rewards, lengths in zip(df1['random_seed'], df1['rewards'], df1['lengths']):
        p = np.where(vals == seed)[0][0]
        reward_results[p].append(np.sum(rewards))
        ideal_results[p].append(np.sum(np.array(rewards) == 1))

    ys = np.zeros((2, num_points))

    for p in range(num_points):
        ys[0, p] = np.mean(reward_results[p])
        ys[1, p] = np.mean(ideal_results[p])


    reward_results = [[] for i in range(num_points)]
    ideal_results = [[] for i in range(num_points)]

    for seed, rewards, lengths in zip(df2['random_seed'], df2['rewards'], df2['lengths']):
        p = np.where(vals == seed)[0][0]
        reward_results[p].append(np.sum(rewards))
        ideal_results[p].append(np.sum(np.array(rewards) == 1))

    xs = np.zeros((2, num_points))

    for p in range(num_points):
        xs[0, p] = np.mean(reward_results[p])
        xs[1, p] = np.mean(ideal_results[p])


    colors = ['r', 'b']

    fig, axes = plt.subplots(1, 2, figsize=(6, 3))
    axes[0].ticklabel_format(style='sci', axis='both', scilimits=(0, 0))

    axes[0].scatter(xs[0, :], ys[0, :], color=[colors[i] for i in ys[0, :] > xs[0, :]])
    axes[1].scatter(xs[1, :], ys[1, :], color=[colors[i] for i in ys[1, :] > xs[1, :]])

    min_val = np.min(np.concatenate([xs[0, :], ys[0, :]]))
    max_val = np.max(np.concatenate([xs[0, :], ys[0, :]]))
    axes[0].plot([min_val, max_val], [min_val, max_val], 'k-')
    axes[0].axis('equal')
    axes[0].set_aspect('equal', 'box')

    min_val = np.min(np.concatenate([xs[1, :], ys[1, :]]))
    max_val = np.max(np.concatenate([xs[1, :], ys[1, :]]))
    axes[1].plot([min_val, max_val], [min_val, max_val], 'k-')
    axes[1].axis('equal')
    axes[1].set_aspect('equal', 'box')

    axes[0].set_ylabel(labels[0])
    axes[0].set_xlabel(labels[1])
    axes[1].set_xlabel(labels[1])

    axes[0].set_title('Reward')
    axes[1].set_title('Ideal Episodes')

    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)

    fig.tight_layout()
    plt.savefig('Plots/PairwiseComparisonPlot.pdf')
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(4, 2))
    axes[0].pie([np.sum(ys[0, :] > xs[0, :]), np.sum(ys[0, :] < xs[0, :])], colors=reversed(colors))
    axes[1].pie([np.sum(ys[1, :] > xs[1, :]), np.sum(ys[1, :] < xs[1, :])], colors=reversed(colors))
    fig.tight_layout()
    plt.savefig('Plots/PairwisePieChart.pdf')
    plt.close()

    return


def PlotMeanSOMLocations(root_dir, df):

    vals = df['type'].values
    vals = np.unique(vals)
    num_plots = vals.shape[0]

    mazes = [[] for i in range(num_plots)]

    for type, directory in zip(df['type'], df['dir']):
        som_locations = np.load(root_dir + directory + '/SOMLocations.npy')

        p = np.where(vals == type)[0][0]
        mazes[p].append(som_locations)

    for i in range(num_plots):
        plt.figure()
        plt.imshow(np.mean(mazes[i], axis=0))#, cmap='plasma')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig('Plots/MeanSOMLocations' + str(i) + '.pdf')
        plt.close()



    return


def PlotRevaluationComparisons(data_frames, labels):

    start = 0
    end = 1000

    var = 'type'
    vals = np.array([])
    for df in data_frames:
        vals = np.concatenate([vals, df[var].values])

    vals = np.unique(vals)
    num_mazes = vals.shape[0]

    f, a = plt.subplots(3, figsize=(6, 6))

    a[2].set_xlabel('Episode')
    a[0].set_ylabel('Episode Length')
    a[1].set_ylabel('Reward')
    a[2].set_ylabel('Ideal Episodes')
    a[0].set_xticks([])
    a[1].set_xticks([])

    a[2].spines['top'].set_visible(False)
    a[2].spines['right'].set_visible(False)
    a[0].spines['top'].set_visible(False)
    a[0].spines['right'].set_visible(False)
    a[0].spines['bottom'].set_visible(False)
    a[1].spines['top'].set_visible(False)
    a[1].spines['right'].set_visible(False)
    a[1].spines['bottom'].set_visible(False)

    a[0].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    a[1].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

    colors = ['b', 'r', 'g', 'k', 'c', 'm']

    for df, label, color in zip(data_frames, labels, colors):

        length_results = [[] for i in range(num_mazes)]
        reward_results = [[] for i in range(num_mazes)]
        ideal_results = [[] for i in range(num_mazes)]

        for v, rewards, lengths, maze in zip(df[var], df['rewards'], df['lengths'], df['maze']):
            p = np.where(vals == v)[0][0]

            length_results[p].append(np.cumsum(lengths))
            reward_results[p].append(np.cumsum(rewards))
            ideal_results[p].append(np.cumsum(np.array(rewards) == 1))

        for p in range(num_mazes):

            num_trials = df['num_trials'][0]

            if(p != 0):
                y = np.array(length_results[p]) + np.expand_dims(np.array(length_results[p - 1])[:, -1], axis=-1)
            else:
                y = length_results[p]

            error = np.std(y, axis=0)
            y = np.mean(y, axis=0)
            x = np.arange(y.shape[0]) + (p * num_trials)

            a[0].plot(x, y, color=color)
            a[0].fill_between(x, y - error, y + error, color=color, alpha=.25)

            if (p != 0):
                y = np.array(reward_results[p]) + np.expand_dims(np.array(reward_results[p - 1])[:, -1], axis=-1)
            else:
                y = reward_results[p]

            error = np.std(y, axis=0)
            y = np.mean(y, axis=0)
            x = np.arange(y.shape[0]) + (p * num_trials)

            a[1].plot(x, y, color=color)
            a[1].fill_between(x, y - error, y + error, color=color, alpha=.25)

            if (p != 0):
                y = np.array(ideal_results[p]) + np.expand_dims(np.array(ideal_results[p-1])[:, -1], axis=-1)
            else:
                y = ideal_results[p]

            error = np.std(y, axis=0)
            y = np.mean(y, axis=0)
            x = np.arange(y.shape[0]) + (p * num_trials)

            if(p==0):
                a[2].plot(x, y, label=label, color=color)
            else:
                a[2].plot(x, y, color=color)
                a[0].axvline(p * num_trials, color='k', linestyle='--', linewidth=2)
                a[1].axvline(p * num_trials, color='k', linestyle='--', linewidth=2)
                a[2].axvline(p * num_trials, color='k', linestyle='--', linewidth=2)

            a[2].fill_between(x, y - error, y + error, color=color, alpha=.25)

    for axis in a:
        axis.set_xlim([start, end])

    for s in a.ravel():
        s.legend()

    f.tight_layout()
    f.savefig('Plots/RevaluationComparisonPlot.pdf')
    plt.close(f)

    return


def PlotAllExplanations(maze, memories_list, actions_list, weights_list, training_rewards, training_lengths,
                        test_rewards, save_name, best_agent):

    rows = 7
    cols = 3

    fig = plt.figure(figsize=(8, 10))
    gs = GridSpec(rows, cols, figure=fig)
    colours = sns.color_palette('colorblind', len(training_rewards))

    axes = []
    for row in range(rows-1):
        for col in range(cols):
            axes.append(fig.add_subplot(gs[row, col]))

    for i, ax in enumerate(axes):
        if i == best_agent:
            color = (1,0,0)
        else:
            color = (.5,.5,.5)
        ax.set_title('Agent: ' + str(i + 1))
        ax.tick_params(color=color, labelcolor=color)
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(5)
        ax.set_xticks([])
        ax.set_yticks([])

    titles = ['Training Reward', 'Training Episode Length', 'Test Reward']
    x_labels = ['Episode', 'Episode', 'Agent']
    x_ticks = [np.arange(0, 1000, 250), np.arange(0, 1000, 250), np.arange(1, 13)]
    for i, title in enumerate(titles):
        axes.append(fig.add_subplot(gs[int(rows-1), i]))
        axes[-1].spines['top'].set_visible(False)
        axes[-1].spines['right'].set_visible(False)
        axes[-1].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        axes[-1].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        axes[-1].set_yscale('symlog')
        axes[-1].set_title(title)
        axes[-1].set_xlabel(x_labels[i])
        axes[-1].set_xticks(x_ticks[i])

    axes[-1].tick_params(axis='x', labelsize=8)

    for ax, memories, actions, weights in zip(axes, memories_list, actions_list, weights_list):
        ax.imshow(maze)
        cmap = plt.cm.get_cmap('hot')
        norm = colors.Normalize(vmin=0, vmax=1)

        for memory, action, weight in zip(memories, actions, weights):
            ax.plot(memory[1], memory[0], marker='*', color=cmap(norm(weight)))
            ax.text(memory[1], memory[0] + .5, action, ha='center')

    for i, (training_reward, training_length, test_reward) in enumerate(zip(training_rewards,
                                                                            training_lengths,
                                                                            test_rewards)):
        if i == best_agent:
            color = (1,0,0)
        else:
            color = (.5,.5,.5)
        training_reward = np.concatenate([[0], training_reward])
        training_reward = np.mean(np.reshape(training_reward, (10, 100)), axis=-1)
        axes[-3].plot(np.arange(training_reward.shape[0]) * 100, training_reward, color=color)
        training_length = np.concatenate([[0], training_length])
        training_length = np.mean(np.reshape(training_length, (10, 100)), axis=-1)
        axes[-2].plot(np.arange(training_length.shape[0]) * 100, training_length, color=color)
        axes[-1].bar(i + 1, test_reward, color=color)

    # plt.tight_layout()
    fig.savefig(save_name)
    plt.close()

    return



def PlotExplanation(maze, memories, actions, weights, observations, save_name, colour_key = "Weighting"):

    fig, ax = plt.subplots(1)
    ax.imshow(maze)
    cmap = plt.get_cmap('hot')
    norm = colors.Normalize(vmin=0, vmax=1)
    actions = np.array(actions, dtype='O')    

    for i in range(len(actions)):
        if memories[i][0] == observations[i][0] and memories[i][1] == observations[i][1]:
            actions[i] = '(' + actions[i] + ')'

    for memory, action, weight in zip(memories, actions, weights):
        ax.plot(memory[1], memory[0], marker='*', color=cmap(norm(weight)))
        ax.text(memory[1], memory[0] + .5, action, ha='center')

    # Add a color bar to explain the weight mapping
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # Required for the colorbar to work properly
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label(colour_key)  # Label for the color bar

    # Save the figure
    fig.savefig(save_name)
    plt.close()

    return

def PlotAllExplanations(maze, memories, actions, weights, save_name, colour_key = "Q-Value", minVal=None, maxVal=None):  

    fig, ax = plt.subplots(1)
    ax.imshow(maze)

    # Set up a diverging colormap that works well with values from -1 to 1
    min_weight = min(weights)
    if minVal is not None:
        min_weight = min(min_weight, minVal)
    max_weight = max(weights)
    if maxVal is not None:
        max_weight = max(max_weight, maxVal)
    cmap = plt.get_cmap('hot')
    norm = colors.Normalize(vmin=min_weight, vmax=max_weight)
    
    for memory, action, weight in zip(memories, actions, weights):
        ax.plot(memory[1], memory[0], marker='*', color=cmap(norm(weight)))
        ax.text(memory[1], memory[0] + .5, action, ha='center', color='white')

    # Add a color bar to explain the weight mapping
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # Required for the colorbar to work properly
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label(colour_key)  # Label for the color bar

    # Save the figure
    fig.savefig(save_name)
    plt.close()

    return min_weight, max_weight

    fig.savefig(save_name)
    plt.close()

    return

def ShowExplanation(maze, memories, actions, weights, observations, title=None):

    fig, ax = plt.subplots(1)
    ax.imshow(maze)
    cmap = plt.get_cmap('hot')
    norm = colors.Normalize(vmin=0, vmax=1)
    actions = np.array(actions, dtype='O')    

    for i in range(len(actions)):
        if memories[i][0] == observations[i][0] and memories[i][1] == observations[i][1]:
            actions[i] = '(' + actions[i] + ')'

    for memory, action, weight in zip(memories, actions, weights):
        ax.plot(memory[1], memory[0], marker='*', color=cmap(norm(weight)))
        ax.text(memory[1], memory[0] + .5, action, ha='center')

    # # Get the current figure
    # fig = plt.gcf()

    # Set the window title (for Matplotlib versions < 3.4)
    if title is not None:
        # fig.canvas.set_window_title(title)
        plt.title(title)
    plt.show()

    return

def ShowDQN(maze, actions, DQN, title=None, save_name=None, minVal=None, maxVal=None):
    if len(actions) > 100:
        return
    START_VAL=2
    actions_dict = {'up': [-1, 0], 'down': [1, 0], 'left': [0, -1], 'right': [0, 1]}
    actions_dict_name = {0: 'up', 1: 'down', 2: 'left', 3: 'right'}

    startState = np.squeeze(np.array(np.where(maze == START_VAL)))
    observations = []
    qActions = []
    qVals = []
    state = startState
    for i in range(0, len(actions)):
        observations.append(state)
        dqnVals = DQN[i]
        max_q = np.max(dqnVals)
        # Get the index of the maximum value
        qAction = np.argmax(dqnVals)
        qVals.append(max_q)
        qActions.append(actions_dict_name[qAction])
        state = state + actions_dict[actions[i]]


    fig, ax = plt.subplots(1)
    ax.imshow(maze)
    cmap = plt.get_cmap('hot')
    if minVal is None:
        min_weight = min(qVals)
    else:
        min_weight = minVal
    if maxVal is None:
        max_weight = max(qVals)
    else:
        max_weight = maxVal
    norm = colors.Normalize(vmin=min_weight, vmax=max_weight)

    for memory, action, weight in zip(observations, qActions, qVals):
        ax.plot(memory[1], memory[0], marker='*', color=cmap(norm(weight)))
        ax.text(memory[1], memory[0] + .5, action, ha='center')

    # Add a color bar to explain the weight mapping
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # Required for the colorbar to work properly
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("Q-Value")  # Label for the color bar
    # # Get the current figure
    # fig = plt.gcf()

    # Set the window title (for Matplotlib versions < 3.4)
    if title is not None:
        # fig.canvas.set_window_title(title)
        plt.title(title)
    if save_name is not None:
        fig.savefig(save_name)
        plt.close()
    else:
        plt.show()

    return


def PlotExplanationResults(var, data_frames, best_agent):

    f, a = plt.subplots(2, figsize=(3, 4))

    a[1].set_xlabel('Episode')

    # a[2].set_xlabel('Episode')
    a[0].set_ylabel('Episode Length')
    a[1].set_ylabel('Reward')
    # a[2].set_ylabel('Ideal Episodes')
    a[0].set_xticks([])
    # a[1].set_xticks([])

    # a[2].spines['top'].set_visible(False)
    # a[2].spines['right'].set_visible(False)
    a[0].spines['top'].set_visible(False)
    a[0].spines['right'].set_visible(False)
    a[0].spines['bottom'].set_visible(False)
    a[1].spines['top'].set_visible(False)
    a[1].spines['right'].set_visible(False)
    # a[1].spines['bottom'].set_visible(False)

    a[0].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    a[1].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

    colors = ['b', 'r', 'g', 'k', 'c', 'm']

    for df, color in zip(data_frames, colors):

        length_results = []
        reward_results = []
        ideal_results = []

        for rewards, lengths, maze in zip(df['rewards'], df['lengths'], df['maze']):

            length_results.append(np.cumsum(lengths))
            reward_results.append(np.cumsum(rewards))
            ideal_results.append(np.cumsum(np.array(rewards) == 1))

        y = np.mean(length_results, axis=0)
        x = np.arange(y.shape[0])
        error = np.std(length_results, axis=0)

        a[0].plot(x, y, color=color)
        a[0].fill_between(x, y-error, y+error, color=color, alpha=.25)
        a[0].plot(x, np.array(length_results)[best_agent, :], color=color, linestyle=':')

        y = np.mean(reward_results, axis=0)
        x = np.arange(y.shape[0])
        error = np.std(reward_results, axis=0)

        a[1].plot(x, y, color=color)
        a[1].fill_between(x, y - error, y + error, color=color, alpha=.25)
        a[1].plot(x, np.array(reward_results)[best_agent, :], color=color, linestyle=':')

        # y = np.mean(ideal_results, axis=0)
        # x = np.arange(y.shape[0])
        # error = np.std(ideal_results, axis=0)
        #
        # a[2].plot(x, y, color=color)
        # a[2].fill_between(x, y - error, y + error, color=color, alpha=.25)
        # a[2].plot(x, np.array(ideal_results)[best_agent, :], color=color, linestyle=':')

    f.tight_layout()
    f.savefig('Plots/Explanation_Results_' + str(var) + '.pdf')
    plt.close(f)

    return

def PlotExplanationHeatMap(name, maze, weights, memories, actions):

    heat_map = np.zeros(maze.shape)

    for w, m, a in zip(weights, memories, actions):

        w = np.array(w)
        m = np.array(m)
        a = np.array(a)

        binned_memories = np.rint(m)
        heat_map[binned_memories[:, 0].astype(int), binned_memories[:, 1].astype(int)] += 1

    plt.figure()
    plt.imshow(maze, alpha=.5)
    plt.imshow(heat_map, cmap='hot')
    plt.colorbar()
    plt.savefig('Plots/Explanation_Heat_Map' + str(name) + '.pdf')
    plt.close()

    return
