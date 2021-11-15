import numpy as np
import pandas as pd
from itertools import combinations


data = np.array([
    [1, 0, 1, 1, 0],
    [1, 1, 0, 1, 0],
    [0, 0, 1, 1, 0],
    [0, 1, 0, 1, 0],
    [1, 0, 1, 0, 1],
    [0, 1, 1, 0, 0],
])

def distance(x1, x2):
    stack = np.column_stack((x1, x2))
    n11, n10, n01, n00 = 0,0,0,0
    for pair in stack:
        if pair[0] == 1 and pair[1] == 1: n11 += 1
        if pair[0] == 1 and pair[1] == 0: n10 += 1
        if pair[0] == 0 and pair[1] == 1: n01 += 1
        if pair[0] == 0 and pair[1] == 0: n00 += 1

    return (n11, n10, n01, n00)

def clustering(data, distance_funtion, linkage_function):
    clusters = [[i] for i in range(len(data))]
    dendogram = {}
    while len(clusters) > 1:
        min_distance = np.inf
        min_x_index, min_y_index = None, None
        for x_index, y_index in combinations(clusters, 2):
            distance = linkage_function(
                distance_funtion,
                list(map(lambda i: data[i], x_index)),
                list(map(lambda i: data[i], y_index)),
            )
            if distance < min_distance:
                min_distance = distance
                min_x_index, min_y_index = x_index, y_index

        clusters.remove(min_x_index)
        clusters.remove(min_y_index)
        clusters.append(min_x_index + min_y_index)
        dendogram[frozenset([frozenset(min_x_index), frozenset(min_y_index)])] = min_distance

    return dendogram

def single_linkage(distance_function, cluster1, cluster2):
    return min([distance_function(x, y) for x in cluster1 for y in cluster2])

def complete_linkage(distance_function, cluster1, cluster2):
    return max([distance_function(x, y) for x in cluster1 for y in cluster2])

def group_average_linkage(distance_function, cluster1, cluster2):
    return (
        sum([distance_function(x, y) for x in cluster1 for y in cluster2]) /
        (len(list(cluster1))*len(list(cluster2)))
    )


def SMC(x1, x2):
    n11, n10, n01, n00 = distance(x1, x2)
    return (n11 + n00) / (n11 + n10 + n01 + n00)

def RC(x1, x2):
    n11, n10, n01, n00 = distance(x1, x2)
    return (n11) / (n11 + n10 + n01 + n00)

def JC(x1, x2):
    n11, n10, n01, n00 = distance(x1, x2)
    return (n11) / (n11 + n10 + n01)


for pair, dist  in clustering(data, RC, single_linkage).items():
    c1, c2 = pair
    inc = lambda x: str(x+1)
    print(f"[{' '.join(map(inc, c1))}], [{' '.join(map(inc, c2))}]::{dist}")
print("")

for pair, dist  in clustering(data, SMC, complete_linkage).items():
    c1, c2 = pair
    inc = lambda x: str(x+1)
    print(f"[{' '.join(map(inc, c1))}], [{' '.join(map(inc, c2))}]::{dist}")
print("")

for pair, dist  in clustering(data, JC, group_average_linkage).items():
    c1, c2 = pair
    inc = lambda x: str(x+1)
    print(f"[{' '.join(map(inc, c1))}], [{' '.join(map(inc, c2))}]::{dist}")
print("")

print("SMC")
df = pd.DataFrame()
for i in range(len(data)):
    lengths = []
    for j in range(len(data)):
        lengths.append(SMC(data[i], data[j]))
    df = df.append(pd.Series(lengths), ignore_index=True)
print(df)

print("\nJC")
df = pd.DataFrame()
for i in range(len(data)):
    lengths = []
    for j in range(len(data)):
        lengths.append(JC(data[i], data[j]))
    df = df.append(pd.Series(lengths), ignore_index=True)
print(df.round(2))

print("\nRC")
df = pd.DataFrame()
for i in range(len(data)):
    lengths = []
    for j in range(len(data)):
        lengths.append(RC(data[i], data[j]))
    df = df.append(pd.Series(lengths), ignore_index=True)
print(df)
