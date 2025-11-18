
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from itertools import product
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score

url = "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv"
df = pd.read_csv(url)

print("Original shape:", df.shape)
print(df.head(), "\n")

df['age_group'] = pd.cut(df['age'],
                         bins=[0, 30, 50, 100],
                         labels=['young', 'middle', 'old'])

df['children_count'] = pd.cut(df['children'],
                              bins=[-1, 1, 3, 10],
                              labels=['low', 'med', 'high'])

df['bmi_risk'] = pd.cut(df['bmi'],
                        bins=[0, 25, 30, 50],
                        labels=['low', 'med', 'high'])

df['recommend'] = (df['charges'] > df['charges'].median()).astype(int)

cols = ['age_group', 'sex', 'smoker', 'region',
        'children_count', 'bmi_risk', 'recommend']
data = df[cols].copy()

data['age_group']      = data['age_group'].fillna('young')
data['children_count'] = data['children_count'].fillna('low')
data['bmi_risk']       = data['bmi_risk'].fillna('low')

print("Cleaned data (first 5 rows):")
print(data.head(), "\n")

cat_maps = {
    'age_group'     : {'young':0, 'middle':1, 'old':2},
    'sex'           : {'female':0, 'male':1},
    'smoker'        : {'no':0, 'yes':1},
    'region'        : {'northeast':0, 'northwest':1, 'southeast':2, 'southwest':3},
    'children_count': {'low':0, 'med':1, 'high':2},
    'bmi_risk'      : {'low':0, 'med':1, 'high':2},
    'recommend'     : {0:0, 1:1}
}

data_int = data.copy()
for col, mapping in cat_maps.items():
    data_int[col] = data[col].map(mapping).fillna(0).astype(int)

print("Integer-encoded data (first 5 rows):")
print(data_int.head(), "\n")

train, test = train_test_split(data_int, test_size=0.2,
                               random_state=42, stratify=data_int['recommend'])


dag = nx.DiGraph()
nodes = ['age_group','sex','bmi_risk','smoker',
         'children_count','region','recommend']
dag.add_nodes_from(nodes)
edges = [
    ('age_group','smoker'), ('sex','smoker'), ('bmi_risk','smoker'),
    ('smoker','recommend'), ('children_count','recommend'), ('region','recommend')
]
dag.add_edges_from(edges)

states = {n: data_int[n].max() + 1 for n in nodes}

class SimpleEM:
    def _init_(self, dag, states, train_df, alpha=1.0):
        self.dag = dag
        self.states = states
        self.alpha = alpha
        self.cpts = self._init_cpts(train_df)
        self._em_fit(train_df, max_iter=20)

    def _init_cpts(self, df):
        cpts = {}
        for node in nx.topological_sort(self.dag):
            parents = list(self.dag.predecessors(node))
            if not parents:                                
                counts = np.bincount(df[node].astype(int),
                                    minlength=self.states[node]) + self.alpha
                cpts[node] = counts / counts.sum()
            else:
                p_dims = [self.states[p] for p in parents]
                counts = np.zeros((np.prod(p_dims), self.states[node]))
                for _, row in df.iterrows():
                    if all(pd.notna(row[p]) for p in parents):
                        par_idx = np.ravel_multi_index(
                            [int(row[p]) for p in parents], p_dims)
                        child_idx = int(row[node])
                        counts[par_idx, child_idx] += 1
                counts += self.alpha
                cpts[node] = counts / counts.sum(axis=1, keepdims=True)
        return cpts

    def _em_fit(self, df, max_iter=20):
        filled = df.copy()
        for it in range(max_iter):
            filled = self._fill_missings(filled)
            for node in nx.topological_sort(self.dag):
                parents = list(self.dag.predecessors(node))
                if not parents:
                    counts = np.bincount(filled[node].astype(int),
                                        minlength=self.states[node]) + self.alpha
                    self.cpts[node] = counts / counts.sum()
                else:
                    p_dims = [self.states[p] for p in parents]
                    counts = np.zeros((np.prod(p_dims), self.states[node]))
                    for _, row in filled.iterrows():
                        par_idx = np.ravel_multi_index(
                            [int(row[p]) for p in parents], p_dims)
                        child_idx = int(row[node])
                        counts[par_idx, child_idx] += 1
                    counts += self.alpha
                    self.cpts[node] = counts / counts.sum(axis=1, keepdims=True)
            if it % 5 == 0:
                print(f"EM iteration {it}")

    def _fill_missings(self, df):
        filled = df.copy()
        for node in nx.topological_sort(self.dag):
            parents = list(self.dag.predecessors(node))
            miss = filled[node].isna()
            if miss.any():
                for idx in filled[miss].index:
                    if all(pd.notna(filled.loc[idx, p]) for p in parents):
                        p_idx = np.ravel_multi_index(
                            [int(filled.loc[idx, p]) for p in parents],
                            [self.states[p] for p in parents])
                        probs = self.cpts[node][p_idx]
                        filled.loc[idx, node] = np.random.choice(
                            self.states[node], p=probs)
        return filled

em = SimpleEM(dag, states, train)

def marginal(em, target, evidence):
    """
    evidence: dict with *all parents* of every node (including the target)
    """
    unobserved = [n for n in nodes if n not in evidence and n != target]
    total = 0.0
    target_p = np.zeros(em.states[target])

    for combo in product(*(range(em.states[n]) for n in unobserved)):
        assign = dict(zip(unobserved, combo))
        assign.update(evidence)                     

        for n in nodes:
            if n not in assign:
                assign[n] = 0

        for t in range(em.states[target]):
            assign[target] = t
            p = 1.0
            for node in nx.topological_sort(dag):
                parents = list(dag.predecessors(node))
                if not parents:
                    p *= em.cpts[node][assign[node]]
                else:
                    p_idx = np.ravel_multi_index(
                        [assign[p] for p in parents],
                        [em.states[p] for p in parents])
                    p *= em.cpts[node][p_idx, assign[node]]
            target_p[t] += p
            total += p

    if total > 0:
        target_p /= total
    return target_p

ev_existing = {
    'age_group':1, 'sex':1, 'bmi_risk':2,
    'children_count':1, 'region':2, 'smoker':1
}
p = marginal(em, 'recommend', ev_existing)
print(f"\nExisting → P(Recommend=1) = {p[1]:.3f} → {'Recommend' if p[1]>0.5 else 'Skip'}")

def cold_start_ffnn(sparse):

    age_idx = cat_maps['age_group'][sparse['age_group']]
    reg_idx = cat_maps['region'][sparse['region']]

    x = np.array([age_idx, reg_idx], dtype=float)

    np.random.seed(0)
    w1 = np.random.randn(2,64)*0.1
    h1 = np.maximum(0, x @ w1)
    h1 *= np.random.binomial(1,0.8,64)

    w2 = np.random.randn(64,64)*0.1
    h2 = np.maximum(0, h1 @ w2)
    h2 *= np.random.binomial(1,0.8,64)

    w_out = np.random.randn(64)*0.1
    logit = h2 @ w_out
    p_smoker = 1/(1+np.exp(-logit))

    pseudo = 1 if p_smoker>0.5 else 0
    print(f"[FFNN] age={sparse['age_group']}, region={sparse['region']} → smoker={pseudo} (p={p_smoker:.2f})")
    full = {n:0 for n in nodes}
    full['age_group'] = age_idx
    full['region']    = reg_idx
    full['smoker']    = pseudo
    return full

prospect = {'age_group':'young', 'region':'northwest'}
prospect_full = cold_start_ffnn(prospect)
p_pro = marginal(em, 'recommend', prospect_full)
print(f"Prospect → P(Recommend=1) = {p_pro[1]:.3f} → {'Recommend' if p_pro[1]>0.5 else 'Skip'}")

X_test = test.drop(columns=['recommend'])
y_test = test['recommend']
preds = []
for _, row in X_test.iterrows():
    ev = row.to_dict()
    p = marginal(em, 'recommend', ev)
    preds.append(1 if p[1]>0.5 else 0)

prec = precision_score(y_test, preds)
print(f"\nPrecision@1 = {prec:.3f} (paper baseline ≈0.28 @3)")

plt.figure(figsize=(8,5))
pos = nx.spring_layout(dag)
nx.draw(dag, pos, with_labels=True, node_color='lightblue',
        node_size=2000, font_size=10, arrowsize=20)
plt.title("Hybrid BN – RecSys 2017 replica")
plt.show()
