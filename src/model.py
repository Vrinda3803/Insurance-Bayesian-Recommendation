import pandas as pd
from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination
from pgmpy.estimators import BayesianEstimator
import matplotlib.pyplot as plt
import networkx as nx

url = "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv"
df = pd.read_csv(url)

df['age_group'] = pd.cut(df['age'], bins=[0,30,50,100], labels=['young','middle','old'])
df['bmi_risk'] = pd.cut(df['bmi'], bins=[0,25,30,50], labels=['normal','overweight','obese'])
df['children_count'] = pd.cut(df['children'], bins=[-1,0,1,3,10], labels=['zero','one','two_three','many'])
df['high_charges'] = (df['charges'] > df['charges'].median()).astype(int)

data = df[['age_group','sex','smoker','region','children_count','bmi_risk','high_charges']]

model = BayesianNetwork([
    ('age_group', 'smoker'), ('sex', 'smoker'), ('bmi_risk', 'smoker'),
    ('smoker', 'high_charges'), ('children_count', 'high_charges'), ('region', 'high_charges')
])

model.fit(data, estimator=BayesianEstimator, prior_type='dirichlet', pseudo_counts=1)
infer = VariableElimination(model)

def predict(evidence):
    prob = infer.query(['high_charges'], evidence=evidence)
    p = prob.values[1]
    return "Recommend Premium" if p > 0.5 else "Standard Plan", round(p, 3)

print("High-risk customer:")
print(predict({'age_group':'middle','sex':'male','bmi_risk':'obese','smoker':'yes','region':'southeast','children_count':'zero'}))

print("\nYoung healthy prospect:")
print(predict({'age_group':'young','sex':'female','bmi_risk':'normal','smoker':'no','region':'northwest','children_count':'one'}))

plt.figure(figsize=(10,7))
nx.draw(model, with_labels=True, node_color='lightcoral', node_size=4000, font_weight='bold', arrowsize=20)
plt.title("Insurance Recommendation Bayesian Network")
plt.savefig("figures/bn_structure.png")
plt.show()
