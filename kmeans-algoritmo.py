import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

numero_clusters = 3

######## KMEANS ##########
df = pd.read_excel('amostra.xlsx', sheet_name='Amostra')

features = ['n° de pessoas desejando ler', 'n° de pessoas lendo', 'n° de avaliações']
data = df[features]

scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

kmeans = KMeans(n_clusters=numero_clusters, random_state=42)
kmeans.fit(data_scaled)

df['Cluster'] = kmeans.labels_

output_file_name = 'saida_kmeans.xlsx'
df.to_excel(output_file_name, index=False)


##########################

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

coluna_x = 'n° de pessoas desejando ler'
coluna_y = 'n° de pessoas lendo'
coluna_z = 'n° de avaliações'

x = df[coluna_x]
y = df[coluna_y]
z = df[coluna_z]
clusters = df['Cluster']

scatter = ax.scatter(x, y, z, c=clusters, cmap='viridis')
ax.set_title('Clusterização K-means')
ax.set_xlabel(coluna_x)
ax.set_ylabel(coluna_y)
ax.set_zlabel(coluna_z)

legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
ax.add_artist(legend1)
plt.show()
