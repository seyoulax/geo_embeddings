# <p align='center'>Geo-embeddings [(Большие Вызовы 2024)](https://konkurs.sochisirius.ru/)</h1>

### Main goals:
- **Firstly**(business part), we need to solve business problem by answering the following question: "Where should I open new business point?" to maximize revenue.
- **Secondly**(research part) and mainly, we should find out if GNNs(Graph Neural Networks) are better for solving this task

### Data(proprietary):
- Transactions on different business points
- Demographic features
- Raster images
- Graph with business points and connections between them (`u and w connected if there is a road between them`)

### Keywords and used techniques:
- **GAT**, **GCN** for semi-supervised learning on node-level property prediction
- **GraphSage** and **GraphInfoMax** for unsupervised learning
- **Baseline**: Boosting and other tabular data models
- **Raster CNN**: adding extra knowledge to GNN

### Publications
- [**Habr**](https://habr.com/ru/companies/vtb/articles/847998/)
- [**Contest publication**](https://bigchallenges.ru/projects2024/bigdata_4)
