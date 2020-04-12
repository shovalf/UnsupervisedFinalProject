import data_preperation
import check_normal_distribution
import clustering
import conclusion_clustering
import dimension_reduction
import features_distribution
import isolation_forest
import labels


data_preperation.main()
features_distribution.main()
isolation_forest.main()
check_normal_distribution.main()
points_pca, points_tsne, points_2_dim, X_scaled = dimension_reduction.main(2)
clustering.main(points_pca, points_tsne, points_2_dim, X_scaled)
labels.main('PCA', 1)
labels.main('PCA', 2)
labels.main('PCA', 3)
labels.main('ae', 1)
labels.main('ae', 2)
labels.main('ae', 3)
conclusion_clustering.main()
