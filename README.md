# Titanic

## Visualize Data

![Pclass_Sex_Embarked](.\images\Pclass_Sex_Embarked.png)

The possibility of surviving is strongly related to `Pclass`, `Sex`, and `Embarked`

## Error Analysis

Accuracy score of all three classifiers is around 80% By exploring the prediction results, it turns out that they make mistakes in the same instances.

![Error_Venn_Plot](.\images\error_venn_plot.png)

Distribution of errors compared with the whole dataset:

![two distributions](D:\Jupyter\Titanic\images\two_distributions.png)

The distribution seems to be reversed! It is true that the survival possibility of male passengers is lower that female. But the classifiers overdo this and resulted in misclassified some survived men.

 

