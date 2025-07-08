

# Dimensionality Reduction
FiftyOne grid plugin to perform dimensionality reduction on a subset and compute visualization
![](https://github.com/prernadh/dimensionality_reduction/blob/main/plugin_in_action.gif)

## Installation

```shell
fiftyone plugins download https://github.com/prernadh/dimensionality_reduction
```

Refer to the [main README](https://github.com/voxel51/fiftyone-plugins) for
more information about managing downloaded plugins and developing plugins
locally.

## Run Example

After installing this plugin, you can try the example yourself on the `quickstart` dataset.
```python
import fiftyone as fo
import fiftyone.zoo as foz

dataset = foz.load_zoo_dataset("quickstart")
model = foz.load_zoo_model("clip-vit-base32-torch")
view = dataset.limit(4)
dataset.save_view("4_images", view)

embeddings = view.compute_embeddings(
    model=model,
    embeddings_field="clip_embeddings"
)

session = fo.launch_app(dataset)
```

Select the `4_images` saved view from the View dropdown on the left of the App. Click on the `Insights` icon in the sample grid -> pass in the embeddings field name (`clip_embeddings`) and click execute. (We recommend that you schedule delegation as dimensionality reduction with UMAP scales with the number of samples in your view.)
Finally you can open the embeddings panel and see the generated embedding visualization.
