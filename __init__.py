import fiftyone as fo
import fiftyone.operators as foo
from fiftyone.operators import types
import fiftyone.brain as fob


class DimensionalityReduction(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="dimensionality_reduction",
            label="Dimensionality reduction",
            description="Performs dimensionality reduction algorithm (UMAP) on selected subset of samples in FiftyOne",
            allow_delegated_execution=True,
            allow_immediate_execution=True,
            default_choice_to_delegated=True,
        )
    
    def resolve_input(self, ctx):
        inputs = types.Object()
        prop = inputs.str(
            "embeddings_field",
            required=True,
            label="Pre-computed embeddings field name",
        )
        view = types.View(label="Compute visualization")
    
        return types.Property(inputs, view=view)
    
    def resolve_placement(self, ctx):
        return types.Placement(
            types.Places.SAMPLES_GRID_ACTIONS,
            types.Button(
                label="Filter by float field",
                icon="insights",
                prompt=True,
            ),
        )
 
    def execute(self, ctx):
        view = ctx.dataset.view()
        embeddings_field = ctx.params["embeddings_field"]
        tag_names = view.distinct("tags")
        brain_key = "tags_" + ("_").join(tag_names) + "_dimensionality_reduction"
        fob.compute_visualization(
            view,
            embeddings=embeddings_field,
            method="umap",
            brain_key=brain_key,
            create_index=True,
            num_dims=2,
            overwrite=True,
        )
    
        return {}


def register(plugin):
    plugin.register(DimensionalityReduction)
