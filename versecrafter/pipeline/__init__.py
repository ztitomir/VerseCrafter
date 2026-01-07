from .pipeline_wan_versecrafter import WanVerseCrafterPipeline

import importlib.util

if importlib.util.find_spec("paifuser") is not None:
    # --------------------------------------------------------------- #
    #   Sparse Attention
    # --------------------------------------------------------------- #
    from paifuser.ops import sparse_reset

    WanVerseCrafterPipeline.__call__ = sparse_reset(WanVerseCrafterPipeline.__call__)
