from mlops_artifacts.validate import validate_manifest_dir

__all__ = ["pack_model_artifact", "validate_manifest_dir"]


def pack_model_artifact(*args, **kwargs):
    # Lazy import to avoid importing torch in code paths that only need manifest validation.
    from mlops_artifacts.pack import pack_model_artifact as _pack_model_artifact

    return _pack_model_artifact(*args, **kwargs)
