from .image_nodes import JFANode

NODE_CLASS_MAPPINGS = {
    "JFANode": JFANode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JFANode": "JFA",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
