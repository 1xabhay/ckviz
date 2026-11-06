"""A plugin that tries to replace the built-in `filter` stage."""
from pydantic import BaseModel

from ckviz.stages import stage


@stage(BaseModel)
def filter(f, c):
    return f, {}
