from external.inhouse_model import InhouseModel
from agents.model_settings import ModelSettings



MODEL_NAME = "vnptai-hackathon-large"
MODEL_NAME_FORMAT = "vnptai-hackathon-small"
# MODEL_NAME = "Qwen3-32B"


inhouse = InhouseModel(
    model=MODEL_NAME
)
inhouse_format = InhouseModel(
    model=MODEL_NAME_FORMAT
)
model_setting = ModelSettings(
    temperature=0,
)