from external.inhouse_model import InhouseModel
from agents.model_settings import ModelSettings



MODEL_NAME = "vnptai-hackathon-small"

inhouse = InhouseModel(
    model=MODEL_NAME
)
model_setting = ModelSettings(
    temperature=0,
)