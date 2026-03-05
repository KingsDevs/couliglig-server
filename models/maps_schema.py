from pydantic import BaseModel, Field

class MapConfigOut(BaseModel):
    id: int
    dock_config_id: int
    map_yaml_path: str
    map_image_path: str

    class Config:
        from_attributes = True