from pydantic import BaseModel


class HousePrice(BaseModel):
    area:float
    bhk:float
    bathroom:float
    age:float
    status:float
    location:str
    builder:str

