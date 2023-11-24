from pydantic import BaseModel


class Point(BaseModel):
    x: float
    y: float


class Square(BaseModel):
    topLeft: Point
    bottomRight: Point
