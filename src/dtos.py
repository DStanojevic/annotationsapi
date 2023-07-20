from pydantic import BaseModel


class Point(BaseModel):
    x: int
    y: int


class Square(BaseModel):
    topLeft: Point
    bottomRight: Point
