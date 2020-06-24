from app import db

#this is our base class, other classes can inherit from this

class Base(db.Model):
    __abstract__  = True

    id            = db.Column(db.Integer, primary_key=True)
    date_created  = db.Column(db.DateTime,  default=db.func.current_timestamp())
    date_modified = db.Column(db.DateTime,  default=db.func.current_timestamp(),
                                           onupdate=db.func.current_timestamp())

class Covidinfo(Base):
    __tablename__ = 'covidinfo_table'

    is_human =  db.Column(db.Boolean)
    mask_on =  db.Column(db.Boolean)
    temp_info = db.Column(db.Integer)
    screening_info = db.Column(db.JSON)

    def __init__(id, is_human, mask_on, temp_info, screening_info):
        self.is_human = is_human
        self.mask_on = mask_on
        self.temp_info = temp_info
        self.screening_info = screening_info

    def __repr__(self):
        return '<User %r>' % (self.is_human)

    def serialize(self):
        return {
            'id': self.id,
            'is_human': self.is_human,
            'mask_on': self.mask_on,
            'temp_info': self.temp_info,
            'screening_info': self.screening_info
        }