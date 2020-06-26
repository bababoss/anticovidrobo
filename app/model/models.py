from app import db

#this is our base class, other classes can inherit from this

class Base(db.Model):
    __abstract__  = True

    id            = db.Column(db.Integer, primary_key=True)
    date_created  = db.Column(db.DateTime,  default=db.func.current_timestamp())
    # date_modified = db.Column(db.DateTime,  default=db.func.current_timestamp(),
    #                                        onupdate=db.func.current_timestamp())

class Covidinfo(Base):
    __tablename__ = 'covidinfo_table'

    mask_on =  db.Column(db.Boolean)
    temp_info = db.Column(db.Integer)
    audio_info = db.Column(db.JSON)
    screening_result = db.Column(db.String)

    def __init__(self, mask_on, temp_info, audio_info, screening_result):

        self.mask_on = mask_on
        self.temp_info = temp_info
        self.audio_info = audio_info
        self.screening_result = screening_result

    def __repr__(self):
        return '<User %r>' % (self.mask_on)

    def serialize(self):
        return {
            'id': self.id,
            'mask_on': self.mask_on,
            'temp_info': self.temp_info,
            'audio_info': self.audio_info,
            'screening_result': self.screening_result
        }