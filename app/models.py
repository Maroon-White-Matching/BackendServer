from app import db

class Users(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100))
    username = db.Column(db.String(100))
    password = db.Column(db.String(100))
    role = db.Column(db.String(100))

    def __init__(self, name, username, password, role):
        self.name = name
        self.username = username
        self.password = password
        self.role = role

    def __repr__(self):
        return '<id {}>'.format(self.id)

    def serialize(self):
        return {
            'id': self.id,
            'name': self.name,
            'username': self.username,
            'password': self.password,
            'role': self.role
        }

class Result(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    coach = db.Column(db.String(100))
    fellow = db.Column(db.String(100))

    def init(self, id, coach, fellow):
        self.id = id
        self.coach = coach
        self.fellow = fellow

    def repr(self):
        return '<id {}>'.format(self.id)

    def serialize(self):
        return {
            'id': self.id,
            'coach': self.coach,
            'fellow': self.fellow
        }