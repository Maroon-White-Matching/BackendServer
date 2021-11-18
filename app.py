from app import app
from app.models import Users
from app import db

# delete
# duser = Users.query.get(5)
# db.session.delete(duser)
# db.session.commit()

# add
# db.session.add(Users(name='test',username='test',password='test', role='test'))
# db.session.commit()

# update
# user = Users.query.get(5)
# user.name = 'New Name'
# db.session.commit()

#result = SomeModel.query.with_entities(SomeModel.col1, SomeModel.col2)


# users = Users.query.all()
# for row in users:
#     print (row.name)