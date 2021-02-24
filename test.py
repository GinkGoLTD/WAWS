class Father(object):
    def __init__(self, name, age):
        self.name = name
        self.age = age
    
    @staticmethod
    def change_age(newage):
        return newage
    
    def change(self, newage):
        self.age = Father.change_age(newage)


if __name__ == "__main__":
    me = Father("ta", 30)
    print(me.age)
    me.change(20)
    print(me.age)