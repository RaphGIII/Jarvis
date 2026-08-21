class Toggle:
    def __init__(self):
        self.on = False

    def flip(self):
        self.on = not self.on

    def state(self):
        return self.on

    def is_on(self):
        return self.state()
