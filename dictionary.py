class Parameters():
    
    def __init__(self):
        self.parameter = {}
        
    def add_val(self, name, value, description):
        self.parameter[name] = [value, description]
        
    def get_val(self, name):
        return  self.parameter[name][0]
    
    def get_description(self, name):
        return  self.parameter[name][1]

    def get_dict(self):
        return self.parameter


