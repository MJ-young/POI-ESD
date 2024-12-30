class Service:
    def __init__(self, service_id,name, size, budget):
        self.service_id = service_id
        self.size = size
        self.budget = budget
        self.name = name

    def __repr__(self):
        return f"Service(id={self.service_id}, size={self.size}, budget={self.budget})"