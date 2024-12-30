class POI:
    def __init__(self, poi_id, location, correlation):
        self.poi_id = poi_id
        self.location = location  # Assume location is a tuple of (x, y)
        self.correlation = correlation
        self.covered_by = []  # List of servers covering this POI
        self.shortest_hops = {}

    def add_covered_by(self, server):
        self.covered_by.append(server)

    def __repr__(self):
        return f"POI(id={self.poi_id}, location={self.location}, correlation={self.correlation}, covered_by={self.covered_by}, shortest_hops={self.shortest_hops})\n"