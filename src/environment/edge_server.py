import random
class EdgeServer:
    def __init__(self, server_id, location, poi_list, neighbors, match_revenue, coverage_radius=200, capacity=1, price_per_unit=1):
        self.server_id = server_id
        self.location = location  # Assume location is a tuple of (x, y)
        self.capacity = capacity
        self.price_per_unit = price_per_unit
        # if random_rate>0:
        #     self.price_per_unit = price_per_unit * (1 + random.uniform(-random_rate, random_rate))
        # else:
        #     self.price_per_unit = price_per_unit
        self.coverage_radius = coverage_radius
        self.covered_pois = poi_list  # List of POIs covered by this server
        self.neighbors = neighbors  # List of neighbor servers
        self.match_revenue = match_revenue
        self.cover_revenue = 0
        self.shortest_hops = {}
        self.rho_pois = {}

    def add_covered_poi(self, poi):
        self.covered_pois.append(poi)

    def add_neighbor(self, neighbor):
        self.neighbors.append(neighbor)

    def __repr__(self):
        return f"EdgeServer(id={self.server_id}, location={self.location}, match_revenue={self.match_revenue}, cover_revenue={self.cover_revenue}, neighbors={self.neighbors}, shortest_hops={self.shortest_hops}, covered_pois={self.covered_pois})\n"