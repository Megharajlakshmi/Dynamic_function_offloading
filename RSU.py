import math
from Config import Config
from scipy.integrate import quad
import random

class RSU:
    def __init__(self, vehicle, relative_x, relative_y=0):

        # New: RSU position is set **relative to vehicle’s coordinates**
        self.x_position = vehicle.x_position + relative_x
        self.y_position = vehicle.y_position + relative_y

        self.freq = Config.RSU_FREQUENCY
        self.power = Config.RSU_POWER
        self.radius = Config.RSU_RADIUS
        self.loadfactor = round(random.uniform(0, 11)) #0  # New: Tracks RSU load dynamically
        # print("RSU-----",self.loadfactor)
        self.height = Config.RSU_HEIGHT
        self.stay_dist = 2 * math.sqrt(self.radius**2 - self.height**2)
        self.bandwidth = Config.BANDWIDTH
        self.pathloss_exponent = Config.PATH_LOSS_EXPONENT
        self.rayleigh_fading_channel = Config.RAYLEIGH_FADING_CHANNEL
        self.noise = Config.NOISE

        # New: Precision handling for RSU position
        self.precision_error = Config.RSU_PRECISION_ERROR

    def compDelay(self, task_size):
        return task_size / (self.freq / Config.RSU_CPI)

    def isVehicleConnected(self, vehicle):
        dist = self.calculateDistance(vehicle)
        flag = dist <= (self.radius * 1.1)  # Allow slight buffer
        return flag
        

    def calculateDistance(self, vehicle):
        return math.sqrt((vehicle.x_position - self.x_position) ** 2 + (vehicle.y_position - self.y_position) ** 2)

    def commDelay(self, task_size, stayTime, vehicleSpeed, vehiclePower):
        avg_rate, _ = quad(self.transRate, 0, stayTime, args=(vehicleSpeed, vehiclePower))
        avg_rate /= stayTime
        return task_size / avg_rate

    def transRate(self, t, vehicleSpeed, vehiclePower):
        d = math.sqrt(self.height ** 2 + (self.stay_dist / 2 - vehicleSpeed * t) ** 2)
        r = self.bandwidth * math.log2(
            1 + ((vehiclePower * (d ** self.pathloss_exponent) * (self.rayleigh_fading_channel ** 2)) / self.noise))
        return r
    
    def compute_energy(self, task_size, stayTime, vehicleSpeed, vehiclePower):
        return self.power * (self.compDelay(task_size) + self.commDelay(task_size, stayTime, vehicleSpeed, vehiclePower) + Config.LATENCY)